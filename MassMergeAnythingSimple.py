import json
import os
import shutil

import torch
import gc
from collections import OrderedDict
from typing import TYPE_CHECKING, Union, List

from torch.nn import functional as F

import yaml
from safetensors.torch import save_model, save_file, load_file

from jobs.process import BaseExtensionProcess
from toolkit.basic import value_map
from toolkit.config_modules import ModelConfig
from toolkit.kohya_model_util import load_vae
from toolkit.metadata import get_meta_for_safetensors, load_metadata_from_safetensors
from toolkit.paths import get_path
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.train_tools import get_torch_dtype
from tqdm import tqdm
import numpy as np
from tinydb import TinyDB, Query

from .tools import get_hash_from_dict

# Type check imports. Prevents circular imports
if TYPE_CHECKING:
    from jobs import ExtensionJob


# extend standard config classes to add weight
class ModelInputConfig(ModelConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight = kwargs.get('weight', 1.0)
        # overwrite default dtype unless user specifies otherwise
        # float 32 will give up better precision on the merging functions
        self.dtype: str = kwargs.get('dtype', 'float32')


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class ModelToMergeConfig:
    def __init__(self, **kwargs):
        self.name_or_path: str = kwargs.get('path', None)
        self.name_or_path: str = kwargs.get('name_or_path', self.name_or_path)

        self.weight: float = kwargs.get('weight', 1.0)
        mappings = kwargs.get('mappings', {})
        mapping_str = kwargs.get('mapping', None)
        if mapping_str and mapping_str in mappings:
            self.keymap = mappings[mapping_str]
        else:
            self.keymap = None

    def get_key(self, orig_key):
        if self.keymap is None:
            return orig_key
        else:
            my_key = orig_key
            for map in self.keymap:
                [search, replace] = map
                my_key = my_key.replace(search, replace)
            return my_key


# this is our main class process
class MassMergeAnythingSimple(BaseExtensionProcess):

    def __init__(
            self,
            process_id: int,
            job: 'ExtensionJob',
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)
        self.save_path = get_path(self.get_conf('save_path', required=True))
        self.working_dir = get_path(self.get_conf('working_dir', required=True))
        self.save_dtype = self.get_conf('save_dtype', default='float16')
        self.device = self.get_conf('device', default='cpu')
        self.report_format = self.get_conf('report_format', default='json')
        self.keep_cache = self.get_conf('keep_cache', default=False, as_type=bool)
        self.subtract_base = self.get_conf('subtract_base', default=False, as_type=bool)
        self.base_layers_to_keep = self.get_conf('base_layers_to_keep', default=[], as_type=list)
        self.base_layers_to_keep_begins_with = self.get_conf('base_layers_to_keep_begins_with', default=[], as_type=list)
        self.base_alpha = self.get_conf('base_alpha', default=0.0, as_type=float)
        self.mappings = self.get_conf('mappings', default={}, as_type=dict)
        self.use_base_meta = self.get_conf('use_base_meta', default=False, as_type=bool)
        if self.report_format == 'null':
            self.report_format = None
        self.db: TinyDB = TinyDB(os.path.join(os.path.dirname(__file__), 'db.json'))

        self.cache_dtype = self.get_conf('cache_dtype', default='float32', as_type=get_torch_dtype)
        # merge_step = self.get_conf('merge_step', default=1, as_type=int)

        self.weight_json = os.path.join(self.working_dir, 'weight.json')
        self.base_model = self.get_conf('base_model', default=None)
        self.differential_loss = self.get_conf('differential_loss', 'mse')

        if self.differential_loss == 'mse':
            self.loss_fn = F.mse_loss
        elif self.differential_loss == 'l1' or self.differential_loss == 'mae':
            self.loss_fn = F.l1_loss
        elif self.differential_loss == 'cosine':
            self.loss_fn = F.cosine_similarity
        else:
            raise ValueError(f"Invalid differential loss {self.differential_loss}")
        # make working dir
        os.makedirs(self.working_dir, exist_ok=True)

        # build models to merge list
        models_to_merge = self.get_conf('models_to_merge', required=True, as_type=list)
        print(f"Models to merge: {models_to_merge}")
        # build list of ModelInputConfig objects. I find it is a good idea to make a class for each config
        # this way you can add methods to it and it is easier to read and code. There are a lot of
        # inbuilt config classes located in toolkit.config_modules as well
        self.models_to_merge = [ModelToMergeConfig(**model, mappings=self.mappings) for model in models_to_merge]
        if self.base_model is None:
            self.base_model = self.models_to_merge[0]
        else:
            self.base_model = ModelToMergeConfig(**self.base_model, mappings=self.mappings)
        # setup is complete. Don't load anything else here, just setup variables and stuff

        self.hash_dict = OrderedDict([
            ('differential_loss', self.differential_loss),
        ])
        if self.subtract_base:
            self.hash_dict['subtract_base'] = self.subtract_base
            self.hash_dict['base_model_path'] = self.base_model.name_or_path

        self.hash = get_hash_from_dict(self.hash_dict)

    # this is the entire run process be sure to call super().run() first
    @torch.no_grad()
    def run(self):
        def load_state_dict(path, device='cpu') -> Union[OrderedDict, dict]:
            # get extension
            ext = os.path.splitext(path)[1].lower()
            if ext == '.safetensors':
                return load_file(path, device)
            else:
                # assume torch 'pt, pth, pthc'
                model = torch.load(path, map_location=device)
                if isinstance(model, dict):
                    return model
                elif isinstance(model, OrderedDict):
                    return model
                else:
                    return model.state_dict()

        # always call first
        super().run()
        print(f"Running process: {self.__class__.__name__}")

        Similarity = Query()
        sim_dir_path = os.path.join(self.working_dir, 'sim_cache')
        # os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(sim_dir_path, exist_ok=True)

        print(f'Caching all weights to disk for each model to {self.working_dir}')

        num_models = len(self.models_to_merge)

        similarity_state_dict_master_cache = {}

        # load first one to get info
        bade_model_device = 'cpu' if not self.subtract_base else self.device
        base_model = load_state_dict(self.base_model.name_or_path, bade_model_device)

        # get all the keys
        tensor_keys = [x for x in base_model.keys()]

        # for key in tensor_keys:
        #     base_model[key] = base_model[key].to(get_torch_dtype('float32'))

        bad_keys = [
            'conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight',
            'conditioner.embedders.1.model.logit_scale'
        ]

        # remove them
        for key in bad_keys:
            if key in tensor_keys:
                tensor_keys.remove(key)

        flush()

        working_dtype = get_torch_dtype('fp32')

        print("Building merged feature ensemble")

        ensemble_state_dict = OrderedDict()

        for key in tensor_keys:
            if self.subtract_base:
                ensemble_state_dict[key] = base_model[key].clone().detach().to(dtype=working_dtype, device=self.device)
            else:
                ensemble_state_dict[key] = torch.zeros_like(base_model[key]).to(dtype=working_dtype, device=self.device)

        pbar = tqdm(total=len(self.models_to_merge), desc="Merging")

        beta = 1.0 - self.base_alpha

        for idx, model_config in enumerate(self.models_to_merge):
            # put model name in description
            pbar.set_description(f"Model: {os.path.splitext(os.path.basename(model_config.name_or_path))[0]}")
            # load model
            model = load_state_dict(model_config.name_or_path, self.device)
            found_non_zero = False
            for key in tensor_keys:
                key_a = model_config.get_key(key)

                if key in self.base_layers_to_keep or key.startswith(tuple(self.base_layers_to_keep_begins_with)):
                    # just use base
                    ensemble_state_dict[key] = base_model[key].clone().detach().to(dtype=working_dtype, device=self.device)
                    continue
                if key_a not in model:
                    print(f"Key {key_a} not found in model {model_config.name_or_path}")
                    tensor = base_model[key].clone().detach().to(dtype=working_dtype, device=self.device)
                else:
                    tensor = model[key_a].to(dtype=working_dtype, device=self.device)
                # ensemble_state_dict[key] += tensor_weights[key][idx] * tensor
                tensor_w = 1 / len(self.models_to_merge)
                if self.subtract_base:
                    tensor = tensor - base_model[key].clone().detach().to(dtype=working_dtype, device=self.device)

                ensemble_state_dict[key] += (tensor * tensor_w) * beta
                # check if nan
                if torch.isnan(ensemble_state_dict[key]).any():
                    # throw an error
                    raise ValueError(f"NaN detected in ensemble state dict for {key}. Try a different dtype")

            # fix alphas if they exist
            # for key in tensor_keys:
            #     if key.endswith('alpha'):
            #         ensemble_state_dict[key] = base_model[key]
            del model
            flush()
            pbar.update(1)

        pbar.close()

        print(f"Saving merged model to {self.save_path}")

        save_state_dict = OrderedDict()
        for key, tensor in ensemble_state_dict.items():
            save_state_dict[key] = tensor.clone().detach().to('cpu', get_torch_dtype(self.save_dtype))

        if self.use_base_meta:
            save_meta = load_metadata_from_safetensors(self.base_model.name_or_path)
        else:
            save_meta = get_meta_for_safetensors(self.meta, self.job.name)

        if self.save_path.endswith('.safetensors'):
            save_file(save_state_dict, self.save_path, save_meta)
        else:
            torch.save(save_state_dict, self.save_path)

        del base_model
        del ensemble_state_dict
        flush()

        print("Finished!")
