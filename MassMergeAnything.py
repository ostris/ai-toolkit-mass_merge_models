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


def max_normalize_with_reference(tensor, reference_tensor):
    """
    Normalize tensor using the absolute maximum value of the reference tensor.

    Args:
    - tensor (torch.Tensor): The tensor to be normalized.
    - reference_tensor (torch.Tensor): The tensor whose absolute max value will be used for normalization.

    Returns:
    - torch.Tensor: Normalized tensor.
    """
    abs_max_reference = reference_tensor.abs().max()
    return tensor * abs_max_reference / tensor.abs().max()


# this is our main class process
class MassMergeAnything(BaseExtensionProcess):

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
        self.base_layers_to_keep_begins_with = self.get_conf('base_layers_to_keep_begins_with', default=[],
                                                             as_type=list)
        self.base_layers_to_keep_contains = self.get_conf('base_layers_to_keep_contains', default=[], as_type=list)
        self.match_base_norm = self.get_conf('match_base_norm', default=False, as_type=bool)
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

        # make sure they all exist and remove ones that dont
        to_remove = []
        for model in self.models_to_merge:
            if not os.path.exists(model.name_or_path):
                print(f"Model {model.name_or_path} not found. Removing from list")
                to_remove.append(model)

        for model in to_remove:
            self.models_to_merge.remove(model)

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
            'conditioner.embedders.1.model.logit_scale',
            "conditioner.embedders.0.transformer.text_model.embeddings.position_ids",
            "cond_stage_model.transformer.text_model.embeddings.position_ids"
        ]

        for begin in self.base_layers_to_keep_begins_with:
            bad_keys += [x for x in tensor_keys if x.startswith(begin)]

        # remove them
        for key in bad_keys:
            if key in tensor_keys:
                tensor_keys.remove(key)

        flush()

        working_dtype = get_torch_dtype('fp32')

        # tensor_similarities = {}
        similarity_state_dict = {}
        for key in tensor_keys:
            similarity_state_dict[key] = torch.zeros(
                (num_models, num_models), device=self.device, dtype=get_torch_dtype(working_dtype)
            )

        pbar = tqdm(total=len(self.models_to_merge) * len(self.models_to_merge), desc="Computing similarity")
        for itx_a, model_a_config in enumerate(self.models_to_merge):
            model_a_name = os.path.splitext(os.path.basename(model_a_config.name_or_path))[0]
            # update description
            pbar.set_description(f"Computing: {model_a_name}")
            model_a = None
            for itx_b, model_b_config in enumerate(self.models_to_merge):
                do_compute = True
                model_b_name = os.path.splitext(os.path.basename(model_b_config.name_or_path))[0]
                # always do them in alphabetical order so we dont do same comparisons
                first_name = model_a_name if model_a_name.lower() < model_b_name.lower() else model_b_name
                second_name = model_b_name if model_a_name.lower() < model_b_name.lower() else model_a_name
                sim_key = f"{first_name}_{second_name}_{self.differential_loss}"
                ssim_name = f"{sim_key}.safetensors"
                ssim_path = os.path.join(sim_dir_path, ssim_name)
                model_similarities = {}

                # check if it exists
                if model_a_name == model_b_name:
                    for key in tensor_keys:
                        model_similarities[key] = torch.tensor(0.0, device='cpu', dtype=working_dtype)
                    do_compute = False
                    similarity_state_dict[key][itx_a, itx_b] = torch.tensor(0.0, device='cpu', dtype=working_dtype)
                    similarity_state_dict[key][itx_b, itx_a] = torch.tensor(0.0, device='cpu', dtype=working_dtype)

                elif os.path.exists(ssim_path):
                    do_compute = False
                    model_similarities = load_file(ssim_path, self.device)
                    for key in tensor_keys:
                        similarity_state_dict[key][itx_a, itx_b] = model_similarities[key].clone().cpu()
                        similarity_state_dict[key][itx_b, itx_a] = model_similarities[key].clone().cpu()
                if do_compute:
                    # load if we haven't loaded yet
                    if model_a is None:
                        # we need to calculate this one, load both models
                        model_a = load_state_dict(model_a_config.name_or_path, self.device)

                    model_b = load_state_dict(model_b_config.name_or_path, self.device)
                    # calculate the similarity matrix for each tensor

                    found_non_zero = False
                    for key in tqdm(tensor_keys):
                        # if the key starts with one of the base layers to keep, skip it
                        skip = False
                        for base_key in self.base_layers_to_keep_begins_with:
                            if key.startswith(base_key):
                                skip = True
                                break
                        if skip:
                            continue
                        key_a = model_a_config.get_key(key)
                        key_b = model_b_config.get_key(key)
                        if key_a not in model_a:
                            raise ValueError(f"Key {key_a} not found in model {model_a_config.name_or_path}")

                        if key_b not in model_b:
                            raise ValueError(f"Key {key_b} not found in model {model_b_config.name_or_path}")

                        try:
                            if self.subtract_base:
                                model_a_weight = model_a[key_a] - base_model[key]
                                model_b_weight = model_b[key_b] - base_model[key]
                            else:
                                model_a_weight = model_a[key_a]
                                model_b_weight = model_b[key_b]
                            if self.differential_loss == 'cosine':
                                model_a_weight = model_a_weight.view(1, -1)
                                model_b_weight = model_b_weight.view(1, -1)
                                sim = self.loss_fn(model_a_weight, model_b_weight)
                                # do abs incase it is cosine similarity
                                sim = torch.abs(sim)
                            else:
                                sim = self.loss_fn(model_a_weight.float(), model_b_weight.float())
                            # del model_a_weight
                            del model_b_weight
                            del model_b[key_b]  # don't need it anymore

                            if torch.isnan(sim).any():
                                # throw an error
                                raise ValueError(f"NaN detected in similarity matrix for {key}. Try a different dtype")

                            model_similarities[key] = sim.clone()
                            similarity_state_dict[key][itx_a, itx_b] = model_similarities[key].clone().cpu()
                            similarity_state_dict[key][itx_b, itx_a] = model_similarities[key].clone().cpu()
                            del sim

                        except Exception as e:
                            print(f"Error computing similarity for {sim_key}")
                            print(e)
                            model_similarities[key] = None

                        if model_similarities[key] is not None and torch.sum(model_similarities[key]) > 0.0:
                            found_non_zero = True

                    if not found_non_zero:
                        print(f"Models_identical {ssim_name} after recompute")

                    # save the similarity matrix to disk
                    save_file(model_similarities, os.path.join(sim_dir_path, ssim_name))
                    del model_b
                    del model_similarities
                    # except Exception as e:
                    #     raise ValueError(e)
                    # flush()
                pbar.update(1)

                # try:
                #     # add them to the matrix
                #     for key in tensor_keys:
                #         try:
                #             similarity_state_dict[key][itx_a, itx_b] = model_similarities[key].clone().cpu()
                #             similarity_state_dict[key][itx_b, itx_a] = model_similarities[key].clone().cpu()
                #         except Exception as e:
                #             print(f"Error adding similarity for {sim_key}")
                #             similarity_state_dict[key][itx_a, itx_b] = torch.tensor(0.0, device=self.device, dtype=working_dtype)
                #             similarity_state_dict[key][itx_b, itx_a] = torch.tensor(0.0, device=self.device, dtype=working_dtype)
                # except Exception as e:
                #     print(f"Error adding similarity for {sim_key}")
                #     print(e)
                #     raise e
                model_similarities = {}
                flush()

            if model_a is not None:
                del model_a
            flush()

        pbar.close()

        del similarity_state_dict_master_cache
        flush()

        print("Reducing feature similarity matrices")

        # do these calculations at float 32
        dtype = get_torch_dtype('float32')

        # compute the weights for each tensor based on the similarity matrix
        tensor_weights = {}

        # will keep the weight at minimum 10% of the model activity scaled to number of models
        min_model_activity = 0.1

        # build model weight scaler
        model_weight_scaler = torch.ones((num_models, num_models), device=self.device, dtype=dtype)
        for itx_a, model_a_config in enumerate(self.models_to_merge):
            for itx_b, model_b_config in enumerate(self.models_to_merge):
                model_weight_scaler[itx_a, itx_b] = model_a_config.weight * model_b_config.weight

        idx = 0
        for key, similarity_matrix in similarity_state_dict.items():
            # move minimum to 0 to compensate for negative similarities
            # similarity_matrix -= torch.min(similarity_matrix)

            # adjust the similarity matrix to the weights of the models
            similarity_matrix = similarity_matrix.to(dtype) * model_weight_scaler

            weights = torch.sum(similarity_matrix, dim=0)

            # todo, try different scaling methods here

            # scale weights so the sum is 1
            weights = weights / torch.sum(weights)

            # print(
            #     f" - {key}: min: {torch.min(weights).item()}, max: {torch.max(weights).item()}, sum: {torch.sum(weights).item()}")

            # scale weights from min_weight_value to 1
            # weights = value_map(weights, 0, torch.max(weights), 0, 1)
            # weights = value_map(weights, torch.min(weights), torch.max(weights), min_weight_value, 1)

            tensor_weights[key] = weights.to(get_torch_dtype(working_dtype))

            idx += 1

        flush()
        ### WEIGHING AND MERGING ###

        print("Building merged feature ensemble")

        ensemble_state_dict = OrderedDict()

        for key in tensor_keys:
            if self.subtract_base:
                ensemble_state_dict[key] = base_model[key].clone().detach().to(dtype=working_dtype, device=self.device)
            else:
                ensemble_state_dict[key] = torch.zeros_like(base_model[key]).to(dtype=working_dtype, device=self.device)
        # see if bad keys are in base and add them if they are
        for key in bad_keys:
            if key in base_model:
                ensemble_state_dict[key] = base_model[key].clone().detach().to(dtype=working_dtype, device=self.device)

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

                if key in self.base_layers_to_keep or key.startswith(
                        tuple(self.base_layers_to_keep_begins_with)) or any(
                    x in key for x in self.base_layers_to_keep_contains):
                    # just use base
                    ensemble_state_dict[key] = base_model[key].clone().detach().to(dtype=working_dtype,
                                                                                   device=self.device)
                    tensor_weights[key][idx] *= 0.0
                    continue
                if key_a not in model:
                    print(f"Key {key_a} not found in model {model_config.name_or_path}")
                    tensor = base_model[key].clone().detach().to(dtype=working_dtype, device=self.device)
                else:
                    tensor = model[key_a].to(dtype=working_dtype, device=self.device)
                # ensemble_state_dict[key] += tensor_weights[key][idx] * tensor
                tensor_w = tensor_weights[key][idx]

                # if tensor_w  is nan, divide by num models
                if torch.isnan(tensor_w).any():
                    raise ValueError(f"NaN detected in tensor weight for {key}. Dividing by number of models")
                    # tensor_w = 1.0 / len(self.models_to_merge)

                if self.subtract_base:
                    tensor = tensor - base_model[key].clone().detach().to(dtype=working_dtype, device=self.device)

                ensemble_state_dict[key] += (tensor * tensor_w) * beta

                if self.match_base_norm:
                    ensemble_state_dict[key] = max_normalize_with_reference(
                        ensemble_state_dict[key],
                        base_model[key].clone().detach().to(dtype=working_dtype, device=self.device)
                    )
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

        if self.report_format is not None:
            # create tensor merge report
            merge_report = OrderedDict()
            merge_report['totals'] = OrderedDict()
            num_total_keys = len(list(tensor_weights.keys()))
            for key, weights in tensor_weights.items():
                weights = weights.detach().cpu().numpy()
                # convert to list of floats
                weights = [float(w) for w in weights]
                clean_key = key
                merge_report.setdefault(clean_key, OrderedDict())
                for weight, model_config in zip(weights, self.models_to_merge):
                    model_path = model_config.name_or_path
                    # get filename no ext. Can be any extension
                    filename_no_ext = os.path.splitext(os.path.basename(model_path))[0]
                    merge_report[clean_key][filename_no_ext] = weight
                    if filename_no_ext not in merge_report['totals']:
                        merge_report['totals'][filename_no_ext] = 0.0
                    merge_report['totals'][filename_no_ext] += (weight / num_total_keys)

            for key, weight_dict in merge_report.items():
                # sort the weight dicts by value
                merge_report[key] = OrderedDict(sorted(weight_dict.items(), key=lambda x: x[1], reverse=True))

            save_path_no_ext = os.path.splitext(self.save_path)[0]

            saved_to = None
            if self.report_format == "yaml":
                with open(f"{save_path_no_ext}_report.yaml", 'w') as f:
                    yaml.dump(merge_report, f)
                    saved_to = f"{save_path_no_ext}_report.yaml"
            elif self.report_format == "json":
                with open(f"{save_path_no_ext}_report.json", 'w') as f:
                    json.dump(merge_report, f, indent=4)
                    saved_to = f"{save_path_no_ext}_report.json"

            print(f"Saved merge report to {saved_to}")
        del base_model
        del ensemble_state_dict
        flush()

        print("Finished!")
