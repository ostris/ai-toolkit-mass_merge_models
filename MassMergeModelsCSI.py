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
from toolkit.paths import get_path
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.train_tools import get_torch_dtype
from tqdm import tqdm
import numpy as np

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


# this is our main class process
class MassMergeModelsCSI(BaseExtensionProcess):
    loss_fn: Union[F.mse_loss, F.l1_loss, F.cosine_similarity]
    models_to_merge: List[ModelInputConfig]

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
        self.working_dtype = self.get_conf('working_dtype', default='fp16')
        self.device = self.get_conf('device', default='cpu')
        self.report_format = self.get_conf('report_format', default='json')
        self.keep_cache = self.get_conf('keep_cache', default=False, as_type=bool)
        self.vae_path = self.get_conf('vae_path', default=None)
        if self.report_format == 'null':
            self.report_format = None

        self.cache_dtype = self.get_conf('cache_dtype', default='float32', as_type=get_torch_dtype)
        self.base_model = ModelInputConfig(**self.get_conf('base_model', required=True))
        # merge_step = self.get_conf('merge_step', default=1, as_type=int)

        self.weight_json = os.path.join(self.working_dir, 'weight.json')
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
        # build list of ModelInputConfig objects. I find it is a good idea to make a class for each config
        # this way you can add methods to it and it is easier to read and code. There are a lot of
        # inbuilt config classes located in toolkit.config_modules as well
        self.models_to_merge = [ModelInputConfig(**model) for model in models_to_merge]
        # setup is complete. Don't load anything else here, just setup variables and stuff

    # this is the entire run process be sure to call super().run() first
    @torch.no_grad()
    def run(self):
        # always call first
        super().run()
        print(f"Running process: {self.__class__.__name__}")
        cache_dir = os.path.join(self.working_dir, self.name, 'cache')
        sim_dir_path = os.path.join(self.working_dir, 'sim_cache')
        # os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(sim_dir_path, exist_ok=True)

        print("Loading base model")
        base_model: StableDiffusion = StableDiffusion(
            device=self.device,
            model_config=self.base_model,
            dtype=self.working_dtype
        )
        base_model.load_model()
        # unload stuff we dont need
        # todo, prevent loading in the first place
        if self.vae_path is not None:
            del base_model.vae
        del base_model.tokenizer
        flush()

        print(f'Caching all weights to disk for each model to {self.working_dir}')
        print(f"Computing pairwise similarity of all cached weights:")

        def get_all_keys(_model):
            all_keys = []
            if isinstance(_model.text_encoder, list):
                te_list = _model.text_encoder
            else:
                te_list = [_model.text_encoder]
            for te_num, te in enumerate(te_list):
                for key in te.state_dict().keys():
                    all_keys.append(f"te{te_num}_{key}")
            for key in _model.unet.state_dict().keys():
                all_keys.append(f"unet_{key}")
            return all_keys

        # get all weight keys from the base model
        base_state_dict_all = {}
        te_keys = []
        if isinstance(base_model.text_encoder, list):
            te_list = base_model.text_encoder
        else:
            te_list = [base_model.text_encoder]
        for te_num, te in enumerate(te_list):
            for key in te.state_dict().keys():
                te_keys.append(f"te{te_num}_{key}")
                base_state_dict_all[f"te{te_num}_{key}"] = te.state_dict()[key]

        unet_keys = [f"unet_{key}" for key in base_model.unet.state_dict().keys()]
        for key in base_model.unet.state_dict().keys():
            base_state_dict_all[f"unet_{key}"] = base_model.unet.state_dict()[key]

        tensor_keys = te_keys + unet_keys

        num_models = len(self.models_to_merge)

        ### COMPUTE COSINE SIMILARITY ###

        # tensor_sim_path = os.path.join(self.working_dir, self.name, 'tensor_sim.pkl')
        # tensor_sim_path = os.path.join(self.working_dir, self.name, f'tensor_sim_{self.differential_loss}.safetensors')
        # if os.path.exists(tensor_sim_path):
        #     tensor_similarities = load_file(tensor_sim_path, self.device)
        #     print(f"Loaded tensor similarities from {tensor_sim_path}")
        #     print(" - Checking for new items to compute")
        # else:

        similarity_state_dict_master_cache = {}

        # if os.path.exists(similarity_state_dict_master_cache_path):
        #     similarity_state_dict_master_cache = load_file(similarity_state_dict_master_cache_path, self.device)
        #     print(f"Loaded similarity state dict master cache from {similarity_state_dict_master_cache_path}")
        #     print(" - Checking for new items to compute")

        # tensor_similarities = {}
        similarity_state_dict = {}
        for key in tensor_keys:
            similarity_state_dict[key] = torch.zeros(
                (num_models, num_models), device=self.device, dtype=get_torch_dtype(self.working_dtype)
            )

        pbar = tqdm(total=len(self.models_to_merge) * len(self.models_to_merge), desc="Computing similarity")
        for itx_a, model_a_config in enumerate(self.models_to_merge):
            model_a_name = os.path.splitext(os.path.basename(model_a_config.name_or_path))[0]
            # update description
            pbar.set_description(f"Computing: {model_a_name}")
            model_a = None

            for itx_b, model_b_config in enumerate(self.models_to_merge):
                model_b_name = os.path.splitext(os.path.basename(model_b_config.name_or_path))[0]
                # always do them in alphabetical order so we dont do same comparisons
                first_name = model_a_name if model_a_name.lower() < model_b_name.lower() else model_b_name
                second_name = model_b_name if model_a_name.lower() < model_b_name.lower() else model_a_name
                sim_key = f"{first_name}_{second_name}_{self.differential_loss}"
                ssim_name = f"{sim_key}.safetensors"
                # check if it exists
                if sim_key in similarity_state_dict_master_cache:
                    model_similarities = similarity_state_dict_master_cache[sim_key]
                elif os.path.exists(os.path.join(sim_dir_path, ssim_name)):
                    # load it
                    model_similarities = load_file(os.path.join(sim_dir_path, ssim_name), self.device)
                else:
                    model_similarities = {}
                    # load if we haven't loaded yet
                    if model_a is None:
                        # we need to calculate this one, load both models
                        model_a = StableDiffusion(
                            device=self.device,
                            model_config=model_a_config,
                            dtype=self.working_dtype
                        )
                        model_a.load_model()
                        # unload stuff we dont need
                        # todo, prevent loading in the first place
                        del model_a.vae
                        del model_a.tokenizer

                    model_b = StableDiffusion(
                        device=self.device,
                        model_config=model_b_config,
                        dtype=self.working_dtype
                    )
                    model_b.load_model()
                    # unload stuff we dont need
                    # todo, prevent loading in the first place
                    del model_b.vae
                    del model_b.tokenizer
                    flush()

                    # calculate the similarity matrix for each tensor
                    for key in tensor_keys:
                        base_model_weight = base_state_dict_all[key]
                        model_a_weight = model_a.get_weight_by_name(key) - base_model_weight
                        model_b_weight = model_b.get_weight_by_name(key) - base_model_weight
                        if self.differential_loss == 'cosine':
                            model_a_weight = model_a_weight.view(1, -1)
                            model_b_weight = model_b_weight.view(1, -1)
                            sim = self.loss_fn(model_a_weight, model_b_weight)
                            # do abs incase it is cosine similarity
                            sim = torch.abs(sim)
                        else:
                            sim = self.loss_fn(model_a_weight, model_b_weight)
                        model_similarities[key] = sim.detach().to('cpu', get_torch_dtype(self.cache_dtype))
                        # check for nans
                        if torch.isnan(model_similarities[key]).any():
                            # throw an error
                            raise ValueError(f"NaN detected in similarity matrix for {key}. Try a different dtype")

                    # save the similarity matrix to disk
                    save_file(model_similarities, os.path.join(sim_dir_path, ssim_name))
                    del model_b
                    flush()
                pbar.update(1)

                # add them to the matrix
                for key in tensor_keys:
                    similarity_state_dict[key][itx_a, itx_b] = model_similarities[key]
                    similarity_state_dict[key][itx_b, itx_a] = model_similarities[key]

            if model_a is not None:
                del model_a
            flush()

        pbar.close()

        del similarity_state_dict_master_cache
        flush()

        # if num_similarities_added > 0:
        #     print(f"Added {num_similarities_added} new similarities")
        #     print(f"Saving tensor similarities to {tensor_sim_path}")
        #     save_file(tensor_similarities, tensor_sim_path)

        ### COMPUTE WEIGHTS FOR EACH TENSOR ###

        print("Reducing feature similarity matrices")

        # do these calculations at float 32
        dtype = get_torch_dtype('float32')

        # compute the weights for each tensor based on the similarity matrix
        tensor_weights = {}

        # we can use a few different methods to compute the weights
        # I will just continue with clustering for now

        # will keep the weight at minimum 10% of the model activity scaled to number of models
        min_model_activity = 0.1
        min_weight_value = min_model_activity / len(self.models_to_merge)

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

            print(f" - {key}: min: {torch.min(weights).item()}, max: {torch.max(weights).item()}, sum: {torch.sum(weights).item()}")

            # scale weights from min_weight_value to 1
            # weights = value_map(weights, 0, torch.max(weights), 0, 1)
            # weights = value_map(weights, torch.min(weights), torch.max(weights), min_weight_value, 1)

            tensor_weights[key] = weights.to(get_torch_dtype(self.working_dtype))

            idx += 1

        ### WEIGHING AND MERGING ###

        print("Building merged feature ensemble")

        ensemble_state_dict = {}

        pbar = tqdm(total=len(self.models_to_merge), desc="Building ensemble")
        for key in tensor_keys:
            ensemble_state_dict[key] = torch.zeros_like(base_state_dict_all[key]).to(dtype)

        for idx, model_config in enumerate(self.models_to_merge):
            # put model name in description
            pbar.set_description(f"Model: {os.path.splitext(os.path.basename(model_config.name_or_path))[0]}")
            # load model
            model = StableDiffusion(
                device=self.device,
                model_config=model_config,
                dtype=self.working_dtype
            )
            model.load_model()
            # unload stuff we dont need
            # todo, prevent loading in the first place
            del model.vae
            del model.tokenizer
            flush()

            for key in tensor_keys:
                tensor = model.get_weight_by_name(key)
                # ensemble_state_dict[key] += tensor_weights[key][idx] * tensor
                ensemble_state_dict[key] += (tensor.to(dtype) - base_state_dict_all[key].to(dtype)) * tensor_weights[key][idx]
                # check if nan
                if torch.isnan(ensemble_state_dict[key]).any():
                    # throw an error
                    raise ValueError(f"NaN detected in ensemble state dict for {key}. Try a different dtype")

            del model
            flush()
            pbar.update(1)

        pbar.close()

        ### MERGE INTO BASE MODEL ###

        print("Merging ensemble model back into base model")

        # merge the ensemble state dict into the base model
        if isinstance(base_model.text_encoder, list):
            te_list = base_model.text_encoder
        else:
            te_list = [base_model.text_encoder]
        for te_num, te in enumerate(te_list):
            te.to(dtype)
            for key in te.state_dict().keys():
                te.state_dict()[key] += ensemble_state_dict[f"te{te_num}_{key}"].to(dtype)

        base_model.unet.to(dtype)
        for key in base_model.unet.state_dict().keys():
            base_model.unet.state_dict()[key] += ensemble_state_dict[f"unet_{key}"].to(dtype)

        if self.vae_path is not None:
            print(f"Loading VAE from {self.vae_path}")
            # load a vae model
            base_model.vae = load_vae(self.vae_path, dtype=self.working_dtype)

        print(f"Saving merged model to {self.save_path}")

        base_model.save(self.save_path, meta=self.meta, save_dtype=self.save_dtype)

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
