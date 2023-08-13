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
        self.device = self.get_conf('device', default='cpu')
        self.report_format = self.get_conf('report_format', default='json')
        self.keep_cache = self.get_conf('keep_cache', default=False, as_type=bool)
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
        os.makedirs(cache_dir, exist_ok=True)

        state_path = os.path.join(self.working_dir, self.name, 'state.json')
        state = {
            'models_cached': []
        }
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)

        def save_state():
            with open(state_path, 'w') as f:
                json.dump(state, f)

        print("Loading base model")
        base_model: StableDiffusion = StableDiffusion(
            device=self.device,
            model_config=self.base_model,
            dtype="float32"
        )
        base_model.load_model()
        print(f'Caching all weights to disk for each model to {self.working_dir}')

        # we have to cache to disk because each model takes up a lot of ram. There is not way to do this without
        # caching to disk. We save each weight instead of the model to keep from loading the whole thing
        # this allows us to merge an unlimited number of models without running out of ram

        pbar = tqdm(self.models_to_merge, desc=f"Caching weights to disk")
        failed_models = []
        num_failed = 0
        for idx, model_config in enumerate(self.models_to_merge):
            if model_config.name_or_path in state['models_cached']:
                print(f"Skipping {model_config.name_or_path} because it is already cached")
                pbar.update(1)
                continue
            try:
                # setup model class with our helper class
                sd_model = StableDiffusion(
                    device=self.device,
                    model_config=model_config,
                    dtype=self.cache_dtype
                )
                sd_model.load_model()

                ### CACHE TEXT ENCODER ###

                # adjust the weight of the text encoder
                if isinstance(sd_model.text_encoder, list):
                    # sdxl model
                    base_te = base_model.text_encoder
                    instance_te = sd_model.text_encoder
                else:
                    # normal model
                    base_te = [base_model.text_encoder]
                    instance_te = [sd_model.text_encoder]

                te_num = 0
                for text_encoder, base_text_encoder in zip(instance_te, base_te):
                    for key, value in text_encoder.state_dict().items():
                        # if it exists , skip it

                        # subtract base model weights from the model we are merging so we don't dilute it with the base
                        value -= base_text_encoder.state_dict()[key]
                        file_name = f"te{te_num}_m{idx}_{key}.safetensors"
                        v = value.detach().to('cpu', get_torch_dtype(self.cache_dtype))

                        save_file({'x': v}, os.path.join(cache_dir, file_name))
                    te_num += 1

                ### CACHE UNET ###

                for key, value in sd_model.unet.state_dict().items():
                    # subtract base model weights from the model we are merging so we don't dilute it with the base
                    value -= base_model.unet.state_dict()[key]
                    file_name = f"unet_m{idx}_{key}.safetensors"
                    v = value.detach().to('cpu', get_torch_dtype(self.cache_dtype))
                    save_file({'x': v}, os.path.join(cache_dir, file_name))

                del sd_model
                del base_te
                del instance_te
                flush()
                state['models_cached'].append(model_config.name_or_path)
                save_state()

            except Exception as e:
                print(f"Failed to cache model {model_config.name_or_path} with error {e}")
                num_failed += 1
                # Add it to failed. Remove it after iterating
                failed_models.append(model_config)
            pbar.update(1)
        # keep failed in for now to keep list in order
        # for failed_model in failed_models:
        #     self.models_to_merge.remove(failed_model)
        print(f"Cached {len(self.models_to_merge) - num_failed} models to disk with {num_failed} failures")
        print(f"Computing pairwise cosine similarity of all cached weights")

        # get all weight keys from the base model
        base_state_dict_all = {}
        te_keys = []
        if isinstance(base_model.text_encoder, list):
            te_list = base_model.text_encoder
        else:
            te_list = [base_model.text_encoder]
        for te_num, te in enumerate(te_list):
            for key in te.state_dict().keys():
                te_keys.append(f"te{te_num}_m[idx]_{key}")
                base_state_dict_all[f"te{te_num}_m[idx]_{key}"] = te.state_dict()[key]

        unet_keys = [f"unet_m[idx]_{key}" for key in base_model.unet.state_dict().keys()]
        for key in base_model.unet.state_dict().keys():
            base_state_dict_all[f"unet_m[idx]_{key}"] = base_model.unet.state_dict()[key]

        tensor_keys = te_keys + unet_keys

        num_models = len(self.models_to_merge)

        ### COMPUTE COSINE SIMILARITY ###

        # tensor_sim_path = os.path.join(self.working_dir, self.name, 'tensor_sim.pkl')
        tensor_sim_path = os.path.join(self.working_dir, self.name, f'tensor_sim_{self.differential_loss}.safetensors')
        if os.path.exists(tensor_sim_path):
            tensor_similarities = load_file(tensor_sim_path, self.device)
            print(f"Loaded tensor similarities from {tensor_sim_path}")
            print(" - Checking for new items to compute")
        else:

            tensor_similarities = {}

            num_similarities_added = 0

            pbar = tqdm(total=len(tensor_keys), desc="Computing similarity")

            # calculate the similarity matrix for each tensor
            for key in tensor_keys:
                similarity_matrix = torch.zeros((num_models, num_models), device=self.device, dtype=self.cache_dtype)

                tensors_paths = [
                    os.path.join(
                        cache_dir, f"{key.replace('[idx]', str(idx))}.safetensors"
                    ) for idx in range(num_models)
                ]

                for i in range(num_models):
                    tensors_i = load_file(tensors_paths[i], self.device)['x']
                    if self.differential_loss == 'cosine':
                        tensors_i = tensors_i.view(1, -1)
                    for j in range(i + 1, num_models):
                        tensors_j = load_file(tensors_paths[j], self.device)['x']
                        if self.differential_loss == 'cosine':
                            tensors_j = tensors_j.view(1, -1)
                            sim = self.loss_fn(tensors_i, tensors_j)
                            # do abs incase it is cosine similarity
                            sim = torch.abs(sim)
                        else:
                            sim = self.loss_fn(tensors_i, tensors_j)
                        similarity_matrix[i, j] = sim
                        similarity_matrix[j, i] = sim
                        del tensors_j
                    del tensors_i
                    flush()

                tensor_similarities[key] = similarity_matrix
                num_similarities_added += 1
                pbar.update(1)

            pbar.close()

            if num_similarities_added > 0:
                print(f"Added {num_similarities_added} new similarities")
                print(f"Saving tensor similarities to {tensor_sim_path}")
                save_file(tensor_similarities, tensor_sim_path)

        ### COMPUTE WEIGHTS FOR EACH TENSOR ###

        print("Reducing feature similarity matrices")

        # compute the weights for each tensor based on the similarity matrix
        tensor_weights = {}

        # we can use a few different methods to compute the weights
        # I will just continue with clustering for now

        # will keep the weight at minimum 10% of the model activity scaled to number of models
        # min_model_activity = 0.1
        # min_weight_value = min_model_activity / len(self.models_to_merge)

        for key, similarity_matrix in tensor_similarities.items():
            # move minimum to 0 to compensate for negative similarities
            # similarity_matrix -= torch.min(similarity_matrix)
            weights = torch.sum(similarity_matrix, dim=0)
            # todo, try different scaling methods here

            weights = value_map(weights, torch.min(weights), torch.max(weights), 0, 1)

            # scale weights so the sum is 1
            weights = weights / torch.sum(weights)


            # scale weights from min_weight_value to 1
            # weights = value_map(weights, 0, torch.max(weights), 0, 1)
            # weights = value_map(weights, torch.min(weights), torch.max(weights), min_weight_value, 1)

            tensor_weights[key] = weights

        ### WEIGHING AND MERGING ###

        print("Building merged feature ensemble")

        ensemble_state_dict = {}

        pbar = tqdm(total=len(tensor_keys), desc="Building ensemble")

        for key in tensor_keys:
            weighted_sum = torch.zeros_like(base_state_dict_all[key])

            for idx in range(num_models):
                tensor = \
                    load_file(
                        os.path.join(cache_dir, f"{key.replace('[idx]', str(idx))}.safetensors"),
                        self.device
                    )['x']
                weighted_sum += tensor_weights[key][idx] * tensor
                del tensor
                flush()

            ensemble_state_dict[key] = weighted_sum

            pbar.update(1)
            flush()

        pbar.close()

        ### MERGE INTO BASE MODEL ###

        print("Merging ensemble model back into base model")

        # merge the ensemble state dict into the base model
        if isinstance(base_model.text_encoder, list):
            te_list = base_model.text_encoder
        else:
            te_list = [base_model.text_encoder]
        for te_num, te in enumerate(te_list):
            for key in te.state_dict().keys():
                te.state_dict()[key] += ensemble_state_dict[f"te{te_num}_m[idx]_{key}"]

        for key in base_model.unet.state_dict().keys():
            base_model.unet.state_dict()[key] += ensemble_state_dict[f"unet_m[idx]_{key}"]

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
                clean_key = key.replace("_m[idx]", "")
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

        if not self.keep_cache:
            print(f"Removing tmp directory {cache_dir}")
            shutil.rmtree(os.path.join(self.working_dir, self.name))
        else:
            print(f"Keeping tmp directory {cache_dir}")
            print(" - Be sure to manually clean it up, as it takes up a lot of space")
        del base_model
        del ensemble_state_dict
        flush()

        print("Finished!")
