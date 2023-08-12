import json
import os

import torch
import gc
from collections import OrderedDict
from typing import TYPE_CHECKING
from jobs.process import BaseExtensionProcess
from toolkit.config_modules import ModelConfig
from toolkit.paths import get_path
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.train_tools import get_torch_dtype
from tqdm import tqdm

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
class MassMergeModels(BaseExtensionProcess):
    def __init__(
            self,
            process_id: int,
            job: 'ExtensionJob',
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)
        self.save_path = get_path(self.get_conf('save_path', required=True))
        self.working_dir = get_path(self.get_conf('working_dir', required=True))
        self.save_dtype = self.get_conf('save_dtype', default='float16', as_type=get_torch_dtype)
        self.device = self.get_conf('device', default='cpu', as_type=torch.device)
        self.passes = self.get_conf('passes', default=1, as_type=int)
        self.base_model = ModelInputConfig(**self.get_conf('base_model', required=True))
        # merge_step = self.get_conf('merge_step', default=1, as_type=int)

        self.weight_json = os.path.join(self.working_dir, 'weight.json')
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
    def run(self):
        # always call first
        super().run()
        print(f"Running process: {self.__class__.__name__}")

        merge_steps = [1]
        for i in range(self.passes):
            merge_steps.append(2)
            merge_steps.append(3)

        # we need step num and index so we can save the model
        for index in range(len(merge_steps)):
            merge_step = merge_steps[index]

            # let's adjust our weights first to normalize them so the total is 1.0
            total_weight = sum([model.weight for model in self.models_to_merge])
            if merge_step == 1:
                weight_adjust = 1.0 / total_weight
                for model in self.models_to_merge:
                    model.weight *= weight_adjust

            model_to_load = self.base_model
            if merge_step == 2:
                model_to_load = ModelInputConfig(**{
                    "name_or_path": os.path.join(self.working_dir, f"tmp_merged_model.safetensors"),
                })
            print("Loading base model")
            base_model: StableDiffusion = StableDiffusion(
                device=self.device,
                model_config=model_to_load,
                dtype="float32"
            )
            base_model.load_model()

            loss_fn = torch.nn.MSELoss()
            # loss_fn = torch.nn.L1Loss()  # MAE

            mse_dict = {}
            sum_dict = {}
            if merge_step == 3:
                # load the dict
                with open(self.weight_json, 'r') as f:
                    mse_dict = json.load(f)

                # find the sum of the weights
                for model_path in mse_dict.keys():
                    for key in mse_dict[model_path].keys():
                        if key not in sum_dict.keys():
                            sum_dict[key] = 0
                        sum_dict[key] += mse_dict[model_path][key]

            output_model: StableDiffusion = None
            num_failed = 0
            # let's do the merge, it is a good idea to use tqdm to show progress
            for model_config in tqdm(self.models_to_merge, desc=f"Merging step {index + 1} of {len(merge_steps)}"):
                if merge_step == 2:
                    mse_dict[model_config.name_or_path] = {}
                try:
                    # setup model class with our helper class
                    sd_model = StableDiffusion(
                        device=self.device,
                        model_config=model_config,
                        dtype="float32"
                    )
                    # load the model
                    sd_model.load_model()

                    # adjust the weight of the text encoder
                    if isinstance(sd_model.text_encoder, list):
                        # sdxl model
                        te_num = 0
                        for text_encoder, base_text_encoder in zip(sd_model.text_encoder, base_model.text_encoder):
                            for key, value in text_encoder.state_dict().items():
                                if merge_step == 1 or merge_step == 3:
                                    # subtract base model weights from the model we are merging so we don't dilute it with the base
                                    value -= base_text_encoder.state_dict()[key]
                                    if merge_step == 3:
                                        # adjust the weight based on the mse
                                        multiplier = mse_dict[model_config.name_or_path][f"te.{te_num}.{key}"] / sum_dict[f"te.{te_num}.{key}"]
                                    else:
                                        # adjust the weight based on the config
                                        multiplier = model_config.weight
                                    value *= multiplier
                                elif merge_step == 2:
                                    # just calculate mse from merge model
                                    mse_dict[model_config.name_or_path][f"te.{te_num}.{key}"] = loss_fn(value, base_text_encoder.state_dict()[key]).item()
                                else:
                                    pass
                            te_num += 1
                    else:
                        # normal model
                        for key, value in sd_model.text_encoder.state_dict().items():
                            if merge_step == 1 or merge_step == 3:
                                # subtract base model weights from the model we are merging so we don't dilute it with the base
                                value -= base_model.text_encoder.state_dict()[key]
                                if merge_step == 3:
                                    # adjust the weight based on the mse
                                    multiplier = mse_dict[model_config.name_or_path][f"te.{key}"] / sum_dict[f"te.{key}"]
                                else:
                                    # adjust the weight based on the config
                                    multiplier = model_config.weight
                                value *= multiplier
                            elif merge_step == 2:
                                mse_dict[model_config.name_or_path][f"te.{key}"] = loss_fn(value, base_model.text_encoder.state_dict()[key]).item()
                            else:
                                pass

                    # adjust the weights of the unet
                    for key, value in sd_model.unet.state_dict().items():
                        if merge_step == 1 or merge_step == 3:
                            # subtract base model weights from the model we are merging so we don't dilute it with the base
                            value -= base_model.unet.state_dict()[key]
                            # adjust the weight based on the config
                            if merge_step == 3:
                                # adjust the weight based on the mse
                                multiplier = mse_dict[model_config.name_or_path][f"unet.{key}"] / sum_dict[f"unet.{key}"]
                            else:
                                # adjust the weight based on the config
                                multiplier = model_config.weight
                            value *= multiplier
                        elif merge_step == 2:
                            mse_dict[model_config.name_or_path][f"unet.{key}"] = loss_fn(value,  base_model.unet.state_dict()[key]).item()
                        else:
                            pass

                    if merge_step != 2:
                        if output_model is None:
                            # use this one as the base
                            output_model = sd_model
                        else:
                            # merge the models
                            # text encoder
                            if isinstance(output_model.text_encoder, list):
                                # sdxl model
                                for i, text_encoder in enumerate(output_model.text_encoder):
                                    for key, value in text_encoder.state_dict().items():
                                        value += sd_model.text_encoder[i].state_dict()[key]
                            else:
                                # normal model
                                for key, value in output_model.text_encoder.state_dict().items():
                                    value += sd_model.text_encoder.state_dict()[key]
                            # unet
                            for key, value in output_model.unet.state_dict().items():
                                value += sd_model.unet.state_dict()[key]

                            # remove the model to free memory
                            del sd_model
                            flush()
                except Exception as e:
                    print(f"Error merging model: {e}")
                    print(e)
                    print(f"{model_config.name_or_path} will be skipped")
                    print(f"{model_config.name_or_path} will be skipped")
                    num_failed += 1
                    continue

            if merge_step != 2:
                if num_failed > 0:
                    print(f"Failed to merge {num_failed} models. Rescaling to compensate")
                    # rescale the model to compensate for the failed models
                    for key, value in output_model.text_encoder.state_dict().items():
                        value *= 1.0 / (1.0 - (num_failed / len(self.models_to_merge)))
                    for key, value in output_model.unet.state_dict().items():
                        value *= 1.0 / (1.0 - (num_failed / len(self.models_to_merge)))

                # add the base model back to the merged model
                # text encoder
                if isinstance(output_model.text_encoder, list):
                    # sdxl model
                    for i, text_encoder in enumerate(output_model.text_encoder):
                        for key, value in text_encoder.state_dict().items():
                            value += base_model.text_encoder[i].state_dict()[key]
                else:
                    # normal model
                    for key, value in output_model.text_encoder.state_dict().items():
                        value += base_model.text_encoder.state_dict()[key]
                # unet
                for key, value in output_model.unet.state_dict().items():
                    value += base_model.unet.state_dict()[key]

            if merge_step == 2:
                # save the mse dict
                with open(self.weight_json, "w") as f:
                    json.dump(mse_dict, f, indent=4)

            is_last = index == len(merge_steps) - 1

            if merge_step != 2:
                out_path = os.path.join(self.working_dir, f"tmp_merged_model.safetensors")
                if is_last and merge_step == 3:
                    out_path = self.save_path

                # merge loop is done, let's save the model
                print(f"Saving merged model to {out_path}")
                output_model.save(out_path, meta=self.meta, save_dtype=self.save_dtype)
                print(f"Saved merged model to {out_path}")
            # do cleanup here
            del output_model
            del base_model
            flush()

        # all steps complete, let's clean up
        flush()
