# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PEFT utilities: Utilities related to peft library
"""
import collections
from typing import Optional


def recurse_remove_peft_layers(model):
    r"""
    Recursively replace all instances of `LoraLayer` with corresponding new layers in `model`.
    """
    from mindone.diffusers._peft.utils import _get_submodules

    key_list = [key for key, _ in model.cells_and_names() if "lora" not in key]
    for key in key_list:
        try:
            parent, target, target_name = _get_submodules(model, key)
        except AttributeError:
            continue
        if hasattr(target, "base_layer"):
            setattr(parent, target_name, target.get_base_layer())
    return model


def scale_lora_layers(model, weight):
    """
    Adjust the weightage given to the LoRA layers of the model.

    Args:
        model (`torch.nn.Module`):
            The model to scale.
        weight (`float`):
            The weight to be given to the LoRA layers.
    """
    from mindone.diffusers._peft.tuners.tuners_utils import BaseTunerLayer

    for _, module in model.cells_and_names():
        if isinstance(module, BaseTunerLayer):
            module.scale_layer(weight)


def unscale_lora_layers(model, weight: Optional[float] = None):
    """
    Removes the previously passed weight given to the LoRA layers of the model.

    Args:
        model (`torch.nn.Module`):
            The model to scale.
        weight (`float`, *optional*):
            The weight to be given to the LoRA layers. If no scale is passed the scale of the lora layer will be
            re-initialized to the correct value. If 0.0 is passed, we will re-initialize the scale with the correct
            value.
    """
    from mindone.diffusers._peft.tuners.tuners_utils import BaseTunerLayer

    for _, module in model.cells_and_names():
        if isinstance(module, BaseTunerLayer):
            if weight is not None and weight != 0:
                module.unscale_layer(weight)
            elif weight is not None and weight == 0:
                for adapter_name in module.active_adapters:
                    # if weight == 0 unscale should re-set the scale to the original value.
                    module.set_scale(adapter_name, 1.0)


def get_peft_kwargs(rank_dict, network_alpha_dict, peft_state_dict, is_unet=True):
    rank_pattern = {}
    alpha_pattern = {}
    r = lora_alpha = list(rank_dict.values())[0]

    if len(set(rank_dict.values())) > 1:
        # get the rank occuring the most number of times
        r = collections.Counter(rank_dict.values()).most_common()[0][0]

        # for modules with rank different from the most occuring rank, add it to the `rank_pattern`
        rank_pattern = dict(filter(lambda x: x[1] != r, rank_dict.items()))
        rank_pattern = {k.split(".lora_B.")[0]: v for k, v in rank_pattern.items()}

    if network_alpha_dict is not None and len(network_alpha_dict) > 0:
        if len(set(network_alpha_dict.values())) > 1:
            # get the alpha occuring the most number of times
            lora_alpha = collections.Counter(network_alpha_dict.values()).most_common()[0][0]

            # for modules with alpha different from the most occuring alpha, add it to the `alpha_pattern`
            alpha_pattern = dict(filter(lambda x: x[1] != lora_alpha, network_alpha_dict.items()))
            if is_unet:
                alpha_pattern = {
                    ".".join(k.split(".lora_A.")[0].split(".")).replace(".alpha", ""): v
                    for k, v in alpha_pattern.items()
                }
            else:
                alpha_pattern = {".".join(k.split(".down.")[0].split(".")[:-1]): v for k, v in alpha_pattern.items()}
        else:
            lora_alpha = set(network_alpha_dict.values()).pop()

    # layer names without the Diffusers specific
    target_modules = list({name.split(".lora")[0] for name in peft_state_dict.keys()})

    lora_config_kwargs = {
        "r": r,
        "lora_alpha": lora_alpha,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "target_modules": target_modules,
    }
    return lora_config_kwargs


def get_adapter_name(model):
    from mindone.diffusers._peft.tuners.tuners_utils import BaseTunerLayer

    for _, module in model.cells_and_names():
        if isinstance(module, BaseTunerLayer):
            return f"default_{len(module.r)}"
    return "default_0"


def set_adapter_layers(model, enabled=True):
    from mindone.diffusers._peft.tuners.tuners_utils import BaseTunerLayer

    for _, module in model.cells_and_names():
        if isinstance(module, BaseTunerLayer):
            # The recent version of PEFT needs to call `enable_adapters` instead
            if hasattr(module, "enable_adapters"):
                module.enable_adapters(enabled=enabled)
            else:
                raise RuntimeError("'BaseTunerLayer' object has no attribute 'enable_adapters'")


def delete_adapter_layers(model, adapter_name):
    from mindone.diffusers._peft.tuners.tuners_utils import BaseTunerLayer

    for _, module in model.cells_and_names():
        if isinstance(module, BaseTunerLayer):
            if hasattr(module, "delete_adapter"):
                module.delete_adapter(adapter_name)
            else:
                raise RuntimeError("'BaseTunerLayer' object has no attribute 'delete_adapter'")

    # For transformers integration - we need to pop the adapter from the config
    if getattr(model, "_hf_peft_config_loaded", False) and hasattr(model, "peft_config"):
        model.peft_config.pop(adapter_name, None)
        # In case all adapters are deleted, we need to delete the config
        # and make sure to set the flag to False
        if len(model.peft_config) == 0:
            del model.peft_config
            model._hf_peft_config_loaded = None


def set_weights_and_activate_adapters(model, adapter_names, weights):
    from mindone.diffusers._peft.tuners.tuners_utils import BaseTunerLayer

    # iterate over each adapter, make it active and set the corresponding scaling weight
    for adapter_name, weight in zip(adapter_names, weights):
        for _, module in model.cells_and_names():
            if isinstance(module, BaseTunerLayer):
                # For backward compatibility with previous PEFT versions
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                else:
                    raise RuntimeError("'BaseTunerLayer' object has no attribute 'set_adapter'")
                module.set_scale(adapter_name, weight)

    # set multiple active adapters
    for _, module in model.cells_and_names():
        if isinstance(module, BaseTunerLayer):
            # For backward compatibility with previous PEFT versions
            if hasattr(module, "set_adapter"):
                module.set_adapter(adapter_names)
            else:
                raise RuntimeError("'BaseTunerLayer' object has no attribute 'set_adapter'")