"""
Dump the param names of pytorch ckpt (https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/resolve/main/x4-upscaler-ema.ckpt) to text.
Build and dump the mindspore network param names to text.

>>> torch_params = torch.load("path/to/x4-upscaler-ema.ckpt", map_location="cpu")["state_dict"]
>>> with open("path/to/pt_names_x4-upscaler-ema.txt", "w") as f:
>>>     for name, value in torch_params.items():
>>>         f.write(f"{name}#{value.shape}#{value.dtype}\n")

In our repo, the pytorch and mindspore network param names text is prepared already. The mindspore ckpt can be obtained by downloading x4-upscaler-ema.ckpt 
and running this script.

$ python convert.py

"""

import os
import sys

import torch
import mindspore as ms
from omegaconf import OmegaConf

# sys.path.append("examples/stable_diffusion_v2")
# from examples.stable_diffusion_v2.ldm.util import instantiate_from_config

__dir__ = os.path.dirname(os.path.abspath(__file__))


# match the param names
skip_pt_names = (
    "betas",
    "alphas_cumprod",
    "alphas_cumprod_prev",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "log_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod",
    "posterior_variance",
    "posterior_log_variance_clipped",
    "posterior_mean_coef1",
    "posterior_mean_coef2",
    "model_ema.decay",
    "model_ema.num_updates",
    "cond_stage_model.model.text_projection",
    "cond_stage_model.model.logit_scale",
    "transformer.resblocks.23",    # prefix
    "low_scale_model",             # prefix
    "token_embedding",             # prefix
    )

skip_ms_names = (
    "token_embedding",
)

# one special case
token_embedding_pt2ms_mapping = {
    "cond_stage_model.model.token_embedding.weight": "cond_stage_model.model.token_embedding.embedding_table"
}


def _skip(skip_names: tuple, line: str) -> bool:
    param_name, *_ = line.strip().split("#")
    for skip_name in skip_names:
        if skip_name in param_name:
            return True
    return False


def pt2ms_weight(pt_names_path: str, ms_names_path: str, pt_weights_path: str, config_path: str) -> None:
    with open(os.path.join(__dir__, pt_names_path), "r") as f:
        pt_lines = f.readlines()
    with open(os.path.join(__dir__, ms_names_path), "r") as f:
        ms_lines = f.readlines()

    # skip some pytorch param names
    pt_lines_filtered = [line for line in pt_lines if not _skip(skip_pt_names, line)]
    ms_lines_filtered = [line for line in ms_lines if not _skip(skip_ms_names, line)]

    assert len(pt_lines_filtered) == len(ms_lines_filtered), "Weights cannot be converted, because numbers of pt and ms params are not equal. Please check the param names and the skip condition."

    print(f"INFO: Converting {len(pt_lines_filtered) + 1} weights...")  # including one special case
    pt_weights = torch.load(pt_weights_path, map_location="cpu")["state_dict"]
    ms_weights = list()
    for pt_line, ms_line in zip(pt_lines_filtered, ms_lines_filtered):
        pt_name, pt_shape, pt_dtype = pt_line.strip().split("#")
        ms_name, ms_shape, ms_dtype = ms_line.strip().split("#")

        # TODO:
        # assert pt_shape == ms_shape, 
        # if pt_dtype != ms_dtype:
        #     print("WARNING:")

        # TODO: positional_embedding in pt param is float32, in ms param is float16

        ms_weight = ms.Tensor(pt_weights[pt_name].numpy())
        ms_weights.append({"name": ms_name, "data": ms_weight})

    # deal with one special case: token embedding
    special_pt_name = "cond_stage_model.model.token_embedding.weight"
    special_ms_name = token_embedding_pt2ms_mapping[special_pt_name]
    ms_weight = ms.Tensor(pt_weights[special_pt_name].numpy())
    # ms_weight = ms.Parameter(pt_weights[special_pt_name].numpy())
    ms_weights.append({"name": special_ms_name, "data": ms_weight})

    # TODO: check param loading into ms network
    # config = OmegaConf.load(config_path)
    # model = instantiate_from_config(config.model)
    # param_not_load, ckpt_not_load = ms.load_param_into_net(network, ms_weights)
    # if param_not_load:
    #     raise ValueError(f"The following parameters in network were not loaded:\n{param_not_load}")
    # if ckpt_not_load:
    #     raise ValueError(f"The following parameters in ckpt were not loaded:\n{ckpt_not_load}")
    print(f"INFO: MindSpore weights converted successfully! Number of params: {len(ms_weights)}.")

    ms_weights_path = os.path.splitext(pt_weights_path)[0] + "_ms.ckpt"
    ms.save_checkpoint(ms_weights, ms_weights_path)
    print(f"INFO: Saved in '{ms_weights_path}'.")


if __name__ == "__main__":
    pt_names_path = "pt_names_x4-upscaler-ema.txt"
    ms_names_path = "ms_names_x4-upscaler-ema.txt"
    pt_weights_path = "/home/lihaoyang/ckpt/x4-upscaler-ema.ckpt"
    config_path = "/home/lihaoyang/code/mindone/examples/stable_diffusion_v2/configs/x4-upscaling.yaml"
    pt2ms_weight(pt_names_path, ms_names_path, pt_weights_path, config_path)
