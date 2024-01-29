import sys
sys.path.append("../../")
import argparse
import copy

from omegaconf import OmegaConf
import mindspore as ms
from mindspore import load_param_into_net
from mindspore.train.serialization import _update_param

from gm.helpers import load_model_from_config


def read_ckpt_params(args):
    param_dict = ms.load_checkpoint(args.pretrained_model_path)
    for pn in param_dict:
        print(pn)


def convert(args):
    config = OmegaConf.load(args.model_config)
    sd_controlnet = load_model_from_config(config.model)
    print("Load SD checkpoint from ", args.pretrained_model_path, "to ", sd_controlnet.__class__.__name__)

    # load sd main
    pretrained_weights = ms.load_checkpoint(args.pretrained_model_path)
    param_not_load, ckpt_not_load = load_param_into_net(sd_controlnet, pretrained_weights)
    print("Net params not load: {}".format(param_not_load))
    print("Checkpoint parm not load: {}".format(param_not_load))

    # copy sd encoder weights to controlnet
    # prior: controlnet param names start with "model.diffusion_model.controlnet", e.g. model.diffusion_model.controlnet.input_blocks.0.0.conv.weight
    # removing "controlnet." from it gives the original param name in sd
    # except for input_hint_block for control image encoding.
    net_params = sd_controlnet.get_parameters()
    for net_param in net_params:
        if "controlnet." in net_param.name:
            sd_param_name = net_param.name.replace("controlnet.", "")
            if sd_param_name in pretrained_weights:
                new_param = copy.deepcopy(pretrained_weights[sd_param_name])
                _update_param(net_param, new_param, strict_load=False)
                print(f"Copied {sd_param_name} -> {net_param.name}")
            else:
                print(
                    f"WARNING: {sd_param_name} not in preatrined_weights! Ignore this warning if the param belongs to input hint block or zero moduels or middle_block_out."
                )

    save_fn = args.pretrained_model_path.replace(".ckpt", "_controlnet_init.ckpt")
    ms.save_checkpoint(sd_controlnet, save_fn)

    print("Finish! Checkpoint saved in : ", save_fn)

    return save_fn


def validate(controlnet_init_ckpt):
    param_dict = ms.load_checkpoint(controlnet_init_ckpt)
    count = 0
    for pn in param_dict:
        if "controlnet." in pn:
            sd_pn = pn.replace("controlnet.", "")
            if sd_pn in param_dict:
                count += 1
                assert param_dict[pn].asnumpy().sum() == param_dict[sd_pn].asnumpy().sum()
    print(f"{count} controlnet params are validated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config",
        default="configs/training/sd_xl_base_finetune_controlnet_910b.yaml",
        type=str,
        help="sdxl with controlnet model config path"
    )
    parser.add_argument(
        "--pretrained_model_path",
        "-p",
        default="checkpoints/sd_xl_base_1.0_ms.ckpt",
        type=str,
        help="Specify the pretrained model from this checkpoint",
    )
    parser.add_argument(
        "--controlnet_init_ckpt",
        type=str,
        help="path to merged sdxl controlnet init ckpt, for validating the correctness of merging",
    )
    args = parser.parse_args()
    convert(args)
    # validate(args.controlnet_init_ckpt)
