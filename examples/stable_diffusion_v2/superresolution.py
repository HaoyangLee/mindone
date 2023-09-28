import os
import sys
import logging
# import torch
import argparse
import numpy as np
# import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
# from einops import repeat, rearrange
# from pytorch_lightning import seed_everything
# from imwatermark import WatermarkEncoder

workspace = os.path.dirname(os.path.abspath(__file__))
print("workspace:", workspace, flush=True)
sys.path.append(workspace)

from mindspore import ops, Tensor
import mindspore as ms
# from scripts.txt2img import put_watermark
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentUpscaleDiffusion, LatentUpscaleFinetuneDiffusion
from ldm.modules.train.tools import set_random_seed
from ldm.modules.logger import set_logger

from ldm.util import exists, instantiate_from_config

logger = logging.getLogger("super_resolution")

##############################################
# torch.set_grad_enabled(False)

# def initialize_model(config, ckpt):
#     config = OmegaConf.load(config)
#     model = instantiate_from_config(config.model)
#     model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

#     device = torch.device(
#         "cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model = model.to(device)
#     sampler = DDIMSampler(model)
#     return sampler

#--------------------------------------------#
def load_model_from_config(config, ckpt, verbose=False):
    if not os.path.isabs(config):
        config = os.path.join(workspace, config)
    config = OmegaConf.load(f"{config}")

    logger.info(f"Loading model from {ckpt}")
    model = instantiate_from_config(config.model)
    ####################
    # dump_net_param(model)
    # import troubleshooter as ts
    # ts.migrator.save_net_and_weight_params(model, path="/home/lihaoyang/ckpt/superresilution_ts/")

    if os.path.exists(ckpt):
        param_dict = ms.load_checkpoint(ckpt)
        if param_dict:
            param_not_load, _ = ms.load_param_into_net(model, param_dict)
            logger.info("Net params not loaded: {}".format(param_not_load))
    else:
        logger.warning(f"!!!Warning!!!: {ckpt} doesn't exist")

    sampler = DDIMSampler(model)

    return sampler

def dump_net_param(net, dump_path="/home/lihaoyang/ckpt/ms_net_param_dis.ckpt"):
    import json
    from mindspore import save_checkpoint
    print("*" * 40)
    print("network structure")
    print("*" * 40)
    print(net)
    net_param = list()
    with open("/home/lihaoyang/code/mindone/examples/stable_diffusion_v2/tools/model_conversion/superresolution/ms_net_param_dis.txt", "w") as f:        
        for param in net.get_parameters():
            net_param.append({"name": param.name, "data": param.data})
            f.write(f"{param.name}#{param.data.shape}#{param.data.dtype}\n")
    save_checkpoint(net_param, dump_path)
    # with open(dump_path, "w") as f:
    #     json.dump(net_param, f)
    print(f"INFO: The mindspore network params are dumped as {dump_path}.")
    return net_param
##############################################

def make_batch_sd(
        image,
        txt,
        num_samples=1,
):
    image = np.array(image.convert("RGB"))
    image = Tensor.from_numpy(image).to(dtype=ms.float32) / 127.5 - 1.0
    batch = {
        ###########################
        # "lr": rearrange(image, 'h w c -> 1 c h w'),
        #-------------------------#
        "lr": ops.transpose(image[None], (0, 3, 1, 2)),
        ###########################
        "txt": num_samples * [txt],
    }
    ###########################
    # batch["lr"] = repeat(batch["lr"].to(device=device),
    #                      "1 ... -> n ...", n=num_samples)
    #-------------------------#
    batch["lr"] = batch["lr"].repeat(num_samples, axis=0)
    ###########################
    return batch


def make_noise_augmentation(model, batch, noise_level=None):
    x_low = batch[model.low_scale_key]
    ###################
    # x_low = x_low.to(memory_format=torch.contiguous_format).float()
    #-----------------#
    x_low = x_low.to(dtype=ms.float32)
    ###################
    x_aug, noise_level = model.low_scale_model(x_low, noise_level)
    return x_aug, noise_level


def paint(sampler, image, prompt, seed, scale, h, w, steps, num_samples=1, callback=None, eta=0., noise_level=None):
    # device = torch.device(
    #     "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model
    #################################
    # seed_everything(seed)
    #--------------------------------#
    set_random_seed(seed)
    #################################

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, model.channels, h, w)
    ##################################
    # start_code = torch.from_numpy(start_code).to(
    #     device=device, dtype=torch.float32)
    #--------------------------------#
    start_code = Tensor.from_numpy(start_code).to(dtype=ms.float32)
    ##################################

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "SDV2"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    #######################################
    # with torch.no_grad(), torch.autocast("cuda"):
    #######################################
    batch = make_batch_sd(
        image, txt=prompt, num_samples=num_samples)
    tokenized_txt = model.tokenize(batch["txt"])
    c = model.cond_stage_model.encode(tokenized_txt)
    c_cat = list()
    if isinstance(model, LatentUpscaleFinetuneDiffusion):
        for ck in model.concat_keys:
            cc = batch[ck]
            if exists(model.reshuffle_patch_size):
                assert isinstance(model.reshuffle_patch_size, int)
                cc = rearrange(cc, 'b c (p1 h) (p2 w) -> b (p1 p2 c) h w',
                                p1=model.reshuffle_patch_size, p2=model.reshuffle_patch_size)
            c_cat.append(cc)
        ##################################
        # c_cat = torch.cat(c_cat, dim=1)
        #--------------------------------#
        c_cat = ops.cat(c_cat, axis=1)
        ##################################
        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}
        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
    elif isinstance(model, LatentUpscaleDiffusion):
        x_augment, noise_level = make_noise_augmentation(
            model, batch, noise_level)
        cond = {"c_concat": [x_augment],
                "c_crossattn": [c], "c_adm": noise_level}
        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [x_augment], "c_crossattn": [
            uc_cross], "c_adm": noise_level}
    else:
        raise NotImplementedError()

    shape = [model.channels, h, w]
    samples, intermediates = sampler.sample(
        steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc_full,
        x_T=start_code,
        callback=callback
    )

    ######################
    # with torch.no_grad():
    #     x_samples_ddim = model.decode_first_stage(samples)
    #---------------------#
    x_samples_ddim = model.decode_first_stage(samples)
    ######################

    result = ops.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    #######################
    # result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    #---------------------#
    result = result.asnumpy().transpose(0, 2, 3, 1) * 255
    #######################

    # return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]
    return result

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded


def predict(image_path, prompt, steps, num_samples, scale, seed, eta, noise_level, **kwargs):
    input_image = Image.open(image_path)
    init_image = input_image.convert("RGB")
    image = pad_image(init_image)  # resize to integer multiple of 32
    width, height = image.size
    
    ##################
    # noise_level = torch.Tensor(
    #     num_samples * [noise_level]).to(sampler.model.device).long()
    #----------------#
    noise_level = Tensor(num_samples * [noise_level]).long()
    ##################
    sampler.make_schedule(steps, ddim_eta=eta, verbose=True)
    result = paint(
        sampler=sampler,
        image=image,
        prompt=prompt,
        seed=seed,
        scale=scale,
        h=height, w=width, steps=steps,
        num_samples=num_samples,
        callback=None,
        noise_level=noise_level
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("-i", "--image_path", type=str, default="data/merged-dog-half.png", help="path to input image")
    parser.add_argument("-p", "--prompt", type=str, default="a beautiful dog", help="prompt text")
    parser.add_argument("-c", "--ckpt", type=str, default="/home/lihaoyang/ckpt/x4-upscaler-ema_ms.ckpt", help="path to ckpt")
    # parser.add_argument("-i", "--image", type=str, required=True, help="path to input image")
    # parser.add_argument("-p", "--prompt", type=str, required=True, help="prompt text")
    # parser.add_argument("-c", "--ckpt", type=str, required=True, help="path to ckpt")
    parser.add_argument("--config", type=str, default="configs/x4-upscaling.yaml", help="path to config file")
    # parser.add_argument("--steps", type=int, default=75, help="DDIM step")
    parser.add_argument("--steps", type=int, default=20, help="DDIM step")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples")
    parser.add_argument("--scale", type=float, default=10.0, help="Scale")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--eta", type=float, default=0.0, help="eta (DDIM)")
    parser.add_argument("--noise_level", type=int, default=20, help="Noise augmentation")
    parser.add_argument("--output_path", default="output/", type=str, help="output directory to save log")
    parser.add_argument("--log_level", type=str, default="logging.INFO", choices=["logging.DEBUG", "logging.INFO", "logging.WARNING", "logging.ERROR"], help="log level")

    args = parser.parse_args()
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.context.set_context(mode=args.ms_mode, device_target="Ascend", device_id=device_id, max_device_memory="30GB")

    #########################
    # sampler = initialize_model(sys.argv[1], sys.argv[2])
    #-----------------------#
    sampler = load_model_from_config(args.config, args.ckpt)
    #########################

    # set logger
    set_logger(
        name="",
        output_dir=args.output_path,
        rank=0,
        log_level=eval(args.log_level),
    )

    result = predict(**vars(args))
    # result = predict(input_image, prompt, steps, num_samples, scale, seed, eta, noise_level)

    # block = gr.Blocks().queue()
    # with block:
    #     with gr.Row():
    #         gr.Markdown("## Stable Diffusion Upscaling")

    #     with gr.Row():
    #         with gr.Column():
    #             input_image = gr.Image(source='upload', type="pil")
    #             gr.Markdown(
    #                 "Tip: Add a description of the object that should be upscaled, e.g.: 'a professional photograph of a cat")
    #             prompt = gr.Textbox(label="Prompt")
    #             run_button = gr.Button(label="Run")
    #             with gr.Accordion("Advanced options", open=False):
    #                 num_samples = gr.Slider(
    #                     label="Number of Samples", minimum=1, maximum=4, value=1, step=1)
    #                 steps = gr.Slider(label="DDIM Steps", minimum=2,
    #                                 maximum=200, value=75, step=1)
    #                 scale = gr.Slider(
    #                     label="Scale", minimum=0.1, maximum=30.0, value=10, step=0.1
    #                 )
    #                 seed = gr.Slider(
    #                     label="Seed",
    #                     minimum=0,
    #                     maximum=2147483647,
    #                     step=1,
    #                     randomize=True,
    #                 )
    #                 eta = gr.Number(label="eta (DDIM)",
    #                                 value=0.0, min=0.0, max=1.0)
    #                 noise_level = None
    #                 if isinstance(sampler.model, LatentUpscaleDiffusion):
    #                     # TODO: make this work for all models
    #                     noise_level = gr.Number(
    #                         label="Noise Augmentation", min=0, max=350, value=20, step=1)

    #         with gr.Column():
    #             gallery = gr.Gallery(label="Generated images", show_label=False).style(
    #                 grid=[2], height="auto")

    #     run_button.click(fn=predict, inputs=[
    #                     input_image, prompt, steps, num_samples, scale, seed, eta, noise_level], outputs=[gallery])

    # block.launch()
