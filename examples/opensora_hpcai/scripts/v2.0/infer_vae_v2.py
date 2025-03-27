import argparse
import logging
import os
import sys
import time
from pathlib import Path
import yaml

import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore import nn
from mindspore.communication.management import get_group_size, get_rank, init

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from opensora.datasets.video_dataset import create_dataloader
from opensora.models.hunyuan_vae import CausalVAE3D_HUNYUAN
from opensora.utils.amp import auto_mixed_precision
from opensora.utils.model_utils import str2bool

from mindone.utils.logger import set_logger
from mindone.utils.misc import to_abspath
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def save_output(output_folder: Path,
                video_name: Path,
                mean: np.array,
                std: np.array,
                ):
    fn = video_name.with_suffix(".npz")
    npz_fp = os.path.join(output_folder, fn)
    if not os.path.exists(os.path.dirname(npz_fp)):
        os.makedirs(os.path.dirname(npz_fp))
    if os.path.exists(npz_fp):
        if args.allow_overwrite:
            logger.info(f"Overwritting {npz_fp}")
    np.savez(
        npz_fp,
        latent_mean=mean.astype(np.float32),
        latent_std=std.astype(np.float32),
    )


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    jit_level: str = "O0",
):
    """
    Initialize MindSpore environment.

    Args:
        mode: MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        seed: The seed value for reproducibility. Default is 42.
        distributed: Whether to enable distributed training. Default is False.
    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)

    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
        ms.reset_auto_parallel_context()

        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )

        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, int(device_num / 8), int(rank_id / 8)]
        logger.info(dict(zip(var_info, var_value)))

    else:
        device_num = 1
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )

    try:
        if jit_level in ["O0", "O1", "O2"]:
            ms.set_context(jit_config={"jit_level": jit_level})
        else:
            logger.warning(
                f"Unsupport jit_level: {jit_level}. The framework automatically selects the execution method"
            )
    except Exception:
        logger.warning(
            "The current jit_level is not suitable because current MindSpore version or mode does not match,"
            "please ensure the MindSpore version >= ms2.3_0615, and use GRAPH_MODE."
        )

    return rank_id, device_num


def main(args):
    set_logger(name="", output_dir="logs/infer_vae_v2")
    rank_id, device_num = init_env(
        args.mode, args.seed, args.use_parallel, device_target=args.device_target,
    )
    print(f"rank_id {rank_id}, device_num {device_num}")

    # build dataloader for large amount of captions
    ds_config = dict(
        csv_path=args.csv_path,
        video_folder=args.video_folder,
        sample_size=args.video_size,
        sample_stride=args.frame_stride,
        micro_batch_size=args.num_frames,
        video_column=args.video_column,
        caption_column=args.caption_column,
        return_frame_data=True,
        resize_by_max_value=args.resize_by_max_value,
        transform_name=args.transform_name,
        filter_data=args.filter_data,
    )
    dataloader, _ = create_dataloader(
        ds_config,
        batch_size=1,
        ds_name="video",
        num_parallel_workers=16,
        max_rowsize=256,
        shuffle=False,  # be in order
        device_num=device_num,
        rank_id=rank_id,
        drop_remainder=False,
        return_dataset=False,
    )
    dataset_size = dataloader.get_dataset_size()
    logger.info(f"Num batches: {dataset_size}")

    # model initiate and weight loading
    logger.info("vae init")
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ae_config = config["ae"]
    ae_config["dtype"] = args.vae_precision
    if args.vae_checkpoint is not None:
        ae_config["from_pretrained"] = args.vae_checkpoint

    model_ae = CausalVAE3D_HUNYUAN(**ae_config).set_train(False)
    del model_ae.decoder
    
    dtype_map = {"fp16": ms.float16, "bf16": ms.bfloat16}
    if args.vae_precision in ["fp16", "bf16"]:
        vae = auto_mixed_precision(
            vae,
            amp_level=args.amp_level,
            dtype=dtype_map[args.vae_precision],
            custom_fp32_cells=[nn.GroupNorm],
        )

    logger.info("Start VAE embedding...")

    # infer
    if args.csv_path is not None:
        if args.output_path in [None, ""]:
            output_folder = os.path.dirname(args.csv_path)
        else:
            output_folder = args.output_path
        os.makedirs(output_folder, exist_ok=True)

        logger.info(f"Output embeddings will be saved: {output_folder}")

        ds_iter = dataloader.create_dict_iterator(1, output_numpy=True)
        for step, data in tqdm(enumerate(ds_iter), total=dataset_size):
            start_time = time.time()

            frame_data = data["frame_data"]
            num_videos = frame_data.shape[0]
            for i in range(num_videos):
                abs_video_path = data["video_path"][i]
                video_path = Path(abs_video_path).relative_to(args.video_folder)
                fn = video_path.with_suffix(".npz")
                npz_fp = os.path.join(output_folder, fn)
                if os.path.exists(npz_fp) and not args.allow_overwrite:
                    logger.info(f"{npz_fp} exists, skip vae encoding")
                    continue

                x = frame_data[i]
                x = np.expand_dims(np.transpose(x, (1, 0, 2, 3)), axis=0)  # [f, c, h, w] -> [b, c, f, h, w], b must be 1

                if x.shape[2] < args.num_frames:
                    msg = f"Video {video_path} is too short! It has {x.shape[2]} frames, but expected >= {args.num_frames} frames."
                    if args.drop_short_video:
                        logger.warning(f"{msg} Drop it")
                        continue
                    else:
                        raise ValueError(f"{msg} Please set a smaller --num_frames or set --drop_short_video=True")
                x = x[:, :, : args.num_frames, :, :]    # keep the first args.num_frames of a video

                logger.info(f"The shape [b, c, f, h, w] of video for vae encoding: {x.shape}")

                x_0, posterior = ms.ops.stop_gradient(model_ae.encode(ms.Tensor(x, ms.float32), return_posterior=True))
                video_latent_mean = posterior.mean.asnumpy()
                video_latent_std = posterior.std.asnumpy()
                save_output(output_folder, video_path, video_latent_mean, video_latent_std)

            end_time = time.time()
            logger.info(f"Time cost: {end_time-start_time:0.3f}s")
        logger.info(f"Done. Embeddings saved in {output_folder}")

    else:
        raise ValueError("Must provide csv file!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="configs/opensora-v2-0/train/image.yaml",
        help="Path to load a config yaml file, but only use the ae config in it.",
    )
    parser.add_argument(
        "--csv_path",
        default=None,
        type=str,
        help="path to csv annotation file, If None, video_caption.csv is expected to live under `data_path`",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="output dir to save the embeddings, if None, will treat the parent dir of csv_path as output_path.",
    )
    parser.add_argument("--video_column", default="video", type=str, help="name of column for videos saved in csv file")
    parser.add_argument(
        "--caption_column", default="caption", type=str, help="name of column for captions saved in csv file"
    )
    parser.add_argument("--video_folder", default="", type=str, help="root dir for the video data")
    parser.add_argument("--filter_data", default=False, type=str2bool, help="Filter non-existing videos.")
    parser.add_argument("--video_size", nargs="+", default=[256, 256], type=int, help="video size for vae encoder input")
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default=None,
        help="VAE checkpoint file path which is used to load vae weight, will overwrite the path in ae_config.",
    )
    parser.add_argument(
        "--vae_precision",
        type=str,
        default="fp32",
        choices=["bf16", "fp16", "fp32"],
        help="Precision mode for the VAE model: fp16, bf16, or fp32.",
    )

    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument("--frame_stride", default=1, type=int, help="frame sampling stride")
    parser.add_argument(
        "--transform_name",
        default="crop_resize",
        type=str,
        help="center or crop_resize, if center, resize by the short side to h \
                then center crop. If crop_resize, center crop maximally according to \
                the AR of target image size then resize, suitable for where target h != target w.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=129,
        help="Crop the video to num_frames frames.",
    )
    parser.add_argument(
        "--drop_short_video",
        type=str2bool,
        default=False,
        help="How to deal with the video whose frames < args.num_frames. True: drop them; False: raise error, set a smaller --num_frames",
    )    
    parser.add_argument(
        "--allow_overwrite",
        type=str2bool,
        default=False,
        help="If True, allow to overwrite the existing npz file. If False, will skip vae encoding if the latent npz file is already existed",
    )
    parser.add_argument("--resize_by_max_value", default=False, type=str2bool, help="resize the image by max instead.")

    __dir__ = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.abspath(os.path.join(__dir__, ".."))
    args = parser.parse_args()
    # convert to absolute path, necessary for modelarts
    args.csv_path = to_abspath(abs_path, args.csv_path)
    args.output_path = to_abspath(abs_path, args.output_path)
    args.video_folder = to_abspath(abs_path, args.video_folder)
    args.vae_checkpoint = to_abspath(abs_path, args.vae_checkpoint)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
