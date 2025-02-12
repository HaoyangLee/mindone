import logging
from typing import Dict, Optional, Tuple, Union

from hyvideo.constants import PRECISION_TO_TYPE
from hyvideo.modules import load_model
from jsonargparse.typing import Path_fr

import mindspore as ms
from mindspore import _no_grad, amp, jit_class, nn

from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.params import load_param_into_net_with_filter

__all__ = ["MODEL_DTYPE", "no_grad", "init_model", "resume_train_net"]

logger = logging.getLogger(__name__)

# MODEL_SPEC = {"llama-1B": llama3_1B, "llama-5B": llama3_5B, "llama-30B": llama3_30B}

MODEL_DTYPE = {
    "fp32": ms.float32,
    "fp16": ms.float16,
    "bf16": ms.bfloat16,
}


def load_ckpt_params(model: nn.Cell, ckpt: Union[str, Dict]) -> None:
    if isinstance(ckpt, str):
        logger.info(f"Loading {ckpt} params into network...")
        param_dict = ms.load_checkpoint(ckpt)
        param_dict = {k.replace("network.model.", ""): v for k, v in param_dict.items()}
    else:
        param_dict = ckpt

    param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict)
    if param_not_load or ckpt_not_load:
        logger.warning(
            f"Exist ckpt params not loaded: {ckpt_not_load} (total: {len(ckpt_not_load)}),\n"
            f"or net params not loaded: {param_not_load} (total: {len(param_not_load)})"
        )


@jit_class
class no_grad(_no_grad):
    """
    A context manager that suppresses gradient memory allocation in PyNative mode.
    """

    def __init__(self):
        super().__init__()
        self._pynative = ms.get_context("mode") == ms.PYNATIVE_MODE

    def __enter__(self):
        if self._pynative:
            super().__enter__()

    def __exit__(self, *args):
        if self._pynative:
            super().__exit__(*args)


def init_model(
    name: str = "",
    in_channels: int = 16,
    out_channels: int = 16,
    pretrained_model_path: Optional[Path_fr] = None,
    zero_stage: Optional[int] = None,
    text_states_dim: int = 4096,
    text_states_dim_2: int = 768,
    resume: bool = False,
    factor_kwargs: dict = {},
    use_fp8: bool = False,
    enable_ms_amp: bool = True,
    amp_level: str = "O2",
):
    dtype = factor_kwargs["dtype"]
    dtype = PRECISION_TO_TYPE[dtype]
    model = load_model(
        name=name,
        zero_stage=zero_stage,
        text_states_dim=text_states_dim,
        text_states_dim_2=text_states_dim_2,
        in_channels=in_channels,
        out_channels=out_channels,
        factor_kwargs=factor_kwargs,
    )

    if resume:
        logger.info("Resume training checkpoint provided, skipping weight loading.")
    elif pretrained_model_path:
        load_ckpt_params(model, pretrained_model_path.absolute)
    else:
        logger.info(f"Initialize {name} model randomly.")

    if use_fp8:
        raise NotImplementedError("fp8 is not supported yet.")

    if enable_ms_amp and dtype != ms.float32:
        logger.warning(f"Use MS auto mixed precision, amp_level: {amp_level}")
        if amp_level == "auto":
            amp.auto_mixed_precision(model, amp_level=amp_level, dtype=dtype)
        else:
            from hyvideo.modules.embed_layers import SinusoidalEmbedding
            from hyvideo.modules.norm_layers import FP32LayerNorm, LayerNorm, RMSNorm

            whitelist_ops = [
                LayerNorm,
                RMSNorm,
                FP32LayerNorm,
                SinusoidalEmbedding,
            ]
            logger.info("custom fp32 cell for dit: ", whitelist_ops)
            model = auto_mixed_precision(model, amp_level=amp_level, dtype=dtype, custom_fp32_cells=whitelist_ops)

    return model


def resume_train_net(
    train_net: TrainOneStepWrapper, resume_ckpt: Optional[Path_fr] = None
) -> Tuple[Union[int, None], Union[int, None]]:
    if resume_ckpt is None:
        return None, None

    state_dict = ms.load_checkpoint(resume_ckpt)
    if "epoch_num" not in state_dict or "cur_step" not in state_dict or "loss_scale" not in state_dict:
        raise ValueError("Resume training checkpoint is invalid. Please check the checkpoint file.")

    start_epoch = state_dict.pop("epoch_num").item()
    global_step = state_dict.pop("cur_step").item()
    logger.info(f"Resuming training of network from {resume_ckpt} at global step {global_step}")

    # FIXME: `EvalSaveCallback` renames `scale_sense` to `loss_scale` when saving the resume checkpoint
    train_net.scale_sense = ms.Parameter(state_dict.pop("loss_scale"), name="scale_sense")
    param_not_load, ckpt_not_load = load_param_into_net_with_filter(train_net, state_dict, filter=state_dict.keys())
    if param_not_load or ckpt_not_load:
        logger.warning(
            f"Exist ckpt params not loaded: {ckpt_not_load} (total: {len(ckpt_not_load)}),\n"
            f"or net params not loaded: {param_not_load} (total: {len(param_not_load)})"
        )

    return start_epoch, global_step
