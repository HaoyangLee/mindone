from typing import List, Optional, Union

from gm.util import append_dims
from omegaconf import ListConfig

import mindspore as ms
from mindspore import nn, ops
from mindspore import numpy as msnp


class StandardDiffusionLoss(nn.Cell):
    def __init__(
        self,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
        keep_loss_fp32=True,
    ):
        super().__init__()

        assert type in ["l2", "l1"]
        self.type = type
        self.offset_noise_level = offset_noise_level
        self.keep_loss_fp32 = keep_loss_fp32

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def get_noise_input(self, pred, noise, sigmas):
        input = pred
        if self.offset_noise_level > 0.0:
            noise = noise + self.offset_noise_level * append_dims(
                ops.randn(input.shape[0], dtype=input.dtype), input.ndim
            )
        noised_input = input + noise * append_dims(sigmas, input.ndim)
        return noised_input

    def construct(self, pred, target, w):
        dtype = pred.dtype
        if self.keep_loss_fp32:
            pred = ops.cast(pred, ms.float32)
            target = ops.cast(target, ms.float32)

        if self.type == "l2":
            loss = ops.mean((w * (pred - target) ** 2).reshape(target.shape[0], -1), 1).astype(dtype)
        elif self.type == "l1":
            loss = ops.mean((w * (pred - target).abs()).reshape(target.shape[0], -1), 1).astype(dtype)
        else:
            loss = 0.0
        return loss


class ControlNetLoss(StandardDiffusionLoss):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    def construct(self, pred, target, w):
        return self.p_losses(pred, target)
        # return super().construct(pred, target, w)


    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = nn.MSELoss(reduction="mean")(target, pred)
            else:
                loss = nn.MSELoss(reduction="none")(target, pred)
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, latent_image, text_emb, t, control=None, **kwargs):
        noise = msnp.randn(latent_image.shape)
        latent_image_noisy = self.q_sample(x_start=latent_image, t=t, noise=noise)

        model_output = self.model.diffusion_model(
            x=latent_image_noisy,
            timesteps=t,
            context=text_emb,
            control=control,
            only_mid_control=False,
            **kwargs,
        )

        if self.parameterization == "x0":
            target = latent_image
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "velocity":
            target = self.get_velocity(latent_image, noise, t)  # TODO: parse train step from randint
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])

        logvar_t = self.logvar[t]
        loss = loss_simple / ops.exp(logvar_t) + logvar_t
        loss = self.l_simple_weight * loss.mean()

        return loss