# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import logging
from functools import partial

import numpy as np
from ldm.modules.diffusionmodules.util import make_beta_schedule
from ldm.util import default, exists, extract_into_tensor, instantiate_from_config

from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import numpy as msnp
from mindspore import ops
from omegaconf import ListConfig

_logger = logging.getLogger(__name__)


def disabled_train(self, mode=True):
    """
    Overwrite model.set_train with this function to make sure train/eval mode does not change anymore.
    """
    self.set_train(False)
    return self


class DDPM(nn.Cell):
    def __init__(
        self,
        unet_config,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        image_size=256,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
        use_fp16=False,
    ):
        """
        Classic DDPM with Gaussian diffusion
        ===============================================================
        Args:
            v_posterior: weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta.
            parameterization:
                eps - epsilon (predicting the noise of the diffusion process)
                x0 - orginal (latent) image (directly predicting the noisy sample)
                velocity - velocity of z (see section 2.4 https://imagen.research.google/video/paper.pdf).
        """

        super().__init__()
        assert parameterization in ["eps", "x0", "velocity"], f"'parameterization' should be 'eps', 'x0' or 'velocity', but got '{parameterization}.'"
        self.parameterization = parameterization
        _logger.debug(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.dtype = mstype.float16 if use_fp16 else mstype.float32
        self.use_scheduler = scheduler_config is not None
        self.use_ema = use_ema
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        assert original_elbo_weight == 0.0, "Variational lower bound loss has been removed."

        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
        self.isnan = ops.IsNan()
        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = Tensor(np.full(shape=(self.num_timesteps,), fill_value=logvar_init).astype(np.float32))
        if self.learn_logvar:
            self.logvar = Parameter(self.logvar, requires_grad=True)
        self.randn_like = ops.StandardNormal()
        self.mse_mean = nn.MSELoss(reduction="mean")
        self.mse_none = nn.MSELoss(reduction="none")

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas have to be defined for each timestep"

        to_mindspore = partial(Tensor, dtype=self.dtype)
        self.betas = to_mindspore(betas)
        self.alphas_cumprod = to_mindspore(alphas_cumprod)
        self.alphas_cumprod_prev = to_mindspore(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_mindspore(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_mindspore(np.sqrt(1.0 - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_mindspore(np.log(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_mindspore(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_mindspore(np.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        ) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = to_mindspore(posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = to_mindspore(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = to_mindspore(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2 = to_mindspore((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))

        # if self.parameterization == "eps":
        #    lvlb_weights = self.betas ** 2 / (
        #                2 * self.posterior_variance * to_mindspore(alphas) * (1 - self.alphas_cumprod))
        # elif self.parameterization == "x0":
        #    lvlb_weights = 0.5 * msnp.sqrt(Tensor(alphas_cumprod)) / (2. * 1 - Tensor(alphas_cumprod))
        # elif self.parameterization == "velocity":
        #    # TODO: confirm
        #    lvlb_weights = self.betas ** 2 / (
        #                2 * self.posterior_variance * to_mindspore(alphas) * (1 - self.alphas_cumprod))
        # else:
        #    raise NotImplementedError("mu not supported")
        # lvlb_weights[0] = lvlb_weights[1]
        # self.lvlb_weights = to_mindspore(lvlb_weights)

    def get_velocity(self, sample, noise, t):
        # TODO: how t affects noise mean and variance here. all variance fixed?
        v = (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, sample.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, sample.shape) * sample
        )
        return v

    # TODO: it's a good practice. May adopt it later.
    # with ema_scopte(): save_model(), run_eval()
    """
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            parameters = self.model.get_parameters()
            trained_parameters = [param for param in parameters if param.requires_grad is True ]
            self.model_ema.store(iter(trained_parameters))
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                parameters = self.model.get_parameters()
                trained_parameters = [param for param in parameters if param.requires_grad is True]
                self.model_ema.restore(iter(trained_parameters))
                if context is not None:
                    print(f"{context}: Restored training weights")
    """

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

    def q_sample(self, x_start, t, noise):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )


class LatentDiffusion(DDPM):
    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        *args,
        **kwargs,
    ):
        """
        main class
        """
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except Exception:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", Tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.uniform_int = ops.UniformInt()

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        self.transpose = ops.Transpose()

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model
        for param in self.first_stage_model.get_parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model
            for param in self.cond_stage_model.get_parameters():
                param.requires_grad = False
        else:
            assert config != "__is_first_stage__"
            assert config != "__is_unconditional__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def tokenize(self, c):
        tokenized_res = self.cond_stage_model.tokenize(c)
        return tokenized_res

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            c = self.cond_stage_model.encode(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_learned_conditioning_fortrain(self, c):
        c = self.cond_stage_model(c)
        return c

    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        return self.first_stage_model.decode(z)

    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(self, z):
        return self.scale_factor * z

    def apply_model(self, x_noisy, t, cond, return_ids=False, **kwargs):
        """
        args:
            cond: it can be a dictionary or a Tensor. When `cond` is a dictionary,
                it passes through `DiffusionWrapper` as keyword arguments. When it
                is a Tensor, it is the input argument of "c_concat" or `c_crossattn`
                depends on the predefined `conditioning_key`.
        """
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond, **kwargs)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def get_input(self, x, c):
        if len(x.shape) == 3:
            x = x[..., None]
        x = self.transpose(x, (0, 3, 1, 2))
        z = ops.stop_gradient(self.get_first_stage_encoding(self.encode_first_stage(x)))
        return z, c

    def construct(self, x, c):
        t = self.uniform_int(
            (x.shape[0],), Tensor(0, dtype=mstype.int32), Tensor(self.num_timesteps, dtype=mstype.int32)
        )
        x, c = self.get_input(x, c)
        c = self.get_learned_conditioning_fortrain(c)
        return self.p_losses(x, c, t)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = msnp.randn(x_start.shape)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(
            x_noisy,
            t,
            cond=cond,
        )

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "velocity":
            # target = sqrt_alpha_cum * noise - sqrt_one_minus_alpha_prod * x_start
            target = self.get_velocity(x_start, noise, t)  # TODO: parse train step from randint
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])

        logvar_t = self.logvar[t]
        loss = loss_simple / ops.exp(logvar_t) + logvar_t
        loss = self.l_simple_weight * loss.mean()

        # NOTE: original_elbo_weight is never set larger than 0. Diffuser remove it too. Let's remove it to save mem.
        # loss_vlb = self.get_loss(model_output, target, mean=False).mean((1, 2, 3))
        # loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        # loss += (self.original_elbo_weight * loss_vlb)

        return loss

    # @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size, null_label=None):
        if null_label is not None:
            xc = null_label
            if isinstance(xc, ListConfig):
                xc = list(xc)
            #####################################
            # if isinstance(xc, dict) or isinstance(xc, list):
            #     c = self.get_learned_conditioning(xc)
            # else:
            #     if hasattr(xc, "to"):
            #         xc = xc.to(self.device)
            #     c = self.get_learned_conditioning(xc)
            #----------------------------------#
            c = self.get_learned_conditioning(self.tokenize(xc))
            ####################################
        else:
            raise NotImplementedError("Not implemented when 'cond_stage_key' is 'class_label' or 'cls'.")
            # if self.cond_stage_key in ["class_label", "cls"]:
            #     xc = self.cond_stage_model.get_unconditional_conditioning(batch_size, device=self.device)
            #     return self.get_learned_conditioning(xc)
            # else:
            #     raise NotImplementedError("todo")
        if isinstance(c, list):  # in case the encoder gives us a list
            for i in range(len(c)):
                #####################
                # c[i] = repeat(c[i], '1 ... -> b ...', b=batch_size).to(self.device)
                #-------------------#
                c[i] = c[i].repeat(batch_size, axis=0)
                #####################
        else:
            #####################
            # c = repeat(c, '1 ... -> b ...', b=batch_size).to(self.device)
            #-------------------#
            c = c.repeat(batch_size, axis=0)
            #####################
        return c


# latent diffusion (unet) forward based on input noised latent and encoded conditions
class DiffusionWrapper(nn.Cell):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, "concat", "crossattn", "hybrid", "adm", "crossattn-adm", "hybrid-adm"]

    def construct(self, x, t, c_concat: list = None, c_crossattn: list = None, c_adm=None , **kwargs):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t, **kwargs)
        elif self.conditioning_key == "concat":
            x_concat = ops.concat((x, c_concat), axis=1)
            out = self.diffusion_model(x_concat, t, **kwargs)
        elif self.conditioning_key == "crossattn":
            context = c_crossattn
            out = self.diffusion_model(x, t, context=context, **kwargs)
        elif self.conditioning_key == "hybrid":
            x_concat = ops.concat((x, c_concat), axis=1)
            context = c_crossattn
            out = self.diffusion_model(x_concat, t, context=context, **kwargs)
        elif self.conditioning_key == "crossattn-adm":
            context = c_crossattn
            out = self.diffusion_model(x, t, context=context, y=c_adm, **kwargs)
        elif self.conditioning_key == "hybrid-adm":
            assert c_adm is not None
            x_concat = ops.concat([x] + c_concat, axis=1)
            cc = ops.concat(c_crossattn, axis=1)
            out = self.diffusion_model(x_concat, t, context=cc, y=c_adm, **kwargs)
        elif self.conditioning_key == "adm":
            cc = c_crossattn
            out = self.diffusion_model(x, t, y=cc, **kwargs)
        else:
            raise NotImplementedError()

        return out


class LatentDiffusionDB(DDPM):
    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        reg_weight=1.0,
        *args,
        **kwargs,
    ):
        """
        main class
        """
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.reg_weight = reg_weight
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except Exception:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", Tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.uniform_int = ops.UniformInt()

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        self.transpose = ops.Transpose()

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model
        for param in self.first_stage_model.get_parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model
        else:
            assert config != "__is_first_stage__"
            assert config != "__is_unconditional__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            c = self.cond_stage_model.encode(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_learned_conditioning_fortrain(self, c):
        c = self.cond_stage_model(c)
        return c

    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        return self.first_stage_model.decode(z)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        x_noisy = ops.cast(x_noisy, self.dtype)
        cond = ops.cast(cond, self.dtype)

        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}
        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def get_input(self, x, c):
        if len(x.shape) == 3:
            x = x[..., None]
        x = self.transpose(x, (0, 3, 1, 2))
        z = ops.stop_gradient(self.scale_factor * self.first_stage_model.encode(x))

        return z, c

    def shared_step(self, x, c):
        x, c = self.get_input(x, c)
        t = self.uniform_int(
            (x.shape[0],), Tensor(0, dtype=mstype.int32), Tensor(self.num_timesteps, dtype=mstype.int32)
        )
        c = self.get_learned_conditioning_fortrain(c)
        loss = self.p_losses(x, c, t)
        return loss

    def construct(self, train_x, train_c, reg_x, reg_c):
        loss_train = self.shared_step(train_x, train_c)
        loss_reg = self.shared_step(reg_x, reg_c)
        loss = loss_train + self.reg_weight * loss_reg
        return loss

    def p_losses(self, x_start, cond, t, noise=None):
        noise = msnp.randn(x_start.shape)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "velocity":
            # target = sqrt_alpha_cum * noise - sqrt_one_minus_alpha_prod * x_start
            target = self.get_velocity(x_start, noise, t)  # TODO: parse train step from randint
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])

        logvar_t = self.logvar[t]
        loss = loss_simple / ops.exp(logvar_t) + logvar_t
        loss = self.l_simple_weight * loss.mean()

        # loss_vlb = self.get_loss(model_output, target, mean=False).mean((1, 2, 3))
        # loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        # loss += (self.original_elbo_weight * loss_vlb)

        return loss


class LatentDiffusionDreamBooth(LatentDiffusion):
    def __init__(self, prior_loss_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_loss_weight = prior_loss_weight

    def shared_step(self, x, c):
        x, c = self.get_input(x, c)
        t = ops.UniformInt()(
            (x.shape[0],), Tensor(0, dtype=mstype.int32), Tensor(self.num_timesteps, dtype=mstype.int32)
        )
        c = self.get_learned_conditioning_fortrain(c)
        loss = self.p_losses(x, c, t)
        return loss

    def construct(self, *args):
        if self.prior_loss_weight != 0:
            train_x, train_c, reg_x, reg_c = args
            loss_train = self.shared_step(train_x, train_c)
            loss_reg = self.shared_step(reg_x, reg_c)
            loss = loss_train + self.prior_loss_weight * loss_reg
        else:
            train_x, train_c = args
            loss_train = self.shared_step(train_x, train_c)
            loss = loss_train
        return loss


class LatentInpaintDiffusion(LatentDiffusion):
    def __init__(
        self,
        concat_keys=("mask", "masked_image"),
        masked_image_key="masked_image",
        finetune_keys=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.masked_image_key = masked_image_key
        assert self.masked_image_key in concat_keys
        self.concat_keys = concat_keys


class LatentDepthDiffusion(LatentDiffusion):
    def __init__(
        self,
        concat_keys=("depth"),
        finetune_keys=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.concat_keys = concat_keys


class ImageEmbeddingConditionedLatentDiffusion(LatentDiffusion):
    def __init__(
        self, embedder_config, embedding_dropout=0.5, freeze_embedder=True, noise_aug_config=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedding_dropout = embedding_dropout
        self._init_embedder(embedder_config, freeze_embedder)
        self._init_noise_aug(noise_aug_config)

    def _init_embedder(self, config, freeze=True):
        self.embedder = instantiate_from_config(config)
        if freeze:
            self.embedder.set_train(False)
            for param in self.embedder.get_parameters():
                param.requires_grad = False

    def _init_noise_aug(self, config):
        if config is not None:
            self.noise_augmentor = instantiate_from_config(config)
            self.noise_augmentor.set_train(False)
        else:
            self.noise_augmentor = None

    def get_input(self, x, c):
        z, c = LatentDiffusion.get_input(self, x, c)
        c_adm = self.embedder(x)
        if self.noise_augmentor is not None:
            c_adm, noise_level_emb = self.noise_augmentor(c_adm)
            # assume this gives embeddings of noise levels
            c_adm = ops.concat((c_adm, noise_level_emb), 1)
        if self.training:
            c_adm = ops.bernoulli((1.0 - self.embedding_dropout) * ops.ones(c_adm.shape[0])[:, None]) * c_adm

        # TODO: training support 3 inputs
        return z, c, c_adm


class LatentUpscaleDiffusion(LatentDiffusion):
    def __init__(self, *args, low_scale_config, low_scale_key="LR", noise_level_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        # assumes that neither the cond_stage nor the low_scale_model contain trainable params
        assert not self.cond_stage_trainable
        self.instantiate_low_stage(low_scale_config)
        self.low_scale_key = low_scale_key
        self.noise_level_key = noise_level_key

    def instantiate_low_stage(self, config):
        model = instantiate_from_config(config)
        ##########################
        # self.low_scale_model = model.eval()
        # self.low_scale_model.train = disabled_train
        #------------------------#
        self.low_scale_model = model
        ##########################
        for param in self.low_scale_model.get_parameters():
            param.requires_grad = False

    # @torch.no_grad()
    def get_input(self, batch, k, cond_key=None, bs=None, log_mode=False):
        if not log_mode:
            z, c = super().get_input(batch, k, force_c_encode=True, bs=bs)
        else:
            z, c, x, xrec, xc = super().get_input(batch, self.first_stage_key, return_first_stage_outputs=True,
                                                  force_c_encode=True, return_original_cond=True, bs=bs)
        x_low = batch[self.low_scale_key][:bs]
        ###########################
        # x_low = rearrange(x_low, 'b h w c -> b c h w')
        # x_low = x_low.to(memory_format=torch.contiguous_format).float()
        #-------------------------#
        x_low = ops.transpose(x_low, (0, 3, 1, 2))
        x_low = x_low.to(dtype=ms.float32)
        ###########################
        zx, noise_level = self.low_scale_model(x_low)
        if self.noise_level_key is not None:
            # get noise level from batch instead, e.g. when extracting a custom noise level for bsr
            raise NotImplementedError()

        all_conds = {"c_concat": [zx], "c_crossattn": [c], "c_adm": noise_level}
        if log_mode:
            # TODO: maybe disable if too expensive
            x_low_rec = self.low_scale_model.decode(zx)
            return z, all_conds, x, xrec, xc, x_low, x_low_rec, noise_level
        return z, all_conds

    # @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   plot_denoise_rows=False, plot_progressive_rows=True, plot_diffusion_rows=True,
                   unconditional_guidance_scale=1., unconditional_guidance_label=None, use_ema_scope=True,
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc, x_low, x_low_rec, noise_level = self.get_input(batch, self.first_stage_key, bs=N,
                                                                          log_mode=True)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["x_lr"] = x_low
        log[f"x_lr_rec_@noise_levels{'-'.join(map(lambda x: str(x), list(noise_level.cpu().numpy())))}"] = x_low_rec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2] // 25)
                log["conditioning"] = xc
            elif self.cond_stage_key in ['class_label', 'cls']:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"], size=x.shape[2] // 25)
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    ####################
                    # t = repeat(Tensor([t]), '1 -> b', b=n_row)
                    # t = t.to(self.device).long()
                    # noise = torch.randn_like(z_start)
                    #------------------#
                    t = Tensor([t]).repeat(n_row, axis=0)
                    t = t.long()
                    noise = ops.randn_like(z_start)
                    ####################
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))
            
            ####################
            # diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            # diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            # diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            #------------------#
            diffusion_row = ops.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = ops.transpose(diffusion_row, (1, 0, 2, 3, 4))
            b, n, c, h, w = diffusion_grid.shape
            diffusion_grid = ops.reshape(diffusion_grid, (b*n, c, h, w))
            ####################
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_tmp = self.get_unconditional_conditioning(N, unconditional_guidance_label)
            # TODO explore better "unconditional" choices for the other keys
            # maybe guide away from empty text label and highest noise level and maximally degraded zx?
            uc = dict()
            for k in c:
                if k == "c_crossattn":
                    assert isinstance(c[k], list) and len(c[k]) == 1
                    uc[k] = [uc_tmp]
                elif k == "c_adm":  # todo: only run with text-based guidance?
                    assert isinstance(c[k], Tensor)
                    #uc[k] = torch.ones_like(c[k]) * self.low_scale_model.max_noise_level
                    uc[k] = c[k]
                elif isinstance(c[k], list):
                    uc[k] = [c[k][i] for i in range(len(c[k]))]
                else:
                    uc[k] = c[k]

            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                 ddim_steps=ddim_steps, eta=ddim_eta,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc,
                                                 )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        if plot_progressive_rows:
            raise NotImplementedError("Function 'progressive_denoising' is not implemented.")
            # with ema_scope("Plotting Progressives"):
            #     img, progressives = self.progressive_denoising(c,
            #                                                    shape=(self.channels, self.image_size, self.image_size),
            #                                                    batch_size=N)
            # prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            # log["progressive_row"] = prog_row

        return log


class LatentFinetuneDiffusion(LatentDiffusion):
    """
         Basis for different finetunas, such as inpainting or depth2image
         To disable finetuning mode, set finetune_keys to None
    """

    def __init__(self,
                 concat_keys: tuple,
                 finetune_keys=("model.diffusion_model.input_blocks.0.0.weight",
                                "model_ema.diffusion_modelinput_blocks00weight"
                                ),
                 keep_finetune_dims=4,
                 # if model was trained without concat mode before and we would like to keep these channels
                 c_concat_log_start=None,  # to log reconstruction of c_concat codes
                 c_concat_log_end=None,
                 *args, **kwargs
                 ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", list())
        super().__init__(*args, **kwargs)
        self.finetune_keys = finetune_keys
        self.concat_keys = concat_keys
        self.keep_dims = keep_finetune_dims
        self.c_concat_log_start = c_concat_log_start
        self.c_concat_log_end = c_concat_log_end
        if exists(self.finetune_keys): assert exists(ckpt_path), 'can only finetune from a given checkpoint'
        if exists(ckpt_path):
            self.init_from_ckpt(ckpt_path, ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

            # make it explicit, finetune by including extra input channels
            if exists(self.finetune_keys) and k in self.finetune_keys:
                new_entry = None
                for name, param in self.named_parameters():
                    if name in self.finetune_keys:
                        print(
                            f"modifying key '{name}' and keeping its original {self.keep_dims} (channels) dimensions only")
                        new_entry = ops.zeros_like(param)  # zero init
                assert exists(new_entry), 'did not find matching parameter to modify'
                new_entry[:, :self.keep_dims, ...] = sd[k]
                sd[k] = new_entry

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    # @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, unconditional_guidance_scale=1., unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, bs=N, return_first_stage_outputs=True)
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2] // 25)
                log["conditioning"] = xc
            elif self.cond_stage_key in ['class_label', 'cls']:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"], size=x.shape[2] // 25)
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if not (self.c_concat_log_start is None and self.c_concat_log_end is None):
            log["c_concat_decoded"] = self.decode_first_stage(c_cat[:, self.c_concat_log_start:self.c_concat_log_end])

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    ###############
                    # t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    # t = t.to(self.device).long()
                    # noise = torch.randn_like(z_start)
                    #-------------#
                    t = Tensor([t]).repeat(n_row, axis=0)
                    t = t.long()
                    noise = ops.randn_like(z_start)
                    ###############
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            ########################
            # diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            # diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            # diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            #------------------#
            diffusion_row = ops.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = ops.transpose(diffusion_row, (1, 0, 2, 3, 4))
            b, n, c, h, w = diffusion_grid.shape
            diffusion_grid = ops.reshape(diffusion_grid, (b*n, c, h, w))
            ########################
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                         batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N, unconditional_guidance_label)
            uc_cat = c_cat
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                 batch_size=N, ddim=use_ddim,
                                                 ddim_steps=ddim_steps, eta=ddim_eta,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc_full,
                                                 )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log


class LatentUpscaleFinetuneDiffusion(LatentFinetuneDiffusion):
    """
        condition on low-res image (and optionally on some spatial noise augmentation)
    """
    def __init__(self, concat_keys=("lr",), reshuffle_patch_size=None,
                 low_scale_config=None, low_scale_key=None, *args, **kwargs):
        super().__init__(concat_keys=concat_keys, *args, **kwargs)
        self.reshuffle_patch_size = reshuffle_patch_size
        self.low_scale_model = None
        if low_scale_config is not None:
            print("Initializing a low-scale model")
            assert exists(low_scale_key)
            self.instantiate_low_stage(low_scale_config)
            self.low_scale_key = low_scale_key

    def instantiate_low_stage(self, config):
        model = instantiate_from_config(config)
        ##########################
        # self.low_scale_model = model.eval()
        # self.low_scale_model.train = disabled_train
        #------------------------#
        self.low_scale_model = model
        ##########################
        for param in self.low_scale_model.get_parameters():
            param.requires_grad = False

    # @torch.no_grad()
    def get_input(self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False):
        # note: restricted to non-trainable encoders currently
        assert not self.cond_stage_trainable, 'trainable cond stages not yet supported for upscaling-ft'
        z, c, x, xrec, xc = super().get_input(batch, self.first_stage_key, return_first_stage_outputs=True,
                                              force_c_encode=True, return_original_cond=True, bs=bs)

        assert exists(self.concat_keys)
        assert len(self.concat_keys) == 1
        # optionally make spatial noise_level here
        c_cat = list()
        noise_level = None
        for ck in self.concat_keys:
            cc = batch[ck]
            cc = rearrange(cc, 'b h w c -> b c h w')
            if exists(self.reshuffle_patch_size):
                assert isinstance(self.reshuffle_patch_size, int)
                cc = rearrange(cc, 'b c (p1 h) (p2 w) -> b (p1 p2 c) h w',
                               p1=self.reshuffle_patch_size, p2=self.reshuffle_patch_size)
            if bs is not None:
                cc = cc[:bs]
                # cc = cc.to(self.device)
            if exists(self.low_scale_model) and ck == self.low_scale_key:
                cc, noise_level = self.low_scale_model(cc)
            c_cat.append(cc)
        c_cat = ops.cat(c_cat, axis=1)
        if exists(noise_level):
            all_conds = {"c_concat": [c_cat], "c_crossattn": [c], "c_adm": noise_level}
        else:
            all_conds = {"c_concat": [c_cat], "c_crossattn": [c]}
        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds

    # @torch.no_grad()
    def log_images(self, *args, **kwargs):
        log = super().log_images(*args, **kwargs)
        log["lr"] = rearrange(args[0]["lr"], 'b h w c -> b c h w')
        return log
