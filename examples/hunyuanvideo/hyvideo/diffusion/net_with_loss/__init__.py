import logging
import math

from hyvideo.utils.communications import prepare_parallel_data
from hyvideo.utils.ms_utils import no_grad
from hyvideo.utils.parallel_states import get_sequence_parallel_state, hccl_info
from hyvideo.utils.utils import explicit_uniform_sampling, get_sigmas

import mindspore as ms
from mindspore import mint, nn, ops

from mindone.diffusers.training_utils import compute_snr

__all__ = ["DiffusionWithLoss"]

logger = logging.getLogger(__name__)


class DiffusionWithLoss(nn.Cell):
    """An training pipeline for diffusion model

    Args:
        model (nn.Cell): A noise prediction model to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        noise_scheduler: (object): A class for noise scheduler, such as DDPM scheduler
        text_encoder / text_encoder_2 (nn.Cell): A text encoding model which accepts token ids and returns text embeddings in shape (T, D).
            T is the number of tokens, and D is the embedding dimension.
        train_with_embed (bool): whether to train with embeddings (no need vae and text encoder to extract latent features and text embeddings)
    """

    def __init__(
        self,
        network: nn.Cell,
        noise_scheduler,
        vae: nn.Cell = None,
        text_encoder: nn.Cell = None,
        text_encoder_2: nn.Cell = None,  # not to use yet
        text_emb_cached: bool = True,
        video_emb_cached: bool = False,
        use_image_num: int = 0,
        dtype=ms.float32,
        noise_offset: float = 0.0,
        snr_gamma=None,
        rank_id: int = 0,
        device_num: int = 1,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        weighting_scheme: str = "logit_normal",
        mode_scale: float = 1.29,
        embedded_guidance_scale: float = 6.0,
    ):
        super().__init__()
        # TODO: is set_grad() necessary?
        self.network = network.set_grad()
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.prediction_type = self.noise_scheduler.config.prediction_type
        self.num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.noise_offset = noise_offset
        self.rank_id = rank_id
        self.device_num = device_num
        self.snr_gamma = snr_gamma
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.weighting_scheme = weighting_scheme
        self.mode_scale = mode_scale
        self.embedded_guidance_scale = embedded_guidance_scale
        if self.network.guidance_embed:
            assert (
                self.embedded_guidance_scale is not None
            ), "embedded_guidance_scale should be set when using guidance embed"

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.dtype = dtype

        self.text_emb_cached = text_emb_cached
        self.video_emb_cached = video_emb_cached

        if self.text_emb_cached:
            self.text_encoder = None
            self.text_encoder_2 = None
            logger.info("Train with text embedding inputs")

        self.use_image_num = use_image_num
        self.reduce_t = None if not get_sequence_parallel_state() else ops.AllReduce(group=hccl_info.group)
        self.sp_size = 1 if not get_sequence_parallel_state() else hccl_info.world_size
        self.all_gather = None if not get_sequence_parallel_state() else ops.AllGather(group=hccl_info.group)

    def get_condition_embeddings(self, text_tokens, encoder_attention_mask, index=0):
        # text conditions inputs for cross-attention
        # optional: for some conditions, concat to latents, or add to time embedding
        # use for loop to avoid OOM?
        B, frame, L = text_tokens.shape  # B T+num_images L = b 1+4, L
        text_emb = []
        assert index in [0, 1], "index should be 0 or 1"
        text_encoder = self.text_encoder if index == 0 else self.text_encoder_2
        for i in range(frame):
            t = text_encoder(text_tokens[:, i], encoder_attention_mask[:, i])
            text_emb.append(t)
        text_emb = ops.stack(text_emb, axis=1)
        return text_emb

    def vae_encode(self, x):
        image_latents = self.vae.encode(x)
        return image_latents

    def vae_decode(self, x):
        """
        Args:
            x: (b c h w), denoised latent
        Return:
            y: (b H W 3), batch of images, normalized to [0, 1]
        """
        # b, c, f, h, w = x.shape
        y = self.vae.decode(x)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

        return y  # b c f h w

    def get_latents(self, x):
        if x.dim() == 5:
            B, C, F, H, W = x.shape
            if C != 3:
                raise ValueError("Expect input shape (b 3 f h w), but get {}".format(x.shape))
            if self.use_image_num == 0:
                z = self.vae_encode(x)  # (b, c, f, h, w)
            else:
                videos, images = x[:, :, : -self.use_image_num], x[:, :, -self.use_image_num :]
                videos = self.vae_encode(videos)  # (b, c, f, h, w)
                # (b, c, f, h, w) -> (b, f, c, h, w) -> (b*f, c, h, w) -> (b*f, c, 1, h, w)
                images = images.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W).unsqueeze(2)
                images = self.vae_encode(images)  # (b*f, c, 1, h, w)
                # (b*f, c, 1, h, w) -> (b*f, c, h, w) -> (b, f, c, h, w) -> (b, c, f, h, w)
                _, c, _, h, w = images.shape
                images = images.squeeze(2).reshape(B, self.use_image_num, c, h, w).permute(0, 2, 1, 3, 4)
                z = mint.cat([videos, images], dim=2)  # b c 16+4, h, w
        else:
            raise ValueError("Incorrect Dimensions of x")
        return z

    def construct(
        self,
        x: ms.Tensor,
        attention_mask: ms.Tensor,
        text_tokens: ms.Tensor,
        encoder_attention_mask: ms.Tensor = None,
        text_tokens_2: ms.Tensor = None,
        encoder_attention_mask_2: ms.Tensor = None,
    ):
        """
        Video diffusion model forward and loss computation for training

        Args:
            x: pixel values of video frames, resized and normalized to shape (b c f+num_img h w)
            attention_mask: the mask for latent features of shape (b t' h' w'), where t' h' w' are the shape of latent features after vae's encoding.
            text_tokens: text tokens padded to fixed shape (B F L) or text embedding of shape (B F L D) if using text embedding cache
            encoder_attention_mask: the mask for text tokens/embeddings of a fixed shape (B F L)

        Returns:
            loss

        Notes:
            - inputs should matches dataloder output order
            - assume model input/output shape: (b c f+num_img h w)
        """
        # 1. get image/video latents z using vae
        x = x.to(self.dtype)
        with no_grad():
            if not self.video_emb_cached:
                x = ops.stop_gradient(self.get_latents(x))

            # 2. get conditions
            if not self.text_emb_cached:
                text_embed = ops.stop_gradient(
                    self.get_condition_embeddings(text_tokens, encoder_attention_mask, index=0)
                )
                if text_tokens_2 is not None:
                    text_embed_2 = ops.stop_gradient(
                        self.get_condition_embeddings(text_tokens_2, encoder_attention_mask_2, index=1)
                    )
                else:
                    text_embed_2 = None
            else:
                text_embed = text_tokens
                if text_tokens_2 is not None:
                    text_embed_2 = text_tokens_2
                else:
                    text_embed_2 = None
        loss = self.compute_loss(
            x, attention_mask, text_embed, encoder_attention_mask, text_embed_2, encoder_attention_mask_2
        )
        return loss

    def apply_model(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def compute_loss(
        self, x, attention_mask, text_embed, encoder_attention_mask, text_embed_2, encoder_attention_mask_2
    ):
        use_image_num = self.use_image_num
        noise = ops.randn_like(x)
        bsz = x.shape[0]

        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * ops.randn((bsz, x.shape[1], 1, 1, 1), dtype=x.dtype)
        current_step_frame = x.shape[2]
        if get_sequence_parallel_state() and current_step_frame > 1:
            x = self.all_gather(x[None])[0]
            (
                x,
                noise,
                text_embed,
                text_embed_2,
                attention_mask,
                encoder_attention_mask,
                encoder_attention_mask_2,
                use_image_num,
            ) = prepare_parallel_data(
                x,
                noise,
                text_embed,
                text_embed_2,
                attention_mask,
                encoder_attention_mask,
                encoder_attention_mask_2,
                use_image_num,
            )

        # sample a random timestep for each image without bias
        t = explicit_uniform_sampling(
            T=self.num_train_timesteps,
            n=self.device_num,
            rank=self.rank_id,
            bsz=bsz,
        )
        # t = ops.randint(0, self.num_train_timesteps, (x.shape[0],), dtype=ms.int32)
        if get_sequence_parallel_state():
            t = self.reduce_t(t) % self.num_train_timesteps
        x_t = self.noise_scheduler.add_noise(x, noise, t)

        # latte forward input match
        # text embed: (b n_tokens  d) -> (b  1 n_tokens d)
        # text_embed = ops.expand_dims(text_embed, axis=1)
        guidance_expand = (
            ms.Tensor(
                [self.embedded_guidance_scale] * bsz,
                dtype=ms.float32,
            ).to(x_t.dtype)
            * 1000.0
            if self.embedded_guidance_scale is not None
            else None
        )
        model_pred = self.apply_model(
            x_t,
            t,
            text_states=text_embed,
            text_mask=encoder_attention_mask,
            text_states_2=text_embed_2,
            guidance=guidance_expand,
            # attention_mask=attention_mask,
            # use_image_num=use_image_num,
        )

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(x, noise, t)
        elif self.prediction_type == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = x
            # We will have to subtract the noise residual from the prediction to get the target sample.
            model_pred = model_pred - noise
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")
        # comment it to avoid graph syntax error
        # if attention_mask is not None and (attention_mask.bool()).all():
        #     attention_mask = None
        if get_sequence_parallel_state():
            assert (attention_mask.bool()).all()
            # assert attention_mask is None
            attention_mask = None
        # (b c t h w),
        bsz, c, _, _, _ = model_pred.shape
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).float().repeat(c, axis=1)  # b t h w -> b c t h w
            attention_mask = attention_mask.reshape(bsz, -1)

        if self.snr_gamma is None:
            # model_pred: b c t h w, attention_mask: b t h w
            loss = ops.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.reshape(bsz, -1)
            if attention_mask is not None:
                loss = (loss * attention_mask).sum() / attention_mask.sum()  # mean loss on unpad patches
            else:
                loss = loss.mean()
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, t)
            mse_loss_weights = ops.stack([snr, self.snr_gamma * ops.ones_like(t)], axis=1).min(axis=1)[0]
            if self.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)
            loss = ops.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.reshape(bsz, -1)
            mse_loss_weights = mse_loss_weights.reshape(bsz, 1)
            if attention_mask is not None:
                loss = (
                    loss * attention_mask * mse_loss_weights
                ).sum() / attention_mask.sum()  # mean loss on unpad patches
            else:
                loss = (loss * mse_loss_weights).mean()

        return loss


class DiffusionWithLossEval(DiffusionWithLoss):
    def construct(
        self,
        x: ms.Tensor,
        attention_mask: ms.Tensor,
        text_tokens: ms.Tensor,
        encoder_attention_mask: ms.Tensor = None,
        text_tokens_2: ms.Tensor = None,
        encoder_attention_mask_2: ms.Tensor = None,
    ):
        """
        Video diffusion model forward and loss computation for training

        Args:
            x: pixel values of video frames, resized and normalized to shape (b c f+num_img h w)
            attention_mask: the mask for latent features of shape (b t' h' w'), where t' h' w' are the shape of latent features after vae's encoding.
            text_tokens: text tokens padded to fixed shape (B F L) or text embedding of shape (B F L D) if using text embedding cache
            encoder_attention_mask: the mask for text tokens/embeddings of a fixed shape (B F L)

        Returns:
            loss

        Notes:
            - inputs should matches dataloder output order
            - assume model input/output shape: (b c f+num_img h w)
        """
        # 1. get image/video latents z using vae
        x = x.to(self.dtype)
        with no_grad():
            if not self.video_emb_cached:
                x = ops.stop_gradient(self.get_latents(x))

            # 2. get conditions
            if not self.text_emb_cached:
                text_embed = ops.stop_gradient(
                    self.get_condition_embeddings(text_tokens, encoder_attention_mask, index=0)
                )
                if text_tokens_2 is not None:
                    text_embed_2 = ops.stop_gradient(
                        self.get_condition_embeddings(text_tokens_2, encoder_attention_mask_2, index=1)
                    )
            else:
                text_embed = text_tokens
                if text_tokens_2 is not None:
                    text_embed_2 = text_tokens_2
        loss, model_pred, target = self.compute_loss(
            x, attention_mask, text_embed, encoder_attention_mask, text_embed_2, encoder_attention_mask_2
        )

        return loss, model_pred, target

    def compute_loss(
        self, x, attention_mask, text_embed, encoder_attention_mask, text_embed_2, encoder_attention_mask_2
    ):
        use_image_num = self.use_image_num
        noise = ops.randn_like(x)
        bsz = x.shape[0]
        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * ops.randn((bsz, x.shape[1], 1, 1, 1), dtype=x.dtype)
        current_step_frame = x.shape[2]
        if get_sequence_parallel_state() and current_step_frame > 1:
            x = self.all_gather(x[None])[0]
            (
                x,
                noise,
                text_embed,
                text_embed_2,
                attention_mask,
                encoder_attention_mask,
                encoder_attention_mask_2,
                use_image_num,
            ) = prepare_parallel_data(
                x,
                noise,
                text_embed,
                text_embed_2,
                attention_mask,
                encoder_attention_mask,
                encoder_attention_mask_2,
                use_image_num,
            )

        # sample a random timestep for each image without bias
        t = explicit_uniform_sampling(
            T=self.num_train_timesteps,
            n=self.device_num,
            rank=self.rank_id,
            bsz=bsz,
        )
        # t = ops.randint(0, self.num_train_timesteps, (x.shape[0],), dtype=ms.int32)
        if get_sequence_parallel_state():
            t = self.reduce_t(t) % self.num_train_timesteps
        x_t = self.noise_scheduler.add_noise(x, noise, t)

        # latte forward input match
        # text embed: (b n_tokens  d) -> (b  1 n_tokens d)
        # text_embed = ops.expand_dims(text_embed, axis=1)
        guidance_expand = (
            ms.Tensor(
                [self.embedded_guidance_scale] * bsz,
                dtype=ms.float32,
            ).to(x_t.dtype)
            * 1000.0
            if self.embedded_guidance_scale is not None
            else None
        )
        model_pred = self.apply_model(
            x_t,
            t,
            text_states=text_embed,
            text_mask=encoder_attention_mask,
            text_states_2=text_embed_2,
            guidance=guidance_expand,
            # attention_mask=attention_mask,
            # use_image_num=use_image_num,
        )

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(x, noise, t)
        elif self.prediction_type == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = x
            # We will have to subtract the noise residual from the prediction to get the target sample.
            model_pred = model_pred - noise
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")
        # comment it to avoid graph syntax error
        # if attention_mask is not None and (attention_mask.bool()).all():
        #     attention_mask = None
        if get_sequence_parallel_state():
            assert (attention_mask.bool()).all()
            # assert attention_mask is None
            attention_mask = None
        # (b c t h w),
        bsz, c, _, _, _ = model_pred.shape
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).float().repeat(c, axis=1)  # b t h w -> b c t h w
            attention_mask = attention_mask.reshape(bsz, -1)

        if self.snr_gamma is None:
            # model_pred: b c t h w, attention_mask: b t h w
            loss = ops.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.reshape(bsz, -1)
            if attention_mask is not None:
                loss = (loss * attention_mask).sum() / attention_mask.sum()  # mean loss on unpad patches
            else:
                loss = loss.mean()
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, t)
            mse_loss_weights = ops.stack([snr, self.snr_gamma * ops.ones_like(t)], axis=1).min(axis=1)[0]
            if self.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)
            loss = ops.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.reshape(bsz, -1)
            mse_loss_weights = mse_loss_weights.reshape(bsz, 1)
            if attention_mask is not None:
                loss = (
                    loss * attention_mask * mse_loss_weights
                ).sum() / attention_mask.sum()  # mean loss on unpad patches
            else:
                loss = (loss * mse_loss_weights).mean()

        return loss, model_pred, target
