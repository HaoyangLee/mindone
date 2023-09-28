import numpy as np
from ldm.models.clip.simple_tokenizer import get_tokenizer
from ldm.modules.diffusionmodules.openaimodel import Timestep
from ldm.modules.diffusionmodules.upscaling import ImageConcatWithNoiseAugmentation
from PIL import Image

import mindspore as ms
import mindspore.dataset.vision as vision
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.dataset.transforms import Compose

from .image_encoder import ImageEncoder
from .text_encoder import OpenClipTextEncoder, TextEncoder


class FrozenCLIPEmbedder(nn.Cell):
    def __init__(
        self,
        use_fp16=False,
        tokenizer_name="WordpieceTokenizer",
        context_length=77,
        vocab_size=49408,
        output_dim=768,
        width=768,
        layers=12,
        heads=12,
        epsilon=1e-5,
        use_quick_gelu=False,
    ):
        super(FrozenCLIPEmbedder, self).__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.context_length = context_length
        self.tokenizer = get_tokenizer(tokenizer_name)
        setattr(self.tokenizer, "context_length", context_length)

        self.transformer = TextEncoder(
            context_length=context_length,
            vocab_size=vocab_size,
            output_dim=output_dim,
            width=width,
            layers=layers,
            heads=heads,
            epsilon=epsilon,
            use_quick_gelu=use_quick_gelu,
            dtype=self.dtype,
        )

    def tokenize(self, texts):
        SOT_TEXT = self.tokenizer.sot_text
        EOT_TEXT = self.tokenizer.eot_text
        CONTEXT_LEN = self.context_length

        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder[SOT_TEXT]
        eot_token = self.tokenizer.encoder[EOT_TEXT]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = np.zeros((len(all_tokens), CONTEXT_LEN), np.int64)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > CONTEXT_LEN:
                tokens = tokens[: CONTEXT_LEN - 1] + [eot_token]

            result[i, : len(tokens)] = np.array(tokens, np.int64)

        return Tensor(result)

    def encode(self, tokenized_text):
        outputs = self.transformer(tokenized_text)
        return outputs

    def construct(self, c):
        outputs = self.transformer(c)
        return outputs


class FrozenOpenCLIPEmbedder(FrozenCLIPEmbedder):
    def __init__(
        self,
        use_fp16=False,
        tokenizer_name="WordpieceTokenizer",
        context_length=77,
        vocab_size=49408,
        output_dim=768,
        width=768,
        layers=12,
        heads=12,
    ):
        super(FrozenCLIPEmbedder, self).__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.context_length = context_length
        self.tokenizer = get_tokenizer(tokenizer_name)
        setattr(self.tokenizer, "context_length", context_length)

        self.model = OpenClipTextEncoder(
            context_length=context_length,
            vocab_size=vocab_size,
            output_dim=output_dim,
            width=width,
            layers=layers,
            heads=heads,
            epsilon=1e-5,
            use_quick_gelu=False,
            dtype=self.dtype,
        )

    def encode(self, tokenized_text):
        outputs = self.model(tokenized_text)
        return outputs

    def construct(self, c):
        outputs = self.model(c)
        return outputs


# class FrozenOpenCLIPEmbedder(nn.Cell):
#     """
#     Uses the OpenCLIP transformer encoder for text
#     """
#     LAYERS = [
#         # "pooled",
#         "last",
#         "penultimate"
#     ]

#     def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
#                  freeze=True, layer="last"):
#         super().__init__()
#         assert layer in self.LAYERS
#         model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
#         del model.visual
#         self.model = model

#         self.device = device
#         self.max_length = max_length
#         if freeze:
#             self.freeze()
#         self.layer = layer
#         if self.layer == "last":
#             self.layer_idx = 0
#         elif self.layer == "penultimate":
#             self.layer_idx = 1
#         else:
#             raise NotImplementedError()

#     def freeze(self):
#         self.model = self.model.eval()
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, text):
#         tokens = open_clip.tokenize(text)
#         z = self.encode_with_transformer(tokens.to(self.device))
#         return z

#     def encode_with_transformer(self, text):
#         x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
#         x = x + self.model.positional_embedding
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.model.ln_final(x)
#         return x

#     def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
#         for i, r in enumerate(self.model.transformer.resblocks):
#             if i == len(self.model.transformer.resblocks) - self.layer_idx:
#                 break
#             if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
#                 x = checkpoint(r, x, attn_mask)
#             else:
#                 x = r(x, attn_mask=attn_mask)
#         return x

#     def encode(self, text):
#         return self(text)


class CLIPImageEmbedder(nn.Cell):
    def __init__(
        self,
        use_fp16=False,
        embed_dim=1024,
        image_resolution=224,
        vision_layers=32,
        vision_width=1024,
        vision_patch_size=14,
        vision_head_width=64,
        ucg_rate=0.0,
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.model = ImageEncoder(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            vision_head_width=vision_head_width,
            epsilon=1e-5,
            use_quick_gelu=True,
            dtype=self.dtype,
        )
        self.ucg_rate = ucg_rate

        self.transform = self._build_transform()

    def _build_transform(self):
        mean = np.array([0.48145466, 0.4578275, 0.40821073]) * 255
        std = np.array([0.26862954, 0.26130258, 0.27577711]) * 255
        transforms = Compose(
            [
                vision.Resize((224, 224), vision.Inter.BICUBIC),
                vision.Normalize(mean.tolist(), std.tolist()),
                vision.HWC2CHW(),
            ]
        )
        return transforms

    def preprocess(self, image: Image.Image) -> Tensor:
        w, h = image.size
        w, h = map(lambda x: x - x % 64, (w, h))
        image = image.resize((w, h), resample=Image.LANCZOS)

        image = np.array(image)
        image = self.transform(image)
        image = image[None, ...]
        if self.use_fp16:
            image = image.astype(np.float16)
        x = ms.Tensor(image)
        return x

    def construct(self, x: Tensor, no_dropout: bool = False):
        out = self.model.encode_image(x)
        out = out.to(x.dtype)
        if self.ucg_rate > 0.0 and not no_dropout:
            out = ops.bernoulli((1.0 - self.ucg_rate) * ops.ones(out.shape[0]))[:, None] * out
        return out


class FrozenOpenCLIPImageEmbedder(CLIPImageEmbedder):
    def __init__(
        self,
        use_fp16=False,
        embed_dim=1024,
        image_resolution=224,
        vision_layers=32,
        vision_width=1024,
        vision_patch_size=14,
        vision_head_width=64,
        ucg_rate=0.0,
    ):
        super(CLIPImageEmbedder, self).__init__()
        self.use_fp16 = use_fp16
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.model = ImageEncoder(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            vision_head_width=vision_head_width,
            epsilon=1e-5,
            use_quick_gelu=False,
            dtype=self.dtype,
        )
        self.ucg_rate = ucg_rate


class CLIPEmbeddingNoiseAugmentation(ImageConcatWithNoiseAugmentation):
    def __init__(self, *args, clip_stats_path=None, timestep_dim=1024, use_fp16=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype = ms.float16 if use_fp16 else ms.float32

        if clip_stats_path is None:
            clip_mean, clip_std = ops.zeros(timestep_dim), ops.ones(timestep_dim)
        else:
            clip = ms.load_checkpoint(clip_stats_path)
            clip_mean, clip_std = clip["mean"], clip["std"]

        self.data_mean = clip_mean[None, :]
        self.data_std = clip_std[None, :]
        self.time_embed = Timestep(timestep_dim).to_float(self.dtype)

    def scale(self, x):
        # re-normalize to centered mean and unit variance
        x = (x - self.data_mean) * 1.0 / self.data_std
        return x

    def unscale(self, x):
        # back to original data stats
        x = (x * self.data_std) + self.data_mean
        return x

    def construct(self, x, noise_level=None):
        if noise_level is None:
            noise_level = ms.numpy.randint(0, self.max_noise_level, (x.shape[0],), dtype=ms.int64)
        x = self.scale(x)
        z = self.q_sample(x, noise_level)
        z = self.unscale(z)
        noise_level = self.time_embed(noise_level)
        z = ops.cast(z, self.dtype)
        noise_level = ops.cast(noise_level, self.dtype)
        return z, noise_level
