from abc import abstractmethod

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import jit

from mindone.utils.amp import auto_mixed_precision

from ..attention import SpatialTransformer
from .motion_module import VanillaTemporalModule, get_motion_module
from .openaimodel import AttentionBlock, Downsample, ResBlock, Upsample, Timestep
from .util import conv_nd, linear, normalization, timestep_embedding, zero_module, default
from .unet3d import rearrange_in, rearrange_out


class TimestepBlock(nn.Cell):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def construct(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.SequentialCell, TimestepBlock):
    def construct(self, x, emb, context=None, video_length=None, norm_in_5d=None, **kwargs):
        for cell in self.cell_list:
            if isinstance(cell, VanillaTemporalModule) or (isinstance(cell, ResBlock) and norm_in_5d):
                if isinstance(cell, TimestepBlock):
                    x = cell(x, emb, video_length=video_length)
                elif isinstance(cell, SpatialTransformer):
                    x = cell(x, context, video_length=video_length)
                else:
                    x = cell(x, video_length=video_length)
            else:
                if isinstance(cell, TimestepBlock):
                    x = cell(x, emb)
                elif isinstance(cell, SpatialTransformer):
                    x = cell(x, context)
                else:
                    x = cell(x)
        return x


class UNet3DModel(nn.Cell):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        # disable_self_attentions=None,
        num_attention_blocks=None,
        # disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        enable_flash_attention=False,
        adm_in_channels=None,
        transformer_depth_middle=None,
        use_recompute=False,
        use_fp16=False,
        # Additional
        use_inflated_groupnorm=True,  # diff, default is to use in mm-v2, which is more reasonable.
        use_motion_module=False,
        motion_module_resolutions=(1, 2, 4, 8),  # used to identify which level to be injected with Motion Module
        motion_module_mid_block=False,
        motion_module_decoder_only=False,
        motion_module_type=None,  # default:
        motion_module_kwargs={},  #
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
    ):
        super().__init__()
        from omegaconf.listconfig import ListConfig
        
        if use_motion_module:
            assert unet_use_cross_frame_attention is False, "not support"
            assert unet_use_temporal_attention is False, "not support"
            assert motion_module_type == "Vanilla", "not support"
        else:
            print("D---: WARNING: not using motion module")

        self.norm_in_5d = not use_inflated_groupnorm

        print("D--: norm in 5d: ", self.norm_in_5d)
        # print("D--: flash attention: ", enable_flash_attention)

        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        elif isinstance(transformer_depth, ListConfig):
            transformer_depth = list(transformer_depth)
        transformer_depth_middle = default(transformer_depth_middle, transformer_depth[-1])

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        # self.num_res_blocks = num_res_blocks
        # if disable_self_attentions is not None:
        #     # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
        #     assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )  # todo: convert to warning

        self.attention_resolutions = attention_resolutions
        self.dropout = 1.0 - dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.dtype = ms.float16 if use_fp16 else ms.float32

        time_embed_dim = model_channels * 4

        self.time_embed = nn.SequentialCell(
            [
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            ]
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Dense(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.SequentialCell(
                    [
                        Timestep(model_channels),
                        nn.SequentialCell(
                            [
                                linear(model_channels, time_embed_dim),
                                nn.SiLU(),
                                linear(time_embed_dim, time_embed_dim),
                            ]
                        ),
                    ]
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.SequentialCell(
                    [
                        nn.SequentialCell(
                            [
                                linear(adm_in_channels, time_embed_dim),
                                nn.SiLU(),
                                linear(time_embed_dim, time_embed_dim),
                            ]
                        )
                    ]
                )
            else:
                raise ValueError()

        self.input_blocks = nn.CellList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1, pad_mode="pad"))]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                # input blocks
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        self.dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm_in_5d=self.norm_in_5d,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    # if disable_self_attentions is not None:
                    #     disabled_sa = disable_self_attentions[level]
                    # else:
                    #     disabled_sa = False

                    if num_attention_blocks is None or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                # disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                enable_flash_attention=enable_flash_attention,
                            )
                        )

                # add MotionModule 1) after SpatialTransformer in DownBlockWithAttn, 3*2 times, or 2) after ResBlock in DownBlockWithoutAttn, 1*2 time.
                if use_motion_module:
                    layers.append(
                        # TODO: set mm fp32/fp16 independently?
                        get_motion_module(  # return VanillaTemporalModule
                            in_channels=ch,
                            motion_module_type=motion_module_type,
                            motion_module_kwargs=motion_module_kwargs,
                            dtype=self.dtype,
                        )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            norm_in_5d=self.norm_in_5d,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        # middle block, add 1 MM
        layers = [
            ResBlock(
                ch,
                time_embed_dim,
                self.dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                norm_in_5d=self.norm_in_5d,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(  # always uses a self-attn
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                # disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                enable_flash_attention=enable_flash_attention,
            )]

        # Add MM after SpatialTrans in MiddleBlock, 1
        if use_motion_module and motion_module_mid_block:
            layers.append(
                get_motion_module(
                    in_channels=ch,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                    dtype=self.dtype,
                )
            )

        layers.append(
            ResBlock(
            ch,
            time_embed_dim,
            self.dropout,
            dims=dims,
            use_scale_shift_norm=use_scale_shift_norm,
            norm_in_5d=self.norm_in_5d,
            )
        )
        self.middle_block = TimestepEmbedSequential(*layers)

        self._feature_size += ch

        self.output_blocks = nn.CellList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        self.dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm_in_5d=self.norm_in_5d,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    # if disable_self_attentions is not None:
                    #     disabled_sa = disable_self_attentions[level]
                    # else:
                    #     disabled_sa = False

                    if num_attention_blocks is None or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                # disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                enable_flash_attention=enable_flash_attention,
                            )
                        )

                # Add MM after ResBlock in UpBlockWithoutAttn (1*3), or after SpatialTransformer in UpBlockWithAttn (3*3)
                if use_motion_module:
                    layers.append(
                        get_motion_module(
                            in_channels=ch,
                            motion_module_type=motion_module_type,
                            motion_module_kwargs=motion_module_kwargs,
                            dtype=self.dtype,
                        )
                    )

                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            norm_in_5d=self.norm_in_5d,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.conv_norm_out = normalization(ch, norm_in_5d=self.norm_in_5d)

        self.out = nn.SequentialCell(
            [
                # normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1, pad_mode="pad")),
            ]
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.SequentialCell(
                [
                    # normalization(ch),
                    conv_nd(dims, model_channels, n_embed, 1),
                    # nn.LogSoftmax(axis=1)  # change to cross_entropy and produce non-normalized logits
                ]
            )

        if use_recompute:
            self.recompute_strategy_v1()

    @jit
    def construct(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # assert (y is not None) == (
        #     self.num_classes is not None
        # ), "must specify y if and only if the model is class-conditional"
        hs, hs_idx = (), -1
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        # 0. rearrange inputs to (b*f, ...) for pseudo 3d until we meet temporal transformer (i.e. motion module)
        B, C, F, H, W = x.shape
        # x: (b c f h w) -> (b*f c h w)
        x = rearrange_in(x)
        # time mixed with other embedding: (b dim_emb) -> (b*f dim_emb)
        emb = emb.repeat_interleave(repeats=F, dim=0)
        # context: (b max_length dim_clip) -> (b*f dim_emb)
        context = context.repeat_interleave(repeats=F, dim=0)

        h = x

        # 1. conv_in and downblocks
        for module in self.input_blocks:
            h = module(h, emb, context, F, self.norm_in_5d)
            hs += (h,)
            hs_idx += 1

        # 2. middle block
        h = self.middle_block(h, emb, context, F, self.norm_in_5d)

        # 3. up blocks
        for module in self.output_blocks:
            h = ops.concat([h, hs[hs_idx]], axis=1)
            hs_idx -= 1
            h = module(h, emb, context, F, self.norm_in_5d)

        if self.norm_in_5d:
            h = self.conv_norm_out(h, video_length=F)
        else:
            h = self.conv_norm_out(h)

        h = self.out(h)

        # rearrange back: (b*f c h w) -> (b c f h w)
        h = rearrange_out(h, f=F)

        return h

    # TODO: adapt to TimestepEmbedSequential
    # def set_mm_amp_level(self, amp_level):
    #     # set motion module precision
    #     print("D--: mm amp level: ", amp_level)
    #     for i, celllist in enumerate(self.input_blocks, 1):
    #         for cell in celllist:
    #             if isinstance(cell, VanillaTemporalModule):
    #                 cell = auto_mixed_precision(cell, amp_level)

    #     for module in self.middle_block:
    #         if isinstance(module, VanillaTemporalModule):
    #             module = auto_mixed_precision(module, amp_level)

    #     for celllist in self.output_blocks:
    #         for cell in celllist:
    #             if isinstance(cell, VanillaTemporalModule):
    #                 cell = auto_mixed_precision(cell, amp_level)

    def recompute_strategy_v1(self):
        # embed
        self.time_embed.recompute()
        self.label_emb.recompute()

        # input blocks
        self.input_blocks[4][0].recompute()  # 4
        self.input_blocks[5][0].recompute()  # 5
        self.input_blocks[7][0].recompute()  # 7
        self.input_blocks[8][0].recompute()  # 8

        # middle block
        self.middle_block[0].recompute()
        self.middle_block[1].recompute()

        # output blocks
        self.output_blocks[0][1].recompute()  # 0
        self.output_blocks[1][1].recompute()  # 1
        self.output_blocks[2][1].recompute()  # 2
        self.output_blocks[2][2].recompute()  # 2
        self.output_blocks[3][1].recompute()  # 3
        self.output_blocks[4][1].recompute()  # 4
        self.output_blocks[5][1].recompute()  # 5
        self.output_blocks[5][2].recompute()  # 5

        print("Turn on recompute with StrategyV1.")