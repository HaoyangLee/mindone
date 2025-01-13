import math
import mindspore as ms
from mindspore import nn, ops
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class VanillaAttention(nn.Cell):
    def __init__(self, head_dim, dropout=0.):
        super().__init__()
        self.scale_factor = 1 / math.sqrt(head_dim)
        self.dropout = dropout

    def construct(self, q, k, v, mask=None):
        '''
        q/k/v: (B S N D)
        mask: (B 1 S S),  1 - for retain, 0 - for drop. e.g. [[1, 1, 0, 0 ..], [1, 1, 0, 0 ..]]
        '''
        input_dtype = q.dtype
        # preapre layout. (B S N D) -> (B N S D)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # q: [B N S D)
        b, num_heads, s, _ = q.shape
        attn = ops.bmm(q, k.transpose(0, 1, 3, 2)) * self.scale_factor
        attn= attn.to(ms.float32)  # (B N Sq Sk)

        if mask is not None:
            # TODO: shape of mask ??
            # mask = self.repeat_interleave(mask.to(ms.int32), h, 0)
            # mask = mask.to(ms.bool_)
            # TODO: get different -inf based on data type
            # import pdb; pdb.set_trace()
            mask = mask.tile((1, num_heads, 1, 1))
            mask = ops.logical_not(mask)  # [1, 1, 0, 0 ..] -> [0, 0, 1, 1, ..]
            attn = ops.masked_fill(attn, mask, -ms.numpy.inf)

        attn = ops.softmax(attn, axis=-1).to(input_dtype)
        attn = ops.dropout(attn, p=self.dropout)
        x = ops.bmm(attn.to(v.dtype), v)  # (B N S D)

        # prepare output layout
        x = ops.transpose(x, (0, 2, 1, 3))

        b, s, a, d = x.shape
        out = x.reshape(b, s, -1)

        return out


class FlashAttention(nn.Cell):
    def __init__(
        self,
        heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        scale_factor = 1 / math.sqrt(head_dim)
        self.flash_attention = FlashAttentionScore(heads, keep_prob=1-dropout, scale_value=scale_factor, input_layout="BNSD")

    def construct(self, q, k, v, mask=None):
        '''
        input:
            q/k/v: (B S N D)
            mask: (B 1 S S), attn mask, 1 - for retain, 0 - for drop. e.g. [[1, 1, 0, 0 ..], [1, 1, 0, 0 ..]]. dtype: bool
        output:
            o (B S N*D)
        '''
        input_dtype = q.dtype
        # preapre layout. (B S N D) -> (B N S D)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # in ms FA, 0 - retain, 1 - discard. shape: `(B, N1, S1, S2)` or `(B, 1, S1, S2)`, dtype: bool or uint8
        if mask is not None:
            mask = ops.logical_not(mask)
        _, _, _, out = self.flash_attention(q, k, v, None, None, None, mask)

        # (B N S D) -> (B S N D)
        out  = ops.transpose(out, (0, 2, 1, 3))

        B, S, N, D = out.shape
        out = out.reshape(B, S, -1)

        return out


