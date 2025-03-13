from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import mint


def liger_rope(pos: Tensor, dim: int, theta: int) -> tuple[Tensor, Tensor]:
    assert dim % 2 == 0
    scale = mint.arange(0, dim, 2, dtype=mstype.float32) / dim
    omega = 1.0 / (theta**scale)
    out = mint.einsum("...n,d->...nd", pos, omega)  # (b, seq, dim//2)
    cos = out.cos()
    sin = out.sin()

    return cos, sin


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = mint.arange(0, dim, 2, dtype=mstype.float64) / dim
    omega = 1.0 / (theta**scale)
    out = mint.einsum("...n,d->...nd", pos, omega)
    out = mint.stack([mint.cos(out), -mint.sin(out), mint.sin(out), mint.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:3], 2, 2)  # b n d (i j) -> b n d i j
    return out.to(mstype.float32)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.to(mstype.float32).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(mstype.float32).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def rearrange_tensor(tensor: Tensor) -> Tensor:
    """
    Rearranges the last dimension (D) of the input tensor based on the specified mapping:
    2d -> d, 2d+1 -> D/2 + d.

    Args:
        tensor (Tensor): Input tensor of shape [B, H, L, D], where D is even.

    Returns:
        Tensor: Tensor with rearranged last dimension, same shape as input.
    """
    B, H, L, D = tensor.shape
    if D % 2 != 0:
        raise ValueError("The last dimension D must be even.")

    half_D = D // 2
    indices = mint.empty(D, dtype=mstype.int64)

    # Fill the indices based on the mapping rule
    indices[:half_D] = mint.arange(0, D, 2)
    indices[half_D:] = mint.arange(1, D, 2)

    # Rearrange the tensor based on the computed indices
    return tensor.index_select(dim=-1, index=indices)


def reverse_rearrange_tensor(tensor: Tensor) -> Tensor:
    """
    Restores the original order of the last dimension (D) of the input tensor based on the reverse mapping:
    d -> 2d, D/2 + d -> 2d + 1.

    Args:
        tensor (Tensor): Input tensor of shape [B, H, L, D], where D is even.

    Returns:
        Tensor: Tensor with restored original last dimension order, same shape as input.
    """
    B, H, L, D = tensor.shape
    if D % 2 != 0:
        raise ValueError("The last dimension D must be even.")

    half_D = D // 2
    reverse_indices = mint.empty(D, dtype=mstype.int64)

    # Fill the reverse indices to restore the original order
    reverse_indices[::2] = mint.arange(half_D)
    reverse_indices[1::2] = mint.arange(half_D, D)

    # Rearrange the tensor based on the reverse indices
    return tensor.index_select(dim=-1, index=reverse_indices)
