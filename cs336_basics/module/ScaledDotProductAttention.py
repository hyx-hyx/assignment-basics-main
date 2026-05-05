import torch
from einops import einsum


def softmax(x: torch.Tensor, dim: int):
    assert (dim < len(x.shape))
    x = x - torch.amax(x, dim=dim, keepdim=True)
    x_sum = torch.sum(torch.exp(x), dim=dim, keepdim=True)
    return torch.exp(x) / x_sum


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
    d_k = k.shape[-1]
    qkt = einsum(q, k, "... queries d_k,... keys d_k -> ... queries keys")
    qkt_norm = qkt / (d_k ** 0.5)
    if mask is not None:
        mask = torch.where(mask, 0.0, -torch.inf)
        qkt_softmax = softmax(qkt_norm + mask, dim=-1)

    else:
        qkt_softmax = softmax(qkt_norm, dim=-1)
    return einsum(qkt_softmax, v, "... queries keys,... keys d_v -> ... queries d_v")
