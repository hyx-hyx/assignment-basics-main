import torch
import torch.nn as nn
from cs336_basics.module.ScaledDotProductAttention import scaled_dot_product_attention
from einops import einsum, rearrange


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()

        self.w_q = nn.Parameter(torch.empty(d_model, d_model))
        self.w_k = nn.Parameter(torch.empty(d_model, d_model))
        self.w_v = nn.Parameter(torch.empty(d_model, d_model))
        self.w_o = nn.Parameter(torch.empty(d_model, d_model))
        self.d_model = d_model
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_k = d_v = int(self.d_model / self.num_heads)
        """
        这里可以进行代码简化：
        Q = einx.dot("... sequence_length d_model,(h dk) d_model -> ... h sequence_length dk",x, self.w_q,h=self.num_heads,dk=d_k)
        等价于：
            Q = einsum(x, self.w_q, "... sequence_length d_model,hdk d_model -> ... sequence_length hdk")
            Q = rearrange(Q, "... sequence_length (h dk) -> ... h sequence_length dk", h=self.num_heads)
        """

        Q = einsum(x, self.w_q, "... sequence_length d_model,hdk d_model -> ... sequence_length hdk")
        K = einsum(x, self.w_k, "... sequence_length d_model,hdk d_model -> ... sequence_length hdk")
        V = einsum(x, self.w_v, "... sequence_length d_model,hdv d_model -> ... sequence_length hdv")
        mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2])).to(torch.bool)
        mask = mask[(None,)]

        # 这里将原来的d_model拆分为两个维度，h 和 dk/dv,便于矩阵相乘，并行计算

        Q = rearrange(Q, "... sequence_length (h dk) -> ... h sequence_length dk", h=self.num_heads)
        K = rearrange(K, "... sequence_length (h dk) -> ... h sequence_length dk", h=self.num_heads)
        V = rearrange(V, "... sequence_length (h dv) -> ... h sequence_length dv", h=self.num_heads)
        multi_head = scaled_dot_product_attention(Q, K, V, mask)

        # 多头注意力计算结果合并
        multi_head = rearrange(multi_head, "... h sequence_length dk -> ... sequence_length (h dk)")

        multi_head_self_attention = einsum(multi_head, self.w_o, "... hdv, d_model hdv -> ... d_model")

        return multi_head_self_attention


if __name__ == "__main__":
    print("test")
