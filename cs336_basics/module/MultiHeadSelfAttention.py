from __future__ import annotations

import einx
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Int

from cs336_basics.module.RotaryPositionalEmbedding import \
    RotaryPositionalEmbedding
from cs336_basics.module.ScaledDotProductAttention import \
    scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()

        self.weights = {}
        self.d_model = d_model
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor,
                rope: RotaryPositionalEmbedding | None = None,
                token_positions: Int[torch.Tensor, " ... sequence_length"] | None = None) -> torch.Tensor:
        d_k = d_v = self.d_model // self.num_heads
        w_q = self.weights["attn.q_proj.weight"]
        w_k = self.weights["attn.k_proj.weight"]
        w_v = self.weights["attn.v_proj.weight"]
        w_o = self.weights["attn.output_proj.weight"]

        """
        这里进行代码简化：
        Q = einx.dot("... sequence_length d_model,(h dk) d_model -> ... h sequence_length dk",x, self.w_q,h=self.num_heads,dk=d_k)
        等价于：
            Q = einsum(x, self.w_q, "... sequence_length d_model,hdk d_model -> ... sequence_length hdk")
            # 将维度按注意力头拆分
            Q = rearrange(Q, "... sequence_length (h dk) -> ... h sequence_length dk", h=self.num_heads)
        """

        Q = einx.dot("... sequence_length d_model,(h dk) d_model -> ... h sequence_length dk", x, w_q,
                     h=self.num_heads, dk=d_k)
        K = einx.dot("... sequence_length d_model,(h dk) d_model -> ... h sequence_length dk", x, w_k,
                     h=self.num_heads, dk=d_k)
        V = einx.dot("... sequence_length d_model,(h dv) d_model -> ... h sequence_length dv", x, w_v,
                     h=self.num_heads, dk=d_v)

        # 加上旋转位置编码
        if rope is not None:
            if token_positions is None:
                seq_len = x.shape[-2]
                token_positions = torch.arange(0, seq_len, dtype=torch.long)

            Q = rope.forward(Q, token_positions)
            K = rope.forward(K, token_positions)

        # 添加因果掩码
        mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2])).to(torch.bool)
        mask = mask[(None,)]
        multi_head = scaled_dot_product_attention(Q, K, V, mask)

        # 多头注意力计算结果合并
        multi_head = rearrange(
            multi_head, "... h sequence_length dk -> ... sequence_length (h dk)")

        # Wo矩阵运算
        multi_head_self_attention = einsum(
            multi_head, w_o, "... hdv, d_model hdv -> ... d_model")

        return multi_head_self_attention


if __name__ == "__main__":
    print("test")
