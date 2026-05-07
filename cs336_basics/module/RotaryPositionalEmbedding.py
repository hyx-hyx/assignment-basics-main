import torch
import torch.nn as nn
from einops import rearrange, einsum


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k=d_k
        idx = torch.linspace(start=0, end=max_seq_len - 1, steps=max_seq_len)
        k = torch.linspace(start=1, end=d_k / 2, steps=int(d_k / 2))
        idx = rearrange(idx, "max_seq_len -> max_seq_len 1")
        k = rearrange(k, "d_k -> 1 d_k")

        theta_i_k = idx / pow(theta, (2 * k - 2) / d_k)

        cos_theta = torch.cos(theta_i_k)  # (12, 32)
        sin_theta = torch.sin(theta_i_k)  # (12, 32)
        r_matrices = torch.zeros(max_seq_len, d_k, d_k)
        for idx in range(d_k // 2):
            # 获取对应块的 cos 和 sin
            cos_val = cos_theta[:, idx]
            sin_val = sin_theta[:, idx]

            start_col = 2 * idx

            # 填充 2×2 块
            r_matrices[:, start_col, start_col] = cos_val
            r_matrices[:, start_col, start_col + 1] = -sin_val
            r_matrices[:, start_col + 1, start_col] = sin_val
            r_matrices[:, start_col + 1, start_col + 1] = cos_val
        self.register_buffer("rotate matrix", r_matrices, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = token_positions.shape[-1]

        one_hot = torch.eye(seq_len)[token_positions]
        all_position_ri = einsum(self.get_buffer("rotate matrix")[0:seq_len,0:self.d_k,0:self.d_k], one_hot,
                                 "seq_len d_k_row d_k_col,... seq_len seq_len_col -> ... seq_len_col d_k_row d_k_col")
        return einsum(all_position_ri, x, "... seq_len_col d_k_row d_k_col, ... seq_len_col d_k_col -> ... seq_len_col d_k_row")


if __name__ == "__main__":
    print("test")
