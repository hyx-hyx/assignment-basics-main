import torch
import torch.nn as nn
import torch.nn.init as init
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        
        # 创建一个权重张量
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        std = 2.0 / (out_features + in_features)
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... x_in,w_out x_in -> ... w_out")


if __name__ == "__main__":
    print("test")
