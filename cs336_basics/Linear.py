import torch
import torch.nn as nn
import torch.nn.init as init
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        super().__init__()
        # 创建一个权重张量
        self.w = torch.empty(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x,self.w,"... x_in,w_out x_in -> ... w_out")


if __name__ == "__main__":
    print("test")
