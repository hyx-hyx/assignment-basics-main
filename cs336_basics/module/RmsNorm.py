import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from einops import reduce


class RmsNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = nn.Parameter(torch.empty(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_square_sum = reduce(x * x, "... d_model -> ...", "sum")
        rms = torch.pow(x_square_sum / self.d_model + self.eps, 0.5)
        result = rearrange(self.g, "d_model -> 1 1 d_model") * x / rearrange(rms, "... -> ... 1")
        return result.to(in_dtype)


if __name__ == "__main__":
    print("test")
