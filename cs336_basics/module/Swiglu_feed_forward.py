import torch
import torch.nn as nn
from cs336_basics.module.Linear import Linear
from tests.conftest import d_model


class Swiglu_feed_forward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1.forward(x)
        w3x = self.w3.forward(x)
        return self.w2.forward(w1x * torch.sigmoid(w1x) * w3x)


if __name__ == "__main__":
    print("test")
