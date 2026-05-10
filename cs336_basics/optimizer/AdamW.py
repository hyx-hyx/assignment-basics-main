from collections.abc import Callable
from typing import Optional

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr, "m": 0, "v": 0, "beta1": 0.9, "beta2": "0.999", "epsilon": 1e-8}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, beta1, beta2, epsilon = group["lr"], group["beta1"], group["beta2"], group["epsilon"]
