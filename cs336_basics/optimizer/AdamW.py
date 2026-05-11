import math
from collections.abc import Callable
from typing import Optional

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr, "m": 0, "v": 0, "beta1": betas[0],
                    "beta2": betas[1], "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self.state = {}

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, beta1, beta2, epsilon, lamda, m, v = group["lr"], group[
                "beta1"], group["beta2"], group["eps"], group["weight_decay"], group["m"], group["v"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get iteration number from the state, or 0.
                t = self.state.get("t", 1)
                # Get the gradient of loss with respect to p.
                grad = p.grad.data
                # Update weight tensor in-place.
                lr_t = lr*pow(1-beta2**t, 0.5)/(1-beta1**t)
                p.data -= lr*lamda*p.data
                m = beta1*m+(1-beta1)*grad
                group["m"] = m
                v = beta2*v+(1-beta2)*(grad**2)
                group["v"] = v
                p.data -= lr_t*m/(pow(v, 0.5)+epsilon)
                self.state["t"] = t + 1  # Increment iteration number
        return loss
