import math
from collections.abc import Callable
from typing import Optional

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {
            "lr": lr, "beta1": betas[0], "beta2": betas[1], "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr, beta1, beta2, epsilon, lamda = group["lr"], group[
                "beta1"], group["beta2"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get the gradient of loss with respect to p.
                grad = p.grad.data
                # 获取参数状态
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                # Update weight tensor in-place.
                lr_t = lr*math.sqrt(1-beta2**t)/(1-beta1**t)
                p.data -= lr*lamda*p.data
                m.mul_(beta1).add_(grad, alpha=(1-beta1))
                v.mul_(beta2).add_((grad**2), alpha=(1-beta2))
                p.data -= lr_t*m/(torch.sqrt(v)+epsilon)
        return loss
