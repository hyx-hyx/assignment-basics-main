from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    l2_norm_square = 0.0
    for p in parameters:
        if p.grad is not None:
            grad = p.grad.data
            l2_norm_square += torch.sum(grad**2)
    for p in parameters:
        if p.grad is not None:
            if l2_norm_square > max_l2_norm**2:
                p.grad.data.mul_(
                    max_l2_norm / (torch.sqrt(l2_norm_square)+1e-6))
