import math
from collections.abc import Callable
from typing import Optional

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)
    for t in range(10):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights ** 2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.

""""
lr=1e-3
loss=
    25.322290420532227
    24.319528579711914
    23.63652992248535
    23.093820571899414
    22.634252548217773
    22.231168746948242
    21.869619369506836
    21.540231704711914
    21.236679077148438
    20.954469680786133
    
lr=1e1
loss=
    22.26663589477539
    21.384878158569336
    20.7842960357666
    20.307079315185547
    19.902965545654297
    19.54852294921875
    19.230602264404297
    18.940959930419922
    18.674039840698242
    18.425884246826172

lr=1e2
loss=
    21.857046127319336
    20.991504669189453
    20.4019718170166
    19.933530807495117
    19.536853790283203
    19.188928604125977
    18.876855850219727
    18.59254264831543
    18.330533981323242
    18.086942672729492
    
lr=1e3
loss=
    21.099275588989258
    20.263744354248047
    19.694652557373047
    19.242450714111328
    18.859525680541992
    18.523664474487305
    18.222410202026367
    17.947952270507812
    17.695030212402344
    17.459880828857422
"""