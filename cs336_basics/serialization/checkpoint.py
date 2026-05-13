import os
import typing

import torch
import torch.nn as nn


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_obj = model.state_dict()
    opt_obj = optimizer.state_dict()
    obj = {"model": model_obj, "optimizer": opt_obj, "iteration": iteration}
    torch.save(obj, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    src_obj = torch.load(src)
    model.load_state_dict(src_obj["model"])
    optimizer.load_state_dict(src_obj["optimizer"])
    return src_obj["iteration"]
