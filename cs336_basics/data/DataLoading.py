from re import L

import numpy as np
import numpy.typing as npt
import torch
from torchgen import context


def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    np.save("dataset.npy", dataset)
    data = np.load("dataset.npy", mmap_mode='r')

    len_data = len(data)
    uniform_custom = np.random.randint(
        0, len(data)-context_length, size=batch_size)

    data_tensor = torch.from_numpy(data)
    uniform_tensor = torch.from_numpy(uniform_custom)

    idx = uniform_tensor.unsqueeze(1)+torch.arange(context_length)
    lt = data_tensor[idx].to(device)
    rt = data_tensor[idx+1].to(device)

    return lt, rt
