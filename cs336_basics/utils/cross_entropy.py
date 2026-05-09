import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int

from cs336_basics.module.ScaledDotProductAttention import (
    scaled_dot_product_attention, softmax)
from tests.conftest import vocab_size


def eval_cross_entropy(o: Float[torch.Tensor, " batch_size vocab_size"], x: Int[torch.Tensor, " batch_size"]):
    vocab_size = o.shape[-1]
    o_1_i = o[:, 0:vocab_size-1]
    log_softmax = -torch.log(softmax(o_1_i, 1))
    i = torch.linspace(0, vocab_size-1, steps=vocab_size, dtype=int)
    one_hot = torch.eye(vocab_size)[x[i+1]]
    logp = einsum(one_hot, log_softmax,
                  "vocab_size_r vocab_size_c,batch_size vocab_size_c -> batch_size vocab_size_r")
    return torch.sum(logp)/o.shape[0]/o.shape[-1]
