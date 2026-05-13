from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.BPE import BpeTrain
from cs336_basics.data.DataLoading import data_loading
from cs336_basics.module.Embedding import Embedding
from cs336_basics.module.Linear import Linear
from cs336_basics.module.MultiHeadSelfAttention import MultiHeadSelfAttention
from cs336_basics.module.RmsNorm import RmsNorm
from cs336_basics.module.RotaryPositionalEmbedding import RotaryPositionalEmbedding
from cs336_basics.module.ScaledDotProductAttention import scaled_dot_product_attention, softmax
from cs336_basics.module.SwigluFeedForward import SwigluFeedForward, silu
from cs336_basics.module.TransformerLM import TransformerLM
from cs336_basics.optimizer import learning_rate_schedule
from cs336_basics.optimizer.AdamW import AdamW
from cs336_basics.serialization.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.utils.cross_entropy import eval_cross_entropy
from cs336_basics.utils.gradient_clipping import gradient_clipping


def train(dataset: npt.NDArray, params: dict, out: str | os.PathLike | BinaryIO | IO[bytes]):

    data_param_list = ["batch_size", "context_length", "device"]
    model_param_list = ["d_model", "num_heads", "d_ff",
                        "rope_theta", "weights", "vocab_size", "num_layers"]
    optimizer_param_list = []
    # embedding

    # data
    batch_size, context_length, device = [params[i] for i in data_param_list]
    data = data_loading(npt, batch_size, context_length, device)

    # embedding
    Embedding(vocab_size, d_model)

    # model
    d_model, num_heads, d_ff, rope_theta, weights, vocab_size, context_length, num_layers = [
        params[i] for i in model_param_list]
    transformer_lm = TransformerLM(d_model, num_heads, d_ff, rope_theta, weights, vocab_size, context_length,
                                   num_layers)

    adamw = AdamW(params, lr, weight_decay, betas, eps)
