import torch
import torch.nn as nn
from cs336_basics.module.MultiHeadSelfAttention import MultiHeadSelfAttention
from RmsNorm import RmsNorm
from cs336_basics.module.RotaryPositionalEmbedding import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int, theta: float, weights: dict[str, torch.Tensor], ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len=max_seq_len
        self.theta = theta
        self.weights = weights

    def forward(self, x: torch.Tensor):
        rms_norm=RmsNorm(self.d_model)
        multi_head_self_attention=MultiHeadSelfAttention(self.d_model,self.num_heads)
        rope=RotaryPositionalEmbedding(self.theta,self.d_model,self.max_seq_len)
        return x + multi_head_self_attention.forward(rms_norm.forward(x),rope)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int):
        super().__init__()

    def forward():
        pass
