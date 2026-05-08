import torch
import torch.nn as nn
from cs336_basics.module.MultiHeadSelfAttention import MultiHeadSelfAttention
from cs336_basics.module.RmsNorm import RmsNorm
from cs336_basics.module.RotaryPositionalEmbedding import RotaryPositionalEmbedding
from cs336_basics.module.SwigluFeedForward import SwigluFeedForward

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
        
        # first sublayer
        rms_norm=RmsNorm(self.d_model)
        rms_norm.load_state_dict({"g": self.weights["ln1.weight"]})
        
        multi_head_self_attention=MultiHeadSelfAttention(self.d_model,self.num_heads)
        multi_head_self_attention.weights={k:self.weights[k] for k in ["attn.q_proj.weight","attn.k_proj.weight","attn.v_proj.weight","attn.output_proj.weight"]}
        
        rope=RotaryPositionalEmbedding(self.theta,self.d_model//self.num_heads,self.max_seq_len)
        y=x + multi_head_self_attention.forward(rms_norm.forward(x),rope)


        # second sublayer
        rms_norm.load_state_dict({"g": self.weights["ln2.weight"]})
        
        swiglu_feed_forward=SwigluFeedForward(self.d_model,self.d_ff)
        swiglu_feed_forward.w1.weight.data = self.weights["ffn.w1.weight"]
        swiglu_feed_forward.w2.weight.data = self.weights["ffn.w2.weight"]
        swiglu_feed_forward.w3.weight.data = self.weights["ffn.w3.weight"]
        return y+swiglu_feed_forward.forward(rms_norm.forward(y))

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int):
        super().__init__()

    def forward():
        pass
