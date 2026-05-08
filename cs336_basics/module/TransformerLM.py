import re

import torch
import torch.nn as nn
from cs336_basics.module.Embedding import Embedding
from cs336_basics.module.Linear import Linear
from cs336_basics.module.MultiHeadSelfAttention import MultiHeadSelfAttention
from cs336_basics.module.RmsNorm import RmsNorm
from cs336_basics.module.RotaryPositionalEmbedding import RotaryPositionalEmbedding
from cs336_basics.module.SwigluFeedForward import SwigluFeedForward


class TransformerLM(nn.Module):
    class TransformerBlock(nn.Module):
        def __init__(self, d_model: int, num_heads: int, d_ff: int,
                     max_seq_len: int, theta: float):
            super().__init__()

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_ff = d_ff
            self.max_seq_len = max_seq_len
            self.theta = theta
            self.weights = {}

        def forward(self, x: torch.Tensor):
            # first sublayer
            rms_norm = RmsNorm(self.d_model)
            rms_norm.load_state_dict({"g": self.weights["ln1.weight"]})

            multi_head_self_attention = MultiHeadSelfAttention(self.d_model, self.num_heads)
            multi_head_self_attention.weights = {k: self.weights[k] for k in
                                                 ["attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight",
                                                  "attn.output_proj.weight"]}

            rope = RotaryPositionalEmbedding(self.theta, self.d_model // self.num_heads, self.max_seq_len)
            y = x + multi_head_self_attention.forward(rms_norm.forward(x), rope)

            # second sublayer
            rms_norm.load_state_dict({"g": self.weights["ln2.weight"]})

            swiglu_feed_forward = SwigluFeedForward(self.d_model, self.d_ff)
            swiglu_feed_forward.w1.weight.data = self.weights["ffn.w1.weight"]
            swiglu_feed_forward.w2.weight.data = self.weights["ffn.w2.weight"]
            swiglu_feed_forward.w3.weight.data = self.weights["ffn.w3.weight"]
            return y + swiglu_feed_forward.forward(rms_norm.forward(y))

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 theta: float, weights: dict[str, torch.Tensor],
                 vocab_size: int, context_length: int, num_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.transformer_blocks = []
        self.weights = weights
        self.d_model = d_model

        for i in range(0, num_layers):
            weight_str_list = ["layers." + str(i) + ".attn.q_proj.weight"
                , "layers." + str(i) + ".attn.k_proj.weight"
                , "layers." + str(i) + ".attn.v_proj.weight"
                , "layers." + str(i) + ".attn.output_proj.weight"
                , "layers." + str(i) + ".ln1.weight"
                , "layers." + str(i) + ".ffn.w1.weight"
                , "layers." + str(i) + ".ffn.w2.weight"
                , "layers." + str(i) + ".ffn.w3.weight"
                , "layers." + str(i) + ".ln2.weight"]

            transformer_block = self.TransformerBlock(d_model, num_heads, d_ff, context_length, theta)
            for weight_str in weight_str_list:
                # 使用正则表达式去掉 layers.{num}. 部分
                result_str = re.sub(r'^layers\.\d+\.', '', weight_str)
                transformer_block.weights[result_str] = self.weights[weight_str]

            self.transformer_blocks.append(transformer_block)

    def forward(self, x: torch.Tensor):

        embedding = Embedding(self.vocab_size, self.d_model)
        embedding.w.data = self.weights["token_embeddings.weight"]
        y = embedding.forward(x)

        for tb in self.transformer_blocks:
            y = tb.forward(y)

        rmsnorm = RmsNorm(self.d_model)
        rmsnorm.g.data = self.weights["ln_final.weight"]
        linear = Linear(self.d_model, self.vocab_size)
        linear.weight.data = self.weights["lm_head.weight"]
        return linear.forward(rmsnorm.forward(y))
