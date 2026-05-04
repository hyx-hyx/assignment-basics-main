import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from einops import rearrange, einsum


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        # 创建一个权重张量
        self.w = torch.empty(num_embeddings, embedding_dim)
        self.vocab_size=num_embeddings
        self.d_model=embedding_dim
        init.trunc_normal_(self.w, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        one_hot=np.eye(self.vocab_size)[token_ids]
        return einsum(one_hot, self.w, "... v_size,v_size d_model -> ... d_model")


if __name__ == "__main__":
    print("test")
