from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, special

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
from cs336_basics.Tokenizer import Tokenizer
from cs336_basics.utils.cross_entropy import eval_cross_entropy
from cs336_basics.utils.gradient_clipping import gradient_clipping
from tests.conftest import d_model, vocab_size


class TransfomerTrainer:
    def __init__(self, model_config, optimizer_config, device):
        self.model = self._set_model(model_config)
        self.optimizer = self._set_adamw(optimizer_config)
        self.device = device
        self.current_iter = 0

    def _set_model(self, model_config):
        model_param_list = ["d_model", "num_heads", "d_ff",
                            "rope_theta", "vocab_size", "num_layers"]
        d_model, num_heads, d_ff, rope_theta, vocab_size, context_length, num_layers = [
            model_config[i] for i in model_param_list]
        transformer_lm = TransformerLM(d_model, num_heads, d_ff, rope_theta, vocab_size, context_length,
                                       num_layers)
        return transformer_lm.to(self.device)

    def _set_adamw(self, optimizer_config):
        optimizer_param_list = ["learning_rate",
                                "weight_decay", "betas", "eps"]
        lr, weight_decay, betas, eps = [
            optimizer_config[i] for i in optimizer_param_list]
        return AdamW(lr, weight_decay, betas, eps)

    def prepare_data(self, input_path, output_dir, bpe_config):
        vocab_size, special_tokens = bpe_config["vocab_size"], bpe_config["special_tokens"]

        # BPE分词器
        trainer = BpeTrain(input_path, vocab_size, special_tokens)
        vocab, merges = trainer.train()

        # 保存 vocab 和 merges
        import json
        vocab_path = os.path.join(output_dir, "vocab.json")
        merges_path = os.path.join(output_dir, "merges.txt")

        # 保存 vocab (需要转换格式)
        vocab_to_save = {k: v.decode('utf-8', errors='replace')
                         for k, v in vocab.items()}
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_to_save, f)

        # 保存 merges
        with open(merges_path, 'w', encoding='utf-8') as f:
            for merge in merges:
                token1 = merge[0].decode('utf-8', errors='replace')
                token2 = merge[1].decode('utf-8', errors='replace')
                f.write(f"{token1} {token2}\n")

        print(f"Saved vocab to {vocab_path}")
        print(f"Saved merges to {merges_path}")

        # Step 2: Tokenization
        print("Step 2: Tokenizing text...")
        self.tokenizer = Tokenizer(vocab, merge, ["<|endoftext|>"])
        token_ids = []
        with open(input_path, 'r') as f:
            for id in self.tokenizer.encode_iterable(f):
                token_ids.append(id)

        # 保存为 numpy 数组
        dataset_path = os.path.join(output_dir, "dataset.npy")
        np.save(dataset_path, np.array(token_ids))
        print(f"Saved dataset to {dataset_path}")
        print(f"Total tokens: {len(token_ids)}")

        return vocab, merges

    def train(self, train_dataset, num_epochs,
              eval_interval=1000, save_interval=5000,
              out_checkpoint_dir: str | os.PathLike | BinaryIO | IO[bytes] = None,
              in_checkpoint_dir: str | os.PathLike | BinaryIO | IO[bytes] = None,
              data_config=None,
              scheduler_config=None):
        """
        完整的训练循环

        Args:
            train_dataset: token IDs 的 numpy 数组
            num_epochs: 总训练迭代次数
            eval_interval: 每多少次迭代评估一次
            save_interval: 每多少次迭代保存一次检查点
            data_config: 数据加载参数配置
            scheduler_config: 学习率调度配置
        """
        print(f"Starting training for {num_epochs} iterations...")

        if in_checkpoint_dir:
            print(f"从检查点恢复继续训练,检查点存储地址:{in_checkpoint_dir}")
            load_checkpoint(in_checkpoint_dir, self.model, self.optimizer)
        else:
            print("未设置检查点地址,训练从0开始!")

        for _ in range(num_epochs):
            # 数据加载
            self.current_iter += 1
            data_param_list = ["batch_size", "context_length", "device"]
            # data
            batch_size, context_length, device = [
                data_config[i] for i in data_param_list]
            x, y = data_loading(train_dataset, batch_size,
                                context_length, device)

            self.train_epoch(x, y)
            # self.evaluate()
            if out_checkpoint_dir:
                save_checkpoint(self.model, self.optimizer,
                                self.current_iter, out_checkpoint_dir)
            else:
                print("未设置checkpoint的保存地址!!")

            # 学习率调整
            max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters = scheduler_config["max_learning_rate"], scheduler_config[
                "min_learning_rate"], scheduler_config["warmup_iters"], scheduler_config["cosine_cycle_iters"]
            current_lr = learning_rate_schedule(
                self.current_iter, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
            for param in self.optimizer.param_groups:
                param['lr'] = current_lr

    def train_epoch(self, batch_x, batch_y):
        # 单个 epoch 的训练逻辑

        # 前向传播
        y_prime = self.model.forward(batch_x)
        # 计算损失
        loss = eval_cross_entropy(y_prime, batch_y)
        # 后向传播
        loss.backward()
        # 梯度裁剪
        gradient_clipping()
        # optimizer
        self.optimizer.step(loss.backward)


if __name__ == "__main__":
    config = {
        # ========== 模型架构参数 ==========
        "model": {
            "d_model": 512,           # 模型维度
            "num_heads": 8,           # 注意力头数
            "d_ff": 2048,             # 前馈网络维度
            "num_layers": 6,          # Transformer 层数
            "rope_theta": 10000.0,    # RoPE 的 theta 参数
            "vocab_size": 50000,      # 词汇表大小
            "context_length": 1024,   # 上下文长度
        },

        # ========== 数据参数 ==========
        "data": {
            "batch_size": 32,         # batch 大小
            "context_length": 1024,   # 序列长度（与模型一致）
            "device": "cuda",         # 训练设备
        },

        # ========== 优化器参数 ==========
        "optimizer": {
            "learning_rate": 3e-4,    # 学习率
            "weight_decay": 0.01,     # 权重衰减
            "betas": (0.9, 0.999),    # AdamW 的 beta 参数
            "eps": 1e-8,              # AdamW 的 epsilon
        },

        # ========== 学习率调度参数 ==========
        "scheduler": {
            "max_learning_rate": 3e-4,
            "min_learning_rate": 3e-5,
            "warmup_iters": 2000,
            "cosine_cycle_iters": 100000,
        },

        # ========== 训练控制参数 ==========
        "training": {
            "num_iterations": 100000,
            "eval_interval": 1000,
            "save_interval": 5000,
            "gradient_clip_norm": 1.0,
        },

        # ========== BPE 训练参数（预处理用）==========
        "bpe": {
            "vocab_size": 50000,
            "special_tokens": ["<|endoftext|>"],
        },
    }
    tt = TransfomerTrainer(config, "cuda")
    tt.prepare_data()
    tt.train()
