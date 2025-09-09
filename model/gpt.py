import math
from typing import Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

from .attn import Attention
from .mlp import MLP, MoE
from .norm import LayerNorm

"""
Features:
    1. MoE flexible top_k, from sparse to dense
    2. few first blocks are Dense, the rest are MoE
"""

@dataclass
class GPTConfig:
    block_size: int = 4096
    vocab_size: int = 50304
    max_vocab_size: int = 50257
    num_layer: int = 12
    num_head: int = 12
    hidden_size: int = 768
    intermediate_size: int = 768 * 4
    dropout: float = 0.0

    # MoE
    use_moe_ratio: float = 1.0
    num_expert: int = 128
    top_k: Union[int, list] = 8  # could be a range from sparse to dense
    moe_intermediate_size: int = 768

class Block(nn.Module):

    def __init__(self, config: GPTConfig, use_moe: bool = True):
        super().__init__()
        self.use_moe = use_moe
        self.ln_1 = LayerNorm(config)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MoE(config) if use_moe else MLP(config)

    def forward(self, x: torch.Tensor, top_k: int = None):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# GPT-like Model

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.pos = None
        use_moe_list = [True if i > 1 - config.use_moe_ratio * config.num_layer else False for i in range(config.num_layer)]
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config, use_moe) for use_moe in use_moe_list]),
            ln_f = LayerNorm(config),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        x = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss