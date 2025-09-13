import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import GPTConfig
from .attn import Attention
from .mlp import MLP, MoE
from .norm import LayerNorm

"""
Features:
    1. MoE flexible top_k, from sparse to dense
    2. few first blocks are Dense, the rest are MoE
"""

class Block(nn.Module):

    def __init__(self, config: GPTConfig, use_moe: bool = True, top_k: int = None):
        super().__init__()
        self.use_moe = use_moe
        self.ln_1 = LayerNorm(config)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MoE(config, top_k) if use_moe else MLP(config)

    def forward(self, x: torch.Tensor):
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
            wte = nn.Embedding(config.vocab_size, config.hidden_size),
            h = nn.ModuleList([Block(config, use_moe) for use_moe in use_moe_list]),
            ln_f = LayerNorm(config),
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tied_lm_head:
            self.lm_head.weight = self.transformer.wte.weight

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