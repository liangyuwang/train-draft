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
        mlp_out = self.mlp(self.ln_2(x))
        x = x + mlp_out[0] if self.use_moe else x + mlp_out
        return x

# GPT-like Model

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.pos = None
        use_moe_list = [True if i > 1 - config.use_moe_ratio * config.num_layer else False for i in range(config.num_layer)]
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config, use_moe) for use_moe in use_moe_list])
        self.lnf = LayerNorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tied_lm_head:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            std = 0.013 # same as openllama, bloom suggests sqrt(2/(NHIDDEN*5)) = 0.0098 or sqrt(2/(NHIDDEN*3)) = 0.009
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        x = self.wte(idx) # token embeddings of shape (B, T, n_embd)
        for block in self.blocks:
            x = block(x)
        x = self.lnf(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# Looped GPT

class LoopedGPT(GPT):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.pos = None
        self.shared_layers = config.shared_layers

        # Default: each layer has its own group (no sharing)
        if self.shared_layers is None:
            self.shared_layers = list(range(config.num_layer))

        # Determine which layers use MoE
        use_moe_list = [
            True if i >= config.num_layer - int(config.use_moe_ratio * config.num_layer) else False
            for i in range(config.num_layer)
        ]

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)

        group2block = {}
        self.blocks = nn.ModuleList()

        for i, gid in enumerate(self.shared_layers):
            use_moe = use_moe_list[i]

            if gid not in group2block:
                # First block in this group: create normally
                block = Block(config, use_moe)
                group2block[gid] = block
                self.blocks.append(block)
            else:
                # Reuse group leader block
                ref_block = group2block[gid]

                # Ensure group consistency: all must be dense or all must be MoE
                if ref_block.use_moe != use_moe:
                    raise ValueError(
                        f"Group {gid} has inconsistent block types: both MoE and dense are present."
                    )

                if use_moe:
                    # Create a new MoE block, then share all params except 'moe_gate'
                    new_block = Block(config, use_moe=True)
                    self._share_params(ref_block, new_block, exclude_suffix=["moe_gate"])
                    self.blocks.append(new_block)
                else:
                    # Dense: just reuse the same block instance
                    self.blocks.append(ref_block)

        self.lnf = LayerNorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tied_lm_head:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _share_params(self, src_block: nn.Module, dst_block: nn.Module, exclude_suffix=None):
        """
        Share parameters between two blocks, except those whose names end with excluded suffixes.
        This method replaces parameters in dst_block with references to src_block's parameters.
        """
        if exclude_suffix is None:
            exclude_suffix = []

        src_params = dict(src_block.named_parameters())

        for name, param in dst_block.named_parameters():
            if any(name.endswith(suffix) for suffix in exclude_suffix):
                continue  # skip excluded params (e.g., 'moe_gate')
            if name not in src_params:
                raise KeyError(f"Parameter {name} not found in source block.")

            shared_param = src_params[name]
            # navigate into submodules to replace parameter
            module_names = name.split(".")
            mod = dst_block
            for sub_name in module_names[:-1]:
                mod = getattr(mod, sub_name)
            setattr(mod, module_names[-1], shared_param)