import torch
import torch.nn as nn

from . import GPTConfig

"""
Features:
    1. DeepGemm for FP8 Grouped GEMM
"""

class MLP(nn.Module):
    """Dense MLP or Single expert in MoE"""
    def __init__(self, config: GPTConfig, use_moe: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size if use_moe else config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))