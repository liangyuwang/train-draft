import torch
import torch.nn as nn

from .config import GPTConfig
from .modules.norm import LayerNorm
from .gpt import MLP, Block, GPT

class LoopedMLPBlock(Block):
    def __init__(self, config: GPTConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.mlp_loop_count = config.mlp_loop_count

    def forward(self, x: torch.Tensor):
        # # Loop the MLP multiple times to simulate deeper computation
        x = x + self.attn(self.ln_1(x))
        for _ in range(self.mlp_loop_count):
            mlp_out = self.mlp(self.ln_2(x))
            x = x + mlp_out[0] if self.use_moe else x + mlp_out
        return x

class MLPLoopedGPT(GPT, nn.Module):
    def __init__(self, config: GPTConfig):
        nn.Module.__init__(self)
        self.config = config
        self.pos = None
        use_moe_list = [True if i >= (1 - config.use_moe_ratio) * config.num_layer else False for i in range(config.num_layer)]
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([LoopedMLPBlock(config, use_moe) for use_moe in use_moe_list])
        self.lnf = LayerNorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tied_lm_head:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)