import torch
import torch.nn as nn

from .config import GPTConfig
from .modules.norm import LayerNorm
from .gpt import Block, GPT

class LoopedAttnBlock(Block):
    def __init__(self, config: GPTConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn_loop_count = config.attn_loop_count

    def forward(self, x: torch.Tensor):
        # # Loop the MLP multiple times to simulate deeper computation
        for _ in range(self.attn_loop_count):
            x = x + self.attn(self.ln_1(x))
        mlp_out = self.mlp(self.ln_2(x))
        x = x + mlp_out[0] if self.use_moe else x + mlp_out
        return x

class AttnLoopedGPT(GPT, nn.Module):
    def __init__(self, config: GPTConfig):
        nn.Module.__init__(self)
        self.config = config
        self.pos = None
        use_moe_list = [True if i >= (1 - config.use_moe_ratio) * config.num_layer else False for i in range(config.num_layer)]
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([LoopedAttnBlock(config, use_moe) for use_moe in use_moe_list])
        self.lnf = LayerNorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tied_lm_head:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)