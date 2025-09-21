from .config import GPTConfig
from .gpt import GPT, Block
from .shared_gpt import SharedGPT
from .looped_gpt import LoopedGPT

from .modules.attn import Attention
from .modules.mlp import MLP, MoE
from .modules.norm import LayerNorm

def gpt(config: GPTConfig):
    if config.use_shared_layers:
        return SharedGPT(config)
    elif config.use_looped_layers:
        if config.looped_layers_range is not None:
            assert len(config.looped_layers_range) == 3, "looped_layers_range [start, end, step] must be provided when use_looped_layers is True"
        return LoopedGPT(config)
    else:
        return GPT(config)