from .config import GPTConfig
from .gpt import GPT, Block
from .shared_gpt import SharedGPT
from .looped_gpt import LoopedGPT

from .modules.attn import Attention
from .modules.mlp import MLP, MoE
from .modules.norm import LayerNorm

def gpt(config: GPTConfig):
    if config.use_shared_layers:
        assert config.shared_layers is not None and len(config.shared_layers) > 0, "shared_layers must be provided when use_shared_layers is True"
        return SharedGPT(config, config.shared_layers)
    elif config.use_looped_layers:
        assert config.looped_layers_range is not None and len(config.looped_layers_range) == 3, "looped_layers_range [start, end, step] must be provided when use_looped_layers is True"
        return LoopedGPT(config, config.looped_layers_range, config.looped_layers_repeats)
    else:
        return GPT(config)