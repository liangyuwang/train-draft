from .config import GPTConfig
from .gpt import GPT, Block
from .shared_gpt import SharedGPT
from .looped_gpt import LoopedGPT
from .mlp_looped_gpt import MLPLoopedGPT

from .modules.attn import Attention
from .modules.mlp import MLP, MoE
from .modules.norm import LayerNorm

def gpt(config: GPTConfig):
    if config.use_shared_layers:
        if config.shared_layers is not None:
            assert len(config.shared_layers) == config.num_layer, "shared_layers must be provided and have length equal to num_layer when use_shared_layers=True"
        return SharedGPT(config)
    elif config.use_looped_layers:
        if config.looped_layers_range is not None:
            assert len(config.looped_layers_range) == 3, "looped_layers_range [start, end, step] must be provided when use_looped_layers=True"
        return LoopedGPT(config)
    elif config.use_mlp_looped:
        return MLPLoopedGPT(config)
    else:
        return GPT(config)