from typing import Union
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 4096
    vocab_size: int = 50304
    max_vocab_size: int = 50257
    num_layer: int = 32
    num_attention_heads: int = 128
    num_key_value_heads: int = 8
    hidden_size: int = 1024
    intermediate_size: int = 4096
    dropout: float = 0.0
    tied_lm_head: bool = True

    # MoE
    use_moe_ratio: float = 1.0  # ratio of layers using MoE
    num_experts: int = 128
    num_experts_per_tok: Union[int, list] = 8  # could be a range from sparse to dense
    moe_intermediate_size: int = 256

    # Shared GPT
    use_shared_layers: bool = False
    shared_layers: Union[list] = None  # None means all layers are shared

    # Looped GPT
    use_looped_layers: bool = False
    looped_layers_range: Union[list] = None  # [start, end, step], None means all layers are looped
    looped_layers_repeats: int = 1

    def __post_init__(self):
        if self.use_shared_layers and self.shared_layers is None:
            self.shared_layers = list(range(self.num_layer))
        if self.use_looped_layers and self.looped_layers_range is None:
            self.looped_layers_range = [0, self.num_layer, 1]
            self.looped_layers_repeats = self.num_layer