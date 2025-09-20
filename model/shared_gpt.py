import torch
import torch.nn as nn

from .config import GPTConfig
from .modules.norm import LayerNorm
from .gpt import Block, GPT

class SharedGPT(GPT, nn.Module):

    def __init__(self, config: GPTConfig):
        """
        SharedGPT / SharedMoE

        This model extends a standard GPT by introducing flexible parameter sharing across
        layers. Instead of allocating a unique set of parameters for every transformer block,
        layers can be grouped into "sharing groups", where all blocks in the same group share
        some or all of their parameters.

        - For dense blocks:
        All parameters within the group are shared. In practice, all layers in the group
        point to the same Block instance, so parameter memory is identical to a single layer,
        while computation is still performed multiple times.

        - For MoE (Mixture-of-Experts) blocks:
        All parameters are shared across the group, except those ending with "moe_gate".
        Each layer in the group keeps its own gating parameters but shares the experts and
        the rest of the architecture with its group leader. This allows diverse routing while
        saving memory on the large shared components.

        Key behavior:
        - The model still has `num_layer` entries in self.blocks, so forward() runs exactly as
        in a standard GPT.
        - The difference is purely in parameter tying: some blocks share their weights, others
        remain independent.
        - The grouping is specified by `shared_layers`, a list of integers of length num_layer,
        where shared_layers[i] indicates the "group id" (or leader) for layer i.

        Effect:
        - Memory: fewer unique parameter sets are stored, reducing model size.
        - Compute: every layer is still executed, so FLOPs remain unchanged.
        """
        nn.Module.__init__(self)
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