import torch
import torch.nn as nn

from . import GPTConfig, Block, SharedGPT

class LoopedGPT(SharedGPT):

    def __init__(self, config: GPTConfig):
        """
        LoopedGPT / LoopedMoE

        This model generalizes SharedGPT by introducing looped layer ranges. Instead of manually
        specifying group assignments, a contiguous range of layers can be "looped": their
        parameters are divided into groups and reused repeatedly across the range.

        Configuration:
        - `looped_layers_range`: [start, end] or [start, end, step]
            Defines which part of the model is subject to looping. If a step is provided,
            the range is split into multiple independent segments of length `step`.
        - `looped_layers_repeats`: int
            Defines how many times the parameter set is repeated within each segment.
            For a segment of length L, L must be divisible by repeats. Each "lane" of size
            L/repeats is tied together.

        Examples:
        - num_layer=24, looped_layers_range=[12,24], looped_layers_repeats=12:
            The last 12 layers all share the same parameters (equivalent to SharedGPT where
            layers 12..23 map to one leader).
        - num_layer=24, looped_layers_range=[12,24], looped_layers_repeats=3:
            The last 12 layers are split into 3 groups of 4 layers. Each group is shared
            internally, producing a repeating structure.
        - num_layer=24, looped_layers_range=[0,24,6], looped_layers_repeats=3:
            The full depth is divided into 4 segments of length 6. Within each segment,
            parameters are tied into 3-way loops, independent from other segments.

        Key behavior:
        - Forward execution is unchanged: self.blocks still contains num_layer entries.
        - Only the parameter references differ: looped layers point back to their "leaders".
        - Dense blocks: entire block is reused across the loop.
        - MoE blocks: only "moe_gate" parameters are unique; all other weights are shared
        across repetitions.

        Effect:
        - Creates architectures where deep stacks are built from a smaller set of parameters,
        enabling efficient depth scaling.
        - When repeats = segment length, this reduces to "repeat a single layer many times",
        a common design in looped transformer research.
        """
        self.looped_layers_range = config.looped_layers_range
        self.looped_layers_repeats = config.looped_layers_repeats
        config.shared_layers = self._make_shared_layers(config)
        # Let SharedGPT handle block creation and param sharing with its _share_params
        super().__init__(config)

    def _make_shared_layers(self, config: GPTConfig):
        """
        Translate loop settings into a shared_layers list.
        Semantics:
          - Range [start, end): only layers in this half-open interval are looped.
          - If a third element 'step' is provided, split [start, end) into segments
            [s, min(s+step, end)) and apply the loop rule independently per segment.
          - In each segment, we partition layers into 'repeats' lanes and share across lanes.
            Formally, for segment length L, require L % repeats == 0.
            Let group_size = L // repeats. For each offset i in [0, group_size):
              indices {base+i + r*group_size | r in [0..repeats-1]} share one leader (base+i).
        """
        num_layer = int(config.num_layer)
        shared_layers = list(range(num_layer))

        # Parse loop range
        if self.looped_layers_range is None:
            start, end, step = 0, num_layer, num_layer  # one big segment
        else:
            rng = self.looped_layers_range
            if len(rng) == 2:
                start, end = int(rng[0]), int(rng[1])
                step = max(1, end - start)  # single segment covering the whole range
            elif len(rng) == 3:
                start, end, step = int(rng[0]), int(rng[1]), int(rng[2])
            else:
                raise ValueError("looped_layers_range must be [start,end] or [start,end,step]")

        # Basic validation
        if not (0 <= start <= end <= num_layer):
            raise ValueError(f"Invalid loop range [{start}, {end}) for num_layer={num_layer}.")
        if step <= 0:
            raise ValueError(f"'step' must be positive, got {step}.")
        if self.looped_layers_repeats <= 0:
            raise ValueError(f"'repeats' must be positive, got {self.looped_layers_repeats}.")

        repeats = self.looped_layers_repeats

        # Walk over segments [seg_start, seg_end)
        seg_start = start
        while seg_start < end:
            seg_end = min(seg_start + step, end)
            seg_len = seg_end - seg_start
            if seg_len == 0:
                seg_start = seg_end
                continue

            # Each segment must be divisible by repeats
            if seg_len % repeats != 0:
                raise AssertionError(
                    f"Segment [{seg_start}, {seg_end}) length {seg_len} must be divisible by repeats={repeats}."
                )
            group_size = seg_len // repeats  # number of distinct leaders inside this segment

            # For each offset i in the segment "lane", bind all repeats to the same leader
            base = seg_start
            for i in range(group_size):
                leader = base + i
                for r in range(repeats):
                    idx = base + i + r * group_size
                    # Map every follower in the lane to the lane leader
                    shared_layers[idx] = leader

            seg_start = seg_end

        return shared_layers