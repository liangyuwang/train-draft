import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPTConfig

"""
Features:
    0.1. Async MoE forward
    1. DeepGemm for FP8 Grouped GEMM
"""

class MLP(nn.Module):
    """Dense MLP or Single expert in MoE"""
    def __init__(self, config: GPTConfig, use_moe: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size if use_moe else config.intermediate_size
        self.use_moe = use_moe

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, config: GPTConfig, top_k: int = None):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = top_k if top_k is not None else config.num_experts_per_tok
        self.hidden_size = config.hidden_size

        self.moe_gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(config, use_moe=True) for _ in range(self.num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ copied from https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py """
        B, N, d = x.shape
        x = x.view(-1, d)
        # router_logits: (batch * N, n_experts)
        router_logits = self.moe_gate(x)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(x.dtype)
        final_x = torch.zeros((B * N, d), dtype=x.dtype, device=x.device)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = x[None, top_x].reshape(-1, d)
            current_x = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_x.index_add_(0, top_x, current_x.to(x.dtype))
        final_x = final_x.reshape(B, N, d)
        return final_x, router_logits