import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig

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
        self.use_moe = use_moe

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def forward_group(self, x_list, experts):
        w_gate = [expert.gate_proj.weight for expert in experts]
        w_up   = [expert.up_proj.weight   for expert in experts]
        w_down = [expert.down_proj.weight for expert in experts]

        g_list = group_gemm(x_list, w_gate)
        u_list = group_gemm(x_list, w_up)
        mid    = [experts[i].act_fn(g) * u for i, (g,u) in enumerate(zip(g_list,u_list))]
        y_list = group_gemm(mid, w_down)
        return y_list


class MoE(nn.Module):
    def __init__(self, config: GPTConfig, top_k: int = None):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = top_k if top_k is not None else config.num_experts_per_tok
        self.hidden_size = config.hidden_size

        # gating network
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        # experts: each is an MLP with use_moe=True (small intermediate size)
        self.experts = nn.ModuleList([MLP(config, use_moe=True) for _ in range(self.num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ copied from https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py """
        B, N, d = x.shape
        x = x.view(-1, d)
        # router_logits: (batch * N, n_experts)
        router_logits = self.gate(x)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ MoE forward with Grouped GEMM version """
        B, N, d = x.shape
        x = x.view(-1, d)
        # router_logits: (batch * N, n_experts)
        router_logits = self.gate(x)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(x.dtype)

        final_x = torch.zeros((B * N, d), dtype=x.dtype, device=x.device)

        # expert assignments
        eid = selected_experts.reshape(-1)                 # [N*K]
        pflat = routing_weights.reshape(-1)                # [N*K]
        tid = torch.arange(B*N, device=x.device).repeat_interleave(self.top_k)

        # order
        order = torch.argsort(eid)
        eid, tid, pflat = eid[order], tid[order], pflat[order]
        counts = torch.bincount(eid, minlength=self.num_experts)
        cumsum = counts.cumsum(0)
        starts = torch.cat([torch.zeros(1, device=x.device, dtype=cumsum.dtype), cumsum[:-1]])
        x_list = [x[tid[s:e]] for s, e in zip(starts.tolist(), cumsum.tolist())]

        # apply for-loop
        y_list = self.experts[0].forward_group(x_list, self.experts)
        final_x.index_add_(0, tid, torch.cat(y_list, 0) * pflat.unsqueeze(-1))
        final_x = final_x.reshape(B, N, d)
        return final_x, router_logits


def group_gemm(x_list, w_list):
    """ Grouped GEMM with DeepGemm """
    import deepgemm     #TODO
    y_list = []
    for x, w in zip(x_list, w_list):
        if x.numel() == 0:
            y_list.append(torch.zeros((0, w.size(0)), dtype=x.dtype, device=x.device))
        else:
            y = deepgemm.linear(x, w.t())
            y_list.append(y)
    return y_list