import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig

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

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
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

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """ MoE forward with async execution version """
    #     B, N, d = x.shape
    #     x = x.view(-1, d)
    #     router_logits = self.gate(x)

    #     routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    #     routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    #     routing_weights = routing_weights.to(x.dtype)
    #     final_x = torch.zeros((B * N, d), dtype=x.dtype, device=x.device)

    #     expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    #     expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    #     streams, buffers, indices = [], [], []
    #     for expert_idx in expert_hit:
    #         s = torch.cuda.Stream()
    #         streams.append(s)
    #         with torch.cuda.stream(s):
    #             idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
    #             current_state = x[None, top_x].reshape(-1, d)
    #             current_x = self.experts[expert_idx](current_state) * routing_weights[top_x, idx, None]
    #             buffers.append(current_x.to(x.dtype))
    #             indices.append(top_x)
    #     for stream, buf, top_x in zip(streams, buffers, indices):
    #         stream.synchronize()
    #         final_x.index_add_(0, top_x, buf)
    #     return final_x.reshape(B, N, d), router_logits

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """ MoE forward with Grouped GEMM version """
    #     B, N, d = x.shape
    #     x = x.view(-1, d)
    #     # router_logits: (batch * N, n_experts)
    #     router_logits = self.gate(x)

    #     routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    #     routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    #     # we cast back to the input dtype
    #     routing_weights = routing_weights.to(x.dtype)

    #     final_x = torch.zeros((B * N, d), dtype=x.dtype, device=x.device)

    #     # expert assignments
    #     eid = selected_experts.reshape(-1)                 # [N*K]
    #     pflat = routing_weights.reshape(-1)                # [N*K]
    #     tid = torch.arange(B*N, device=x.device).repeat_interleave(self.top_k)

    #     # order
    #     order = torch.argsort(eid)
    #     eid, tid, pflat = eid[order], tid[order], pflat[order]
    #     counts = torch.bincount(eid, minlength=self.num_experts)
    #     cumsum = counts.cumsum(0)
    #     starts = torch.cat([torch.zeros(1, device=x.device, dtype=cumsum.dtype), cumsum[:-1]])
    #     x_list = [x[tid[s:e]] for s, e in zip(starts.tolist(), cumsum.tolist())]

    #     # apply group gemm for for-loop
    #     y_list = moe_group_experts_forward(x_list, self.experts)
    #     final_x.index_add_(0, tid, torch.cat(y_list, 0) * pflat.unsqueeze(-1))
    #     final_x = final_x.reshape(B, N, d)
    #     return final_x, router_logits


def moe_group_experts_forward(x_list, experts):
    w_gate = [expert.gate_proj.weight for expert in experts]
    w_up   = [expert.up_proj.weight   for expert in experts]
    w_down = [expert.down_proj.weight for expert in experts]

    g_list = group_gemm(x_list, w_gate)
    u_list = group_gemm(x_list, w_up)
    mid    = [experts[i].act_fn(g) * u for i, (g,u) in enumerate(zip(g_list,u_list))]
    y_list = group_gemm(mid, w_down)
    return y_list


def group_gemm(a_list, b_list, trans_b: bool = True):
    """
    Grouped GEMM over lists:
      - a_list: list of [n_i, in_features] tensors (tokens routed to expert i)
      - b_list: list of [out_features, in_features] weight tensors (expert i)
    Returns:
      - outs: list of [n_i, out_features] tensors, same list order as inputs
    Note:
      - Default computes a @ b.T (trans_b=True), matching torch.nn.Linear(weight=[out,in]).
      - Handles empty experts (n_i == 0).
    """
    assert len(a_list) == len(b_list)
    outs = []
    #TODO: use deepgemm for grouped gemm
    for a, b in zip(a_list, b_list):
        if a.numel() == 0:
            # produce an empty [0, out_features] tensor on the same device/dtype as a
            out_features = b.shape[0] if trans_b else b.shape[1]
            outs.append(a.new_zeros((0, out_features)))
        else:
            outs.append(a @ (b.transpose(-1, -2) if trans_b else b))
    return outs