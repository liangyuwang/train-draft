def get_dense_model_params(
    num_layer: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
):
    """
    Compute parameter counts for a dense Transformer model.
    Args:
        num_layer (int): number of transformer layers (L)
        hidden_size (int): hidden dimension (H)
        intermediate_size (int): feed-forward intermediate size (I)
        vocab_size (int): vocabulary size (V)
    Returns:
        dict: total parameter counts in billions
    """
    H = hidden_size
    L = num_layer
    I = intermediate_size
    V = vocab_size

    # Embedding + tied output head
    P_embed = V * H

    # Dense transformer layer parameters:
    # Attention: ~4*H^2 (QKV + out projection)
    # FFN: ~3*H*I (three linear layers: H->I and I->H)
    P_dense_layer = 4 * H * H + 3 * H * I
    P_dense_all = L * P_dense_layer

    # Total parameters
    P_total = P_embed + P_dense_all

    return {
        "total_params_B": P_total / 1e9,
        "dense_params_B": P_total / 1e9,
    }

def get_moe_model_params(
    num_layer: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
    num_expert: int,
    top_k: int,
    moe_intermediate_size: int,
):
    """
    Compute parameter counts for a Transformer model with MoE layers.
    
    Args:
        num_layer (int): number of transformer layers (L)
        hidden_size (int): hidden dimension (H)
        intermediate_size (int): feed-forward intermediate size (I)
        vocab_size (int): vocabulary size (V)
        num_expert (int): number of experts per MoE layer (E)
        top_k (int): number of experts activated per token (k)
        moe_intermediate_size (int): intermediate size for MoE experts (I_moe)
    
    Returns:
        dict: total and active parameter counts in billions
    """

    H = hidden_size
    L = num_layer
    I = intermediate_size
    E = num_expert
    k = top_k
    I_moe = moe_intermediate_size
    V = vocab_size

    # Embedding + tied output head
    P_embed = V * H

    # Dense transformer layer parameters:
    # Attention: ~4*H^2 (QKV + out projection)
    # FFN: ~2*H*I (two linear layers: H->I and I->H)
    P_dense_layer = 4 * H * H + 2 * H * I
    P_dense_all = L * P_dense_layer

    # Router parameters per layer: H * E
    P_router_layer = H * E
    P_router_all = L * P_router_layer

    # MoE experts:
    # Each expert: 3 * H * I_moe  (gate_proj + up_proj + down_proj)
    P_expert = 3 * H * I_moe
    P_moe_all = L * E * P_expert

    # Total parameters
    P_total = P_embed + P_dense_all + P_router_all + P_moe_all

    # Active MoE parameters per forward:
    # k experts activated per layer
    P_moe_active = L * k * P_expert

    # Active total
    P_active = P_embed + P_dense_all + P_router_all + P_moe_active

    return {
        "total_params_B": P_total / 1e9,
        "active_params_B": P_active / 1e9,
        "dense_params_B": (P_embed + P_dense_all + P_router_all) / 1e9,
        "moe_total_B": P_moe_all / 1e9,
        "moe_active_B": P_moe_active / 1e9,
    }