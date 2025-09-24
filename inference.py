import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.checkpoint import state_dict_loader
from torch.distributed.checkpoint.filesystem import FileSystemReader

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    """
    for _ in range(max_new_tokens):
        # truncate to block size
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

        # forward
        logits, _ = model(idx_cond)

        # only keep last time step
        logits = logits[:, -1, :] / temperature

        # top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # softmax -> probs
        probs = F.softmax(logits, dim=-1)

        # sample
        idx_next = torch.multinomial(probs, num_samples=1)

        # append
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def load_model(ckpt_prefix, model):
    state_dict_loader.load(
        state_dict=model.state_dict(),
        storage_reader=FileSystemReader(f"{ckpt_prefix}_model.pt"),
    )
    return model


def main():
    import argparse
    from transformers import AutoTokenizer, set_seed
    from .model import GPTConfig, gpt

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="tokenizer name or path")
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint prefix (without _model.pt)")
    parser.add_argument("--prompt", type=str, required=True, help="input text prompt")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    # Model hyperparameters
    parser.add_argument("--block_size", type=int, default=4096, help="Context length")
    parser.add_argument("--vocab_size", type=int, default=50304, help="Vocabulary size")
    parser.add_argument("--max_vocab_size", type=int, default=50257, help="Maximum vocabulary size")
    parser.add_argument("--num_layer", type=int, default=32, help="Number of transformer layers")
    parser.add_argument("--num_attention_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--num_key_value_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size of the model")
    parser.add_argument("--intermediate_size", type=int, default=4096, help="Intermediate size of the model")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--tied_lm_head", action="store_true", help="Tie the weights of the LM head and embedding layer")
    parser.add_argument("--use_moe_ratio", type=float, default=1.0, help="Ratio of layers using MoE")
    parser.add_argument("--num_experts", type=int, default=128, help="Number of experts in MoE")
    parser.add_argument("--num_experts_per_tok", type=int, default=8, help="Top-k experts to use in MoE")
    parser.add_argument("--moe_intermediate_size", type=int, default=256, help="Intermediate size for MoE layers")
    parser.add_argument("--use_shared_layers", action="store_true", help="Use shared MoE model")
    parser.add_argument("--shared_layers", type=int, nargs='+', default=None, help="Indices of layers to share in shared MoE model")
    parser.add_argument("--use_looped_layers", action="store_true", help="Use looped MoE model")
    parser.add_argument("--looped_layers_range", type=int, nargs='+', default=None, help="Range [start, end, step] for looped layers")
    parser.add_argument("--looped_layers_repeats", type=int, default=1, help="Number of repeats for looped layers")
    args = parser.parse_args()

    set_seed(args.seed)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        raise RuntimeError("Distributed package is not initialized")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model_config = GPTConfig()
    for k, v in vars(args).items():
        if hasattr(model_config, k):
            setattr(model_config, k, v)
    model = gpt(model_config)
    model = load_model(args.ckpt, model)
    model.to(f"{args.device}:{rank}")
    model.eval()

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)

    out = generate(
        model,
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    result = tokenizer.decode(out[0].tolist())

    if rank == 0:
        print("=" * 40)
        print("Prompt:", args.prompt)
        print("Generated:", result)
        print("=" * 40)


if __name__ == "__main__":
    main()
