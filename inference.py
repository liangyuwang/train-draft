import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.checkpoint import state_dict_loader
from torch.distributed.checkpoint.filesystem import FileSystemReader

from utils import get_compiled_to_uncompiled_mapping

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
        logits, _ = model(idx_cond, idx_cond)

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
    """
    Loads model weights and dynamically handles the '_orig_mod.' prefix
    introduced by torch.compile.
    """
    storage_reader = FileSystemReader(f"{ckpt_prefix}_model.pt")

    # 1. Read the checkpoint's metadata to get all key names
    metadata = storage_reader.read_metadata()
    compiled_keys = metadata.state_dict_metadata.keys()

    # 2. Create the mapping from compiled keys to the model's parameters
    print("Creating a key mapping for loading compiled checkpoint into uncompiled model...")
    custom_state_dict = get_compiled_to_uncompiled_mapping(model, compiled_keys)

    # 3. Use this custom state_dict for loading.
    #    The state_dict_loader will iterate through the keys of custom_state_dict (the compiled keys),
    #    read the corresponding data from the checkpoint file, and load it
    #    directly into the associated value (the model's actual parameter tensor).
    print("Loading state dict with custom key mapping...")
    state_dict_loader.load(
        state_dict=custom_state_dict,
        storage_reader=storage_reader,
    )
    print("Model loaded successfully.")
    
    return model

@torch.no_grad()
def test(args, tokenizer, model, prompt, rank):
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=args.device)

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
        print(f"Prompt: {prompt} ...")
        print("Generated:", result)
        print("=" * 40)


test_prompts = [
    # --- Story & Creative Writing Starters ---
    "Once upon a time, in a kingdom far, far away, there lived a",
    "The year is 2099. The megacity of Neo-Veridia is powered by glowing crystals, but tonight, the largest crystal began to",
    "The recipe for the perfect chocolate chip cookie is as follows: First, preheat your oven to 375°F (190°C). Then, in a large bowl, cream together",
    "It was a dark and stormy night. The rain fell in torrents, and the wind howled like a banshee. Suddenly, a knock came at the door",
    "My favorite memory from childhood is the summer I spent at my grandparents' house. Every morning, we would",

    # --- Factual Knowledge & Explanation Starters ---
    "The planet Mars is the fourth planet from the Sun. It is often called the 'Red Planet' because",
    "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy. This process starts with",
    "The three main branches of the United States government are the Legislative, the Executive, and the",
    "A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data. It works by",
    "The capital of Japan is Tokyo. The capital of France is Paris. The capital of Brazil is",

    # --- List & Pattern Completion ---
    "Here is a list of common kitchen utensils:\n- Spoon\n- Fork\n- Knife\n-",
    "The top 5 most populated countries in the world are:\n1. India\n2. China\n3. United States\n4.",
    "English: Hello\nSpanish: Hola\nEnglish: Goodbye\nSpanish: Adiós\nEnglish: Thank you\nSpanish:",
    "Q: What is the capital of Canada? A: Ottawa.\nQ: What is 2 + 2? A: 4.\nQ: What color is a banana? A:",
    
    # --- Code Completion Starters ---
    "import pandas as pd\n\ndef load_and_clean_csv(file_path):\n    \"\"\"This function loads a CSV file and removes duplicate rows.\"\"\"\n    df = pd.read_csv(",
    "def factorial(n):\n    \"\"\"Calculates the factorial of a non-negative integer n.\"\"\"\n    if n == 0:\n        return 1\n    else:\n        return n *",
    "/* A simple 'Hello, World!' program in the C programming language */\n#include <stdio.h>\n\nint main() {\n   printf(\"Hello, World!\");\n   return",
    "SELECT customer_name, order_date, total_amount\nFROM customers\nINNER JOIN orders ON customers.customer_id = "
]

def main():
    import os
    import argparse
    from transformers import AutoTokenizer, set_seed
    from model import GPTConfig, gpt

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="tokenizer name or path")
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint prefix (without _model.pt)")
    parser.add_argument("--prompt", type=str, default=None, help="input text prompt")
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

    dist.init_process_group(backend='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model_config = GPTConfig()
    for k, v in vars(args).items():
        if hasattr(model_config, k):
            setattr(model_config, k, v)
    model = gpt(model_config)
    model.to(f"{args.device}:{rank}")
    model = load_model(args.ckpt, model)
    model.eval()

    if args.prompt is None:
        for prompt in test_prompts:
            test(args, tokenizer, model, prompt, rank)
    else:
        test(args, tokenizer, model, args.prompt, rank)


if __name__ == "__main__":
    main()