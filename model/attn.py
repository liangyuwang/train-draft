import torch
import torch.nn as nn
import torch.nn.functional as F

from . import GPTConfig

"""
Features:
    1. FlashMLA for long context
    2. Slide Window
    3. MTP
"""

def rope_impl(q, k, position_ids, rope_theta=10000.0):
    """
    Shared RoPE (Rotary Positional Embedding) implementation for Qwen3 family models.
    Supports both single and batched position_ids
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Create frequency tensor
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=q.device) / head_dim))
    
    # Handle position_ids - support both single batch and multi-batch scenarios
    if position_ids.dim() > 1 and position_ids.shape[0] > 1:
        # Multi-batch case: handle each batch separately
        freqs = []
        for i in range(batch_size):
            if i < position_ids.shape[0]:
                batch_freqs = torch.outer(position_ids[i].float(), inv_freq)
            else:
                # Use first batch if not enough position_ids
                batch_freqs = torch.outer(position_ids[0].float(), inv_freq)
            freqs.append(batch_freqs)
        freqs = torch.stack(freqs, dim=0)  # [batch_size, seq_len, head_dim//2]
        
        # Create cos and sin - repeat to match full head_dim
        cos = torch.cos(freqs).unsqueeze(1).repeat(1, 1, 1, 2)  # [batch_size, 1, seq_len, head_dim]
        sin = torch.sin(freqs).unsqueeze(1).repeat(1, 1, 1, 2)  # [batch_size, 1, seq_len, head_dim]
    else:
        # Single batch case - take the first sequence if batched
        if position_ids.dim() > 1:
            t = position_ids[0].float()  # [seq_len] - use first batch
        else:
            t = position_ids.float()     # [seq_len]
        
        # Create position encodings
        freqs = torch.outer(t, inv_freq)    # [seq_len, head_dim//2]
        
        # Duplicate to create full cos/sin tensors
        cos = freqs.cos()  # [seq_len, head_dim//2]
        sin = freqs.sin()  # [seq_len, head_dim//2]
        
        # Expand to full head_dim by repeating each element
        cos = torch.stack([cos, cos], dim=-1).flatten(-2)  # [seq_len, head_dim]
        sin = torch.stack([sin, sin], dim=-1).flatten(-2)  # [seq_len, head_dim]
        
        # Reshape to match q and k dimensions: [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

class Attention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.hidden_size % config.num_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # regularization
        self.n_head = config.num_head
        self.n_embd = config.hidden_size
        self.pos = None

    def forward(self, x: torch.Tensor):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        v = v.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        if self.pos is None:
            self.pos = torch.arange(T, device=x.device).unsqueeze(0)
        q, k = rope_impl(q, k, self.pos)
        y = F.scaled_dot_product_attention(q, k, v, dropout=self.training, dropout_p=False)
        y = y.view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y