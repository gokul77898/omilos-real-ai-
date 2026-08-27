"""Grouped Query Attention (GQA) with RoPE and Causal Masking."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig
from src.rope import RotaryEmbedding, apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key or value heads n_rep times to match query head count.

    Input shape: [batch, num_kv_heads, seq_len, head_dim]
    Output shape: [batch, num_attention_heads, seq_len, head_dim]
    """
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) with Rotary Position Embeddings and Causal Masking.

    Args:
        config: ModelConfig containing hidden_size, num_attention_heads, num_kv_heads, etc.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.max_seq_len = config.max_seq_len

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")

        # Independent linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Rotary Positional Embedding
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_seq_len,
            base=config.rope_theta,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights using scaled Gaussian distribution."""
        std = 0.02
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std / math.sqrt(2.0))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for Grouped Query Attention.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask tensor.

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        # 1. Projections
        q = self.q_proj(x)  # [B, T, H_q * D]
        k = self.k_proj(x)  # [B, T, H_kv * D]
        v = self.v_proj(x)  # [B, T, H_kv * D]

        # 2. Reshape into heads: [B, H, T, D]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE to Q and K
        cos, sin = self.rotary_emb(x, seq_len)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # 4. Repeat KV heads for GQA
        k = repeat_kv(k, self.num_kv_groups)  # [B, H_q, T, D]
        v = repeat_kv(v, self.num_kv_groups)  # [B, H_q, T, D]

        # 5. Scaled Dot-Product Attention with Causal Masking
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=(attention_mask is None),
        )

        # 6. Transpose and reshape back to [B, T, H]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # 7. Output projection
        return self.o_proj(attn_output)
