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
        self.attention_window = min(config.attention_window, config.max_seq_len)
        self.attention_chunk_size = config.attention_chunk_size

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
            scaling_type=config.rope_scaling_type,
            scaling_factor=config.rope_scaling_factor,
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

        # 5. Bounded local causal attention.  We build only [query_chunk, window]
        # masks, never a dense [T, T] mask.  A supplied padding mask is combined
        # with (not substituted for) the causal mask.
        padding_mask = self._normalize_padding_mask(attention_mask, batch_size, seq_len, x.device)
        chunks = []
        for q_start in range(0, seq_len, self.attention_chunk_size):
            q_end = min(seq_len, q_start + self.attention_chunk_size)
            k_start = max(0, q_start - self.attention_window + 1)
            q_chunk = q[:, :, q_start:q_end, :]
            k_chunk = k[:, :, k_start:q_end, :]
            v_chunk = v[:, :, k_start:q_end, :]

            q_positions = torch.arange(q_start, q_end, device=x.device)[:, None]
            k_positions = torch.arange(k_start, q_end, device=x.device)[None, :]
            allowed = (k_positions <= q_positions) & (k_positions >= (q_positions - self.attention_window + 1))
            if padding_mask is not None:
                allowed = allowed[None, None, :, :] & padding_mask[:, None, None, k_start:q_end]
            else:
                allowed = allowed[None, None, :, :]
            chunks.append(F.scaled_dot_product_attention(
                q_chunk, k_chunk, v_chunk, attn_mask=allowed, dropout_p=0.0, is_causal=False,
            ))
        attn_output = torch.cat(chunks, dim=2)

        # 6. Transpose and reshape back to [B, T, H]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # 7. Output projection
        return self.o_proj(attn_output)

    @staticmethod
    def _normalize_padding_mask(
        attention_mask: Optional[torch.Tensor], batch_size: int, seq_len: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Return a [B, T] boolean valid-token mask; reject ambiguous mask shapes."""
        if attention_mask is None:
            return None
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (batch_size, seq_len):
            raise ValueError("attention_mask must have shape [batch_size, seq_len] with 1/True for valid tokens")
        return attention_mask.to(device=device, dtype=torch.bool)
