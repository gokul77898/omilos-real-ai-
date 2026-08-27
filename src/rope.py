"""Rotary Position Embeddings (RoPE) implementation."""

from typing import Optional, Tuple
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor.

    Splits the last dimension [..., d] into [..., d/2] and [..., d/2],
    negating the second half: [-x2, x1].
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).

    Calculates rotary frequencies Theta = base^(-2*(i-1)/dim) and precomputes
    cosine and sine tables up to max_position_embeddings.

    Args:
        dim: Head dimension size (must be even).
        max_position_embeddings: Maximum sequence length supported by cache.
        base: Base theta value for geometric frequency calculation (default: 10000.0).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # inv_freq: [dim // 2]
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute initial cos and sin caches
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        """Precompute cosine and sine embeddings up to seq_len."""
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        # freqs: [seq_len, dim // 2]
        freqs = torch.outer(t, self.inv_freq)
        # emb: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin tensors for sequence length seq_len.

        Args:
            x: Input tensor to determine device and dtype.
            seq_len: Current sequence length.

        Returns:
            Tuple of (cos, sin) tensors shaped [seq_len, dim].
        """
        if seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len)

        cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        return cos, sin


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key tensors.

    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine tensor of shape [seq_len, head_dim] or [1, 1, seq_len, head_dim]
        sin: Sine tensor of shape [seq_len, head_dim] or [1, 1, seq_len, head_dim]

    Returns:
        Tuple of rotated (q_embed, k_embed).
    """
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
