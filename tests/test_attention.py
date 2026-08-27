"""Unit tests for Grouped Query Attention (GQA) and Causal Masking."""

import pytest
import torch

from src.attention import GroupedQueryAttention, repeat_kv
from src.config import ModelConfig


def test_repeat_kv():
    """Verify repeat_kv expands KV heads correctly."""
    batch_size, num_kv_heads, seq_len, head_dim = 2, 2, 4, 32
    x = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    repeated = repeat_kv(x, n_rep=3)
    assert repeated.shape == (batch_size, 6, seq_len, head_dim)


def test_gqa_forward_shape_mha():
    """Verify GQA forward output shape when num_kv_heads == num_heads (MHA mode)."""
    config = ModelConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_kv_heads=8,
        max_seq_len=128,
    )
    attn = GroupedQueryAttention(config)
    x = torch.randn(2, 16, 256)
    out = attn(x)
    assert out.shape == (2, 16, 256)


def test_gqa_forward_shape_gqa():
    """Verify GQA forward output shape when num_kv_heads < num_heads (GQA mode)."""
    config = ModelConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_kv_heads=2,  # 8 query heads share 2 KV heads (4 per group)
        max_seq_len=128,
    )
    attn = GroupedQueryAttention(config)
    x = torch.randn(2, 16, 256)
    out = attn(x)
    assert out.shape == (2, 16, 256)


def test_gqa_strict_causality():
    """Verify causal masking: modifying token at position T does NOT affect positions t < T."""
    config = ModelConfig(
        hidden_size=128,
        num_attention_heads=4,
        num_kv_heads=2,
        max_seq_len=64,
    )
    attn = GroupedQueryAttention(config)
    attn.eval()

    # Input 1: Sequence of length 5
    x1 = torch.randn(1, 5, 128)
    # Input 2: Same prefix (tokens 0..3), but token 4 is completely modified
    x2 = x1.clone()
    x2[0, 4, :] = x2[0, 4, :] + 10.0

    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)

    # Outputs at tokens 0..3 must be bitwise / float-equal
    assert torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-6), (
        "Causal leak detected: modifying token at position 4 altered earlier tokens!"
    )
    # Output at token 4 should differ
    assert not torch.allclose(out1[:, 4, :], out2[:, 4, :], atol=1e-4)


def test_gqa_invalid_config_raises_error():
    """Verify indivisible head configuration raises ValueError."""
    with pytest.raises(ValueError, match="must be divisible"):
        GroupedQueryAttention(
            ModelConfig(hidden_size=256, num_attention_heads=7, num_kv_heads=2)
        )
