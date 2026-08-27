"""Unit tests for Rotary Positional Embeddings (RoPE)."""

import pytest
import torch

from src.rope import RotaryEmbedding, apply_rotary_emb, rotate_half


def test_rotate_half_transformation():
    """Verify rotate_half transforms [x1, x2] into [-x2, x1]."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    rotated = rotate_half(x)
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    assert torch.equal(rotated, expected)


def test_rope_output_shapes():
    """Verify RoPE generates cosine and sine caches with expected shapes."""
    head_dim = 64
    max_seq_len = 128
    rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=max_seq_len)

    x = torch.randn(2, 4, 32, head_dim)
    cos, sin = rope(x, seq_len=32)

    assert cos.shape == (32, head_dim)
    assert sin.shape == (32, head_dim)


def test_apply_rotary_emb_preserves_shape():
    """Verify apply_rotary_emb maintains Q and K shapes."""
    batch_size, num_heads, num_kv_heads, seq_len, head_dim = 2, 8, 4, 16, 64
    rope = RotaryEmbedding(dim=head_dim)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)

    cos, sin = rope(q, seq_len=seq_len)
    q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_position_sensitivity():
    """Verify tokens at different positions receive distinct rotary transformations."""
    head_dim = 64
    rope = RotaryEmbedding(dim=head_dim)
    q = torch.ones(1, 1, 4, head_dim)  # Identical queries at 4 positions
    k = torch.ones(1, 1, 4, head_dim)

    cos, sin = rope(q, seq_len=4)
    q_rot, _ = apply_rotary_emb(q, k, cos, sin)

    # Position 0 vs Position 1 should differ
    assert not torch.allclose(q_rot[0, 0, 0], q_rot[0, 0, 1])
    assert not torch.allclose(q_rot[0, 0, 1], q_rot[0, 0, 2])


def test_rope_invalid_dim_raises_error():
    """Verify odd head dimension raises ValueError."""
    with pytest.raises(ValueError, match="RoPE dimension must be even"):
        RotaryEmbedding(dim=65)
