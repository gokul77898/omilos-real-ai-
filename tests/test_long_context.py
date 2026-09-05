"""Regression tests for bounded causal long-context mechanics."""
import pytest
import torch

from src.attention import GroupedQueryAttention
from src.config import ModelConfig
from src.rope import RotaryEmbedding
from src.model import LegalCausalLM


@pytest.mark.parametrize("length", [2048, 8192, 32768, 65536, 131072])
def test_rope_supports_required_lengths(length):
    rope = RotaryEmbedding(8, max_position_embeddings=2048, scaling_type="linear", scaling_factor=64)
    cos, sin = rope(torch.empty(1, 1, 1, 8), length)
    assert cos.shape == (length, 8)
    assert sin.shape == (length, 8)
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all()


def test_padding_mask_never_disables_causality():
    cfg = ModelConfig(vocab_size=32, hidden_size=16, num_layers=1, num_attention_heads=2, num_kv_heads=1,
                      intermediate_size=32, max_seq_len=16, attention_window=8, attention_chunk_size=4)
    module = GroupedQueryAttention(cfg).eval()
    x = torch.randn(1, 8, 16)
    changed = x.clone()
    changed[:, 7] += 100
    mask = torch.ones(1, 8, dtype=torch.long)
    with torch.no_grad():
        baseline = module(x, mask)
        altered = module(changed, mask)
    assert torch.allclose(baseline[:, :7], altered[:, :7], atol=1e-6)


def test_attention_uses_bounded_masks_not_full_sequence_masks(monkeypatch):
    cfg = ModelConfig(vocab_size=32, hidden_size=16, num_layers=1, num_attention_heads=2, num_kv_heads=1,
                      intermediate_size=32, max_seq_len=256, attention_window=32, attention_chunk_size=8)
    module = GroupedQueryAttention(cfg)
    seen = []
    original = torch.nn.functional.scaled_dot_product_attention
    def wrapped(q, k, v, **kwargs):
        seen.append(kwargs["attn_mask"].shape)
        return original(q, k, v, **kwargs)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", wrapped)
    assert module(torch.randn(1, 256, 16)).shape == (1, 256, 16)
    assert seen and max(shape[-1] for shape in seen) <= 39
    assert max(shape[-2] for shape in seen) <= 8


def test_tiny_full_model_forward_at_128k_is_finite():
    """Exercise embedding, RoPE, local attention, block and logits at 128K cheaply."""
    cfg = ModelConfig(vocab_size=8, hidden_size=8, num_layers=1, num_attention_heads=2, num_kv_heads=1,
                      intermediate_size=16, max_seq_len=131072, attention_window=16, attention_chunk_size=512,
                      rope_scaling_factor=64)
    model = LegalCausalLM(cfg).eval()
    with torch.no_grad():
        output = model(torch.ones(1, 131072, dtype=torch.long))
    assert output.logits.shape == (1, 131072, 8)
    assert torch.isfinite(output.logits).all()
