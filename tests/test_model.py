"""Unit and integration tests for LegalCausalLM decoder-only model."""

import pytest
import torch

from src.config import ModelConfig
from src.model import CausalLMOutput, LegalCausalLM


def test_model_forward_shape():
    """Verify LegalCausalLM outputs logits of shape [B, T, vocab_size]."""
    config = ModelConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=176,
        max_seq_len=32,
    )
    model = LegalCausalLM(config)
    input_ids = torch.randint(0, 100, (2, 8))

    output = model(input_ids)
    assert isinstance(output, CausalLMOutput)
    assert output.logits.shape == (2, 8, 100)
    assert output.loss is None


def test_model_forward_with_labels():
    """Verify forward with labels computes scalar loss and propagates gradients."""
    config = ModelConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=176,
        max_seq_len=32,
    )
    model = LegalCausalLM(config)
    input_ids = torch.randint(0, 100, (2, 8))

    output = model(input_ids=input_ids, labels=input_ids)
    assert output.loss is not None
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)

    output.loss.backward()
    assert model.embed_tokens.weight.grad is not None
    assert torch.isfinite(model.embed_tokens.weight.grad).all()


def test_weight_tying_enabled():
    """Verify weight tying shares parameter between embed_tokens and lm_head."""
    config = ModelConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=176,
        tie_word_embeddings=True,
    )
    model = LegalCausalLM(config)
    assert model.lm_head.weight is model.embed_tokens.weight

    summary = model.get_parameter_summary()
    assert summary["lm_head"] == 0
    assert summary["tie_word_embeddings"] is True


def test_weight_tying_disabled():
    """Verify weight tying disabled creates independent LM head parameter."""
    config = ModelConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=176,
        tie_word_embeddings=False,
    )
    model = LegalCausalLM(config)
    assert model.lm_head.weight is not model.embed_tokens.weight

    summary = model.get_parameter_summary()
    assert summary["lm_head"] == 100 * 64
    assert summary["tie_word_embeddings"] is False


def test_strict_end_to_end_causality():
    """Verify that changing a future token does NOT alter logits at earlier positions."""
    config = ModelConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=176,
        max_seq_len=16,
    )
    model = LegalCausalLM(config)
    model.eval()

    seq_a = torch.tensor([[10, 20, 30, 40, 50]])
    seq_b = torch.tensor([[10, 20, 30, 40, 99]])  # Only token 4 differs

    with torch.no_grad():
        out_a = model(seq_a).logits
        out_b = model(seq_b).logits

    # Logits at positions 0, 1, 2, 3 must be identical
    assert torch.allclose(out_a[:, :4, :], out_b[:, :4, :], atol=1e-5), (
        "Causal leakage: Modifying token at index 4 altered earlier logits!"
    )
    # Logits at position 4 should differ
    assert not torch.allclose(out_a[:, 4, :], out_b[:, 4, :], atol=1e-3)


def test_input_validation_errors():
    """Verify invalid input shapes, types, or token ID ranges raise ValueError."""
    config = ModelConfig(vocab_size=100, hidden_size=64, max_seq_len=16)
    model = LegalCausalLM(config)

    # 1D tensor
    with pytest.raises(ValueError, match="must be 2D"):
        model(torch.tensor([1, 2, 3]))

    # Exceeding max_seq_len
    with pytest.raises(ValueError, match="exceeds configured maximum"):
        model(torch.randint(0, 100, (1, 20)))

    # Negative token ID
    with pytest.raises(ValueError, match="range"):
        model(torch.tensor([[-1, 10, 20]]))

    # Out of vocab token ID
    with pytest.raises(ValueError, match="range"):
        model(torch.tensor([[10, 105, 20]]))
