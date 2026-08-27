"""Unit tests for causal language model loss computation."""

import pytest
import torch

from src.loss import compute_causal_lm_loss


def test_causal_loss_shifting_and_shape():
    """Verify compute_causal_lm_loss shifts logits and labels properly."""
    batch_size, seq_len, vocab_size = 2, 5, 10
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = compute_causal_lm_loss(logits, labels)
    assert loss.ndim == 0  # Scalar
    assert loss.item() > 0
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_causal_loss_ignore_index():
    """Verify that ignore_index (-100) positions are excluded from loss computation."""
    batch_size, seq_len, vocab_size = 1, 4, 8
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # All target positions set to -100 except one
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    labels[0, 1] = 3  # Only position 0 -> position 1 is active

    loss = compute_causal_lm_loss(logits, labels, ignore_index=-100)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_causal_loss_invalid_shapes_raise_error():
    """Verify mismatching shapes raise ValueError."""
    logits = torch.randn(2, 4, 10)
    labels = torch.randint(0, 10, (2, 5))  # Wrong seq_len
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_causal_lm_loss(logits, labels)
