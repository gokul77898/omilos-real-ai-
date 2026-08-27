"""Unit tests for gradient accumulation equivalence, clipping, and checkpointing."""

import pytest
import torch
from torch.optim import SGD

from src.config import ModelConfig
from src.model import LegalCausalLM


def test_gradient_accumulation_equivalence():
    """Verify micro_batch=1 with accumulation=2 produces identical gradients to batch=2."""
    config = ModelConfig(vocab_size=100, hidden_size=64, num_layers=2, max_seq_len=16)

    torch.manual_seed(42)
    model_accum = LegalCausalLM(config)
    torch.manual_seed(42)
    model_batch = LegalCausalLM(config)

    # 2 samples
    x1 = torch.tensor([[10, 20, 30, 40]])
    x2 = torch.tensor([[15, 25, 35, 45]])
    x_combined = torch.cat([x1, x2], dim=0)

    # 1. Gradient accumulation pass: 2 steps with scaled loss
    model_accum.zero_grad()
    loss1 = model_accum(x1, labels=x1).loss / 2.0
    loss1.backward()
    loss2 = model_accum(x2, labels=x2).loss / 2.0
    loss2.backward()

    # 2. Direct full-batch pass
    model_batch.zero_grad()
    loss_full = model_batch(x_combined, labels=x_combined).loss
    loss_full.backward()

    # Verify all parameter gradients match
    for p_acc, p_bat in zip(model_accum.parameters(), model_batch.parameters()):
        if p_acc.grad is not None:
            assert torch.allclose(p_acc.grad, p_bat.grad, atol=1e-5)


def test_gradient_checkpointing_correctness():
    """Verify enabling gradient checkpointing produces identical forward outputs and finite gradients."""
    config = ModelConfig(vocab_size=100, hidden_size=64, num_layers=2, max_seq_len=16)
    torch.manual_seed(42)
    model = LegalCausalLM(config)

    x = torch.randint(0, 100, (2, 8))

    # Without gradient checkpointing
    model.gradient_checkpointing_disable()
    out1 = model(x, labels=x)
    loss1 = out1.loss
    loss1.backward()
    grads1 = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    # With gradient checkpointing
    model.zero_grad()
    model.gradient_checkpointing_enable()
    out2 = model(x, labels=x)
    loss2 = out2.loss
    loss2.backward()
    grads2 = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    assert torch.allclose(out1.logits, out2.logits, atol=1e-5)
    assert torch.allclose(loss1, loss2, atol=1e-5)
    for g1, g2 in zip(grads1, grads2):
        assert torch.allclose(g1, g2, atol=1e-4)
