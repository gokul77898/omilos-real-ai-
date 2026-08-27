"""Unit tests for SwiGLU MLP."""

import pytest
import torch

from src.config import ModelConfig
from src.mlp import SwiGLUMLP


def test_swiglu_shape_preservation():
    """Verify SwiGLU MLP preserves input tensor shape."""
    config = ModelConfig(hidden_size=256, intermediate_size=688)
    mlp = SwiGLUMLP(config)
    x = torch.randn(2, 16, 256)
    out = mlp(x)
    assert out.shape == (2, 16, 256)


def test_swiglu_gradients():
    """Verify gradients propagate through gate, up, and down projections."""
    config = ModelConfig(hidden_size=128, intermediate_size=344)
    mlp = SwiGLUMLP(config)
    x = torch.randn(2, 8, 128, requires_grad=True)
    out = mlp(x)
    loss = out.sum()
    loss.backward()

    assert mlp.gate_proj.weight.grad is not None
    assert mlp.up_proj.weight.grad is not None
    assert mlp.down_proj.weight.grad is not None
    assert x.grad is not None
    assert torch.isfinite(mlp.gate_proj.weight.grad).all()
    assert torch.isfinite(mlp.up_proj.weight.grad).all()
    assert torch.isfinite(mlp.down_proj.weight.grad).all()


def test_swiglu_non_linearity():
    """Verify SwiGLU exhibits non-linear behavior (f(a+b) != f(a) + f(b))."""
    config = ModelConfig(hidden_size=64, intermediate_size=176)
    mlp = SwiGLUMLP(config)
    a = torch.randn(1, 4, 64)
    b = torch.randn(1, 4, 64)

    with torch.no_grad():
        f_a_plus_b = mlp(a + b)
        f_a_plus_f_b = mlp(a) + mlp(b)

    assert not torch.allclose(f_a_plus_b, f_a_plus_f_b, atol=1e-3)
