"""Unit tests for RMSNorm layer."""

import pytest
import torch

from src.norm import RMSNorm


def test_rmsnorm_shape_preservation():
    """Verify that RMSNorm preserves tensor shape."""
    dim = 256
    norm = RMSNorm(dim=dim, eps=1e-6)
    x = torch.randn(4, 16, dim)
    y = norm(x)
    assert y.shape == x.shape


def test_rmsnorm_gradients():
    """Verify gradients propagate to learnable weight parameter."""
    dim = 128
    norm = RMSNorm(dim=dim)
    x = torch.randn(2, 8, dim, requires_grad=True)
    y = norm(x)
    loss = y.sum()
    loss.backward()

    assert norm.weight.grad is not None
    assert x.grad is not None
    assert torch.isfinite(norm.weight.grad).all()
    assert torch.isfinite(x.grad).all()


def test_rmsnorm_numerical_stability():
    """Verify RMSNorm handles zero or extremely small input values without NaNs."""
    dim = 64
    norm = RMSNorm(dim=dim, eps=1e-6)
    zeros = torch.zeros(2, 4, dim)
    y_zeros = norm(zeros)
    assert not torch.isnan(y_zeros).any()
    assert not torch.isinf(y_zeros).any()
    assert torch.allclose(y_zeros, torch.zeros_like(y_zeros))

    small = torch.full((2, 4, dim), 1e-12)
    y_small = norm(small)
    assert not torch.isnan(y_small).any()
    assert not torch.isinf(y_small).any()


def test_rmsnorm_dtype_handling():
    """Verify RMSNorm respects input tensor dtype (float32, float16)."""
    dim = 128
    norm = RMSNorm(dim=dim)

    # Float32
    x_f32 = torch.randn(2, 4, dim, dtype=torch.float32)
    assert norm(x_f32).dtype == torch.float32

    # Float16
    norm_f16 = RMSNorm(dim=dim).to(torch.float16)
    x_f16 = torch.randn(2, 4, dim, dtype=torch.float16)
    assert norm_f16(x_f16).dtype == torch.float16
