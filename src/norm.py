"""Root Mean Square Layer Normalization (RMSNorm) implementation."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    Formula:
        RMS(x) = sqrt( mean(x^2, dim=-1, keepdim=True) + eps )
        RMSNorm(x) = (x / RMS(x)) * weight

    Args:
        dim: The hidden dimension size of the input tensor.
        eps: Small epsilon constant for numerical stability (default: 1e-6).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate RMS norm with float32 casting for numerical stability."""
        input_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(variance + self.eps)
        return normed.to(input_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor and apply learnable affine scale."""
        return self._norm(x) * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
