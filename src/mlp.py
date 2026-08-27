"""SwiGLU Multi-Layer Perceptron (Feed-Forward Network)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig


class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network.

    Formula:
        SwiGLU(x) = (SiLU(W_gate x) * (W_up x)) W_down

    Args:
        config: ModelConfig containing hidden_size and intermediate_size.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std / math.sqrt(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: SiLU(gate(x)) * up(x) -> down(x)"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
