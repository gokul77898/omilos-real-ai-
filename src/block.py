"""Complete Pre-LayerNorm Transformer Block with GQA, RoPE, and SwiGLU."""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from src.attention import GroupedQueryAttention
from src.config import ModelConfig
from src.mlp import SwiGLUMLP
from src.norm import RMSNorm


def count_parameters(module: nn.Module) -> Dict[str, int]:
    """Calculate trainable, non-trainable, and total parameter counts for a module."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    buffer_params = sum(b.numel() for b in module.buffers())
    return {
        "trainable": trainable,
        "non_trainable": non_trainable + buffer_params,
        "total": trainable + non_trainable + buffer_params,
    }


class TransformerBlock(nn.Module):
    """Qwen-style Decoder-Only Transformer Block.

    Architecture (Pre-Norm with Residual Stream):
        x_norm1 = input_layernorm(x)
        h1 = x + attention(x_norm1)
        h1_norm = post_attention_layernorm(h1)
        out = h1 + mlp(h1_norm)

    Args:
        config: ModelConfig containing layer hyperparameters.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        # 1. Pre-Attention Normalization & GQA
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)

        # 2. Post-Attention Normalization & SwiGLU MLP
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through Transformer Block.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask tensor.

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # 1. Attention residual block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        # 2. MLP residual block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
