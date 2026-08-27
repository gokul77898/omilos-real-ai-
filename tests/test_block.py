"""Unit and integration tests for TransformerBlock and parameter counting."""

import pytest
import torch
import torch.nn as nn

from src.block import TransformerBlock, count_parameters
from src.config import ModelConfig


def test_transformer_block_forward_shape():
    """Verify complete Transformer block maintains input shape."""
    config = ModelConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_kv_heads=4,
        intermediate_size=688,
        max_seq_len=128,
    )
    block = TransformerBlock(config)
    x = torch.randn(2, 16, 256)
    out = block(x)
    assert out.shape == (2, 16, 256)


def test_transformer_block_full_gradient_flow():
    """Verify all trainable parameters in the block receive non-null finite gradients."""
    config = ModelConfig(
        hidden_size=128,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=344,
        max_seq_len=64,
    )
    block = TransformerBlock(config)
    x = torch.randn(2, 8, 128, requires_grad=True)
    out = block(x)
    loss = out.sum()
    loss.backward()

    # Check RMSNorm parameters
    assert block.input_layernorm.weight.grad is not None
    assert block.post_attention_layernorm.weight.grad is not None

    # Check Attention projection parameters
    assert block.self_attn.q_proj.weight.grad is not None
    assert block.self_attn.k_proj.weight.grad is not None
    assert block.self_attn.v_proj.weight.grad is not None
    assert block.self_attn.o_proj.weight.grad is not None

    # Check SwiGLU projection parameters
    assert block.mlp.gate_proj.weight.grad is not None
    assert block.mlp.up_proj.weight.grad is not None
    assert block.mlp.down_proj.weight.grad is not None

    # Verify all gradients are finite (no NaNs or Infs)
    for name, param in block.named_parameters():
        assert param.grad is not None, f"Parameter {name} has None grad!"
        assert torch.isfinite(param.grad).all(), f"Parameter {name} grad has NaNs or Infs!"


def test_transformer_block_parameter_counts():
    """Verify parameter counting utility accurately computes trainable and buffer counts."""
    config = ModelConfig(
        hidden_size=128,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=344,
        max_seq_len=64,
    )
    block = TransformerBlock(config)
    counts = count_parameters(block)

    assert counts["trainable"] > 0
    assert counts["total"] >= counts["trainable"]
    assert isinstance(counts["trainable"], int)


def test_stacked_two_blocks_integration():
    """Verify a minimal 2-block stack forward and backward execution with no NaNs/Infs."""
    config = ModelConfig(
        hidden_size=128,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=344,
        max_seq_len=64,
    )
    block1 = TransformerBlock(config)
    block2 = TransformerBlock(config)

    x = torch.randn(2, 10, 128, requires_grad=True)
    h1 = block1(x)
    h2 = block2(h1)

    assert h2.shape == (2, 10, 128)
    assert not torch.isnan(h2).any()
    assert not torch.isinf(h2).any()

    loss = h2.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
