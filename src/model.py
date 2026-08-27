"""Full Decoder-Only Causal Language Model (LegalCausalLM)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.block import TransformerBlock, count_parameters
from src.config import ModelConfig
from src.loss import compute_causal_lm_loss
from src.norm import RMSNorm


@dataclass
class CausalLMOutput:
    """Output structure returned by LegalCausalLM."""
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class LegalCausalLM(nn.Module):
    """Decoder-only Causal Language Model tailored for Indian Legal Reasoning.

    Architecture:
        input_ids [B, T]
            ↓
        Token Embedding (nn.Embedding)
            ↓
        TransformerBlock × num_layers (with optional gradient checkpointing)
            ↓
        Final RMSNorm
            ↓
        LM Head (Linear hidden_size -> vocab_size, optional weight tying)
            ↓
        Logits [B, T, vocab_size]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.max_seq_len = config.max_seq_len
        self.tie_word_embeddings = config.tie_word_embeddings
        self.gradient_checkpointing = False

        # 1. Token Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # 2. Transformer Stack
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # 3. Final Pre-Head Normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 4. Language Model Prediction Head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 5. Weight Tying
        if self.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embedding and projection weights."""
        std = 0.02
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)
        if not self.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing across Transformer blocks."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def _validate_inputs(self, input_ids: torch.Tensor) -> None:
        """Validate input tensor rank, sequence length, and token ID ranges."""
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"input_ids must be a torch.Tensor, got {type(input_ids).__name__}")
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D tensor [batch_size, seq_len], got shape {list(input_ids.shape)}")
        if not input_ids.dtype in (torch.int32, torch.int64, torch.long):
            raise TypeError(f"input_ids must be integer dtype, got {input_ids.dtype}")

        batch_size, seq_len = input_ids.shape
        if seq_len == 0:
            raise ValueError("Sequence length must be greater than 0.")
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds configured maximum sequence length ({self.max_seq_len})."
            )

        min_val = input_ids.min().item()
        max_val = input_ids.max().item()
        if min_val < 0 or max_val >= self.vocab_size:
            raise ValueError(
                f"Token IDs must be in range [0, {self.vocab_size}), but found token ID min={min_val}, max={max_val}."
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        """Forward pass through the causal language model.

        Args:
            input_ids: Tensor of token IDs with shape [batch_size, seq_len].
            labels: Optional target token IDs with shape [batch_size, seq_len].
            attention_mask: Optional attention mask.

        Returns:
            CausalLMOutput with logits and optional loss.
        """
        self._validate_inputs(input_ids)

        # 1. Token Embeddings: [B, T] -> [B, T, H]
        hidden_states = self.embed_tokens(input_ids)

        # 2. Transformer Blocks (with optional gradient checkpointing)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)

        # 3. Final Normalization
        hidden_states = self.norm(hidden_states)

        # 4. Project to vocabulary logits: [B, T, H] -> [B, T, V]
        logits = self.lm_head(hidden_states)

        # 5. Optional loss calculation
        loss = None
        if labels is not None:
            loss = compute_causal_lm_loss(logits, labels)

        return CausalLMOutput(logits=logits, loss=loss)

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Compute detailed breakdown of trainable, non-trainable, and buffer parameters."""
        embed_params = self.embed_tokens.weight.numel()
        blocks_trainable = sum(
            sum(p.numel() for p in block.parameters() if p.requires_grad)
            for block in self.layers
        )
        blocks_buffers = sum(
            sum(b.numel() for b in block.buffers())
            for block in self.layers
        )
        final_norm_params = self.norm.weight.numel()

        if self.tie_word_embeddings:
            lm_head_params = 0
        else:
            lm_head_params = self.lm_head.weight.numel()

        total_trainable = (
            embed_params + blocks_trainable + final_norm_params + lm_head_params
        )
        total_buffers = blocks_buffers
        total_params = total_trainable + total_buffers

        return {
            "token_embeddings": embed_params,
            "transformer_blocks_trainable": blocks_trainable,
            "final_norm": final_norm_params,
            "lm_head": lm_head_params,
            "total_trainable": total_trainable,
            "total_buffers": total_buffers,
            "total_parameters": total_params,
            "tie_word_embeddings": self.tie_word_embeddings,
        }
