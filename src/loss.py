"""Causal Language Model loss computation with shifted predictions and targets."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute autoregressive next-token cross-entropy loss.

    Aligns predictions and targets such that token at position t predicts token at t+1:
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

    Args:
        logits: Unnormalized model predictions of shape [batch_size, seq_len, vocab_size].
        labels: Target token IDs of shape [batch_size, seq_len].
        ignore_index: Target index to ignore in loss calculation (default: -100).

    Returns:
        Scalar loss tensor.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits of rank 3 [B, T, V], got shape {list(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"Expected labels of rank 2 [B, T], got shape {list(labels.shape)}")
    if logits.shape[0] != labels.shape[0] or logits.shape[1] != labels.shape[1]:
        raise ValueError(
            f"Shape mismatch between logits ({list(logits.shape)}) and labels ({list(labels.shape)})"
        )

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten for cross entropy
    vocab_size = shift_logits.shape[-1]
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

    return loss
