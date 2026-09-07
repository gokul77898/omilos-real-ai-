"""Memory-efficient causal language-model loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_causal_lm_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head: torch.nn.Module,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute causal next-token loss without materializing full 128K logits.

    Instead of constructing [B, T, V] logits for the whole sequence, the
    vocabulary projection and cross-entropy are evaluated in time chunks.
    This preserves the exact causal-LM objective while keeping peak memory
    bounded for very long sequences.
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            f"Expected hidden_states rank 3 [B, T, H], "
            f"got {list(hidden_states.shape)}"
        )
    if labels.ndim != 2:
        raise ValueError(
            f"Expected labels rank 2 [B, T], got {list(labels.shape)}"
        )
    if hidden_states.shape[0] != labels.shape[0]:
        raise ValueError("Batch size mismatch between hidden_states and labels")
    if hidden_states.shape[1] != labels.shape[1]:
        raise ValueError("Sequence length mismatch between hidden_states and labels")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    # Causal shift: position t predicts token t+1.
    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]

    batch_size, seq_len_minus_one, hidden_size = shift_hidden.shape
    del hidden_size

    total_loss = None
    total_tokens = 0

    for start in range(0, seq_len_minus_one, chunk_size):
        end = min(start + chunk_size, seq_len_minus_one)

        hidden_chunk = shift_hidden[:, start:end, :]
        label_chunk = shift_labels[:, start:end]

        logits_chunk = lm_head(hidden_chunk)

        # Sum here so the final loss is the exact global mean over tokens,
        # rather than an unweighted mean of chunk means.
        chunk_loss = F.cross_entropy(
            logits_chunk.reshape(-1, logits_chunk.shape[-1]),
            label_chunk.reshape(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )

        valid_tokens = int((label_chunk != ignore_index).sum().item())

        if valid_tokens:
            total_loss = (
                chunk_loss if total_loss is None
                else total_loss + chunk_loss
            )
            total_tokens += valid_tokens

    if total_loss is None or total_tokens == 0:
        # Preserve autograd connectivity and return a finite scalar.
        return (hidden_states.sum() * 0.0)

    return total_loss / total_tokens
