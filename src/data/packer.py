"""Sequence packing into fixed 2048-token context windows with explicit document boundaries."""

from __future__ import annotations

from typing import Iterator, List, Tuple
import numpy as np


class SequencePacker:
    """Packs variable-length tokenized documents into fixed max_seq_len chunks."""

    def __init__(self, max_seq_len: int = 2048, bos_id: int = 2, eos_id: int = 3, final_sequence_policy: str = "discard") -> None:
        if max_seq_len < 3:
            raise ValueError("max_seq_len must be at least 3")
        if (bos_id, eos_id) != (2, 3):
            raise ValueError("Omilos tokenizer contract requires BOS=2 and EOS=3")
        if final_sequence_policy not in {"discard", "error"}:
            raise ValueError("final_sequence_policy must be 'discard' or 'error'; padding is intentionally unsupported")
        self.max_seq_len = max_seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.final_sequence_policy = final_sequence_policy

        self.buffer: List[int] = []
        self.total_tokens_in = 0
        self.total_packed_sequences = 0
        self.total_packed_tokens = 0
        self.finalized = False

    def add_document(self, token_ids: List[int]) -> List[np.ndarray]:
        """Add a tokenized document and yield all completed packed sequences of length max_seq_len.

        Document boundaries: <s> ... </s>
        """
        doc = [self.bos_id] + token_ids + [self.eos_id]
        self.total_tokens_in += len(doc)
        self.buffer.extend(doc)

        packed = []
        while len(self.buffer) >= self.max_seq_len:
            chunk = np.array(self.buffer[:self.max_seq_len], dtype=np.uint16)
            self.buffer = self.buffer[self.max_seq_len:]
            self.total_packed_sequences += 1
            self.total_packed_tokens += self.max_seq_len
            packed.append(chunk)

        return packed

    def get_stats(self) -> dict:
        """Compute sequence packing statistics and efficiency."""
        discarded = len(self.buffer)
        efficiency = (self.total_packed_tokens / max(1, self.total_tokens_in)) * 100.0
        return {
            "total_tokens_in": self.total_tokens_in,
            "packed_sequences": self.total_packed_sequences,
            "total_packed_tokens": self.total_packed_tokens,
            "discarded_remainder": discarded,
            "packing_efficiency_pct": round(efficiency, 2),
            "final_sequence_policy": self.final_sequence_policy,
            "finalized": self.finalized,
        }

    def finalize(self) -> List[np.ndarray]:
        """Finalize packing without inventing PAD tokens in training examples.

        Fixed-length shards cannot represent a partial sequence safely.  The caller
        must explicitly choose to discard its tail or fail, making data loss visible.
        """
        self.finalized = True
        if not self.buffer:
            return []
        if self.final_sequence_policy == "error":
            raise ValueError(f"Unpacked tail contains {len(self.buffer)} tokens")
        return []
