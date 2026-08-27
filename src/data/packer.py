"""Sequence packing into fixed 2048-token context windows with explicit document boundaries."""

from __future__ import annotations

from typing import Iterator, List, Tuple
import numpy as np


class SequencePacker:
    """Packs variable-length tokenized documents into fixed max_seq_len chunks."""

    def __init__(self, max_seq_len: int = 2048, bos_id: int = 1, eos_id: int = 2) -> None:
        self.max_seq_len = max_seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.buffer: List[int] = []
        self.total_tokens_in = 0
        self.total_packed_sequences = 0
        self.total_packed_tokens = 0

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
        }
