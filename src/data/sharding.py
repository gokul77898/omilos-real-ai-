"""Memory-mapped binary shard writing and PyTorch ShardedDataset for streaming pretraining."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class ShardIntegrityError(ValueError):
    """Raised when a token shard or its metadata is malformed or inconsistent."""


class ShardWriter:
    """Writes packed uint16 token buffers into fixed-size binary shards on disk."""

    def __init__(self, output_dir: Union[str, Path], shard_prefix: str = "shard", max_seqs_per_shard: int = 1000) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_prefix = shard_prefix
        self.max_seqs_per_shard = max_seqs_per_shard

        self.current_shard_idx = 0
        self.current_buffer: List[np.ndarray] = []
        self.shards_written: List[Path] = []

    def write_sequence(self, seq: np.ndarray) -> Optional[Path]:
        """Add a 2048-token sequence and write shard when capacity is reached."""
        array = np.asarray(seq)
        if array.ndim != 1 or array.size == 0:
            raise ShardIntegrityError("A sequence must be a non-empty one-dimensional array")
        if np.any(array < 0) or np.any(array > np.iinfo(np.uint16).max):
            raise ShardIntegrityError("Token IDs must fit uint16 storage")
        if self.current_buffer and array.size != self.current_buffer[0].size:
            raise ShardIntegrityError("All sequences in a shard must have the same length")
        self.current_buffer.append(array.astype(np.uint16, copy=False))
        if len(self.current_buffer) >= self.max_seqs_per_shard:
            return self._flush_shard()
        return None

    def close(self) -> Optional[Path]:
        """Flush any remaining sequences in the buffer."""
        if self.current_buffer:
            return self._flush_shard()
        return None

    def _flush_shard(self) -> Path:
        shard_bin = self.output_dir / f"{self.shard_prefix}_{self.current_shard_idx:05d}.bin"
        shard_meta = self.output_dir / f"{self.shard_prefix}_{self.current_shard_idx:05d}.json"

        data = np.stack(self.current_buffer, axis=0).astype(np.uint16)
        tmp_bin = shard_bin.with_suffix(".bin.tmp")
        data.tofile(tmp_bin)
        os.replace(tmp_bin, shard_bin)

        meta = {
            "shard_index": self.current_shard_idx,
            "num_sequences": len(self.current_buffer),
            "seq_len": int(data.shape[1]),
            "dtype": "uint16",
            "bytes": int(data.nbytes),
        }
        tmp_meta = shard_meta.with_suffix(".json.tmp")
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_meta, shard_meta)

        self.shards_written.append(shard_bin)
        self.current_shard_idx += 1
        self.current_buffer.clear()
        return shard_bin


class ShardedDataset(Dataset):
    """Memory-mapped dataset reading directly from pre-tokenized binary shards on disk without RAM bloat."""

    def __init__(self, shard_dir: Union[str, Path], seq_len: int = 2048, vocab_size: Optional[int] = None, reject_pad: bool = False) -> None:
        self.shard_dir = Path(shard_dir)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        if seq_len <= 0:
            raise ShardIntegrityError("seq_len must be positive")
        self.bin_files = sorted(list(self.shard_dir.glob("*.bin")))

        self.shard_offsets: List[int] = [0]
        self.shard_lengths: List[int] = []
        self.mmaps: List[np.ndarray] = []

        total = 0
        for bin_file in self.bin_files:
            file_bytes = bin_file.stat().st_size
            meta_file = bin_file.with_suffix(".json")
            if not meta_file.exists():
                raise ShardIntegrityError(f"Missing metadata for shard: {bin_file.name}")
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ShardIntegrityError(f"Unreadable metadata for shard: {bin_file.name}") from exc
            expected_bytes = int(meta.get("num_sequences", -1)) * int(meta.get("seq_len", -1)) * 2
            if meta.get("dtype") != "uint16" or meta.get("seq_len") != seq_len or meta.get("bytes") != expected_bytes:
                raise ShardIntegrityError(f"Invalid metadata for shard: {bin_file.name}")
            if file_bytes != expected_bytes or file_bytes % (seq_len * 2) != 0:
                raise ShardIntegrityError(f"Truncated or misaligned shard: {bin_file.name}")
            num_seqs = file_bytes // (seq_len * 2)  # uint16 is 2 bytes
            if num_seqs == 0:
                raise ShardIntegrityError(f"Shard contains zero sequences: {bin_file.name}")
            self.shard_lengths.append(num_seqs)
            total += num_seqs
            self.shard_offsets.append(total)
            # Memory map
            m = np.memmap(bin_file, dtype=np.uint16, mode="r", shape=(num_seqs, seq_len))
            if vocab_size is not None and (vocab_size <= 0 or np.any(m >= vocab_size)):
                raise ShardIntegrityError(f"Token ID outside vocabulary range in shard: {bin_file.name}")
            if reject_pad and np.any(m == 0):
                raise ShardIntegrityError(f"PAD=0 found in packed training shard: {bin_file.name}")
            self.mmaps.append(m)

        self.total_sequences = total

    def __len__(self) -> int:
        return self.total_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.total_sequences:
            raise IndexError(f"Index {idx} out of range [0, {self.total_sequences})")

        # Find which shard contains this index
        shard_idx = 0
        while shard_idx < len(self.shard_lengths) and idx >= self.shard_offsets[shard_idx + 1]:
            shard_idx += 1

        local_idx = idx - self.shard_offsets[shard_idx]
        tokens = torch.from_numpy(self.mmaps[shard_idx][local_idx].astype(np.int64))
        # For causal language modeling: input_ids and labels
        return {"input_ids": tokens, "labels": tokens}
