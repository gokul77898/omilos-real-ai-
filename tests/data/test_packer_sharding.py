"""Unit tests for sequence packing, shard writing, and memory-mapped DataLoader reading."""

import tempfile
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.packer import SequencePacker
from src.data.sharding import ShardWriter, ShardedDataset, ShardIntegrityError


def test_sequence_packer_and_sharding():
    """Verify 2048-token sequence packing and memory-mapped shard reading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        packer = SequencePacker(max_seq_len=64, bos_id=2, eos_id=3)
        writer = ShardWriter(tmpdir, shard_prefix="test_shard", max_seqs_per_shard=2)

        # Feed 3 documents each of length 30 (30 + 2 special tokens = 32 tokens each -> 2 docs make 1 64-token chunk)
        for i in range(4):
            doc = [10 + i] * 30
            chunks = packer.add_document(doc)
            for c in chunks:
                writer.write_sequence(c)

        writer.close()
        stats = packer.get_stats()
        assert stats["packed_sequences"] == 2
        assert stats["total_packed_tokens"] == 128

        # Read back with ShardedDataset
        dataset = ShardedDataset(tmpdir, seq_len=64)
        assert len(dataset) == 2

        loader = DataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (2, 64)
        assert batch["labels"].shape == (2, 64)


def test_sharded_dataset_rejects_corrupt_metadata_and_pad():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ShardWriter(tmpdir, max_seqs_per_shard=1)
        writer.write_sequence(np.array([0, 2, 3, 4], dtype=np.uint16))
        writer.close()
        with pytest.raises(ShardIntegrityError, match="PAD=0"):
            ShardedDataset(tmpdir, seq_len=4, vocab_size=10, reject_pad=True)
        meta = next(__import__("pathlib").Path(tmpdir).glob("*.json"))
        meta.write_text('{"bad": true}', encoding="utf-8")
        with pytest.raises(ShardIntegrityError, match="Invalid metadata"):
            ShardedDataset(tmpdir, seq_len=4)
