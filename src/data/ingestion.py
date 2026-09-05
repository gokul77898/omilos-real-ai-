"""Reproducible, provenance-preserving ingestion for legally sourced documents.

This module deliberately does not download data or infer a licence/provenance.  An
operator supplies a JSONL source manifest and the evidence fields are copied into
the resulting corpus manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

from src.data.cleaner import LegalTextCleaner
from src.data.dedup import DocumentDeduplicator
from src.data.packer import SequencePacker
from src.data.sharding import ShardWriter
from src.tokenizer import LegalTokenizer

PREPROCESSING_VERSION = "2.0.0"
SUPPORTED_LANGUAGES = {"en", "hi", "kn", "ta", "te", "ml", "bn", "mr", "gu", "pa", "ur", "sa"}
REQUIRED_FIELDS = {"document_id", "source_id", "source_url", "source_revision", "license", "language", "text"}


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reason: str = ""


def stable_split(document_id: str, validation_percent: int = 5) -> str:
    """Stable source-independent split identity; avoids rerun drift."""
    if not 0 < validation_percent < 100:
        raise ValueError("validation_percent must be in 1..99")
    bucket = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "validation" if bucket < validation_percent else "train"


def validate_document(doc: dict[str, Any]) -> ValidationResult:
    missing = sorted(k for k in REQUIRED_FIELDS if not doc.get(k))
    if missing:
        return ValidationResult(False, f"missing required evidence fields: {', '.join(missing)}")
    if str(doc["language"]).lower() not in SUPPORTED_LANGUAGES:
        return ValidationResult(False, f"unsupported language tag: {doc['language']}")
    text = doc["text"]
    if not isinstance(text, str) or len(text.strip()) < 50:
        return ValidationResult(False, "empty or extremely short text")
    try:
        text.encode("utf-8", "strict")
    except UnicodeError:
        return ValidationResult(False, "invalid Unicode")
    if re.search(r"(.)\1{30,}", text):
        return ValidationResult(False, "excessive character repetition")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and max(lines.count(line) for line in set(lines)) > 10:
        return ValidationResult(False, "excessive repeated lines")
    return ValidationResult(True)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read an operator-provided JSONL manifest without fabricating fields."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at line {line_no}")
                yield value


def build_corpus(
    documents: Iterable[dict[str, Any]], tokenizer: LegalTokenizer, output_dir: str | Path,
    seq_len: int, validation_percent: int = 5, max_seqs_per_shard: int = 500,
) -> dict[str, Any]:
    """Validate, deduplicate, deterministically split, pack and shard documents."""
    if tokenizer.vocab_size != 32000 or (tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) != (0, 1, 2, 3):
        raise ValueError("The canonical corpus requires the 32K tokenizer with PAD/UNK/BOS/EOS IDs 0/1/2/3")
    root = Path(output_dir)
    train_writer = ShardWriter(root / "train", "train_shard", max_seqs_per_shard)
    validation_writer = ShardWriter(root / "validation", "validation_shard", max_seqs_per_shard)
    packers = {split: SequencePacker(seq_len, tokenizer.bos_token_id, tokenizer.eos_token_id) for split in ("train", "validation")}
    writers = {"train": train_writer, "validation": validation_writer}
    dedup = DocumentDeduplicator()
    rejected: list[dict[str, str]] = []
    accepted = 0
    source_ids: set[str] = set()
    for doc in documents:
        result = validate_document(doc)
        if not result.accepted:
            rejected.append({"document_id": str(doc.get("document_id", "")), "reason": result.reason})
            continue
        cleaned = LegalTextCleaner.clean(doc["text"])
        duplicate, reason = dedup.is_duplicate(cleaned, citation=str(doc.get("document_id")))
        if duplicate:
            rejected.append({"document_id": str(doc["document_id"]), "reason": reason})
            continue
        ids = tokenizer.encode(cleaned).ids
        if not ids or min(ids) < 0 or max(ids) >= tokenizer.vocab_size:
            rejected.append({"document_id": str(doc["document_id"]), "reason": "tokenizer failure or invalid token IDs"})
            continue
        split = stable_split(str(doc["document_id"]), validation_percent)
        for sequence in packers[split].add_document(ids):
            if 0 in sequence:
                raise ValueError("Packed training data unexpectedly contains PAD=0")
            writers[split].write_sequence(sequence)
        accepted += 1
        source_ids.add(str(doc["source_id"]))
    for packer in packers.values():
        packer.finalize()
    train_writer.close()
    validation_writer.close()
    if not train_writer.shards_written or not validation_writer.shards_written:
        raise ValueError("Corpus has no complete train or validation shards; adjust sequence length or supply more data")
    manifest = {
        "format_version": "2.0.0", "preprocessing_version": PREPROCESSING_VERSION,
        "tokenizer": {"vocab_size": tokenizer.vocab_size, "special_token_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3}},
        "sequence_length": seq_len, "validation_percent": validation_percent,
        "split_strategy": "sha256(document_id) modulo 100", "source_ids": sorted(source_ids),
        "accepted_documents": accepted, "rejected_documents": rejected,
        "packing": {split: packer.get_stats() for split, packer in packers.items()},
        "shards": {"train": [p.name for p in train_writer.shards_written], "validation": [p.name for p in validation_writer.shards_written]},
    }
    (root / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
