"""Corpus generation orchestrator and 25 High Court coverage tracker."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np

from src.data.cleaner import LegalTextCleaner
from src.data.court_classifier import ALL_25_HIGH_COURTS, CourtClassifier
from src.data.dedup import DocumentDeduplicator
from src.data.packer import SequencePacker
from src.data.sharding import ShardWriter
from src.tokenizer import LegalTokenizer


class CorpusBuilder:
    """Builds and audits the complete Indian Legal Pretraining Corpus."""

    def __init__(self, tokenizer_path: str = "artifacts/tokenizer/tokenizer.json") -> None:
        p = Path(tokenizer_path)
        if p.is_file():
            self.tokenizer = LegalTokenizer.load(p.parent)
        elif p.is_dir() and (p / "tokenizer.json").exists():
            self.tokenizer = LegalTokenizer.load(p)
        else:
            self.tokenizer = LegalTokenizer()
        self.cleaner = LegalTextCleaner()
        self.classifier = CourtClassifier()
        self.dedup = DocumentDeduplicator()

        # 25 High Court statistics
        self.hc_coverage: Dict[str, Dict[str, Any]] = {
            hc: {
                "court": hc,
                "document_count": 0,
                "usable_document_count": 0,
                "estimated_tokens": 0,
                "languages": ["en", "hi"],
                "duplicate_count": 0,
                "status": "ACTIVE_SOURCE",
            }
            for hc in ALL_25_HIGH_COURTS
        }

        # Jurisdiction summaries
        self.court_counts = {
            "Supreme Court": {"docs": 0, "tokens": 0},
            "High Courts": {"docs": 0, "tokens": 0},
            "District Courts": {"docs": 0, "tokens": 0},
            "Tribunals": {"docs": 0, "tokens": 0},
            "Legislation / Constitution": {"docs": 0, "tokens": 0},
        }

        # Language distribution
        self.language_tokens = {
            "English": 0, "Hindi": 0, "Kannada": 0, "Tamil": 0,
            "Telugu": 0, "Malayalam": 0, "Bengali": 0, "Marathi": 0,
            "Gujarati": 0, "Punjabi": 0, "Urdu": 0, "Sanskrit": 0,
        }

        # Domain distribution
        self.domain_tokens = {
            "Constitutional": 0, "Criminal": 0, "Civil": 0, "Corporate / Commercial": 0,
            "Tax": 0, "Intellectual Property": 0, "Arbitration": 0, "Special Acts": 0, "General": 0,
        }

        self.exact_dups_removed = 0
        self.near_dups_removed = 0
        self.total_raw_docs = 0
        self.total_unique_docs = 0
        self.total_unique_tokens = 0

    def process_and_shard_corpus(
        self,
        raw_documents: List[Dict[str, Any]],
        train_dir: str = "data/tokenized/train",
        val_dir: str = "data/tokenized/validation",
        train_split_ratio: float = 0.95,
    ) -> Dict[str, Any]:
        """Clean, classify, deduplicate, split (95/5), pack into 2048-token sequences, and write memory-mapped shards."""
        train_packer = SequencePacker(max_seq_len=2048, bos_id=1, eos_id=2)
        val_packer = SequencePacker(max_seq_len=2048, bos_id=1, eos_id=2)

        train_writer = ShardWriter(train_dir, shard_prefix="train_shard", max_seqs_per_shard=500)
        val_writer = ShardWriter(val_dir, shard_prefix="val_shard", max_seqs_per_shard=500)

        np.random.seed(42)

        for doc in raw_documents:
            self.total_raw_docs += 1
            raw_text = doc.get("text", "")
            citation = doc.get("citation", None)
            lang = doc.get("language", "English")

            # 1. Clean
            cleaned = self.cleaner.clean(raw_text)
            if not cleaned or len(cleaned) < 10:
                continue

            # 2. Deduplicate
            is_dup, reason = self.dedup.is_duplicate(cleaned, citation=citation)
            if is_dup:
                if "Exact" in reason or "citation" in reason:
                    self.exact_dups_removed += 1
                else:
                    self.near_dups_removed += 1
                continue

            self.total_unique_docs += 1

            # 3. Classify Court & Domain
            court_level, hc_name, domain = self.classifier.classify(cleaned, doc.get("metadata"))

            # 4. Tokenize
            encoded = self.tokenizer.encode(cleaned)
            token_ids = encoded.ids
            num_tokens = len(token_ids)
            self.total_unique_tokens += num_tokens

            # Update stats
            if court_level == "Supreme Court":
                self.court_counts["Supreme Court"]["docs"] += 1
                self.court_counts["Supreme Court"]["tokens"] += num_tokens
            elif court_level == "Tribunals":
                self.court_counts["Tribunals"]["docs"] += 1
                self.court_counts["Tribunals"]["tokens"] += num_tokens
            elif hc_name and hc_name in self.hc_coverage:
                self.hc_coverage[hc_name]["document_count"] += 1
                self.hc_coverage[hc_name]["usable_document_count"] += 1
                self.hc_coverage[hc_name]["estimated_tokens"] += num_tokens
                self.court_counts["High Courts"]["docs"] += 1
                self.court_counts["High Courts"]["tokens"] += num_tokens
            else:
                self.court_counts["Legislation / Constitution"]["docs"] += 1
                self.court_counts["Legislation / Constitution"]["tokens"] += num_tokens

            self.language_tokens[lang] = self.language_tokens.get(lang, 0) + num_tokens
            self.domain_tokens[domain] = self.domain_tokens.get(domain, 0) + num_tokens

            # 5. Split 95% Train / 5% Validation at document level
            is_train = (np.random.rand() < train_split_ratio)
            if is_train:
                seqs = train_packer.add_document(token_ids)
                for s in seqs:
                    train_writer.write_sequence(s)
            else:
                seqs = val_packer.add_document(token_ids)
                for s in seqs:
                    val_writer.write_sequence(s)

        train_writer.close()
        val_writer.close()

        train_stats = train_packer.get_stats()
        val_stats = val_packer.get_stats()

        # Build manifest
        manifest = {
            "version": "1.0.0",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tokenizer": "artifacts/tokenizer/tokenizer.json",
            "sequence_length": 2048,
            "raw_documents": self.total_raw_docs,
            "unique_documents": self.total_unique_docs,
            "exact_duplicates_removed": self.exact_dups_removed,
            "near_duplicates_removed": self.near_dups_removed,
            "total_unique_tokens": self.total_unique_tokens,
            "target_tokens": 10_000_000_000,
            "token_deficit": max(0, 10_000_000_000 - self.total_unique_tokens),
            "train_tokens": train_stats["total_packed_tokens"],
            "validation_tokens": val_stats["total_packed_tokens"],
            "train_shards": len(train_writer.shards_written),
            "validation_shards": len(val_writer.shards_written),
            "court_counts": self.court_counts,
            "high_court_coverage": self.hc_coverage,
            "language_tokens": self.language_tokens,
            "domain_tokens": self.domain_tokens,
        }

        with open("data/manifests/final_corpus.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest
