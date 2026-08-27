"""Exact and near-duplicate legal document deduplication."""

from __future__ import annotations

import hashlib
import re
from typing import Dict, Optional, Set, Tuple


class DocumentDeduplicator:
    """Manages exact hash deduplication, citation identifier tracking, and 64-bit SimHash near-duplicate detection."""

    def __init__(self) -> None:
        self.exact_hashes: Set[str] = set()
        self.citation_ids: Set[str] = set()
        self.simhashes: Set[int] = set()

    @staticmethod
    def compute_sha256(text: str) -> str:
        """Compute SHA-256 hash over normalized whitespace-collapsed text."""
        norm = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_simhash(text: str) -> int:
        """Compute 64-bit SimHash fingerprint over 3-gram word shingles."""
        words = re.findall(r"\w+", text.lower())
        if len(words) < 3:
            return 0

        v = [0] * 64
        for i in range(len(words) - 2):
            shingle = f"{words[i]}_{words[i+1]}_{words[i+2]}"
            h = int(hashlib.md5(shingle.encode("utf-8")).hexdigest()[:16], 16)
            for bit in range(64):
                if (h >> bit) & 1:
                    v[bit] += 1
                else:
                    v[bit] -= 1

        fingerprint = 0
        for bit in range(64):
            if v[bit] > 0:
                fingerprint |= (1 << bit)
        return fingerprint

    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        """Compute Hamming distance between two 64-bit fingerprints."""
        x = h1 ^ h2
        dist = 0
        while x > 0:
            dist += x & 1
            x >>= 1
        return dist

    def is_duplicate(self, text: str, citation: Optional[str] = None, max_hamming_dist: int = 8) -> Tuple[bool, str]:
        """Check if document is an exact or near-duplicate.

        Returns:
            Tuple of (is_dup, reason)
        """
        # 1. Citation / CNR Deduplication
        if citation and citation.strip():
            cit_norm = citation.lower().strip()
            if cit_norm in self.citation_ids:
                return True, f"Duplicate citation ID: {citation}"
            self.citation_ids.add(cit_norm)

        # 2. Exact Normalized Hash Deduplication
        text_hash = self.compute_sha256(text)
        if text_hash in self.exact_hashes:
            return True, "Exact text hash match"
        self.exact_hashes.add(text_hash)

        # 3. Near-Duplicate SimHash Detection
        sh = self.compute_simhash(text)
        if sh != 0:
            for existing_sh in self.simhashes:
                if self.hamming_distance(sh, existing_sh) <= max_hamming_dist:
                    return True, "Near-duplicate SimHash match"
            self.simhashes.add(sh)

        return False, "Unique"
