"""Legal text normalization and statutory preservation."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


class LegalTextCleaner:
    """Cleans raw legal text while strictly preserving citations, section numbers, dates, and court names."""

    # Preserved citation patterns e.g. (2023) 5 SCC 123, 2024 INSC 123, AIR 1980 SC 1789, § 125, Section 302 IPC
    CITATION_PATTERNS = [
        r"\d{4}\s+INSC\s+\d+",
        r"AIR\s+\d{4}\s+(?:SC|HC|Bom|Del|Mad|Cal|All)\s+\d+",
        r"\(\d{4}\)\s+\d+\s+SCC\s+\d+",
        r"ILR\s+\d{4}\s+\w+\s+\d+",
        r"§\s*\d+[A-Z]?",
        r"Section\s+\d+[A-Za-z]*(?:\s+of\s+the\s+[A-Za-z\s,]+)?",
        r"Article\s+\d+[A-Za-z]*(?:\s+of\s+the\s+Constitution)?",
    ]

    @classmethod
    def clean(cls, text: str) -> str:
        """Apply Unicode normalization and cleanup without altering legal semantic content."""
        if not text or not isinstance(text, str):
            return ""

        # 1. Unicode Normalization (NFC preserves combined characters across Indian Indic scripts)
        text = unicodedata.normalize("NFC", text)

        # 2. Replace non-standard whitespace / control chars (preserve standard newlines)
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n").replace("\v", "\n")
        text = re.sub(r"[ \t]+", " ", text)

        # 3. Strip zero-width and invisible control bytes except Indic zero-width joiners
        text = re.sub(r"[\u200b\u200e\u200f\ufeff]", "", text)

        # 4. Normalize quotes and dashes
        text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        text = text.replace("—", "-").replace("–", "-")

        # 5. Remove excessive repetitive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
