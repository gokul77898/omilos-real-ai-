"""Unit tests for legal text cleaner and citation preservation."""

from src.data.cleaner import LegalTextCleaner


def test_cleaner_preserves_citations_and_sections():
    """Verify cleaner normalizes whitespace while strictly preserving citations, dates, and section numbers."""
    raw = "  Section  302   of the Indian Penal Code\r\n(2023)  5 SCC 123.  AIR 1980 SC 1789.   § 125 CrPC. "
    cleaned = LegalTextCleaner.clean(raw)
    assert "Section 302 of the Indian Penal Code" in cleaned
    assert "(2023) 5 SCC 123" in cleaned
    assert "AIR 1980 SC 1789" in cleaned
    assert "§ 125" in cleaned


def test_cleaner_indic_unicode_normalization():
    """Verify Unicode NFC normalization across Indic scripts."""
    hindi_raw = "भारतीय  न्याय\tसंहिता, 2023"
    cleaned = LegalTextCleaner.clean(hindi_raw)
    assert cleaned == "भारतीय न्याय संहिता, 2023"
