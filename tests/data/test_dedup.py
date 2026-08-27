"""Unit tests for document deduplicator."""

from src.data.dedup import DocumentDeduplicator


def test_exact_hash_and_citation_dedup():
    """Verify exact hash match and citation duplicate detection."""
    dedup = DocumentDeduplicator()
    text = "Section 302 IPC. The appellant was convicted of murder."
    citation = "2024 INSC 456"

    # First insertion
    is_dup1, _ = dedup.is_duplicate(text, citation=citation)
    assert is_dup1 is False

    # Second exact insertion
    is_dup2, reason2 = dedup.is_duplicate(text, citation=citation)
    assert is_dup2 is True
    assert "Duplicate citation" in reason2 or "Exact" in reason2


def test_simhash_near_duplicate_detection():
    """Verify SimHash catches near-identical documents with trivial word variations."""
    dedup = DocumentDeduplicator()
    t1 = "In the Supreme Court of India. Criminal Appeal 100 of 2024. The judgment of the trial court is confirmed."
    t2 = "In the Supreme Court of India. Criminal Appeal 100 of 2024. The judgment of the trial court is affirmed."

    is_dup1, _ = dedup.is_duplicate(t1)
    assert is_dup1 is False

    is_dup2, reason2 = dedup.is_duplicate(t2)
    assert is_dup2 is True
    assert "Near-duplicate" in reason2
