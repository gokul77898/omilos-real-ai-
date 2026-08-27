"""Unit tests for dataset legal categorization and duplicate flagging."""

import pytest
from src.data.hf_discovery import DatasetCategory, HFDatasetDiscovery, RecommendationStatus


def test_dataset_category_inference():
    """Verify keyword and tag based legal category inference."""
    cat = HFDatasetDiscovery.classify_dataset("123/Constitution-India", ["constitution", "articles"])
    assert cat == DatasetCategory.CONSTITUTIONAL

    cat = HFDatasetDiscovery.classify_dataset("navaneeth/BNS_detailed", ["bns", "acts"])
    assert cat == DatasetCategory.LEGISLATION

    cat = HFDatasetDiscovery.classify_dataset("debkanchan/supreme-court-judgements", ["court", "judgments"])
    assert cat == DatasetCategory.CASE_LAW


def test_duplicate_candidate_exclusion():
    """Verify that duplicate/chunked variants are marked as EXCLUDE."""
    record = HFDatasetDiscovery.audit_candidate(
        dataset_id="vihaannnn/Chunked-Indian-Supreme-Court-Judgements",
        visibility="public",
        raw_license="mit",
        tags=["chunked", "supreme-court"],
        duplicate_of="debkanchan/supreme-court-of-india-judgements",
    )
    assert record.recommendation_status == RecommendationStatus.EXCLUDE
    assert record.recommended is False
    assert "Duplicate" in record.reason
