"""Unit tests for 25 High Court classifier and legal domain categorization."""

from src.data.court_classifier import ALL_25_HIGH_COURTS, CourtClassifier


def test_all_25_high_courts_list_count():
    """Verify all 25 High Courts are explicitly accounted for."""
    assert len(ALL_25_HIGH_COURTS) == 25
    assert "Bombay" in ALL_25_HIGH_COURTS
    assert "Calcutta" in ALL_25_HIGH_COURTS
    assert "Delhi" in ALL_25_HIGH_COURTS
    assert "Madras" in ALL_25_HIGH_COURTS
    assert "Karnataka" in ALL_25_HIGH_COURTS


def test_court_classification_accuracy():
    """Verify court level and High Court jurisdiction detection."""
    text1 = "In the Supreme Court of India, Criminal Appellate Jurisdiction. 2024 INSC 123."
    court, hc, domain = CourtClassifier.classify(text1)
    assert court == "Supreme Court"
    assert hc is None
    assert domain == "Criminal"

    text2 = "In the High Court of Judicature at Bombay, Nagpur Bench. Criminal Appeal."
    court, hc, domain = CourtClassifier.classify(text2)
    assert court == "High Courts"
    assert hc == "Bombay"
    assert domain == "Criminal"
