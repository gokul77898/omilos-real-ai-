import json
import pytest
from src.data.ingestion import stable_split, validate_document
from src.data.packer import SequencePacker

def test_special_token_contract_and_no_padding_policy():
    packer = SequencePacker(max_seq_len=8)
    assert packer.bos_id == 2 and packer.eos_id == 3
    with pytest.raises(ValueError, match="BOS=2"):
        SequencePacker(max_seq_len=8, bos_id=1, eos_id=2)
    packer.add_document([10, 11])
    assert packer.finalize() == []
    assert packer.get_stats()["discarded_remainder"] == 4

def test_provenance_validation_and_stable_split():
    doc = {"document_id": "a", "source_id": "court", "source_url": "https://example.test/a", "source_revision": "2026-01",
           "license": "recorded-not-verified", "language": "en", "text": "A legal document with enough original text to pass the minimum quality test and retain its citations."}
    assert validate_document(doc).accepted
    assert stable_split("a") == stable_split("a")
    doc["language"] = "xx"
    assert not validate_document(doc).accepted
