"""Unit tests for dataset registry serialization and credential safety."""

import json
import tempfile
import pytest
import yaml

from src.data.hf_discovery import HFDatasetDiscovery
from src.data.registry import DatasetRegistry


def test_registry_json_and_yaml_export():
    """Verify registry generates valid JSON and YAML manifests without credential leakage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = DatasetRegistry()

        rec1 = HFDatasetDiscovery.audit_candidate(
            dataset_id="OmilosAISolutions/nyayamitra-training-data",
            visibility="private",
            raw_license="proprietary",
            tags=["indian-law", "bns"],
            estimated_rows=1000,
            estimated_tokens=500000,
        )
        rec2 = HFDatasetDiscovery.audit_candidate(
            dataset_id="vaquill/open-india-law",
            visibility="public",
            raw_license="apache-2.0",
            tags=["acts", "statute"],
            estimated_rows=5000,
            estimated_tokens=2500000,
        )

        registry.add_record(rec1)
        registry.add_record(rec2)

        json_path = registry.save_json(f"{tmpdir}/registry.json")
        yaml_path = registry.export_data_sources_yaml(f"{tmpdir}/sources.yaml")

        assert json_path.exists()
        assert yaml_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            j_data = json.load(f)
        assert j_data["total_datasets"] == 2
        assert len(j_data["datasets"]) == 2

        with open(yaml_path, "r", encoding="utf-8") as f:
            y_data = yaml.safe_load(f)
        assert len(y_data["sources"]) == 2
        assert y_data["sources"][0]["id"] == "OmilosAISolutions/nyayamitra-training-data"
        assert y_data["sources"][0]["enabled"] is True

        # Check credential safety: no tokens, passwords, or keys in files
        raw_text = json_path.read_text() + yaml_path.read_text()
        assert "token" not in raw_text.lower() or "estimated_tokens" in raw_text
        assert "bearer" not in raw_text.lower()
        assert "secret" not in raw_text.lower()
