"""Dataset registry persistence and configuration management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from src.data.hf_discovery import DatasetAuditRecord, RecommendationStatus


class DatasetRegistry:
    """Maintains the canonical registry of audited datasets and exports ingestion policies."""

    def __init__(self, records: Optional[List[DatasetAuditRecord]] = None) -> None:
        self.records: Dict[str, DatasetAuditRecord] = {}
        if records:
            for r in records:
                self.records[r.dataset_id] = r

    def add_record(self, record: DatasetAuditRecord) -> None:
        self.records[record.dataset_id] = record

    def save_json(self, json_path: str | Path = "data/manifests/dataset_registry.json") -> Path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0.0",
            "total_datasets": len(self.records),
            "datasets": [r.to_dict() for r in self.records.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def export_data_sources_yaml(self, yaml_path: str | Path = "configs/data_sources.yaml") -> Path:
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sources = []
        for r in self.records.values():
            entry = {
                "id": r.dataset_id,
                "visibility": r.visibility,
                "category": r.category.value,
                "status": r.recommendation_status.value,
                "enabled": r.recommended,
                "estimated_tokens": r.estimated_tokens,
                "reason": r.reason,
            }
            sources.append(entry)

        content = {
            "version": "1.0",
            "description": "Indian Legal Reasoning Model Data Ingestion Policy",
            "sources": sources,
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(content, f, sort_keys=False, indent=2)
        return path
