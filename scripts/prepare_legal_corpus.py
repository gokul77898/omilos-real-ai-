#!/usr/bin/env python3

"""
Stream open Indian legal datasets from Hugging Face and produce a
provenance-bearing JSONL manifest compatible with src.data.ingestion.

NO gated datasets are used.

Sources:
  - overthelex/indian-court-decisions (supreme_court)
  - overthelex/indian-court-decisions (high_courts)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import orjson
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm


# ============================================================
# DATASET DEFINITIONS
# ============================================================

SOURCES = [
    {
        "dataset_id": "overthelex/indian-court-decisions",
        "config": "supreme_court",
        "text_field": "full_text",
        "source_id": "overthelex/indian-court-decisions/supreme_court",
        "license": "CC-BY-4.0",
        "language": "en",
    },
    {
        "dataset_id": "overthelex/indian-court-decisions",
        "config": "high_courts",
        "text_field": "full_text",
        "source_id": "overthelex/indian-court-decisions/high_courts",
        "license": "CC-BY-4.0",
        "language": "en",
    },
]

# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize Unicode whitespace.
    text = re.sub(r"[ \t\f\v]+", " ", text)

    # Remove trailing whitespace.
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Collapse pathological blank-line runs.
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


# ============================================================
# QUALITY FILTER
# ============================================================

def quality_ok(text: str) -> bool:
    if len(text) < 500:
        return False

    # Unicode replacement character usually indicates damaged extraction.
    replacement_ratio = text.count("\ufffd") / max(len(text), 1)

    if replacement_ratio > 0.01:
        return False

    # Require meaningful alphabetic content.
    alphabetic = sum(ch.isalpha() for ch in text)

    if alphabetic < 200:
        return False

    # Reject pathological repeated characters.
    if re.search(r"(.)\1{30,}", text):
        return False

    # Reject documents where one line is repeated excessively.
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if lines:
        counts = {}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1

        if max(counts.values()) > 10:
            return False

    return True


# ============================================================
# DOCUMENT ID
# ============================================================

def get_document_id(row: dict[str, Any], source: dict[str, Any]) -> str:
    candidates = [
        row.get("case_metadata_id"),
        row.get("cnr"),
        row.get("case_id"),
        row.get("parser_record_id"),
        row.get("id"),
    ]

    for value in candidates:
        if value is not None and str(value).strip():
            return f"{source['source_id']}::{value}"

    # Last-resort deterministic ID.
    raw = json.dumps(
        row,
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    return f"{source['source_id']}::row-{digest}"


# ============================================================
# SOURCE REVISION
# ============================================================

def get_revision(dataset_id: str) -> str:
    """
    Resolve the currently available Hub revision.

    If the Hub API cannot provide a revision, return "main".
    """

    try:
        api = HfApi()

        info = api.repo_info(
            repo_id=dataset_id,
            repo_type="dataset",
        )

        sha = getattr(info, "sha", None)

        if sha:
            return str(sha)

    except Exception as exc:
        print(
            f"WARNING: could not resolve revision for "
            f"{dataset_id}: {exc}"
        )

    return "main"


# ============================================================
# SOURCE URL
# ============================================================

def source_url(dataset_id: str) -> str:
    return (
        "https://huggingface.co/datasets/"
        + dataset_id
    )


# ============================================================
# STREAM ONE DATASET
# ============================================================

def process_source(
    source: dict[str, Any],
    output_handle,
    seen_hashes: set[str],
    max_documents: int | None,
) -> dict[str, Any]:

    dataset_id = source["dataset_id"]
    config = source["config"]

    print()
    print("=" * 80)
    print("DATASET:", dataset_id)
    print("CONFIG:", config or "default")
    print("=" * 80)

    revision = get_revision(dataset_id)

    print("REVISION:", revision)

    ds = load_dataset(
        dataset_id,
        config,
        split="train",
        streaming=True,
    )

    accepted = 0
    rejected = 0
    duplicates = 0
    scanned = 0

    for row in tqdm(
        ds,
        desc=source["source_id"],
    ):
        scanned += 1

        if (
            max_documents is not None
            and accepted >= max_documents
        ):
            break

        raw_text = row.get(
            source["text_field"],
            "",
        )

        text = normalize_text(raw_text)

        if not quality_ok(text):
            rejected += 1
            continue

        # ----------------------------------------------------
        # Content-level deduplication.
        #
        # This catches identical judgments even if they have
        # different source IDs.
        # ----------------------------------------------------

        content_sha256 = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        if content_sha256 in seen_hashes:
            duplicates += 1
            continue

        seen_hashes.add(content_sha256)

        document_id = get_document_id(
            row,
            source,
        )

        record = {
            "document_id": document_id,

            "source_id": source["source_id"],

            "source_url": source_url(
                dataset_id
            ),

            "source_revision": revision,

            "license": source["license"],

            "language": source["language"],

            "text": text,

            # Extra provenance fields.
            "content_sha256": content_sha256,

            "source_config": config,

            "source_cnr": (
                str(row["cnr"])
                if row.get("cnr") is not None
                else None
            ),

            "title": (
                str(
                    row.get("case_title")
                    or row.get("title")
                )
                if (
                    row.get("case_title")
                    or row.get("title")
                )
                else None
            ),

            "court": (
                str(
                    row.get("court_name")
                    or row.get("court_name_normalized")
                )
                if (
                    row.get("court_name")
                    or row.get("court_name_normalized")
                )
                else None
            ),

            "decision_date": (
                str(row["decision_date"])
                if row.get("decision_date") is not None
                else None
            ),
        }

        output_handle.write(
            orjson.dumps(record)
        )
        output_handle.write(b"\n")

        accepted += 1

    return {
        "dataset_id": dataset_id,
        "config": config,
        "revision": revision,
        "scanned": scanned,
        "accepted": accepted,
        "rejected": rejected,
        "duplicates": duplicates,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="data/legal_clean/open_indian_legal.jsonl",
    )

    parser.add_argument(
        "--max-documents-per-source",
        type=int,
        default=None,
        help="Optional limit for testing. Omit for full corpus.",
    )

    args = parser.parse_args()

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    seen_hashes: set[str] = set()

    statistics = []

    print("=" * 80)
    print("OPEN INDIAN LEGAL CORPUS PREPARATION")
    print("=" * 80)
    print("GATED DATASETS: NONE")
    print()

    with output.open("wb") as handle:

        for source in SOURCES:

            stats = process_source(
                source,
                handle,
                seen_hashes,
                args.max_documents_per_source,
            )

            statistics.append(stats)

            print(
                json.dumps(
                    stats,
                    indent=2,
                )
            )

    manifest = {
        "corpus_name": "omilos-indian-legal-pretrain-v1",

        "sources": SOURCES,

        "statistics": statistics,

        "unique_documents": len(seen_hashes),

        "output": str(output),

        "deduplication": {
            "method": "sha256(normalized_text)",
            "scope": "all sources",
        },

        "quality_filter": {
            "minimum_characters": 500,
            "minimum_alphabetic_characters": 200,
            "replacement_character_ratio_max": 0.01,
            "maximum_repeated_line_count": 10,
        },

        "gated_datasets_used": [],

        "license_policy": {
            "Overthelex": "CC-BY-4.0",
        },
    }

    manifest_path = (
        output.parent / "source_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("SOURCE INGESTION COMPLETE")
    print("=" * 80)
    print(
        "Unique documents:",
        len(seen_hashes),
    )
    print(
        "JSONL:",
        output,
    )
    print(
        "Manifest:",
        manifest_path,
    )


if __name__ == "__main__":
    main()
