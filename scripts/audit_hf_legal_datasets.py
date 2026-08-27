#!/usr/bin/env python3
"""Comprehensive Hugging Face Indian Legal Dataset Discovery and Audit Script."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hf_discovery import DatasetCategory, HFDatasetDiscovery
from src.data.registry import DatasetRegistry


# Master list of candidate and private datasets to audit
CANDIDATES = [
    # 1. Private Datasets (Authoritative In-House Sources)
    {
        "dataset_id": "OmilosAISolutions/nyayamitra-training-data",
        "visibility": "private",
        "raw_license": "proprietary",
        "tags": ["indian-law", "acts", "supreme-court", "high-court", "bns", "legal-reasoning"],
        "description": "Authoritative multilingual Indian legal corpus with BNS, BNSS, BSA, Constitution, Supreme Court, and High Court judgments.",
        "estimated_rows": 150_000,
        "estimated_tokens": 120_000_000,
        "text_fields": ["content", "text"],
        "languages": ["en", "hi", "kn", "ta", "te", "ml", "mr", "bn", "gu", "pa", "ur", "sa"],
        "legal_coverage": ["Supreme Court", "High Courts", "BNS", "BNSS", "BSA", "Constitution", "Special Acts"],
    },
    {
        "dataset_id": "OmilosAISolutions/vakeels-legal-training-data",
        "visibility": "private",
        "raw_license": "proprietary",
        "tags": ["case-laws", "advocate-arguments", "legal-qa", "statute-interpretation"],
        "description": "Curated Indian case law transcripts, advocate argument formulations, and statutory reasoning pairs.",
        "estimated_rows": 85_000,
        "estimated_tokens": 75_000_000,
        "text_fields": ["judgment_text", "arguments"],
        "languages": ["en", "hi"],
        "legal_coverage": ["Supreme Court", "High Courts", "Civil & Criminal Procedure"],
    },
    {
        "dataset_id": "OmilosAISolutions/highcourt-2023-full-metadata",
        "visibility": "private",
        "raw_license": "proprietary",
        "tags": ["highcourt", "metadata", "citations", "orders"],
        "description": "Complete metadata, citation graph, and structured case summaries for 2023 High Court proceedings across India.",
        "estimated_rows": 220_000,
        "estimated_tokens": 45_000_000,
        "text_fields": ["order_text", "summary", "citations"],
        "languages": ["en", "hi", "ta", "mr", "bn"],
        "legal_coverage": ["25 Indian High Courts", "Orders", "Notifications"],
    },

    # 2. Public Candidates
    {
        "dataset_id": "vaquill/open-india-law",
        "visibility": "public",
        "raw_license": "apache-2.0",
        "tags": ["indian-law", "acts", "supreme-court"],
        "description": "Open collection of Indian Central Acts, State Acts, and Supreme Court rulings with structured section indices.",
        "estimated_rows": 45_000,
        "estimated_tokens": 50_000_000,
        "text_fields": ["act_text", "section_content"],
        "languages": ["en"],
        "legal_coverage": ["Central Acts", "State Acts", "Supreme Court"],
    },
    {
        "dataset_id": "LH2-data-labs/indian-legal-records",
        "visibility": "public",
        "raw_license": "cc-by-4.0",
        "tags": ["legal-records", "judgments", "tribunals"],
        "description": "Comprehensive digitized legal records and tribunal proceedings from Indian judicial archives.",
        "estimated_rows": 60_000,
        "estimated_tokens": 65_000_000,
        "text_fields": ["record_text", "tribunal_order"],
        "languages": ["en"],
        "legal_coverage": ["NCLT", "ITAT", "CAT", "High Courts"],
    },
    {
        "dataset_id": "KanoonGPT/indian-case-laws",
        "visibility": "public",
        "raw_license": "mit",
        "tags": ["case-laws", "judgments", "indian-kanoon"],
        "description": "Cleaned full-text Supreme Court and High Court judgments extracted from public Indian Kanoon repository.",
        "estimated_rows": 110_000,
        "estimated_tokens": 95_000_000,
        "text_fields": ["judgment", "text"],
        "languages": ["en"],
        "legal_coverage": ["Supreme Court", "High Courts"],
    },
    {
        "dataset_id": "Sumitedu/indian-case-laws",
        "visibility": "public",
        "raw_license": "mit",
        "tags": ["case-laws", "judgments"],
        "description": "Indian case laws with citation indices and headnotes.",
        "estimated_rows": 30_000,
        "estimated_tokens": 28_000_000,
        "text_fields": ["text"],
        "languages": ["en"],
        "legal_coverage": ["Supreme Court", "High Courts"],
    },
    {
        "dataset_id": "debkanchan/supreme-court-of-india-judgements",
        "visibility": "public",
        "raw_license": "apache-2.0",
        "tags": ["supreme-court", "judgments", "sci"],
        "description": "Chronological Supreme Court of India judgments from 1950 to 2021.",
        "estimated_rows": 40_000,
        "estimated_tokens": 80_000_000,
        "text_fields": ["text", "judgment_content"],
        "languages": ["en"],
        "legal_coverage": ["Supreme Court of India (1950-2021)"],
    },
    {
        "dataset_id": "123Divyansh/Constitution.-of-India-TXT_File",
        "visibility": "public",
        "raw_license": "cc0-1.0",
        "tags": ["constitution", "fundamental-rights", "articles"],
        "description": "Full text of the Constitution of India including all Articles, Schedules, and Amendments.",
        "estimated_rows": 500,
        "estimated_tokens": 450_000,
        "text_fields": ["text"],
        "languages": ["en", "hi"],
        "legal_coverage": ["Constitution of India", "Articles 1-395", "Amendments"],
    },
    {
        "dataset_id": "SnehaDeshmukh/IndianBailJudgments-1200",
        "visibility": "public",
        "raw_license": "mit",
        "tags": ["bail", "criminal-law", "crpc", "bns"],
        "description": "1,200 curated Indian High Court and Supreme Court bail orders and reasoning judgments.",
        "estimated_rows": 1_200,
        "estimated_tokens": 3_500_000,
        "text_fields": ["judgment_text", "bail_reasoning"],
        "languages": ["en"],
        "legal_coverage": ["Bail Orders", "Section 437/439 CrPC", "High Courts"],
    },
    {
        "dataset_id": "navaneeth005/BNS_definitions",
        "visibility": "public",
        "raw_license": "mit",
        "tags": ["bns", "definitions", "criminal-law", "statute"],
        "description": "Bharatiya Nyaya Sanhita (BNS) 2023 definitions, section mappings, and IPC cross-references.",
        "estimated_rows": 800,
        "estimated_tokens": 600_000,
        "text_fields": ["definition", "section_text"],
        "languages": ["en", "hi"],
        "legal_coverage": ["BNS 2023", "IPC Mapping"],
    },
    {
        "dataset_id": "navaneeth005/BNS_detailed",
        "visibility": "public",
        "raw_license": "mit",
        "tags": ["bns", "detailed", "acts", "clauses"],
        "description": "Detailed statutory provisions of Bharatiya Nyaya Sanhita, Bharatiya Nagarik Suraksha Sanhita, and Bharatiya Sakshya Adhiniyam.",
        "estimated_rows": 2_500,
        "estimated_tokens": 2_200_000,
        "text_fields": ["clause_text", "explanation"],
        "languages": ["en", "hi"],
        "legal_coverage": ["BNS", "BNSS", "BSA (2023 Criminal Laws)"],
    },
    {
        "dataset_id": "navaneeth005/special_acts",
        "visibility": "public",
        "raw_license": "mit",
        "tags": ["special-acts", "pocso", "ndps", "uapa", "pmla"],
        "description": "Indian Special Acts including NDPS, POCSO, PMLA, UAPA, and Motor Vehicles Act provisions.",
        "estimated_rows": 3_200,
        "estimated_tokens": 3_000_000,
        "text_fields": ["act_text", "provisions"],
        "languages": ["en"],
        "legal_coverage": ["NDPS Act", "POCSO Act", "PMLA", "UAPA", "Negotiable Instruments Act"],
    },
    {
        "dataset_id": "Shreyasrao/Indian-law-supreme-court-judgements-2016",
        "visibility": "public",
        "raw_license": "apache-2.0",
        "tags": ["supreme-court", "2016"],
        "description": "Supreme Court of India rulings from the calendar year 2016.",
        "estimated_rows": 1_800,
        "estimated_tokens": 4_200_000,
        "text_fields": ["text"],
        "languages": ["en"],
        "legal_coverage": ["Supreme Court (2016)"],
    },

    # 3. Chunked / Duplicate Candidates to Audit for Overlap
    {
        "dataset_id": "vihaannnn/Chunked-Indian-Supreme-Court-Judgements",
        "visibility": "public",
        "raw_license": "mit",
        "tags": ["chunked", "supreme-court"],
        "description": "Pre-chunked version of Supreme Court judgments.",
        "estimated_rows": 80_000,
        "estimated_tokens": 40_000_000,
        "text_fields": ["chunk"],
        "languages": ["en"],
        "legal_coverage": ["Supreme Court"],
        "duplicate_of": "debkanchan/supreme-court-of-india-judgements",
    },
    {
        "dataset_id": "vihaannnn/Indian-Supreme-Court-Judgements-Chunked",
        "visibility": "public",
        "raw_license": "mit",
        "tags": ["chunked", "supreme-court"],
        "description": "Secondary pre-chunked clone of Supreme Court judgments.",
        "estimated_rows": 75_000,
        "estimated_tokens": 38_000_000,
        "text_fields": ["chunk"],
        "languages": ["en"],
        "legal_coverage": ["Supreme Court"],
        "duplicate_of": "debkanchan/supreme-court-of-india-judgements",
    },
]


def main() -> None:
    registry = DatasetRegistry()

    print("=" * 70)
    print("HUGGING FACE INDIAN LEGAL DATASET DISCOVERY & AUDIT")
    print("Target Model: ~482.8M Indian Legal Foundation Model (Omilos Own AI)")
    print("=" * 70)

    total_tokens_recommended = 0
    total_tokens_review = 0

    for cand in CANDIDATES:
        record = HFDatasetDiscovery.audit_candidate(
            dataset_id=cand["dataset_id"],
            visibility=cand.get("visibility", "public"),
            raw_license=cand.get("raw_license", "unknown"),
            tags=cand.get("tags", []),
            description=cand.get("description", ""),
            estimated_rows=cand.get("estimated_rows", 0),
            estimated_tokens=cand.get("estimated_tokens", 0),
            text_fields=cand.get("text_fields", ["text"]),
            languages=cand.get("languages", ["en"]),
            legal_coverage=cand.get("legal_coverage", []),
            duplicate_of=cand.get("duplicate_of", None),
        )
        registry.add_record(record)

        if record.recommendation_status.value == "INCLUDE":
            total_tokens_recommended += record.estimated_tokens
        elif record.recommendation_status.value == "REVIEW":
            total_tokens_review += record.estimated_tokens

        # Format output card
        print(f"\nDataset:          {record.dataset_id}")
        print(f"Type:             {record.category.value}")
        print(f"Public/Private:   {record.visibility}")
        print(f"License:          {record.license} ({record.license_status.value})")
        print(f"Provenance:       {record.provenance} ({record.provenance_status.value})")
        print(f"Languages:        {', '.join(record.languages)}")
        print(f"Rows:             {record.estimated_rows:,}")
        print(f"Estimated Tokens: {record.estimated_tokens:,} (~{record.estimated_tokens / 1e6:.1f}M)")
        print(f"Text Fields:      {', '.join(record.text_fields)}")
        print(f"Legal Coverage:   {', '.join(record.legal_coverage)}")
        print(f"Quality Score:    {record.quality_score:.1f} / 40.0")
        print(f"Recommendation:   {record.recommendation_status.value}")
        print(f"Reason:           {record.reason}")
        print("-" * 70)

    # Save artifacts
    json_path = registry.save_json(PROJECT_ROOT / "data" / "manifests" / "dataset_registry.json")
    yaml_path = registry.export_data_sources_yaml(PROJECT_ROOT / "configs" / "data_sources.yaml")

    include_count = sum(1 for r in registry.records.values() if r.recommendation_status.value == "INCLUDE")
    review_count = sum(1 for r in registry.records.values() if r.recommendation_status.value == "REVIEW")
    exclude_count = sum(1 for r in registry.records.values() if r.recommendation_status.value == "EXCLUDE")

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY & REGISTRY EXPORT")
    print("=" * 70)
    print(f"Total Datasets Audited:      {len(registry.records)}")
    print(f"Recommended for Inclusion:   {include_count}")
    print(f"Requiring Deeper Review:     {review_count}")
    print(f"Excluded (Duplicates/NC):    {exclude_count}")
    print(f"Recommended Token Volume:    {total_tokens_recommended:,} (~{total_tokens_recommended / 1e6:.1f}M tokens)")
    print(f"Review Pool Token Volume:    {total_tokens_review:,} (~{total_tokens_review / 1e6:.1f}M tokens)")
    print(f"Master Registry JSON:        {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Data Sources YAML:           {yaml_path.relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
