"""Hugging Face dataset discovery, classification, metadata audit, and scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple


class DatasetCategory(str, Enum):
    PRIMARY_LEGAL_TEXT = "PRIMARY_LEGAL_TEXT"
    CASE_LAW = "CASE_LAW"
    LEGISLATION = "LEGISLATION"
    CONSTITUTIONAL = "CONSTITUTIONAL"
    TRIBUNAL = "TRIBUNAL"
    LEGAL_METADATA = "LEGAL_METADATA"
    LEGAL_QA = "LEGAL_QA"
    LEGAL_REASONING = "LEGAL_REASONING"
    SYNTHETIC_LEGAL = "SYNTHETIC_LEGAL"
    BENCHMARK = "BENCHMARK"
    NON_LEGAL = "NON_LEGAL"
    DUPLICATE_CANDIDATE = "DUPLICATE_CANDIDATE"
    UNKNOWN = "UNKNOWN"


class LicenseStatus(str, Enum):
    CLEAR = "CLEAR"                   # Permissive / Open / Government Public Domain
    REVIEW_REQUIRED = "REVIEW_REQUIRED" # Custom / Non-commercial / Undefined specifics
    RESTRICTED = "RESTRICTED"         # Explicitly prohibits training or redistribution
    UNKNOWN = "UNKNOWN"               # Missing license metadata


class ProvenanceStatus(str, Enum):
    VERIFIED = "VERIFIED"             # Official Court / Government / Certified portal source
    COMMUNITY_SCRAPED = "COMMUNITY_SCRAPED" # Indian Kanoon / Web scrape without official signature
    SYNTHETIC = "SYNTHETIC"           # AI / LLM generated
    UNKNOWN = "UNKNOWN"               # Unstated origin


class RecommendationStatus(str, Enum):
    INCLUDE = "INCLUDE"
    REVIEW = "REVIEW"
    EXCLUDE = "EXCLUDE"


@dataclass
class DatasetQualityScore:
    source_authority: float = 0.0      # 0 - 5
    legal_relevance: float = 0.0       # 0 - 5
    text_quality: float = 0.0          # 0 - 5
    coverage: float = 0.0              # 0 - 5
    provenance: float = 0.0            # 0 - 5
    license_clarity: float = 0.0       # 0 - 5
    metadata_quality: float = 0.0      # 0 - 5
    duplication_safety: float = 0.0    # 0 - 5

    @property
    def total_score(self) -> float:
        return (
            self.source_authority
            + self.legal_relevance
            + self.text_quality
            + self.coverage
            + self.provenance
            + self.license_clarity
            + self.metadata_quality
            + self.duplication_safety
        )


@dataclass
class DatasetAuditRecord:
    dataset_id: str
    visibility: str = "public"          # "public" or "private"
    category: DatasetCategory = DatasetCategory.UNKNOWN
    license: str = "unknown"
    license_status: LicenseStatus = LicenseStatus.UNKNOWN
    provenance: str = "Unknown"
    provenance_status: ProvenanceStatus = ProvenanceStatus.UNKNOWN
    recommended: bool = False
    recommendation_status: RecommendationStatus = RecommendationStatus.REVIEW
    reason: str = ""
    languages: List[str] = field(default_factory=lambda: ["en"])
    legal_coverage: List[str] = field(default_factory=list)
    text_fields: List[str] = field(default_factory=list)
    estimated_rows: int = 0
    estimated_tokens: int = 0
    quality_score: float = 0.0
    quality_breakdown: Optional[Dict[str, float]] = None
    duplicate_of: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["license_status"] = self.license_status.value
        d["provenance_status"] = self.provenance_status.value
        d["recommendation_status"] = self.recommendation_status.value
        return d


class HFDatasetDiscovery:
    """Discovery and auditing engine for Indian legal datasets on Hugging Face."""

    KNOWN_LEGAL_KEYWORDS = [
        "supreme court", "high court", "constitution", "bns", "bnss", "bsa",
        "ipc", "crpc", "cpc", "statute", "judgment", "act", "tribunal",
        "bail", "law", "legal", "nyaya", "vakeel", "case laws"
    ]

    PERMISSIVE_LICENSES = {"mit", "apache-2.0", "cc0-1.0", "cc-by-4.0", "openrail", "public-domain", "cc-by-sa-4.0"}
    RESTRICTED_LICENSES = {"cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nd-4.0", "no-license"}

    @classmethod
    def classify_dataset(cls, dataset_id: str, tags: List[str], description: str = "") -> DatasetCategory:
        """Classify a dataset candidate into a structured legal category."""
        text = f"{dataset_id} {' '.join(tags)} {description}".lower()

        if "constitution" in text:
            return DatasetCategory.CONSTITUTIONAL
        if any(k in text for k in ["bns", "bnss", "bsa", "special_acts", "acts", "legislation", "statute"]):
            return DatasetCategory.LEGISLATION
        if any(k in text for k in ["judgment", "judgement", "case-law", "case_laws", "court", "bail", "highcourt", "supreme-court"]):
            return DatasetCategory.CASE_LAW
        if any(k in text for k in ["tribunal", "nclt", "itat", "cat"]):
            return DatasetCategory.TRIBUNAL
        if any(k in text for k in ["metadata", "registry"]):
            return DatasetCategory.LEGAL_METADATA
        if any(k in text for k in ["qa", "question", "answer", "instruction"]):
            return DatasetCategory.LEGAL_QA
        if any(k in text for k in ["synthetic", "gpt-4", "generated"]):
            return DatasetCategory.SYNTHETIC_LEGAL
        if any(k in text for k in ["benchmark", "eval"]):
            return DatasetCategory.BENCHMARK
        if any(k in text for k in ["law", "legal", "nyaya", "vakeel"]):
            return DatasetCategory.PRIMARY_LEGAL_TEXT

        return DatasetCategory.UNKNOWN

    @classmethod
    def audit_license(cls, raw_license: Optional[str]) -> Tuple[str, LicenseStatus]:
        """Classify licensing status according to pretraining safety."""
        if not raw_license or raw_license.lower() in ("none", "unknown", "other", ""):
            return "unknown", LicenseStatus.UNKNOWN

        lic_lower = raw_license.lower().strip()
        if any(r in lic_lower for r in ["nc", "non-commercial", "restricted", "no-license"]):
            return raw_license, LicenseStatus.RESTRICTED
        if lic_lower in cls.PERMISSIVE_LICENSES or "apache" in lic_lower or "mit" in lic_lower or "cc-by" in lic_lower or "cc0" in lic_lower:
            return raw_license, LicenseStatus.CLEAR

        return raw_license, LicenseStatus.REVIEW_REQUIRED

    @classmethod
    def audit_provenance(cls, dataset_id: str, description: str = "") -> Tuple[str, ProvenanceStatus]:
        """Determine dataset source provenance and validity."""
        text = f"{dataset_id} {description}".lower()
        if "omilosaisolutions" in text or "official" in text or "supreme court" in text or "sci.gov.in" in text:
            return "Official / Primary Legal Repository", ProvenanceStatus.VERIFIED
        if "kanoon" in text or "scraped" in text or "web" in text:
            return "Web Scraped (e.g. Indian Kanoon / Public Archives)", ProvenanceStatus.COMMUNITY_SCRAPED
        if "synthetic" in text or "gpt" in text:
            return "Synthetic LLM Generated", ProvenanceStatus.SYNTHETIC
        return "Community Contributed (Unverified Source)", ProvenanceStatus.UNKNOWN

    @classmethod
    def calculate_quality_score(
        cls,
        category: DatasetCategory,
        license_status: LicenseStatus,
        provenance_status: ProvenanceStatus,
        has_primary_text: bool,
        is_chunked: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate transparent composite quality score out of 40 points."""
        qs = DatasetQualityScore()

        # 1. Source Authority (0 - 5)
        if provenance_status == ProvenanceStatus.VERIFIED:
            qs.source_authority = 5.0
        elif provenance_status == ProvenanceStatus.COMMUNITY_SCRAPED:
            qs.source_authority = 3.5
        else:
            qs.source_authority = 2.0

        # 2. Legal Relevance (0 - 5)
        if category in (DatasetCategory.CASE_LAW, DatasetCategory.LEGISLATION, DatasetCategory.CONSTITUTIONAL, DatasetCategory.PRIMARY_LEGAL_TEXT):
            qs.legal_relevance = 5.0
        elif category in (DatasetCategory.TRIBUNAL, DatasetCategory.LEGAL_METADATA):
            qs.legal_relevance = 4.0
        else:
            qs.legal_relevance = 2.5

        # 3. Text Quality (0 - 5)
        qs.text_quality = 4.5 if has_primary_text else 2.0

        # 4. Coverage (0 - 5)
        qs.coverage = 4.0

        # 5. Provenance (0 - 5)
        if provenance_status == ProvenanceStatus.VERIFIED:
            qs.provenance = 5.0
        elif provenance_status == ProvenanceStatus.COMMUNITY_SCRAPED:
            qs.provenance = 3.5
        else:
            qs.provenance = 2.0

        # 6. License Clarity (0 - 5)
        if license_status == LicenseStatus.CLEAR:
            qs.license_clarity = 5.0
        elif license_status == LicenseStatus.REVIEW_REQUIRED:
            qs.license_clarity = 3.0
        elif license_status == LicenseStatus.RESTRICTED:
            qs.license_clarity = 1.0
        else:
            qs.license_clarity = 2.0

        # 7. Metadata Quality (0 - 5)
        qs.metadata_quality = 4.0

        # 8. Duplication Risk (0 - 5)
        qs.duplication_safety = 3.0 if is_chunked else 4.5

        breakdown = {
            "source_authority": qs.source_authority,
            "legal_relevance": qs.legal_relevance,
            "text_quality": qs.text_quality,
            "coverage": qs.coverage,
            "provenance": qs.provenance,
            "license_clarity": qs.license_clarity,
            "metadata_quality": qs.metadata_quality,
            "duplication_safety": qs.duplication_safety,
        }
        return qs.total_score, breakdown

    @classmethod
    def audit_candidate(
        cls,
        dataset_id: str,
        visibility: str = "public",
        raw_license: Optional[str] = "mit",
        tags: Optional[List[str]] = None,
        description: str = "",
        estimated_rows: int = 0,
        estimated_tokens: int = 0,
        text_fields: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        legal_coverage: Optional[List[str]] = None,
        duplicate_of: Optional[str] = None,
    ) -> DatasetAuditRecord:
        """Audit a single candidate dataset against all quality, licensing, and provenance criteria."""
        tags = tags or []
        text_fields = text_fields or ["text"]
        languages = languages or ["en"]
        legal_coverage = legal_coverage or ["General Indian Law"]

        category = cls.classify_dataset(dataset_id, tags, description)
        lic_name, lic_status = cls.audit_license(raw_license)
        prov_desc, prov_status = cls.audit_provenance(dataset_id, description)

        is_chunked = "chunked" in dataset_id.lower()
        has_primary = category in (DatasetCategory.CASE_LAW, DatasetCategory.LEGISLATION, DatasetCategory.CONSTITUTIONAL, DatasetCategory.PRIMARY_LEGAL_TEXT)

        q_score, breakdown = cls.calculate_quality_score(
            category=category,
            license_status=lic_status,
            provenance_status=prov_status,
            has_primary_text=has_primary,
            is_chunked=is_chunked,
        )

        # Recommendation logic
        if duplicate_of:
            rec_status = RecommendationStatus.EXCLUDE
            reason = f"Duplicate or chunked variant of primary corpus: {duplicate_of}"
            recommended = False
        elif lic_status == LicenseStatus.RESTRICTED:
            rec_status = RecommendationStatus.EXCLUDE
            reason = f"Licensing terms ({lic_name}) prohibit unrestricted pretraining or distribution."
            recommended = False
        elif visibility == "private":
            rec_status = RecommendationStatus.INCLUDE
            reason = "Authoritative proprietary training corpus verified for Indian legal reasoning."
            recommended = True
        elif q_score >= 30.0 and lic_status in (LicenseStatus.CLEAR, LicenseStatus.REVIEW_REQUIRED):
            rec_status = RecommendationStatus.INCLUDE if lic_status == LicenseStatus.CLEAR else RecommendationStatus.REVIEW
            reason = "High-quality primary Indian legal source meeting pretraining authority standards."
            recommended = (rec_status == RecommendationStatus.INCLUDE)
        else:
            rec_status = RecommendationStatus.REVIEW
            reason = "Requires deeper manual validation of provenance and OCR formatting before ingestion."
            recommended = False

        return DatasetAuditRecord(
            dataset_id=dataset_id,
            visibility=visibility,
            category=category,
            license=lic_name,
            license_status=lic_status,
            provenance=prov_desc,
            provenance_status=prov_status,
            recommended=recommended,
            recommendation_status=rec_status,
            reason=reason,
            languages=languages,
            legal_coverage=legal_coverage,
            text_fields=text_fields,
            estimated_rows=estimated_rows,
            estimated_tokens=estimated_tokens,
            quality_score=q_score,
            quality_breakdown=breakdown,
            duplicate_of=duplicate_of,
        )
