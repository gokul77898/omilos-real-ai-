# Indian Legal Dataset Selection & Audit Report

## 1. Overview

To pretrain the **~482.8M Parameter Indian Legal Foundation Model** (**Omilos Own AI** / `indian-legal-reasoning`), this audit comprehensively evaluates open Hugging Face datasets alongside proprietary in-house datasets.

---

## 2. Ingestion & Quality Scoring Methodology

Every dataset is evaluated against 8 dimensions (0 - 5 points each, composite 0 - 40):
1. **Source Authority (0-5):** Preference for official Gazette, Supreme Court / High Court repositories.
2. **Legal Relevance (0-5):** Direct statutory, case-law, or constitutional provisions.
3. **Text Quality (0-5):** Uncorrupted full-text legal phrasing.
4. **Coverage (0-5):** Representation across Central Acts, State Acts, BNS/BNSS/BSA, and High Courts.
5. **Provenance (0-5):** Transparent origin without undocumented synthetic insertions.
6. **License Clarity (0-5):** Permissive pretraining and redistribution rights.
7. **Metadata Quality (0-5):** Citation indexes, section numbers, date metadata.
8. **Duplication Safety (0-5):** Penalizes chunked clones of known corpora.

---

## 3. Dataset Audit Matrix

| Dataset ID | Visibility | Type | License | Provenance | Est. Tokens | Quality Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `OmilosAISolutions/nyayamitra-training-data` | Private | Primary Text | In-House | Official Archive | 120.0M | 37.0 / 40 | **INCLUDE** |
| `OmilosAISolutions/vakeels-legal-training-data` | Private | Case Law | In-House | Official Archive | 75.0M | 37.0 / 40 | **INCLUDE** |
| `OmilosAISolutions/highcourt-2023-full-metadata` | Private | Legal Metadata | In-House | Official Archive | 45.0M | 36.0 / 40 | **INCLUDE** |
| `vaquill/open-india-law` | Public | Primary Text | Apache-2.0 | Primary Repository | 50.0M | 36.5 / 40 | **INCLUDE** |
| `LH2-data-labs/indian-legal-records` | Public | Case Law | CC-BY-4.0 | Judicial Archives | 65.0M | 36.5 / 40 | **INCLUDE** |
| `KanoonGPT/indian-case-laws` | Public | Case Law | MIT | Web Scraped (Kanoon) | 95.0M | 35.0 / 40 | **INCLUDE** |
| `Sumitedu/indian-case-laws` | Public | Case Law | MIT | Case Laws Archive | 28.0M | 35.0 / 40 | **INCLUDE** |
| `debkanchan/supreme-court-of-india-judgements` | Public | Case Law | Apache-2.0 | SC Archive (1950-2021)| 80.0M | 36.5 / 40 | **INCLUDE** |
| `123Divyansh/Constitution.-of-India-TXT_File` | Public | Constitution | CC0-1.0 | Official Constitution | 0.45M | 36.5 / 40 | **INCLUDE** |
| `SnehaDeshmukh/IndianBailJudgments-1200` | Public | Case Law | MIT | HC/SC Bail Orders | 3.5M | 35.0 / 40 | **INCLUDE** |
| `navaneeth005/BNS_definitions` | Public | Legislation | MIT | BNS 2023 Code | 0.6M | 35.0 / 40 | **INCLUDE** |
| `navaneeth005/BNS_detailed` | Public | Legislation | MIT | BNS/BNSS/BSA Code | 2.2M | 35.0 / 40 | **INCLUDE** |
| `navaneeth005/special_acts` | Public | Legislation | MIT | Special Acts Statutes | 3.0M | 35.0 / 40 | **INCLUDE** |
| `Shreyasrao/Indian-law-supreme-court-judgements-2016`| Public | Case Law | Apache-2.0 | SC Archive (2016) | 4.2M | 36.5 / 40 | **INCLUDE** |
| `vihaannnn/Chunked-Indian-Supreme-Court-Judgements` | Public | Case Law | MIT | Duplicate / Chunked | 40.0M | 33.5 / 40 | **EXCLUDE** (Duplicate) |
| `vihaannnn/Indian-Supreme-Court-Judgements-Chunked` | Public | Case Law | MIT | Duplicate / Chunked | 38.0M | 33.5 / 40 | **EXCLUDE** (Duplicate) |

---

## 4. Ingestion Governance

The selection policy is recorded in [`configs/data_sources.yaml`](file:///Users/gokul/.gemini/antigravity/scratch/omilos-own-ai/configs/data_sources.yaml) with explicit enable flags, and machine-readable metadata is maintained in [`data/manifests/dataset_registry.json`](file:///Users/gokul/.gemini/antigravity/scratch/omilos-own-ai/data/manifests/dataset_registry.json).
