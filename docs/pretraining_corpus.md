# Complete Indian Legal Pretraining Corpus

## 1. Overview

The **Omilos Own AI** foundation pretraining corpus has been engineered specifically for the **~482.8M parameter Indian Legal Reasoning Model**.

---

## 2. Ingestion & Sharded Storage Pipeline

```text
HF Datasets (Public + Private)
            │
            ▼
Legal Text Normalization (NFC Unicode, strict statutory citation preservation)
            │
            ▼
Court Classifier (Supreme Court, 25 High Courts, Tribunals, Criminal Codes)
            │
            ▼
Multi-Stage Deduplication (SHA-256 + Citation ID + 64-bit SimHash)
            │
            ▼
Document-Level Split (95% Train / 5% Validation)
            │
            ▼
Sequence Packing (max_seq_len = 2048 with <s> and </s> boundaries)
            │
            ▼
Binary Memory-Mapped Sharded Array Storage
data/tokenized/train/shard_00000.bin
data/tokenized/validation/shard_00000.bin
```

---

## 3. All 25 High Courts Coverage

| High Court | Status | Coverage |
| :--- | :--- | :--- |
| **Allahabad** | VERIFIED | Criminal & Civil Appeals, Lucknow Bench |
| **Andhra Pradesh** | VERIFIED | Constitutional & Statutory Orders, Amaravati |
| **Bombay** | VERIFIED | Commercial, Criminal, Nagpur & Goa Benches |
| **Calcutta** | VERIFIED | Constitutional, Appellate, Port Blair Bench |
| **Chhattisgarh** | VERIFIED | Criminal & Bail Orders, Bilaspur |
| **Delhi** | VERIFIED | IPR, Commercial, Arbitration, Constitutional |
| **Gauhati** | VERIFIED | Northeast Jurisdiction, Assam Orders |
| **Gujarat** | VERIFIED | Corporate, Tax, Criminal Orders, Ahmedabad |
| **Himachal Pradesh** | VERIFIED | Environmental, Civil, Shimla Orders |
| **Jammu & Kashmir and Ladakh** | VERIFIED | Srinagar & Jammu Proceedings |
| **Jharkhand** | VERIFIED | Mining, Tribal, Criminal Orders, Ranchi |
| **Karnataka** | VERIFIED | Tech/Cyber, Commercial, Dharwad Bench |
| **Kerala** | VERIFIED | Constitutional, Maritime, Ernakulam |
| **Madhya Pradesh** | VERIFIED | Criminal, Jabalpur, Indore, Gwalior |
| **Madras** | VERIFIED | Constitutional, Commercial, Madurai Bench |
| **Manipur** | VERIFIED | Imphal Bench Orders |
| **Meghalaya** | VERIFIED | Shillong Bench Proceedings |
| **Orissa** | VERIFIED | Mining, Civil, Cuttack Orders |
| **Patna** | VERIFIED | Criminal, Constitutional Proceedings |
| **Punjab & Haryana** | VERIFIED | Agriculture, Service, Chandigarh Orders |
| **Rajasthan** | VERIFIED | Land, Criminal, Jodhpur & Jaipur Benches |
| **Sikkim** | VERIFIED | Gangtok Civil & Criminal Judgments |
| **Telangana** | VERIFIED | Commercial, Tech, Hyderabad Orders |
| **Tripura** | VERIFIED | Agartala Orders |
| **Uttarakhand** | VERIFIED | Environmental, Civil, Nainital Orders |

---

## 4. 10B Target Token Deficit Analysis

- **Target Pretraining Volume:** `10,000,000,000` tokens (10.0B)
- **Verified Unique Usable Token Pool:** `~572,000,000` tokens (~572M)
- **Exact Deficit:** `~9,428,000,000` tokens (~9.43B tokens)
- *Policy Compliance: We strictly adhere to zero synthetic repetition and zero artificial duplication.*
