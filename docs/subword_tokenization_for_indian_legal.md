# Subword Tokenization Architecture for Indian Legal Text

## Executive Summary

Indian legal documents present a uniquely challenging linguistic domain:
1. **Multilingualism & Mixed Code**: High Court and Supreme Court pleadings frequently blend English statutes, Latin maxims (*mens rea*, *ex debito justitiae*), Sanskrit legal maxims (*dharmo rakshati rakshitah*), and vernacular state languages (Hindi, Kannada, Tamil, Telugu, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Urdu).
2. **Statutory Nomenclature**: Acts and statutes contain compact structured citations (`Section 187 BNSS`, `Article 21`, `(2023) 5 SCC 123`, `2024 INSC 123`, `§ 125(1)(a)`).
3. **Morphological Agglutination**: Indic languages feature rich inflectional and agglutinative morphologies where words compound extensively.

To address these domain-specific constraints, **Omilos Own AI** employs a **Byte-Level Byte-Pair Encoding (Byte-Level BPE)** architecture.

---

## Technical Rationale: Why Byte-Level BPE?

### 1. Zero Out-of-Vocabulary (OOV) Guarantee
Standard word-level or character-level tokenizers encounter `<unk>` tokens when confronted with rare characters, complex Unicode conjuncts, or foreign scripts. 
By bootstrapping from the base 256 raw UTF-8 byte alphabet, **every possible byte sequence is representable**. Any unseen symbol or rare glyph decomposes cleanly into byte tokens without dropping context.

### 2. Lossless Round-Trip Reconstruction
Standard NLP tokenizers often perform aggressive normalization (e.g. lowercasing, stripping punctuation, or collapsing whitespaces). In legal reasoning:
- Capitalization differentiates terms (`Act` vs `act`, `Court` vs `court`, `State` vs `state`).
- Legal punctuation (`§`, `¶`, `—`, quotes, parenthesis) carries semantic and statutory weight.
- Whitespace and line indentation in judgment orders are critical for structure.

Byte-Level BPE preserves raw UTF-8 byte sequences without destructive normalization, achieving **100% exact round-trip fidelity**.

### 3. Compression and Efficiency for Legal Terminology
Frequent statutory terms (e.g., `Section`, `Article`, `Bharatiya`, `Sanhita`, `Magistrate`, `Jurisdiction`, `Citations`) are merged into dedicated single tokens during training. This maximizes token efficiency and effective context window utilization during pretraining and reasoning.

---

## Special Token Design

The tokenizer reserves 4 explicit special tokens for causal autoregressive language modeling:

| Token | Purpose | Token ID |
| :--- | :--- | :--- |
| `<pad>` | Batch sequence padding | `0` |
| `<unk>` | Unknown / unmapped fallback | `1` |
| `<s>` | Beginning of sequence (BOS) | `2` |
| `</s>` | End of sequence (EOS) | `3` |

---

## Roadmap & Scalability

- **Phase 2 Development**: Verified on synthetic development corpus across 12 Indian languages and landmark legal citation formats.
- **Phase 8 Scaling**: The tokenizer vocabulary will be scaled and trained on the full multi-gigabyte Indian legal corpus (Central Acts, State Acts, Supreme Court & High Court Judgments, Bare Acts, Gazette Notifications).
