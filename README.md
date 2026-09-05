# Omilos Own AI — Indian Legal Reasoning Model

> Complete from-scratch decoder-only Transformer tailored for Indian legal reasoning and jurisprudence.

---

## Current Status: long-context pretraining stack (training intentionally not started)

This repository contains the engineering foundation, custom subword tokenizer, core Qwen-style Transformer components, assembled `LegalCausalLM`, and a high-performance PyTorch training engine supporting our target **481.6M parameter model**.

### 15-Phase Development Roadmap

1. **Phase 1: Project Foundation and Training Environment** *(Completed)*
2. **Phase 2: Indian Legal Tokenizer** *(Completed)*
3. **Phase 3: Core Qwen-Style Transformer Components** *(Completed)*
4. **Phase 4: Full Decoder-Only Language Model** *(Completed)*
5. **Phase 5: 500M Model Training Engine** *(Completed)*
6. **Phase 6:** Pretraining Data Pipeline & Streaming
7. **Phase 7:** Pretraining Loop & Optimizer Setup
8. **Phase 8:** Instruction Tuning (SFT) for Legal Tasks
9. **Phase 9:** Multi-Step Legal Chain-of-Thought & Reasoning Training
10. **Phase 10:** Preference Optimization (GRPO / DPO)
11. **Phase 11:** Long-Context Window Expansion
12. **Phase 12:** Indian Legal RAG Integration (Acts, Statutes, Judgments)
13. **Phase 13:** Comprehensive Legal Benchmark & Evaluation
14. **Phase 14:** Quantization & Production Export
15. **Phase 15:** High-Throughput Inference Server

---

## Training Engine Usage & Diagnostics

### 1. View 500M Model Training Memory Estimates

```bash
python scripts/training_memory_estimate.py
```

### 2. Run End-to-End Training Convergence Test

```bash
python scripts/tiny_train.py
```

### 3. Run Hardware Throughput Benchmark

```bash
python scripts/benchmark_training.py
```

### 4. Run Complete Test Suite

```bash
pytest -q
```

## Canonical pretraining data path

The real entrypoint is `scripts/train_pretrain.py`; it reads only validated binary
shards from `data/tokenized_128k_32k/`.  It never creates synthetic training data.
Build those shards from an operator-supplied JSONL provenance manifest first:

```bash
python scripts/build_long_context_corpus.py --input-jsonl /path/to/legal_sources.jsonl
```

Each source record must include `document_id`, `source_id`, `source_url`,
`source_revision`, `license`, `language`, and `text`.  The builder records supplied
evidence; it does not claim that a source is licensed or verified on its own.
