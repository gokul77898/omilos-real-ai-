# Training Engine Architecture & Optimization Guide

## 1. Overview

The **Omilos Own AI** training engine is an explicit, pure PyTorch optimization pipeline designed to scale from compact development models (40M) up to the target **500M parameter Indian legal reasoning foundation model**.

---

## 2. Core Optimization Pipeline

```text
Data Batch [B, T]
        │
        ▼
Device Placement (CUDA / MPS / CPU)
        │
        ▼
Automatic Mixed Precision Context (torch.autocast FP16/BF16)
        │
        ▼
Forward Pass -> Shifted Causal Loss
        │
        ▼
Loss Scaling: loss = loss / gradient_accumulation_steps
        │
        ▼
Backward Pass (torch.amp.GradScaler on CUDA)
        │
        ▼
Accumulation Boundary Check (micro_step % accumulation == 0)
        │
        ├── 1. Unscale Gradients
        ├── 2. Gradient Norm Clipping (max_norm = 1.0)
        ├── 3. Optimizer Step (AdamW)
        ├── 4. Learning Rate Scheduler Step (Warmup + Cosine Decay)
        └── 5. Zero Gradients
```

---

## 3. 500M Target Model Architectural Sizing

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Vocabulary Size ($V$)** | `32,000` | Custom Byte-Level BPE Tokenizer |
| **Hidden Size ($H$)** | `1,152` | Model residual dimension |
| **Number of Layers ($L$)** | `24` | Stacked Pre-Norm Transformer blocks |
| **Attention Heads ($N_q$)** | `18` | Query head count ($d_{	ext{head}} = 64$) |
| **KV Heads ($N_{kv}$)** | `6` | Grouped Query Attention ratio $3	imes$ |
| **Intermediate Size** | `3,840` | SwiGLU hidden dimension ($pprox 3.33 H$) |
| **Context Length ($T$)** | `2,048` | Maximum sequence length |
| **Weight Tying** | `Enabled` | LM head tied to token embeddings |
| **Total Trainable Parameters** | **`481,623,552`** | **~481.6M Target Foundation Model** |

---

## 4. Memory Estimation Breakdown (500M Model)

For micro-batch size $B=4$, sequence length $T=2048$, precision FP16:

1. **Parameters (FP16)**: $481.6	ext{M} 	imes 2	ext{ bytes} pprox 0.90	ext{ GB}$
2. **Gradients (FP32)**: $481.6	ext{M} 	imes 4	ext{ bytes} pprox 1.79	ext{ GB}$
3. **AdamW Optimizer States (FP32)**: $481.6	ext{M} 	imes 8	ext{ bytes} pprox 3.59	ext{ GB}$
4. **Static State Total**: $pprox 6.28	ext{ GB}$
5. **Activation Memory**:
   - Without Checkpointing: $pprox 10.82	ext{ GB}$ (Total: ~17.10 GB VRAM)
   - With Gradient Checkpointing: $pprox 1.27	ext{ GB}$ (Total: ~7.55 GB VRAM)
