# Full Decoder-Only Language Model Architecture (`LegalCausalLM`)

## 1. Overview

`LegalCausalLM` is a modular, from-scratch, autoregressive decoder-only Transformer built for Indian legal reasoning. It incorporates Qwen-style modern architectural principles (RMSNorm, RoPE, Grouped-Query Attention, SwiGLU) with optional weight tying and next-token causal loss.

---

## 2. Complete Architectural Pipeline

```text
Input Token IDs [B, T]
        │
        ▼
Token Embedding: nn.Embedding(vocab_size, hidden_size)  ──► [B, T, H]
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ TransformerBlock × num_layers                          │
│                                                        │
│   ┌────────────────────────────────────────────────┐   │
│   │ Residual Stream 1                              │   │
│   │   x ──► RMSNorm ──► GQA + RoPE ──► (+) ──► x1  │   │
│   └────────────────────────────────────────────────┘   │
│   ┌────────────────────────────────────────────────┐   │
│   │ Residual Stream 2                              │   │
│   │   x1 ──► RMSNorm ──► SwiGLU ──► (+) ──► x2     │   │
│   └────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
        │
        ▼
Final Pre-Head RMSNorm(hidden_size)                     ──► [B, T, H]
        │
        ▼
Language Model Head: nn.Linear(hidden_size, vocab_size) ──► [B, T, V]
        │
        ▼
Next-Token Causal Loss (Cross-Entropy on shifted logits)
```

---

## 3. Weight Tying (`tie_word_embeddings`)

When `model.tie_word_embeddings = true`:
- The output projection matrix `lm_head.weight` is tied directly to the input embedding matrix `embed_tokens.weight`:
  $$	ext{lm\_head.weight} \equiv 	ext{embed\_tokens.weight}$$
- **Parameter Savings**: Eliminates $	ext{vocab\_size} 	imes 	ext{hidden\_size}$ parameters (e.g. $32,000 	imes 512 = 16,384,000$ params / $65.5	ext{ MB}$ in FP32).
- **Regularization**: Forces output token representations to remain in the same semantic metric space as input embeddings.

---

## 4. Shifted Causal Loss

Autoregressive training minimizes negative log-likelihood of predicting token $t_{i+1}$ given context $t_0, \dots, t_i$:

$$\mathcal{L} = -rac{1}{N} \sum_{i=0}^{T-2} \log P(t_{i+1} \mid t_0, \dots, t_i)$$

Implementation:
```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index=-100)
```
