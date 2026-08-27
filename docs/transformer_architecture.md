# Qwen-Style Transformer Architecture & Component Design

## 1. Overview

**Omilos Own AI** is an independently implemented decoder-only Transformer tailored for Indian legal reasoning. It incorporates state-of-the-art architectural patterns from modern foundation models (such as Qwen 2.5 / Llama 3) while remaining a clean from-scratch PyTorch implementation.

---

## 2. Core Components & Mathematical Formulations

### A. Root Mean Square Normalization (RMSNorm)
RMSNorm replaces standard LayerNorm by assuming zero-mean activations, regularizing the root mean square of inputs without computing mean displacement.

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}$$

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma$$

- **Benefits**: ~20% faster than LayerNorm due to avoiding mean calculation; high numerical stability across mixed-precision training.

---

### B. Rotary Position Embeddings (RoPE)
RoPE encodes token position directly into query and key representations via orthogonal rotation matrices in 2D coordinate pairs.

Given 2D sub-vector $(x_{2i}, x_{2i+1})$ at position $m$ with frequency $\theta_i = b^{-2i/d}$:

$$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

Vectorized formulation:
$$\text{RoPE}(x) = x \odot \cos(m\Theta) + \text{rotate\_half}(x) \odot \sin(m\Theta)$$

- **Benefits**: Relative position decay emerges naturally: $q_m^T k_n = f(q, k, m - n)$.

---

### C. Grouped-Query Attention (GQA) & Causal Masking
In standard Multi-Head Attention (MHA), each query head has a private key and value head ($N_q = N_{kv}$). In Grouped-Query Attention (GQA), multiple query heads share a smaller set of key-value heads ($N_{kv} < N_q$).

$$n_{\text{rep}} = \frac{N_q}{N_{kv}}$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot \text{repeat\_kv}(K)^T}{\sqrt{d_{\text{head}}}} + M_{\text{causal}}\right) \cdot \text{repeat\_kv}(V)$$

- **Causal Mask**: Upper-triangular elements $j > i$ are masked to $-\infty$ so position $i$ only attends to past and current tokens.
- **Memory Reduction**: KV-cache bandwidth and VRAM requirements during autoregressive inference are reduced by a factor of $n_{\text{rep}}$ (e.g. $2\times$ to $8\times$).

---

### D. SwiGLU Multi-Layer Perceptron
SwiGLU uses a Swish (SiLU) gated linear unit with three projection weight matrices:

$$\text{SwiGLU}(x) = \left(\text{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x)\right) W_{\text{down}}$$

- **Intermediate Size**: Typically set to $\approx \frac{8}{3} d_{\text{model}}$ (rounded to multiples of 64 or 256) to maintain parameter parity with standard $4 d_{\text{model}}$ MLPs while delivering superior non-linear capacity.

---

## 3. Data Flow & Tensor Shape Progression

Inside a single `TransformerBlock`:

| Step | Operation | Input Tensor Shape | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| **0** | Input activations | — | $[B, T, H]$ |
| **1** | `input_layernorm` | $[B, T, H]$ | $[B, T, H]$ |
| **2** | $Q, K, V$ Projections | $[B, T, H]$ | $Q: [B, T, H_q \cdot D]$, $K, V: [B, T, H_{kv} \cdot D]$ |
| **3** | Head Reshaping & Transpose | — | $Q: [B, H_q, T, D]$, $K, V: [B, H_{kv}, T, D]$ |
| **4** | Apply RoPE to $Q, K$ | $[B, H, T, D]$ | $[B, H, T, D]$ |
| **5** | GQA KV-Repeat | $K, V: [B, H_{kv}, T, D]$ | $K, V: [B, H_q, T, D]$ |
| **6** | Causal Scaled Dot-Product Attn | $Q, K, V: [B, H_q, T, D]$ | $[B, H_q, T, D]$ |
| **7** | Transpose & $O$ Projection | $[B, H_q, T, D]$ | $[B, T, H]$ |
| **8** | Attention Residual Addition | $[B, T, H] + [B, T, H]$ | $[B, T, H]$ |
| **9** | `post_attention_layernorm` | $[B, T, H]$ | $[B, T, H]$ |
| **10** | SwiGLU Gating & Projections | $[B, T, H]$ | $[B, T, H]$ |
| **11** | MLP Residual Addition | $[B, T, H] + [B, T, H]$ | $[B, T, H]$ |
