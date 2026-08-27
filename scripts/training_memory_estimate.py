#!/usr/bin/env python3
"""Detailed memory estimation profiler for the 500M target model."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.model import LegalCausalLM


def main() -> None:
    config_500m_path = PROJECT_ROOT / "configs" / "500m.yaml"
    cfg = load_config(config_500m_path).model

    model = LegalCausalLM(cfg)
    summary = model.get_parameter_summary()

    N = summary["total_trainable"]
    B = 4          # Micro-batch size
    T = 2048       # Sequence length
    L = cfg.num_layers
    H = cfg.hidden_size
    V = cfg.vocab_size

    # Memory calculations
    param_fp32_bytes = N * 4
    param_fp16_bytes = N * 2
    grad_fp32_bytes = N * 4

    # AdamW maintains 2 states per parameter in float32: momentum (4 bytes) and variance (4 bytes) = 8 bytes/param
    optimizer_fp32_bytes = N * 8

    # Activation memory estimate per token per layer:
    # Standard: ~ 34 * B * T * H * L * bytes (for FP16 = 2 bytes)
    # With Gradient Checkpointing: ~ 2 * B * T * H * L * bytes + small overhead
    act_no_ckpt_bytes = 34 * B * T * H * L * 2
    act_with_ckpt_bytes = 4 * B * T * H * L * 2

    total_train_fp16_no_ckpt = param_fp16_bytes + grad_fp32_bytes + optimizer_fp32_bytes + act_no_ckpt_bytes
    total_train_fp16_with_ckpt = param_fp16_bytes + grad_fp32_bytes + optimizer_fp32_bytes + act_with_ckpt_bytes

    print("=" * 65)
    print("500M MODEL TRAINING MEMORY ESTIMATE")
    print("=" * 65)
    print(f"Model Architecture:   L={L}, H={H}, Heads={cfg.num_attention_heads}, KV-Heads={cfg.num_kv_heads}")
    print(f"Intermediate Size:    {cfg.intermediate_size} (SwiGLU)")
    print(f"Context Length:       {T}")
    print(f"Micro-batch size:     {B}")
    print(f"Total Parameters:     {N:,} (~{N / 1e6:.2f}M)")

    print("\n" + "-" * 65)
    print("STATIC MEMORY BREAKDOWN")
    print("-" * 65)
    print(f"Model Parameters (FP16):          {param_fp16_bytes / (1024**3):>8.2f} GB ({param_fp16_bytes / (1024**2):.1f} MB)")
    print(f"Gradients (FP32):                 {grad_fp32_bytes / (1024**3):>8.2f} GB ({grad_fp32_bytes / (1024**2):.1f} MB)")
    print(f"AdamW Optimizer States (FP32):    {optimizer_fp32_bytes / (1024**3):>8.2f} GB ({optimizer_fp32_bytes / (1024**2):.1f} MB)")
    static_total = (param_fp16_bytes + grad_fp32_bytes + optimizer_fp32_bytes) / (1024**3)
    print(f"Static Training State Total:      {static_total:>8.2f} GB")

    print("\n" + "-" * 65)
    print("DYNAMIC ACTIVATION MEMORY ESTIMATE")
    print("-" * 65)
    print(f"Standard (No Checkpointing):      {act_no_ckpt_bytes / (1024**3):>8.2f} GB")
    print(f"With Gradient Checkpointing:      {act_with_ckpt_bytes / (1024**3):>8.2f} GB (Savings: ~{(1 - act_with_ckpt_bytes/act_no_ckpt_bytes)*100:.1f}%)")

    print("\n" + "-" * 65)
    print("ESTIMATED TOTAL TRAINING VRAM REQUIREMENTS")
    print("-" * 65)
    print(f"FP16 Mixed Precision + No Checkpoint:   {total_train_fp16_no_ckpt / (1024**3):>8.2f} GB VRAM")
    print(f"FP16 Mixed Precision + Grad Checkpoint: {total_train_fp16_with_ckpt / (1024**3):>8.2f} GB VRAM (Recommended for 16GB/24GB GPUs)")
    print("\n* Note: Estimates assume FP16 mixed precision with FP32 master weights/optimizer. Actual usage may vary with PyTorch allocator behavior.")
    print("=" * 65)


if __name__ == "__main__":
    main()
