#!/usr/bin/env python3
"""Model summary, parameter breakdown, and memory estimation CLI tool."""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.model import LegalCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an Omilos model configuration")
    parser.add_argument("--config", default="configs/500m.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    app_config = load_config(config_path)
    model_cfg = app_config.model

    model = LegalCausalLM(model_cfg)
    summary = model.get_parameter_summary()

    # Memory estimates for parameter storage only
    total_trainable = summary["total_trainable"]
    total_params = summary["total_parameters"]
    fp32_mb = (total_params * 4) / (1024 ** 2)
    fp16_mb = (total_params * 2) / (1024 ** 2)

    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Model:                LegalCausalLM (Qwen-style Decoder-Only)")
    print(f"Vocabulary:           {model_cfg.vocab_size:,}")
    print(f"Hidden size:          {model_cfg.hidden_size}")
    print(f"Layers:               {model_cfg.num_layers}")
    print(f"Attention heads:      {model_cfg.num_attention_heads}")
    print(f"KV heads:             {model_cfg.num_kv_heads}")
    print(f"Head dimension:       {model_cfg.head_dim}")
    print(f"Intermediate size:    {model_cfg.intermediate_size}")
    print(f"Context length:       {model_cfg.max_seq_len}")
    print(f"RoPE theta:           {model_cfg.rope_theta}")
    print(f"Weight tying:         {'Enabled (lm_head = embed_tokens)' if model_cfg.tie_word_embeddings else 'Disabled'}")

    print("\n" + "-" * 50)
    print("PARAMETERS")
    print("-" * 50)
    print(f"Token embeddings:     {summary['token_embeddings']:>12,}")
    print(f"Transformer blocks:   {summary['transformer_blocks_trainable']:>12,}")
    print(f"Final RMSNorm:        {summary['final_norm']:>12,}")
    print(f"LM head:              {summary['lm_head']:>12,} {'(Tied)' if summary['tie_word_embeddings'] else ''}")
    print(f"\nTrainable:            {summary['total_trainable']:>12,}")
    print(f"Non-trainable:        {0:>12,}")
    print(f"Buffers (RoPE cache): {summary['total_buffers']:>12,}")
    print(f"Total:                {summary['total_parameters']:>12,}")

    print("\n" + "-" * 50)
    print("MEMORY ESTIMATE (Model Parameters Only)")
    print(f"FP32 parameter memory: {fp32_mb:>10.2f} MB")
    print(f"FP16 parameter memory: {fp16_mb:>10.2f} MB")
    print("\n* Note: Parameter memory estimate does not include optimizer states, activations, or gradients.")
    print("=" * 50)


if __name__ == "__main__":
    main()
