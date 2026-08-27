#!/usr/bin/env python3
"""Tiny synthetic overfitting experiment to verify end-to-end learning convergence."""

import sys
from pathlib import Path
import torch
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ModelConfig
from src.model import LegalCausalLM
from src.seed import set_seed


def main() -> None:
    set_seed(42)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("=" * 60)
    print("TINY OVERFITTING EXPERIMENT")
    print("=" * 60)
    print(f"Device: {device}")

    # Compact model configuration for rapid overfitting
    config = ModelConfig(
        vocab_size=256,
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=352,
        max_seq_len=64,
        tie_word_embeddings=True,
    )
    model = LegalCausalLM(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)

    # Tiny synthetic batch: 2 sequences of length 8
    synthetic_tokens = torch.tensor([
        [10, 20, 30, 40, 50, 60, 70, 80],
        [15, 25, 35, 45, 55, 65, 75, 85],
    ], dtype=torch.long, device=device)

    print(f"Input batch shape: {list(synthetic_tokens.shape)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("-" * 60)

    initial_loss = None
    final_loss = None
    num_steps = 50

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        output = model(input_ids=synthetic_tokens, labels=synthetic_tokens)
        loss = output.loss

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0 or step == num_steps:
            print(f"Step {step:2d}/{num_steps} | Loss: {loss.item():.6f}")

        final_loss = loss.item()

    print("-" * 60)
    print(f"Initial Loss: {initial_loss:.6f}")
    print(f"Final Loss:   {final_loss:.6f}")
    reduction_pct = ((initial_loss - final_loss) / initial_loss) * 100
    print(f"Loss Reduction: {reduction_pct:.2f}%")

    if final_loss < 0.2:
        print("\n✓ SUCCESS: Model successfully overfit and memorized the synthetic dataset!")
    else:
        print("\n✗ WARNING: Loss reduction was lower than expected.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
