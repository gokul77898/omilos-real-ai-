#!/usr/bin/env python3
"""Hardware throughput benchmark for training engine."""

from pathlib import Path
import sys
import time
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, ModelConfig, TrainingConfig
from src.model import LegalCausalLM
from src.trainer import Trainer


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 60)
    print("TRAINING ENGINE THROUGHPUT BENCHMARK")
    print("=" * 60)

    app_config = AppConfig()
    app_config.model = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_attention_heads=8,
        num_kv_heads=4,
        intermediate_size=704,
        max_seq_len=256,
        tie_word_embeddings=True,
    )
    app_config.training = TrainingConfig(
        batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        warmup_steps=5,
        max_steps=20,
    )

    B = 4
    T = 128
    dummy_data = torch.randint(0, 1000, (64, T), dtype=torch.long)
    dataloader = DataLoader(TensorDataset(dummy_data, dummy_data), batch_size=B)

    model = LegalCausalLM(app_config.model)
    summary = model.get_parameter_summary()

    trainer = Trainer(
        model=model,
        config=app_config,
        train_dataloader=dataloader,
        device=device,
    )

    # Warmup step
    trainer.train_step(next(iter(dataloader)))

    # Timed run: 20 steps
    start_time = time.time()
    steps = 0
    total_tokens = 0

    for batch in dataloader:
        trainer.train_step(batch)
        steps += 1
        total_tokens += B * T
        if steps >= 20:
            break

    elapsed = time.time() - start_time
    steps_per_sec = steps / elapsed
    tokens_per_sec = total_tokens / elapsed

    print(f"Device:               {device}")
    print(f"Model Parameters:     {summary['total_trainable']:,}")
    print(f"Sequence Length:      {T}")
    print(f"Micro-Batch Size:     {B}")
    print(f"Grad Accumulation:    {app_config.training.gradient_accumulation_steps}")
    print(f"Precision:            Float32")
    print(f"Steps / Second:       {steps_per_sec:.2f}")
    print(f"Tokens / Second:      {tokens_per_sec:,.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
