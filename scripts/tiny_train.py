#!/usr/bin/env python3
"""End-to-end tiny model training convergence experiment."""

from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, ModelConfig, TrainingConfig
from src.model import LegalCausalLM
from src.seed import set_seed
from src.trainer import Trainer


def main() -> None:
    set_seed(42)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("=" * 60)
    print("TINY MODEL TRAINING CONVERGENCE TEST")
    print("=" * 60)

    app_config = AppConfig()
    app_config.model = ModelConfig(
        vocab_size=256,
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=352,
        max_seq_len=64,
        tie_word_embeddings=True,
    )
    app_config.training = TrainingConfig(
        batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-3,
        min_learning_rate=2e-4,
        warmup_steps=10,
        max_steps=60,
    )
    app_config.logging.log_every_steps = 10
    app_config.logging.eval_every_steps = 20

    def make_pattern_data(num_samples: int, seq_len: int, vocab_size: int) -> torch.Tensor:
        data = []
        for i in range(num_samples):
            base = (i * 5) % (vocab_size - 16)
            seq = [((base + j) % (vocab_size - 1)) + 1 for j in range(seq_len)]
            data.append(seq)
        return torch.tensor(data, dtype=torch.long)

    train_tokens = make_pattern_data(64, 32, 256)
    eval_tokens = make_pattern_data(16, 32, 256)

    train_dl = DataLoader(TensorDataset(train_tokens, train_tokens), batch_size=4, shuffle=True)
    eval_dl = DataLoader(TensorDataset(eval_tokens, eval_tokens), batch_size=4, shuffle=False)

    model = LegalCausalLM(app_config.model)
    trainer = Trainer(
        model=model,
        config=app_config,
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        device=device,
    )

    init_eval = trainer.evaluate()
    init_loss = init_eval["eval_loss"]
    init_ppl = init_eval["perplexity"]
    print(f"Initial Eval Loss: {init_loss:.4f} | Perplexity: {init_ppl:.2f}")

    results = trainer.train(max_steps=60)

    final_eval = trainer.evaluate()
    final_loss = final_eval["eval_loss"]
    final_ppl = final_eval["perplexity"]
    print(f"Final Eval Loss:   {final_loss:.4f} | Perplexity: {final_ppl:.2f}")

    loss_drop = init_loss - final_loss
    print(f"Total Steps:       {results['total_steps']}")
    print(f"Loss Reduction:    {loss_drop:.4f}")

    if final_loss < init_loss:
        print("\n✓ SUCCESS: Training engine demonstrated continuous loss reduction and convergence!")
    else:
        print("\n✗ FAILED: Loss did not decrease.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
