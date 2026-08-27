#!/usr/bin/env python3
"""CLI Training runner for Omilos Own AI."""

import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint import load_checkpoint
from src.config import load_config
from src.model import LegalCausalLM
from src.seed import set_seed
from src.trainer import Trainer


def create_synthetic_dataloader(vocab_size: int, seq_len: int, num_samples: int = 100, batch_size: int = 4) -> DataLoader:
    """Create a synthetic tensor dataset and DataLoader for training validation."""
    data = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)
    dataset = TensorDataset(data, data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Omilos Own AI Training CLI")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--device", type=str, default=None, help="Override target device (cpu/mps/cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    app_config = load_config(args.config)
    seed = args.seed if args.seed is not None else app_config.project.seed
    set_seed(seed)

    model = LegalCausalLM(app_config.model)

    train_dl = create_synthetic_dataloader(
        vocab_size=app_config.model.vocab_size,
        seq_len=min(app_config.model.max_seq_len, 64),
        num_samples=200,
        batch_size=app_config.training.batch_size,
    )
    eval_dl = create_synthetic_dataloader(
        vocab_size=app_config.model.vocab_size,
        seq_len=min(app_config.model.max_seq_len, 64),
        num_samples=50,
        batch_size=app_config.training.batch_size,
    )

    trainer = Trainer(
        model=model,
        config=app_config,
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        device=args.device,
    )

    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        meta = load_checkpoint(
            checkpoint_dir=args.resume,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            device=trainer.device,
        )
        trainer.global_step = meta.get("step", 0)
        trainer.epoch = meta.get("epoch", 0)
        print(f"Resumed at global step {trainer.global_step}")

    trainer.train(max_steps=args.max_steps)


if __name__ == "__main__":
    main()
