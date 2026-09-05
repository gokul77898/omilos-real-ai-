#!/usr/bin/env python3
"""The real continued-pretraining entry point.  It never creates synthetic data."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.checkpoint import load_checkpoint
from src.config import load_config
from src.data.sharding import ShardedDataset
from src.model import LegalCausalLM
from src.seed import set_seed
from src.trainer import Trainer

def main() -> None:
    parser = argparse.ArgumentParser(description="Continue pretraining from validated Omilos token shards")
    parser.add_argument("--config", default="configs/500m.yaml")
    parser.add_argument("--data-dir", default="data/tokenized_128k_32k")
    parser.add_argument("--resume", required=True, help="Verified local checkpoint directory")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)
    set_seed(cfg.project.seed)
    root = PROJECT_ROOT / args.data_dir
    manifest = root / "corpus_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"No canonical corpus manifest: {manifest}")
    train = ShardedDataset(root / "train", cfg.model.max_seq_len, cfg.model.vocab_size, reject_pad=True)
    validation = ShardedDataset(root / "validation", cfg.model.max_seq_len, cfg.model.vocab_size, reject_pad=True)
    if not len(train) or not len(validation):
        raise ValueError("Canonical train and validation datasets must both be non-empty")
    train_dl = DataLoader(train, batch_size=cfg.training.batch_size, shuffle=True)
    eval_dl = DataLoader(validation, batch_size=1, shuffle=False)
    trainer = Trainer(LegalCausalLM(cfg.model), cfg, train_dl, eval_dl, device=args.device)
    meta = load_checkpoint(args.resume, trainer.model, trainer.optimizer, trainer.scheduler, trainer.scaler, trainer.device)
    trainer.global_step, trainer.epoch = meta["step"], meta["epoch"]
    if trainer.global_step <= 0:
        raise ValueError("Resume checkpoint lacks a positive global step")
    trainer.train(max_steps=args.max_steps)

if __name__ == "__main__":
    main()
