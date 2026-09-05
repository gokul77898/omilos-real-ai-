#!/usr/bin/env python3
"""Read-only compatibility check for an existing model checkpoint."""
import argparse
from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import load_config
from src.model import LegalCausalLM

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/app/hf-5000/checkpoints/step_5000")
    parser.add_argument("--config", default="configs/500m.yaml")
    args = parser.parse_args()
    path = Path(args.checkpoint) / "model.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not available: {path}")
    model = LegalCausalLM(load_config(PROJECT_ROOT / args.config).model)
    state = torch.load(path, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    print(f"parameter_count={sum(p.numel() for p in model.parameters()):,}")
    print(f"missing_keys={list(result.missing_keys)}")
    print(f"unexpected_keys={list(result.unexpected_keys)}")
    if result.missing_keys or result.unexpected_keys:
        raise SystemExit(1)
    print("CHECKPOINT VERIFIED")

if __name__ == "__main__":
    main()
