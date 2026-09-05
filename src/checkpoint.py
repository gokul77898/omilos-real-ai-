"""Stateful checkpoint management and rotation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def save_checkpoint(
    save_dir: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    step: int = 0,
    epoch: int = 0,
    config: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
    keep_last_n: int = 3,
) -> Path:
    """Save a full training checkpoint with model weights, optimizer, and RNG state.

    Args:
        save_dir: Base directory to save checkpoints.
        model: PyTorch model instance.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler instance.
        scaler: AMP GradScaler instance.
        step: Current global training step.
        epoch: Current training epoch.
        config: Optional AppConfig instance.
        metrics: Optional metric dictionary (e.g. loss, perplexity).
        keep_last_n: Maximum number of previous checkpoints to retain.

    Returns:
        Path to the saved checkpoint directory.
    """
    base_path = Path(save_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    step_dir = base_path / f"checkpoint-step-{step}"
    staging_dir = Path(tempfile.mkdtemp(prefix=f".checkpoint-step-{step}-", dir=base_path))

    # 1. Model weights
    model_path = staging_dir / "model.pt"
    # Extract bare state_dict if wrapped in DDP or module
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(raw_model.state_dict(), model_path)

    # 2. Training state
    training_state = {
        "step": step,
        "epoch": epoch,
        "metrics": metrics or {},
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "rng_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(training_state, staging_dir / "training_state.pt")

    # 3. Metadata JSON
    metadata = {
        "step": step,
        "epoch": epoch,
        "metrics": metrics or {},
    }
    with open(staging_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Verify all files are present and readable before publishing the checkpoint.
    try:
        if not all((staging_dir / name).is_file() for name in ("model.pt", "training_state.pt", "metadata.json")):
            raise RuntimeError("checkpoint files missing after save")
        torch.load(staging_dir / "model.pt", map_location="cpu", weights_only=True)
        torch.load(staging_dir / "training_state.pt", map_location="cpu", weights_only=False)
        json.loads((staging_dir / "metadata.json").read_text(encoding="utf-8"))
        if step_dir.exists():
            shutil.rmtree(step_dir)
        os.replace(staging_dir, step_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    # 4. Rotation policy: keep only the latest keep_last_n checkpoint directories
    if keep_last_n > 0:
        all_ckpts = sorted(
            [d for d in base_path.glob("checkpoint-step-*") if d.is_dir()],
            key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0,
        )
        if len(all_ckpts) > keep_last_n:
            for old_ckpt in all_ckpts[:-keep_last_n]:
                shutil.rmtree(old_ckpt, ignore_errors=True)

    return step_dir


def load_checkpoint(
    checkpoint_dir: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Restore model, optimizer, scheduler, and RNG state from a checkpoint.

    Args:
        checkpoint_dir: Path to the specific checkpoint directory.
        model: Model instance to populate.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        scaler: Optional GradScaler to restore.
        device: Device to map tensors onto.

    Returns:
        Dictionary with restored training state metadata.
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    model_file = ckpt_path / "model.pt"
    state_file = ckpt_path / "training_state.pt"

    if not model_file.exists():
        raise FileNotFoundError(f"Model state file not found in: {ckpt_path}")

    # 1. Restore model state
    map_loc = device or "cpu"
    model_state = torch.load(model_file, map_location=map_loc, weights_only=True)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(model_state)

    training_meta = {"step": 0, "epoch": 0, "metrics": {}}

    # 2. Restore training state if available
    if state_file.exists():
        training_state = torch.load(state_file, map_location=map_loc, weights_only=False)
        training_meta["step"] = training_state.get("step", 0)
        training_meta["epoch"] = training_state.get("epoch", 0)
        training_meta["metrics"] = training_state.get("metrics", {})

        if optimizer and training_state.get("optimizer_state_dict"):
            optimizer.load_state_dict(training_state["optimizer_state_dict"])

        if scheduler and training_state.get("scheduler_state_dict"):
            scheduler.load_state_dict(training_state["scheduler_state_dict"])

        if scaler and training_state.get("scaler_state_dict"):
            scaler.load_state_dict(training_state["scaler_state_dict"])

        # Restore RNG states
        rng_data = training_state.get("rng_states", {})
        if "python" in rng_data:
            random.setstate(rng_data["python"])
        if "numpy" in rng_data:
            np.random.set_state(rng_data["numpy"])
        if "torch_cpu" in rng_data and rng_data["torch_cpu"] is not None:
            torch.set_rng_state(rng_data["torch_cpu"])
        if "torch_cuda" in rng_data and rng_data["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_data["torch_cuda"])

    return training_meta
