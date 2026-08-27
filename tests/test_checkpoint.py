"""Unit tests for checkpoint saving, loading, rotation, and end-to-end resume."""

from pathlib import Path
import tempfile
import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import AppConfig, ModelConfig, TrainingConfig
from src.model import LegalCausalLM
from src.scheduler import get_cosine_schedule_with_warmup
from src.trainer import Trainer


def test_checkpoint_save_and_load():
    """Verify full state restoration across model, optimizer, scheduler, and step metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ModelConfig(vocab_size=100, hidden_size=64, num_layers=2, max_seq_len=32)
        model = LegalCausalLM(config)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=5, max_steps=50)

        # Advance step
        for _ in range(10):
            optimizer.step()
            scheduler.step()

        saved_path = save_checkpoint(
            save_dir=tmpdir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=10,
            epoch=1,
            metrics={"loss": 2.45},
            keep_last_n=2,
        )

        assert (saved_path / "model.pt").exists()
        assert (saved_path / "training_state.pt").exists()
        assert (saved_path / "metadata.json").exists()

        # Create fresh model & optimizer to restore into
        restored_model = LegalCausalLM(config)
        restored_optimizer = AdamW(restored_model.parameters(), lr=1e-3)
        restored_scheduler = get_cosine_schedule_with_warmup(restored_optimizer, warmup_steps=5, max_steps=50)

        meta = load_checkpoint(
            checkpoint_dir=saved_path,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
        )

        assert meta["step"] == 10
        assert meta["epoch"] == 1
        assert meta["metrics"]["loss"] == 2.45

        # Verify weights match identically
        for p1, p2 in zip(model.parameters(), restored_model.parameters()):
            assert torch.equal(p1, p2)


def test_checkpoint_rotation_keep_last_n():
    """Verify rotation removes older checkpoints beyond keep_last_n."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ModelConfig(vocab_size=50, hidden_size=32, num_layers=1, max_seq_len=16)
        model = LegalCausalLM(config)
        base = Path(tmpdir)

        # Save 4 checkpoints with keep_last_n=2
        for step in [100, 200, 300, 400]:
            save_checkpoint(save_dir=base, model=model, step=step, keep_last_n=2)

        existing = sorted([d.name for d in base.glob("checkpoint-step-*")])
        assert existing == ["checkpoint-step-300", "checkpoint-step-400"]


def test_checkpoint_end_to_end_resume():
    """Verify training N steps, saving checkpoint, resuming into fresh trainer, and continuing seamlessly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        app_config = AppConfig()
        app_config.model = ModelConfig(vocab_size=50, hidden_size=32, num_layers=1, max_seq_len=16)
        app_config.training = TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            warmup_steps=2,
            max_steps=20,
        )
        app_config.checkpoint.output_dir = tmpdir

        data = torch.randint(0, 50, (20, 8))
        dl = DataLoader(TensorDataset(data, data), batch_size=2)

        # 1. Train first 5 steps
        model1 = LegalCausalLM(app_config.model)
        trainer1 = Trainer(model=model1, config=app_config, train_dataloader=dl, device="cpu")
        trainer1.train(max_steps=5)
        assert trainer1.global_step == 5

        # Save checkpoint
        saved_dir = save_checkpoint(
            save_dir=tmpdir,
            model=trainer1.model,
            optimizer=trainer1.optimizer,
            scheduler=trainer1.scheduler,
            scaler=trainer1.scaler,
            step=5,
            epoch=trainer1.epoch,
            config=app_config,
        )

        # 2. Resume in fresh trainer
        model2 = LegalCausalLM(app_config.model)
        trainer2 = Trainer(model=model2, config=app_config, train_dataloader=dl, device="cpu")

        meta = load_checkpoint(
            checkpoint_dir=saved_dir,
            model=trainer2.model,
            optimizer=trainer2.optimizer,
            scheduler=trainer2.scheduler,
            scaler=trainer2.scaler,
            device=trainer2.device,
        )
        trainer2.global_step = meta["step"]
        trainer2.epoch = meta["epoch"]
        assert trainer2.global_step == 5

        # 3. Continue training to step 10
        trainer2.train(max_steps=10)
        assert trainer2.global_step == 10
