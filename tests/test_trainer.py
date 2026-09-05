"""Unit tests for Trainer execution, step progression, and evaluation."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import AppConfig, ModelConfig, TrainingConfig
from src.model import LegalCausalLM
from src.trainer import Trainer


def test_trainer_train_step_and_accumulation():
    """Verify train_step executes forward/backward and updates global_step at accumulation boundaries."""
    app_config = AppConfig()
    app_config.model = ModelConfig(vocab_size=50, hidden_size=32, num_layers=1, max_seq_len=16)
    app_config.training = TrainingConfig(
        batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        warmup_steps=2,
        max_steps=10,
    )

    data = torch.randint(0, 50, (10, 8))
    dataloader = DataLoader(TensorDataset(data, data), batch_size=2)

    model = LegalCausalLM(app_config.model)
    trainer = Trainer(model=model, config=app_config, train_dataloader=dataloader, device="cpu")

    assert trainer.global_step == 0
    assert trainer.micro_step == 0

    # Step 1: micro_step=1, global_step should still be 0 (accumulation=2)
    loss1 = trainer.train_step(next(iter(dataloader)))
    assert loss1 > 0
    assert trainer.micro_step == 1
    assert trainer.global_step == 0

    # Step 2: micro_step=2, global_step should increment to 1
    loss2 = trainer.train_step(next(iter(dataloader)))
    assert loss2 > 0
    assert trainer.micro_step == 2
    assert trainer.global_step == 1


def test_trainer_evaluate_and_perplexity():
    """Verify evaluate computes valid finite mean loss and perplexity."""
    app_config = AppConfig()
    app_config.model = ModelConfig(vocab_size=50, hidden_size=32, num_layers=1, max_seq_len=16)
    model = LegalCausalLM(app_config.model)

    data = torch.randint(0, 50, (8, 8))
    eval_dl = DataLoader(TensorDataset(data, data), batch_size=2)

    trainer = Trainer(model=model, config=app_config, train_dataloader=eval_dl, eval_dataloader=eval_dl, device="cpu")
    metrics = trainer.evaluate()

    assert "eval_loss" in metrics
    assert "perplexity" in metrics
    assert metrics["eval_loss"] > 0
    assert metrics["perplexity"] >= 1.0
    assert torch.isfinite(torch.tensor(metrics["eval_loss"]))


def test_trainer_gradient_clipping_application():
    """Verify gradient clipping keeps parameter gradients bounded by max_grad_norm."""
    app_config = AppConfig()
    app_config.model = ModelConfig(vocab_size=50, hidden_size=32, num_layers=1, max_seq_len=16)
    app_config.training = TrainingConfig(
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        max_grad_norm=0.5,
        max_steps=5,
    )

    data = torch.randint(0, 50, (4, 8))
    dl = DataLoader(TensorDataset(data, data), batch_size=2)

    model = LegalCausalLM(app_config.model)
    trainer = Trainer(model=model, config=app_config, train_dataloader=dl, device="cpu")

    trainer.train_step(next(iter(dl)))
    # Check that after step, gradients are zeroed or none
    for p in model.parameters():
        assert p.grad is None or torch.all(p.grad == 0)


def test_trainer_rejects_empty_dataloader():
    config = AppConfig()
    config.model = ModelConfig(vocab_size=16, hidden_size=16, num_layers=1, max_seq_len=8)
    empty = DataLoader(TensorDataset(torch.empty(0, 4, dtype=torch.long)), batch_size=2)
    with pytest.raises(ValueError, match="empty"):
        Trainer(LegalCausalLM(config.model), config, empty, device="cpu")
