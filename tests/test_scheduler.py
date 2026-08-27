"""Unit tests for warmup-cosine learning rate scheduler."""

import pytest
import torch
from torch.optim import AdamW

from src.scheduler import get_cosine_schedule_with_warmup


def test_scheduler_warmup_and_cosine_decay():
    """Verify linear warmup slope and cosine decay toward min_lr."""
    model = torch.nn.Linear(10, 10)
    base_lr = 1e-3
    optimizer = AdamW(model.parameters(), lr=base_lr)
    warmup_steps = 10
    max_steps = 100
    min_lr_ratio = 0.1

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr_ratio=min_lr_ratio,
    )

    # Step 0: LR should be 0
    assert scheduler.get_last_lr()[0] == 0.0

    # Step 5 (halfway through warmup): LR should be 0.5 * base_lr
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    assert pytest.approx(scheduler.get_last_lr()[0], rel=1e-3) == 0.5 * base_lr

    # Step 10 (end of warmup): LR should equal base_lr
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    assert pytest.approx(scheduler.get_last_lr()[0], rel=1e-3) == base_lr

    # Step 100 (end of schedule): LR should equal min_lr_ratio * base_lr
    for _ in range(90):
        optimizer.step()
        scheduler.step()
    assert pytest.approx(scheduler.get_last_lr()[0], rel=1e-3) == min_lr_ratio * base_lr


def test_scheduler_zero_warmup():
    """Verify scheduler behavior with warmup_steps=0."""
    model = torch.nn.Linear(10, 10)
    base_lr = 1e-3
    optimizer = AdamW(model.parameters(), lr=base_lr)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=0,
        max_steps=50,
        min_lr_ratio=0.05,
    )

    # At step 0, with 0 warmup, lr should start at base_lr
    assert pytest.approx(scheduler.get_last_lr()[0], rel=1e-3) == base_lr


def test_scheduler_beyond_max_steps():
    """Verify scheduler maintains min_lr beyond max_steps."""
    model = torch.nn.Linear(10, 10)
    base_lr = 1e-3
    optimizer = AdamW(model.parameters(), lr=base_lr)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=5,
        max_steps=20,
        min_lr_ratio=0.1,
    )

    for _ in range(30):
        optimizer.step()
        scheduler.step()

    # Step 30 is > max_steps (20), should stay at min_lr (1e-4)
    assert pytest.approx(scheduler.get_last_lr()[0], rel=1e-3) == 1e-4
