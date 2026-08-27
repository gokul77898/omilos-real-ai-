"""Seed utilities for reproducible random number generation across backends."""

from __future__ import annotations

import os
import random
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seeds across Python random, NumPy, and PyTorch (CPU & CUDA).

    Note on Determinism:
        Setting seeds provides pseudo-random sequence consistency within the same
        execution environment and hardware architecture. However, full bitwise
        reproducibility across different GPU architectures, CUDA versions, cuDNN
        kernel implementations, or distributed multi-node topologies is not
        guaranteed by seeding alone.

    Args:
        seed: Integer seed value to apply.
        deterministic: If True, enables PyTorch deterministic algorithms and disables
            cuDNN benchmarking for enhanced reproducibility at the expense of performance.
    """
    # 1. Python built-in random
    random.seed(seed)

    # 2. Environment hash seed for dict/set ordering
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 3. NumPy
    np.random.seed(seed)

    # 4. PyTorch CPU
    torch.manual_seed(seed)

    # 5. PyTorch CUDA (all available devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 6. Optional strict deterministic configuration
    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
