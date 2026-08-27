"""Tests for seed reproducibility across random, numpy, and torch."""

import random
import numpy as np
import torch

from src.seed import set_seed


def test_seed_random_determinism():
    """Verify repeated seeding produces identical sequences for Python random."""
    set_seed(1234)
    seq1 = [random.random() for _ in range(10)]

    set_seed(1234)
    seq2 = [random.random() for _ in range(10)]

    assert seq1 == seq2


def test_seed_numpy_determinism():
    """Verify repeated seeding produces identical arrays for NumPy."""
    set_seed(5678)
    arr1 = np.random.randn(5, 5)

    set_seed(5678)
    arr2 = np.random.randn(5, 5)

    np.testing.assert_array_equal(arr1, arr2)


def test_seed_torch_determinism():
    """Verify repeated seeding produces identical tensors for PyTorch."""
    set_seed(9999)
    t1 = torch.randn(10, 10)

    set_seed(9999)
    t2 = torch.randn(10, 10)

    assert torch.equal(t1, t2)


def test_different_seeds_produce_different_values():
    """Verify different seeds produce different pseudorandom values."""
    set_seed(100)
    t1 = torch.randn(5, 5)

    set_seed(200)
    t2 = torch.randn(5, 5)

    assert not torch.equal(t1, t2)
