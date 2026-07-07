"""Tests for seed utilities."""

import torch
import numpy as np

from udl.utils.seed import set_seed


class TestSetSeed:
    """Test set_seed function."""

    def test_torch_reproducibility(self):
        """Setting seed should make torch random numbers reproducible."""
        set_seed(42)
        a = torch.randn(10)

        set_seed(42)
        b = torch.randn(10)

        assert torch.allclose(a, b)

    def test_numpy_reproducibility(self):
        """Setting seed should make numpy random numbers reproducible."""
        set_seed(42)
        a = np.random.randn(10)

        set_seed(42)
        b = np.random.randn(10)

        assert np.allclose(a, b)

    def test_different_seeds_different_results(self):
        """Different seeds should produce different results."""
        set_seed(42)
        a = torch.randn(10)

        set_seed(123)
        b = torch.randn(10)

        assert not torch.allclose(a, b)

    def test_deterministic_mode(self):
        """Deterministic mode should not raise errors."""
        # Just ensure it doesn't crash
        set_seed(42, deterministic=True)
        _ = torch.randn(10)
