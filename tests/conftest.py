"""Shared pytest fixtures for practiceudl tests."""

import pytest
import torch


@pytest.fixture
def cifar_batch():
    """Create a batch of CIFAR-10 sized images (3x32x32)."""
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def mnist_batch():
    """Create a batch of MNIST sized images (1x28x28)."""
    return torch.randn(4, 1, 28, 28)


@pytest.fixture
def classification_batch():
    """Create random logits and targets for classification testing."""
    logits = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    return logits, targets


@pytest.fixture
def device():
    """Get available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
