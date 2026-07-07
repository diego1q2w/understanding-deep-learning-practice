"""Reproducibility utilities for setting random seeds."""

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility across PyTorch, NumPy, and Python.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but reproducible)

    Example:
        >>> set_seed(42)  # Basic reproducibility
        >>> set_seed(42, deterministic=True)  # Full determinism (slower)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.8+ deterministic mode
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
