"""Device detection utilities for PyTorch."""

import torch


def get_device(preferred: str = None) -> torch.device:
    """
    Get the best available device for PyTorch.

    Args:
        preferred: "cuda", "mps", "cpu", or None for auto-detect

    Returns:
        torch.device: The selected device

    Example:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device("cpu")  # Force CPU
        >>> model.to(device)
    """
    if preferred:
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def device_info() -> dict:
    """
    Return information about available compute devices.

    Returns:
        dict: Device availability and selection info

    Example:
        >>> info = device_info()
        >>> print(info)
        {'cuda_available': False, 'mps_available': True, 'selected': 'mps'}
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "selected": str(get_device()),
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
    return info
