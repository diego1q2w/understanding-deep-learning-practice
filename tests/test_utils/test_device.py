"""Tests for device utilities."""

import torch
from udl.utils.device import get_device, device_info


class TestGetDevice:
    """Test get_device function."""

    def test_returns_torch_device(self):
        """get_device should return a torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_preferred_cpu(self):
        """Should return CPU when explicitly requested."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_auto_detect_returns_valid_device(self):
        """Auto-detected device should be usable."""
        device = get_device()
        # Should be able to create a tensor on this device
        tensor = torch.zeros(1, device=device)
        assert tensor.device.type == device.type


class TestDeviceInfo:
    """Test device_info function."""

    def test_returns_dict(self):
        """device_info should return a dictionary."""
        info = device_info()
        assert isinstance(info, dict)

    def test_contains_required_keys(self):
        """device_info should contain required keys."""
        info = device_info()
        assert "cuda_available" in info
        assert "mps_available" in info
        assert "selected" in info

    def test_selected_matches_get_device(self):
        """Selected device should match get_device output."""
        info = device_info()
        device = get_device()
        assert info["selected"] == str(device)
