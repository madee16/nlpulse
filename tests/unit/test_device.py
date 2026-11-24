"""Unit tests for hardware detection and DeviceManager."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.device import DeviceManager, HardwareInfo
from src.core.exceptions import DeviceUnavailableError


def test_device_manager_defaults_to_cpu_when_no_cuda():
    """Verify that auto preference falls back to CPU when CUDA is unavailable."""
    with patch.object(DeviceManager, "is_cuda_available", return_value=False):
        dm = DeviceManager(preference="auto")
        assert dm.target_device == "cpu"
        assert dm.preference == "auto"

        hw = dm.get_hardware_info()
        assert hw.device_type == "cpu"
        assert hw.cuda_available is False
        assert hw.device_name == "CPU"


def test_device_manager_selects_cuda_when_available():
    """Verify that auto preference selects CUDA when CUDA is available."""
    mock_props = MagicMock()
    mock_props.total_memory = 6 * 1024 * 1024 * 1024  # 6 GB
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4050 Laptop GPU"
    mock_torch.cuda.get_device_properties.return_value = mock_props
    mock_torch.cuda.memory_allocated.return_value = 500 * 1024 * 1024
    mock_torch.cuda.memory_reserved.return_value = 1024 * 1024 * 1024

    with patch.dict("sys.modules", {"torch": mock_torch}):
        with patch.object(DeviceManager, "is_cuda_available", return_value=True):
            dm = DeviceManager(preference="auto")
            assert dm.target_device == "cuda"

            hw = dm.get_hardware_info()
            assert hw.device_type == "cuda"
            assert hw.cuda_available is True
            assert hw.device_name == "NVIDIA GeForce RTX 4050 Laptop GPU"
            assert hw.device_count == 1
            assert hw.total_vram_mb is not None
            assert hw.total_vram_mb > 5000.0


def test_device_manager_explicit_cuda_unavailable_raises_error():
    """Verify requesting CUDA explicitly when unavailable raises DeviceUnavailableError."""
    with patch.object(DeviceManager, "is_cuda_available", return_value=False):
        with pytest.raises(DeviceUnavailableError) as exc_info:
            DeviceManager(preference="cuda")
        assert exc_info.value.status_code == 400
        assert "cuda" in str(exc_info.value.message)


def test_device_manager_resolve_device_override():
    """Verify resolving device with runtime overrides."""
    with patch.object(DeviceManager, "is_cuda_available", return_value=False):
        dm = DeviceManager(preference="cpu")
        assert dm.resolve_device(None) == "cpu"
        assert dm.resolve_device("cpu") == "cpu"
        assert dm.resolve_device("auto") == "cpu"

        with pytest.raises(DeviceUnavailableError):
            dm.resolve_device("cuda")

        with pytest.raises(DeviceUnavailableError):
            dm.resolve_device("tpu")


def test_device_manager_cuda_oom_mitigation():
    """Verify handle_cuda_oom executes garbage collection and empty_cache."""
    mock_torch = MagicMock()
    with patch.dict("sys.modules", {"torch": mock_torch}):
        with patch.object(DeviceManager, "is_cuda_available", return_value=True):
            dm = DeviceManager(preference="cpu")
            dm.handle_cuda_oom()
            mock_torch.cuda.empty_cache.assert_called_once()
