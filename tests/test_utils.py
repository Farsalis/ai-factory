"""Comprehensive tests for utils.py module.

Tests cover all functions and classes including:
- bitsandbytes availability detection
- Environment class initialization and methods
- Device name detection with error handling
- Backend configuration for CUDA
- String representation
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

import src.utils as utils


@contextmanager
def _tf32_restore_after_false() -> Iterator[None]:
    """Set ``allow_tf32`` False before the block, restore previous value after."""
    matmul = torch.backends.cuda.matmul
    if not hasattr(matmul, "allow_tf32"):
        yield
        return
    previous = bool(matmul.allow_tf32)
    matmul.allow_tf32 = False
    try:
        yield
    finally:
        matmul.allow_tf32 = previous


# ============================================================================
# Tests for is_bitsandbytes_available
# ============================================================================


@pytest.mark.unit
def test_is_bitsandbytes_available_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test is_bitsandbytes_available returns True when library is available."""
    monkeypatch.setattr(utils.importlib.util, "find_spec", lambda name: object())
    assert utils.is_bitsandbytes_available() is True


@pytest.mark.unit
def test_is_bitsandbytes_available_when_not_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test is_bitsandbytes_available returns False when library is not available."""
    monkeypatch.setattr(utils.importlib.util, "find_spec", lambda name: None)
    assert utils.is_bitsandbytes_available() is False


# ============================================================================
# Tests for Environment.__init__
# ============================================================================


@pytest.mark.unit
def test_environment_cpu_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Environment initialization with CPU (no CUDA)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: False)

    env = utils.Environment()

    assert env.cuda_available is False
    assert env.device_name == "CPU"
    assert env.compute_dtype == torch.float16
    assert env.bf16_supported is False
    assert env.bnb_available is False


@pytest.mark.unit
def test_environment_cuda_available_bf16_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Environment initialization with CUDA and bf16 support."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "Test GPU")
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    env = utils.Environment()

    assert env.cuda_available is True
    assert env.bf16_supported is True
    assert env.compute_dtype == torch.bfloat16
    assert env.bnb_available is True
    assert env.device_name == "Test GPU"


@pytest.mark.unit
def test_environment_cuda_available_no_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Environment initialization with CUDA but no bf16 support."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "Test GPU")
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    env = utils.Environment()

    assert env.cuda_available is True
    assert env.bf16_supported is False
    assert env.compute_dtype == torch.float16
    assert env.device_name == "Test GPU"


@pytest.mark.unit
def test_environment_windows_prefers_float16(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Environment on Windows always prefers float16 even with bf16 support."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "Test GPU")
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Windows")

    env = utils.Environment()

    assert env.cuda_available is True
    assert env.bf16_supported is True
    # Windows should prefer float16 regardless of bf16 support
    assert env.compute_dtype == torch.float16


@pytest.mark.unit
def test_environment_non_windows_with_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Environment on non-Windows uses bf16 when supported."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "Test GPU")
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    env = utils.Environment()

    assert env.compute_dtype == torch.bfloat16


@pytest.mark.unit
def test_environment_non_windows_without_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Environment on non-Windows uses float16 when bf16 not supported."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "Test GPU")
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    env = utils.Environment()

    assert env.compute_dtype == torch.float16


@pytest.mark.unit
def test_environment_bitsandbytes_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Environment correctly detects bitsandbytes availability."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    # Test with bitsandbytes available
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    env = utils.Environment()
    assert env.bnb_available is True

    # Test with bitsandbytes not available
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: False)
    env = utils.Environment()
    assert env.bnb_available is False


# ============================================================================
# Tests for Environment._get_device_name
# ============================================================================


@pytest.mark.unit
def test_get_device_name_cuda_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _get_device_name returns 'CPU' when CUDA is not available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: False)

    env = utils.Environment()
    assert env.device_name == "CPU"


@pytest.mark.unit
def test_get_device_name_cuda_available_with_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test _get_device_name returns device name when CUDA is available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "NVIDIA RTX 4090")
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    env = utils.Environment()
    assert env.device_name == "NVIDIA RTX 4090"


@pytest.mark.unit
def test_get_device_name_cuda_available_no_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test _get_device_name returns 'CPU' when device_count is 0."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    env = utils.Environment()
    assert env.device_name == "CPU"


@pytest.mark.unit
def test_get_device_name_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _get_device_name handles RuntimeError and falls back to CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda idx: (_ for _ in ()).throw(RuntimeError("Device error")),
    )
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    env = utils.Environment()
    assert env.device_name == "CPU"


@pytest.mark.unit
def test_get_device_name_attribute_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _get_device_name handles AttributeError and falls back to CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda idx: (_ for _ in ()).throw(AttributeError("No attribute")),
    )
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    env = utils.Environment()
    assert env.device_name == "CPU"


@pytest.mark.unit
def test_get_device_name_device_count_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _get_device_name handles error in device_count and falls back to CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: (_ for _ in ()).throw(RuntimeError("Device count error")),
    )
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    env = utils.Environment()
    assert env.device_name == "CPU"


# ============================================================================
# Tests for Environment.setup_backends
# ============================================================================


@pytest.mark.unit
def test_setup_backends_cuda_not_available(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test setup_backends logs message when CUDA is not available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: False)

    with caplog.at_level(logging.INFO, logger="src.utils"):
        env = utils.Environment()
        env.setup_backends()

    assert "CUDA not available" in caplog.text or "Using CPU" in caplog.text


@pytest.mark.unit
def test_setup_backends_ampere_plus_enables_tf32(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test setup_backends enables TF32 for Ampere+ GPUs (compute capability >= 8.0)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "NVIDIA A100")
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda idx: (8, 0)
    )  # Ampere
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    with (
        _tf32_restore_after_false(),
        patch("torch.set_float32_matmul_precision") as mock_set_precision,
    ):
        with caplog.at_level(logging.INFO, logger="src.utils"):
            env = utils.Environment()
            env.setup_backends()

        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            assert torch.backends.cuda.matmul.allow_tf32 is True
        mock_set_precision.assert_called_once_with("high")
        assert "TF32" in caplog.text or "Ampere" in caplog.text


@pytest.mark.unit
def test_setup_backends_ampere_plus_higher_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test setup_backends enables TF32 for GPUs with compute capability > 8.0."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "NVIDIA H100")
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda idx: (9, 0)
    )  # Hopper
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    with (
        _tf32_restore_after_false(),
        patch("torch.set_float32_matmul_precision") as mock_set_precision,
    ):
        env = utils.Environment()
        env.setup_backends()

        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            assert torch.backends.cuda.matmul.allow_tf32 is True
        mock_set_precision.assert_called_once_with("high")


@pytest.mark.unit
def test_setup_backends_pre_ampere_no_tf32(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test setup_backends does not enable TF32 for pre-Ampere GPUs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "NVIDIA GTX 1080")
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda idx: (6, 1)
    )  # Pascal
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    with (
        _tf32_restore_after_false(),
        patch("torch.set_float32_matmul_precision") as mock_set_precision,
    ):
        with caplog.at_level(logging.DEBUG, logger="src.utils"):
            env = utils.Environment()
            env.setup_backends()

        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            assert torch.backends.cuda.matmul.allow_tf32 is False
        mock_set_precision.assert_not_called()
        assert (
            "TF32 not enabled" in caplog.text
            or "compute capability" in caplog.text.lower()
        )


@pytest.mark.unit
def test_setup_backends_capability_7_5_no_tf32(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test setup_backends does not enable TF32 for compute capability 7.5."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "NVIDIA RTX 2080")
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda idx: (7, 5)
    )  # Turing
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    with (
        _tf32_restore_after_false(),
        patch("torch.set_float32_matmul_precision") as mock_set_precision,
    ):
        env = utils.Environment()
        env.setup_backends()

        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            assert torch.backends.cuda.matmul.allow_tf32 is False
        mock_set_precision.assert_not_called()


@pytest.mark.unit
def test_setup_backends_runtime_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test setup_backends handles RuntimeError when getting device capability."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "Test GPU")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda idx: (_ for _ in ()).throw(RuntimeError("Capability error")),
    )
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    env = utils.Environment()
    env.setup_backends()

    assert (
        "Failed to configure CUDA backends" in caplog.text
        or "error" in caplog.text.lower()
    )


@pytest.mark.unit
def test_setup_backends_attribute_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test setup_backends handles AttributeError when getting device capability."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "Test GPU")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda idx: (_ for _ in ()).throw(AttributeError("No attribute")),
    )
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    env = utils.Environment()
    env.setup_backends()

    assert (
        "Failed to configure CUDA backends" in caplog.text
        or "error" in caplog.text.lower()
    )


@pytest.mark.unit
def test_setup_backends_logs_device_info(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test setup_backends logs device information when CUDA is available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "NVIDIA RTX 4090")
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda idx: (8, 9)
    )  # Ada Lovelace
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    with (
        _tf32_restore_after_false(),
        patch("torch.set_float32_matmul_precision"),
    ):
        with caplog.at_level(logging.INFO, logger="src.utils"):
            env = utils.Environment()
            env.setup_backends()

        assert "CUDA is available" in caplog.text or "device" in caplog.text.lower()


# ============================================================================
# Tests for Environment.__repr__
# ============================================================================


@pytest.mark.unit
def test_environment_repr_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Environment.__repr__ for CPU environment."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: False)

    env = utils.Environment()
    repr_str = repr(env)

    assert "Environment" in repr_str
    assert "cuda=False" in repr_str
    assert "bnb=False" in repr_str
    assert "bf16=False" in repr_str
    assert "device=CPU" in repr_str


@pytest.mark.unit
def test_environment_repr_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Environment.__repr__ for CUDA environment."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "NVIDIA RTX 4090")
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    env = utils.Environment()
    repr_str = repr(env)

    assert "Environment" in repr_str
    assert "cuda=True" in repr_str
    assert "bnb=True" in repr_str
    assert "bf16=True" in repr_str
    assert "device=NVIDIA RTX 4090" in repr_str
    assert "dtype" in repr_str


@pytest.mark.unit
def test_environment_repr_includes_all_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Environment.__repr__ includes all key attributes."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "Test GPU")
    monkeypatch.setattr(utils, "is_bitsandbytes_available", lambda: True)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    env = utils.Environment()
    repr_str = repr(env)

    # Check all attributes are present
    assert "cuda=" in repr_str
    assert "bnb=" in repr_str
    assert "bf16=" in repr_str
    assert "dtype=" in repr_str
    assert "device=" in repr_str
