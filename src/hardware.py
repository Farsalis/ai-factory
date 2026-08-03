"""Hardware profile for the Model Optimizer.

Extends Environment with VRAM, system RAM, GPU count, compute capability,
and OS so the recommendation engine can suggest config.yaml settings.
See OVERVIEW.md for stability rules applied by the optimizer.
"""

import logging
import platform

import torch

from src.utils import Environment

logger = logging.getLogger(__name__)

# Default when CUDA is unavailable
VRAM_DEFAULT_BYTES = 0


def _get_vram_bytes() -> int:
    """Get total GPU VRAM in bytes for device 0, or 0 if no CUDA."""
    if not torch.cuda.is_available():
        return VRAM_DEFAULT_BYTES
    try:
        if torch.cuda.device_count() > 0:
            return torch.cuda.get_device_properties(0).total_memory
    except (RuntimeError, AttributeError) as e:
        logger.warning("Could not get GPU VRAM: %s", e)
    return VRAM_DEFAULT_BYTES


def _get_compute_capability() -> tuple[int, int] | None:
    """Get (major, minor) compute capability for device 0, or None if no CUDA."""
    if not torch.cuda.is_available():
        return None
    try:
        if torch.cuda.device_count() > 0:
            return torch.cuda.get_device_capability(0)
    except (RuntimeError, AttributeError):
        pass
    return None


def _get_system_ram_bytes() -> int:
    """Get total system RAM in bytes via psutil."""
    try:
        import psutil

        return psutil.virtual_memory().total
    except Exception as e:
        logger.warning("Could not get system RAM: %s", e)
        return 0


class HardwareProfile(Environment):
    """Hardware profile extending Environment with VRAM, RAM, OS, and GPU details.

    Used by the Model Optimizer to recommend config settings. See plan and
    TRAINING_CRASH_DIAGNOSIS.md for how these values drive recommendations.

    :attributes:
        Inherits all Environment attributes (cuda_available, bnb_available,
        bf16_supported, compute_dtype, device_name).
        vram_bytes: Total GPU VRAM in bytes (0 if no CUDA).
        system_ram_bytes: Total system RAM in bytes.
        gpu_count: Number of CUDA devices.
        compute_capability: (major, minor) for device 0, or None.
        os_name: platform.system() (e.g. "Windows", "Linux").
    """

    def __init__(self) -> None:
        super().__init__()
        self.vram_bytes = _get_vram_bytes()
        self.system_ram_bytes = _get_system_ram_bytes()
        self.gpu_count = torch.cuda.device_count() if self.cuda_available else 0
        self.compute_capability = _get_compute_capability()
        self.os_name = platform.system()

    def __repr__(self) -> str:
        return (
            f"HardwareProfile({super().__repr__()}, "
            f"vram_bytes={self.vram_bytes}, system_ram_bytes={self.system_ram_bytes}, "
            f"gpu_count={self.gpu_count}, os={self.os_name!r})"
        )
