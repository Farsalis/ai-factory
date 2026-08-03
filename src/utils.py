"""Utility functions and environment detection for hardware/software setup.

This module provides helper functions for detecting available hardware capabilities
and configuring the environment for optimal PyTorch performance.
"""

import importlib.util
import logging
import platform

import torch

logger = logging.getLogger(__name__)


def is_bitsandbytes_available() -> bool:
    """Check if the bitsandbytes is installed.

    Returns:
        True if bitsandbytes is available, False otherwise.
    """
    return importlib.util.find_spec("bitsandbytes") is not None


class Environment:
    """Container for detected hardware and software environment.

    Automatically detects CUDA availability, bitsandbytes support, and optimal
    compute dtype based on the platform and hardware capabilities.

    Attributes:
        cuda_available: Whether CUDA is available.
        bnb_available: Whether bitsandbytes library is installed.
        bf16_supported: Whether bfloat16 is supported (requires CUDA).
        compute_dtype: Recommended compute dtype (float16 or bfloat16).
        device_name: Name of the CUDA device or "CPU".
    """

    def __init__(self) -> None:
        """Initialize environment detection."""
        self.cuda_available = torch.cuda.is_available()
        self.bnb_available = is_bitsandbytes_available()
        self.bf16_supported = self.cuda_available and torch.cuda.is_bf16_supported()

        # On Windows, prefer float16 for wider library compatibility
        # (e.g., bitsandbytes)
        if platform.system() == "Windows":
            self.compute_dtype = torch.float16
        else:
            self.compute_dtype = (
                torch.bfloat16 if self.bf16_supported else torch.float16
            )

        # Safely get device name, handling edge case where CUDA is available
        # but no device is present
        self.device_name = self._get_device_name()

    def _get_device_name(self) -> str:
        """Get the CUDA device name safely.

        Returns:
            Device name if CUDA is available and device exists, "CPU" otherwise.
        """
        if not self.cuda_available:
            return "CPU"

        try:
            if torch.cuda.device_count() > 0:
                return torch.cuda.get_device_name(0)
        except (RuntimeError, AttributeError) as e:
            logger.warning(
                f"CUDA reported as available but device access failed: {e}. "
                "Falling back to CPU."
            )

        return "CPU"

    def setup_backends(self) -> None:
        """Configure PyTorch backends for optimal performance.

        Enables TF32 for Ampere+ GPUs (compute capability >= 8.0) which provides
        improved performance with minimal accuracy loss.
        """
        if not self.cuda_available:
            logger.info("CUDA not available. Using CPU.")
            return

        logger.info(f"CUDA is available. Using device: {self.device_name}")

        try:
            # torch.cuda.get_device_capability returns a (major, minor) tuple
            major, minor = torch.cuda.get_device_capability(0)
            if (major, minor) >= (8, 0):
                logger.info("Enabling TF32 for Ampere+ GPUs for improved performance.")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            else:
                logger.debug(
                    f"GPU compute capability ({major}.{minor}) < 8.0. TF32 not enabled."
                )
        except (RuntimeError, AttributeError) as e:
            logger.warning(
                f"Failed to configure CUDA backends: {e}. "
                "Continuing with default settings."
            )

    def __repr__(self) -> str:
        """Return a string representation of the environment.

        Returns:
            String representation showing key environment attributes.
        """
        return (
            f"Environment(cuda={self.cuda_available}, "
            f"bnb={self.bnb_available}, "
            f"bf16={self.bf16_supported}, "
            f"dtype={self.compute_dtype}, "
            f"device={self.device_name})"
        )
