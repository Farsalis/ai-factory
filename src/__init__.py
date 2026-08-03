"""
Training package for QLoRA fine-tuning pipeline.

This file makes the `training` directory a Python package so that
relative imports (e.g., `from .config import ScriptConfig`) work when
running as a module: `python -m training.main`.
"""

# --- Native library import-order guard (must run before any torch import) ---
#
# On Windows, importing `torch` *before* `datasets` (which loads HuggingFace's
# native extensions / Arrow-adjacent DLLs) causes a hard interpreter crash
# (exit code 139 / access violation) with no Python traceback. The reverse
# order (`datasets` then `torch`) is safe. See:
#     import torch;  import datasets   -> segfault
#     import datasets; import torch    -> ok
#
# Because this package is imported (via `src/__init__.py`) before any `src.*`
# submodule — and the entry points defer their `import torch` until after the
# first `from src... import` — pre-importing `datasets` here guarantees the
# safe load order for every code path (pipeline, DPO, inference, tests).
#
# Wrapped in a guard so lightweight tooling that lacks `datasets` installed can
# still import the package; the real pipeline will surface a clear ImportError
# later if the dependency is genuinely missing.
try:  # pragma: no cover - environment-dependent native load order
    import datasets as _datasets  # noqa: F401
except ImportError:
    pass

__all__: list[str] = []
