"""Import smoke tests for data-pipeline modules (CLI entrypoints guarded)."""

import importlib

import pytest

pytest.importorskip("transformers")


@pytest.mark.unit
def test_augment_dataset_module_imports() -> None:
    """``augment_dataset`` defines helpers; main is behind ``if __name__``."""
    importlib.import_module("src.data.augment_dataset")


@pytest.mark.unit
def test_generate_icdu_dataset_module_imports() -> None:
    """``generate_icdu_dataset`` is importable without running CLI."""
    importlib.import_module("src.data.generate_icdu_dataset")
