from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from src.config import DataConfig


@pytest.mark.unit
def test_data_config_requires_files(tmp_path: Path) -> None:
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"
    train_file.write_text("{}\n", encoding="utf-8")
    val_file.write_text("{}\n", encoding="utf-8")

    config = DataConfig(train_file=train_file, validation_file=val_file)

    assert config.train_file == train_file
    assert config.validation_file == val_file


@pytest.mark.unit
def test_data_config_missing_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.jsonl"
    existing_file = tmp_path / "existing.jsonl"
    existing_file.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        DataConfig(train_file=missing_file, validation_file=existing_file)
