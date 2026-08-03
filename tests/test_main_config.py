"""Tests for loading and validating configuration from YAML."""

from pathlib import Path

import pytest

pytest.importorskip("yaml")
pytest.importorskip("pydantic")

from src.main import _find_model_path, load_config_from_yaml

_TESTS_DIR = Path(__file__).resolve().parent
TEST_CONFIG_YAML = _TESTS_DIR / "configs" / "test_config.yaml"

_MINIMAL_BODY = """
data:
  train_file: train.jsonl
  validation_file: val.jsonl
model:
  name: mistralai/Mistral-7B-Instruct-v0.3
  max_length: 256
  attn_implementation: sdpa
  trust_remote_code: true
quantization:
  enabled: false
  quant_type: nf4
  use_double_quant: true
lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules:
    - q_proj
    - v_proj
training:
  output_dir: training_out
  seed: 42
  num_train_epochs: 1
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 1
  optim: adamw_torch
  learning_rate: 0.0002
  weight_decay: 0.0
  max_grad_norm: 1.0
  warmup_ratio: 0.0
  lr_scheduler_type: constant
  evaluation_strategy: steps
  eval_steps: 10
  save_strategy: steps
  save_steps: 10
  save_total_limit: 1
  logging_steps: 5
  group_by_length: false
  gradient_checkpointing: false
  report_to: none
  load_best_model_at_end: false
  metric_for_best_model: eval_loss
  greater_is_better: false
  remove_unused_columns: true
  dataloader_num_workers: 0
"""


def _write_minimal_config_tree(cfg_dir: Path, extra_yaml: str = "") -> Path:
    """Create train/val JSONL files and config.yaml; return path to config."""
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (cfg_dir / "val.jsonl").write_text("{}\n", encoding="utf-8")
    (cfg_dir / "config.yaml").write_text(_MINIMAL_BODY + extra_yaml, encoding="utf-8")
    return cfg_dir / "config.yaml"


@pytest.mark.unit
def test_load_config_from_yaml_resolves_relative_paths() -> None:
    """Relative paths in YAML are resolved against the config file directory."""
    config = load_config_from_yaml(TEST_CONFIG_YAML)

    assert config.data.train_file.is_absolute()
    assert config.data.validation_file.is_absolute()
    assert config.training.output_dir.is_absolute()
    assert config.model.name == "mistralai/Mistral-7B-Instruct-v0.3"


@pytest.mark.unit
def test_load_config_from_yaml_file_not_found(tmp_path: Path) -> None:
    """Missing config file raises FileNotFoundError."""
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config_from_yaml(missing)


@pytest.mark.unit
def test_load_config_from_yaml_empty_file(tmp_path: Path) -> None:
    """Empty YAML file raises ValueError."""
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_config_from_yaml(p)


@pytest.mark.unit
def test_load_config_from_yaml_invalid_syntax(tmp_path: Path) -> None:
    """Invalid YAML raises ValueError wrapping parse error."""
    p = tmp_path / "bad.yaml"
    p.write_text("[ unclosed bracket", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config_from_yaml(p)


@pytest.mark.unit
def test_load_config_from_yaml_validation_failure(tmp_path: Path) -> None:
    """Schema validation errors surface as ValueError."""
    p = tmp_path / "incomplete.yaml"
    p.write_text("data: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid configuration"):
        load_config_from_yaml(p)


@pytest.mark.unit
def test_load_config_from_yaml_resolves_dpo_relative_paths(tmp_path: Path) -> None:
    """dpo.train_file and dpo.output_dir are resolved relative to the config file."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "dpo_prefs.jsonl").write_text("{}\n", encoding="utf-8")
    extra = """
dpo:
  train_file: dpo_prefs.jsonl
  output_dir: dpo_runs
"""
    config_path = _write_minimal_config_tree(cfg_dir, extra_yaml=extra)
    config = load_config_from_yaml(config_path)
    assert config.dpo is not None
    assert config.dpo.train_file.is_absolute()
    assert config.dpo.train_file.resolve() == (cfg_dir / "dpo_prefs.jsonl").resolve()
    assert config.dpo.output_dir is not None
    assert config.dpo.output_dir.is_absolute()
    assert config.dpo.output_dir.resolve() == (cfg_dir / "dpo_runs").resolve()


@pytest.mark.unit
def test_find_model_path_prefers_dpo(tmp_path: Path) -> None:
    """When dpo_model exists under output_dir, it is chosen over final_merged_model."""
    cfg_dir = tmp_path / "cfg"
    config_path = _write_minimal_config_tree(cfg_dir)
    config = load_config_from_yaml(config_path)
    out = config.training.output_dir
    (out / "dpo_model").mkdir(parents=True)
    (out / "final_merged_model").mkdir(parents=True)
    chosen = _find_model_path(config)
    assert chosen.resolve() == (out / "dpo_model").resolve()


@pytest.mark.unit
def test_find_model_path_falls_back_to_merged(tmp_path: Path) -> None:
    """When only final_merged_model exists, that path is returned."""
    cfg_dir = tmp_path / "cfg"
    config_path = _write_minimal_config_tree(cfg_dir)
    config = load_config_from_yaml(config_path)
    out = config.training.output_dir
    (out / "final_merged_model").mkdir(parents=True)
    chosen = _find_model_path(config)
    assert chosen.resolve() == (out / "final_merged_model").resolve()


@pytest.mark.unit
def test_find_model_path_raises_when_no_model_dirs(tmp_path: Path) -> None:
    """FileNotFoundError lists checked paths when neither model directory exists."""
    cfg_dir = tmp_path / "cfg"
    config_path = _write_minimal_config_tree(cfg_dir)
    config = load_config_from_yaml(config_path)
    with pytest.raises(FileNotFoundError, match="No model found"):
        _find_model_path(config)


@pytest.mark.unit
def test_find_model_path_artifact_contract_dpo_over_merged(tmp_path: Path) -> None:
    """Artifact contract: dpo_model is always preferred when it exists."""
    cfg_dir = tmp_path / "cfg"
    config_path = _write_minimal_config_tree(cfg_dir)
    config = load_config_from_yaml(config_path)
    out = config.training.output_dir

    (out / "dpo_model").mkdir(parents=True)
    (out / "final_merged_model").mkdir(parents=True)

    chosen = _find_model_path(config)
    assert chosen.name == "dpo_model"


@pytest.mark.unit
def test_find_model_path_artifact_contract_merged_only(tmp_path: Path) -> None:
    """Artifact contract: final_merged_model is used when dpo_model is absent."""
    cfg_dir = tmp_path / "cfg"
    config_path = _write_minimal_config_tree(cfg_dir)
    config = load_config_from_yaml(config_path)
    out = config.training.output_dir

    (out / "final_merged_model").mkdir(parents=True)

    chosen = _find_model_path(config)
    assert chosen.name == "final_merged_model"
