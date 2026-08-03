"""Unit tests for Model Optimizer: hardware profile, recommendation engine, merge."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
TEST_CONFIG_YAML = _TESTS_DIR / "configs" / "test_config.yaml"

pytest.importorskip("yaml")
pytest.importorskip("pydantic")
pytest.importorskip("torch")

from src.hardware import HardwareProfile
from src.model_optimizer import (
    PRESET_CONFIG,
    PRESETS,
    _dataloader_workers,
    _max_batch_for_vram,
    _vram_tier,
    merge_config,
    recommend,
    run_optimizer,
)

# --- Hardware detection (mocked) ---


@pytest.mark.unit
def test_vram_tier() -> None:
    assert _vram_tier(0) == "low"
    assert _vram_tier(6 * 1024**3) == "low"
    assert _vram_tier(8 * 1024**3) == "medium"
    assert _vram_tier(12 * 1024**3) == "high"
    assert _vram_tier(24 * 1024**3) == "high"


@pytest.mark.unit
def test_max_batch_for_vram() -> None:
    # low VRAM
    assert _max_batch_for_vram(4 * 1024**3, "low_memory") == 2
    assert _max_batch_for_vram(4 * 1024**3, "fast") == 4
    # medium
    assert _max_batch_for_vram(10 * 1024**3, "quality") == 4
    # high
    assert _max_batch_for_vram(24 * 1024**3, "fast") == 8
    assert _max_batch_for_vram(24 * 1024**3, "quality") == 4


@pytest.mark.unit
def test_dataloader_workers_windows() -> None:
    profile = MagicMock(spec=HardwareProfile)
    profile.os_name = "Windows"
    profile.system_ram_bytes = 32 * 1024**3
    assert _dataloader_workers(profile, "low_memory") == 0
    assert _dataloader_workers(profile, "fast") == 2


@pytest.mark.unit
def test_dataloader_workers_linux() -> None:
    profile = MagicMock(spec=HardwareProfile)
    profile.os_name = "Linux"
    profile.system_ram_bytes = 8 * 1024**3
    assert _dataloader_workers(profile, "fast") == 0
    profile.system_ram_bytes = 32 * 1024**3
    assert _dataloader_workers(profile, "fast") == 4
    assert _dataloader_workers(profile, "low_memory") == 2


@pytest.mark.unit
@patch("src.hardware._get_system_ram_bytes", return_value=32 * 1024**3)
@patch("src.hardware._get_vram_bytes", return_value=8 * 1024**3)
def test_hardware_profile_with_cuda(mock_vram: MagicMock, mock_ram: MagicMock) -> None:
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.device_count", return_value=1):
            with patch("torch.cuda.get_device_properties") as mock_prop:
                mock_prop.return_value.total_memory = 8 * 1024**3
                mock_prop.return_value.major = 8
                mock_prop.return_value.minor = 9
                with patch("torch.cuda.get_device_name", return_value="RTX 4070"):
                    with patch("torch.cuda.get_device_capability", return_value=(8, 9)):
                        profile = HardwareProfile()
    assert profile.vram_bytes == 8 * 1024**3
    assert profile.system_ram_bytes == 32 * 1024**3
    assert profile.gpu_count == 1
    assert profile.os_name in ("Windows", "Linux", "Darwin")
    mock_vram.assert_called_once()
    mock_ram.assert_called_once()


@pytest.mark.unit
@patch("src.hardware._get_system_ram_bytes", return_value=16 * 1024**3)
@patch("src.hardware._get_vram_bytes", return_value=0)
def test_hardware_profile_cpu_only(mock_vram: MagicMock, mock_ram: MagicMock) -> None:
    with patch("torch.cuda.is_available", return_value=False):
        profile = HardwareProfile()
    assert profile.vram_bytes == 0
    assert profile.system_ram_bytes == 16 * 1024**3
    assert profile.gpu_count == 0


# --- Recommendation engine ---


@pytest.mark.unit
def test_recommend_unknown_preset() -> None:
    profile = MagicMock(spec=HardwareProfile)
    profile.cuda_available = True
    profile.vram_bytes = 8 * 1024**3
    with pytest.raises(ValueError, match="Unknown preset"):
        recommend(profile, "invalid", None)


@pytest.mark.unit
def test_recommend_save_steps_multiple_of_eval_steps() -> None:
    profile = MagicMock(spec=HardwareProfile)
    profile.cuda_available = True
    profile.vram_bytes = 8 * 1024**3
    profile.gpu_count = 1
    profile.os_name = "Linux"
    profile.system_ram_bytes = 32 * 1024**3
    for preset in PRESETS:
        overrides = recommend(profile, preset, {})
        eval_s = overrides["training"]["eval_steps"]
        save_s = overrides["training"]["save_steps"]
        assert save_s % eval_s == 0, f"preset={preset} save_steps % eval_steps != 0"
        assert overrides["training"]["save_only_model"] is True


@pytest.mark.unit
def test_recommend_preset_values() -> None:
    profile = MagicMock(spec=HardwareProfile)
    profile.cuda_available = True
    profile.vram_bytes = 8 * 1024**3
    profile.gpu_count = 1
    profile.os_name = "Linux"
    profile.system_ram_bytes = 32 * 1024**3
    overrides_fast = recommend(profile, "fast", {})
    overrides_quality = recommend(profile, "quality", {})
    assert overrides_fast["lora"]["r"] == PRESET_CONFIG["fast"]["lora_r"]
    assert overrides_quality["lora"]["r"] == PRESET_CONFIG["quality"]["lora_r"]
    assert overrides_fast["model"]["max_length"] == 2048
    assert overrides_quality["model"]["max_length"] == 4096
    assert overrides_fast["training"]["num_train_epochs"] == 1
    assert overrides_quality["training"]["num_train_epochs"] == 2


@pytest.mark.unit
def test_recommend_with_dpo_section() -> None:
    profile = MagicMock(spec=HardwareProfile)
    profile.cuda_available = True
    profile.vram_bytes = 8 * 1024**3
    profile.gpu_count = 1
    profile.os_name = "Linux"
    profile.system_ram_bytes = 32 * 1024**3
    base = {"dpo": {"train_file": "/some/dpo.jsonl", "max_steps": 100}}
    overrides = recommend(profile, "fast", base)
    assert "dpo" in overrides
    assert "per_device_train_batch_size" in overrides["dpo"]
    assert overrides["dpo"].get("torch_compile") is True


# --- Merge ---


@pytest.mark.unit
def test_merge_config_preserves_base() -> None:
    base = {
        "training": {"output_dir": "/keep", "seed": 42},
        "model": {"name": "Qwen/Qwen3-8B"},
    }
    overrides = {"training": {"seed": 99, "eval_steps": 150}}
    merged = merge_config(base, overrides)
    assert merged["training"]["output_dir"] == "/keep"
    assert merged["training"]["seed"] == 99
    assert merged["training"]["eval_steps"] == 150
    assert merged["model"]["name"] == "Qwen/Qwen3-8B"


# --- Integration: run_optimizer with test config ---


@pytest.mark.unit
def test_run_optimizer_validates_and_returns() -> None:
    if not TEST_CONFIG_YAML.exists():
        pytest.skip("Test config not found")
    merged = run_optimizer(
        config_path=str(TEST_CONFIG_YAML),
        preset="balanced",
        output_path=None,
    )
    assert "training" in merged
    assert merged["training"]["save_only_model"] is True
    assert merged["training"]["save_steps"] % merged["training"]["eval_steps"] == 0


@pytest.mark.unit
def test_run_optimizer_writes_yaml(tmp_path: Path) -> None:
    if not TEST_CONFIG_YAML.exists():
        pytest.skip("Test config not found")
    out = tmp_path / "optimized.yaml"
    run_optimizer(
        config_path=str(TEST_CONFIG_YAML),
        preset="quality",
        output_path=str(out),
    )
    assert out.exists()
    from src.main import load_config_from_yaml

    config = load_config_from_yaml(out)
    assert config.training.save_only_model is True
    assert config.training.save_steps % config.training.eval_steps == 0
