"""Model Optimizer: recommend config.yaml settings from hardware and preset.

Uses HardwareProfile (VRAM, RAM, OS) and a preset (fast, quality, balanced,
low_memory) to produce config overrides that respect TRAINING_CRASH_DIAGNOSIS
rules (staggered eval/save, save_only_model, safe dataloader workers, etc.).
Requires --config-path in v1 so paths are preserved and validation succeeds.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Literal

import yaml

from src.config import ScriptConfig
from src.hardware import HardwareProfile


def _resolve_config_paths(
    config_dict: dict[str, Any], base_dir: Path
) -> dict[str, Any]:
    """Resolve relative paths in config dict (mirrors main._resolve_config_paths)."""
    data_section = config_dict.get("data", {}) or {}
    for key in ("train_file", "validation_file"):
        if key in data_section:
            candidate = Path(data_section[key])
            if not candidate.is_absolute():
                data_section[key] = str((base_dir / candidate).resolve())
    config_dict["data"] = data_section

    training_section = config_dict.get("training", {}) or {}
    if "output_dir" in training_section:
        p = Path(training_section["output_dir"])
        if not p.is_absolute():
            training_section["output_dir"] = str((base_dir / p).resolve())
    config_dict["training"] = training_section

    dpo_section = config_dict.get("dpo")
    if dpo_section and isinstance(dpo_section, dict):
        if "output_dir" in dpo_section:
            p = Path(dpo_section["output_dir"])
            if not p.is_absolute():
                dpo_section["output_dir"] = str((base_dir / p).resolve())
        if "train_file" in dpo_section:
            p = Path(dpo_section["train_file"])
            if not p.is_absolute():
                dpo_section["train_file"] = str((base_dir / p).resolve())
        config_dict["dpo"] = dpo_section

    return config_dict


logger = logging.getLogger(__name__)

PresetName = Literal["fast", "quality", "balanced", "low_memory"]
PRESETS: tuple[PresetName, ...] = ("fast", "quality", "balanced", "low_memory")

# VRAM tier thresholds (bytes): 6GB, 8GB, 12GB, 24GB
VRAM_6GB = 6 * 1024**3
VRAM_8GB = 8 * 1024**3
VRAM_12GB = 12 * 1024**3
VRAM_24GB = 24 * 1024**3

# Preset targets: effective batch range, eval_steps, save_steps
# (save_steps = 2 * eval_steps), lora_r, lora_alpha, max_length,
# num_train_epochs, gradient_checkpointing, dpo_batch, dpo_torch_compile
_PRESET_FAST = {
    "target_effective_batch_min": 16,
    "target_effective_batch_max": 32,
    "eval_steps": 200,
    "save_steps": 400,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_length": 2048,
    "num_train_epochs": 1,
    "gradient_checkpointing": False,
    "dpo_per_device_batch": 2,
    "dpo_torch_compile": True,
}
_PRESET_QUALITY = {
    "target_effective_batch_min": 8,
    "target_effective_batch_max": 16,
    "eval_steps": 100,
    "save_steps": 200,
    "lora_r": 64,
    "lora_alpha": 64,
    "max_length": 4096,
    "num_train_epochs": 2,
    "gradient_checkpointing": True,
    "dpo_per_device_batch": 1,
    "dpo_torch_compile": False,
}
_PRESET_BALANCED = {
    "target_effective_batch_min": 8,
    "target_effective_batch_max": 16,
    "eval_steps": 150,
    "save_steps": 300,
    "lora_r": 32,
    "lora_alpha": 32,
    "max_length": 4096,
    "num_train_epochs": 1,
    "gradient_checkpointing": True,
    "dpo_per_device_batch": 2,
    "dpo_torch_compile": False,
}
_PRESET_LOW_MEMORY = {
    "target_effective_batch_min": 4,
    "target_effective_batch_max": 8,
    "eval_steps": 150,
    "save_steps": 300,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_length": 2048,
    "num_train_epochs": 1,
    "gradient_checkpointing": True,
    "dpo_per_device_batch": 1,
    "dpo_torch_compile": False,
}

PRESET_CONFIG: dict[PresetName, dict[str, Any]] = {
    "fast": _PRESET_FAST,
    "quality": _PRESET_QUALITY,
    "balanced": _PRESET_BALANCED,
    "low_memory": _PRESET_LOW_MEMORY,
}


def _vram_tier(vram_bytes: int) -> Literal["low", "medium", "high"]:
    """Classify VRAM into tier for batch size caps."""
    if vram_bytes <= 0:
        return "low"
    if vram_bytes < VRAM_8GB:
        return "low"
    if vram_bytes < VRAM_12GB:
        return "medium"
    return "high"


def _max_batch_for_vram(
    vram_bytes: int,
    preset: PresetName,
) -> int:
    """Max per_device_train_batch_size for given VRAM and preset."""
    tier = _vram_tier(vram_bytes)
    if tier == "low":
        return 2 if preset == "low_memory" else 4
    if tier == "medium":
        return 4
    return 8 if preset == "fast" else 4


def _dataloader_workers(profile: HardwareProfile, preset: PresetName) -> int:
    """Recommend dataloader_num_workers per TRAINING_CRASH_DIAGNOSIS (Windows 0/2)."""
    if profile.os_name == "Windows":
        return 0 if preset == "low_memory" else 2
    ram_gb = profile.system_ram_bytes / (1024**3)
    if ram_gb < 16:
        return 0
    if ram_gb < 32:
        return 2
    return 4 if preset != "low_memory" else 2


def _choose_batch_and_accumulation(
    profile: HardwareProfile,
    preset: PresetName,
) -> tuple[int, int]:
    """Return (per_device_train_batch_size, gradient_accumulation_steps) for SFT."""
    target = PRESET_CONFIG[preset]
    lo, hi = target["target_effective_batch_min"], target["target_effective_batch_max"]
    max_batch = _max_batch_for_vram(profile.vram_bytes, preset)
    max_batch = max(1, max_batch)
    # Prefer larger batch when VRAM allows; otherwise use accumulation to hit target.
    per_device = min(max_batch, hi)
    per_device = max(1, per_device)
    effective = per_device * max(1, profile.gpu_count)
    if effective >= lo:
        accum = max(1, lo // effective) if effective < lo else 1
    else:
        accum = max(1, (lo + effective - 1) // effective)
    return per_device, accum


def recommend(
    profile: HardwareProfile,
    preset: PresetName,
    base_config_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recommend config overrides for the given hardware and preset.

    Returns a nested dict of overrides (training, lora, model, dpo) that can be
    merged into a base config. Respects TRAINING_CRASH_DIAGNOSIS: save_steps
    is a multiple of eval_steps, save_only_model True, safe workers.

    Args:
        profile: Detected hardware (VRAM, RAM, OS).
        preset: One of fast, quality, balanced, low_memory.
        base_config_dict: Optional existing config dict (paths and model name
            are preserved; only optimized keys are overridden).

    Returns:
        Nested overrides dict with keys training, lora, model, and optionally dpo.
    """
    if preset not in PRESET_CONFIG:
        raise ValueError(f"Unknown preset: {preset}. Use one of {PRESETS}.")
    target = PRESET_CONFIG[preset]

    per_device, grad_accum = _choose_batch_and_accumulation(profile, preset)
    eval_steps = target["eval_steps"]
    save_steps = target["save_steps"]
    if save_steps % eval_steps != 0:
        raise ValueError("save_steps must be multiple of eval_steps")

    workers = _dataloader_workers(profile, preset)

    overrides: dict[str, Any] = {
        "training": {
            "per_device_train_batch_size": per_device,
            "per_device_eval_batch_size": per_device,
            "gradient_accumulation_steps": grad_accum,
            "eval_steps": eval_steps,
            "save_steps": save_steps,
            "save_total_limit": 2,
            "save_only_model": True,
            "gradient_checkpointing": target["gradient_checkpointing"],
            "dataloader_num_workers": workers,
            "num_train_epochs": target["num_train_epochs"],
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "lr_scheduler_type": "cosine",
        },
        "lora": {
            "r": target["lora_r"],
            "alpha": target["lora_alpha"],
        },
        "model": {
            "max_length": target["max_length"],
        },
    }

    # Prefer flash_attention_2 if available (model_setup resolves at load time)
    if profile.cuda_available:
        overrides["model"]["attn_implementation"] = "flash_attention_2"
    else:
        overrides["model"]["attn_implementation"] = "sdpa"

    if base_config_dict and "dpo" in base_config_dict and base_config_dict["dpo"]:
        overrides["dpo"] = {
            "per_device_train_batch_size": target["dpo_per_device_batch"],
            "per_device_eval_batch_size": target["dpo_per_device_batch"],
            "gradient_accumulation_steps": 2,
            "eval_steps": 50,
            "save_steps": 50,
            "save_total_limit": 2,
            "gradient_checkpointing": True,
            "lora_rank": target["lora_r"],
        }
        # torch_compile is not on DPOConfig; we add it to the output YAML for CLI/docs
        if target.get("dpo_torch_compile"):
            overrides["dpo"]["torch_compile"] = True

    return overrides


def merge_config(
    base_dict: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Deep-merge overrides into a copy of base_dict. Paths in base are preserved."""
    merged = copy.deepcopy(base_dict)
    for section, section_overrides in overrides.items():
        if section not in merged:
            merged[section] = {}
        if isinstance(merged[section], dict) and isinstance(section_overrides, dict):
            for k, v in section_overrides.items():
                merged[section][k] = v
        else:
            merged[section] = section_overrides
    return merged


def run_optimizer(
    config_path: str,
    preset: PresetName = "balanced",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Load base config, run recommendation, merge, validate, and optionally write YAML.

    Args:
        config_path: Path to existing config.yaml (required in v1).
        preset: fast | quality | balanced | low_memory.
        output_path: If set, write merged config to this YAML file.

    Returns:
        Merged config dict (validated with ScriptConfig).

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValueError: If config is invalid or preset unknown.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        base_dict = yaml.safe_load(f)
    if not base_dict:
        raise ValueError("Configuration file is empty")

    base_dir = path.parent.resolve()
    base_dict = _resolve_config_paths(base_dict, base_dir)

    profile = HardwareProfile()
    overrides = recommend(profile, preset, base_dict)
    merged = merge_config(base_dict, overrides)

    # ScriptConfig validation (path validators run; paths from base exist)
    ScriptConfig(**_merged_for_validation(merged))

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        to_dump = _to_yaml_friendly(merged)
        with open(out, "w", encoding="utf-8") as f:
            yaml.dump(
                to_dump,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        logger.info("Wrote optimized config to %s", out)

    return merged


def _merged_for_validation(merged: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of merged suitable for ScriptConfig (coerce path strings)."""
    return copy.deepcopy(merged)


def _to_yaml_friendly(config: dict[str, Any]) -> dict[str, Any]:
    """Convert config to YAML-serializable form (Path -> str)."""
    out: dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, dict):
            out[k] = _to_yaml_friendly(v)
        elif hasattr(v, "resolve"):
            out[k] = str(v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Model Optimizer: suggest config.yaml from hardware and "
            "preset (fast/quality/balanced/low_memory)."
        )
    )
    parser.add_argument(
        "config_path",
        help="Path to base config YAML (required).",
    )
    parser.add_argument(
        "-p",
        "--preset",
        choices=list(PRESETS),
        default="balanced",
        help="Preset (default: balanced).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write optimized config to this YAML path.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    try:
        run_optimizer(
            config_path=args.config_path,
            preset=args.preset,
            output_path=args.output,
        )
    except Exception as e:
        logging.exception("Optimizer failed: %s", e)
        raise SystemExit(1) from e
