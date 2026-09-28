"""Hugging Face model cards and run summaries for training artifacts.

Cards are generated with ``huggingface_hub``'s ``ModelCard`` /
``ModelCardData`` APIs against the official model card template, so every
artifact card carries the standard YAML frontmatter (``base_model``,
``library_name``, ``license``, ``tags``, ...) and section headings. Stage
detail (SFT vs. DPO) is injected through the template's named variables.

This module is import-light on purpose: it depends only on ``src.config``
and ``huggingface_hub`` (no torch/transformers), so orchestration code can
import it without pulling in the training stack.

Artifact layout the cards document (dirnames configurable via config.yaml):

    training_output/README.md                      overall run summary
    training_output/final_adapter/                 SFT LoRA adapter (no card)
    training_output/final_merged_model/README.md   SFT stage card
    training_output/dpo_model/checkpoint-<step>/README.md   DPO checkpoint card
    training_output/dpo_model/dpo_merged_model/README.md    DPO merged card
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from huggingface_hub import ModelCard, ModelCardData

from src.config import ScriptConfig

logger = logging.getLogger(__name__)

MODEL_CARD_FILENAME = "README.md"

# Packages worth recording on cards for reproducibility.
_SOFTWARE_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "datasets",
    "accelerate",
)

_STAGE_TAGS: dict[str, list[str]] = {
    "sft": ["sft", "qlora", "lora", "trl"],
    "dpo": ["dpo", "qlora", "lora", "trl", "preference-optimization"],
}

_STAGE_MODEL_TYPE: dict[str, str] = {
    "sft": "Causal language model (QLoRA supervised fine-tuning)",
    "dpo": "Causal language model (Direct Preference Optimization with QLoRA)",
}


@dataclass(frozen=True)
class DPOCardContext:
    """Metadata for DPO-stage model cards, decoupled from ScriptConfig.

    ``src.dpo.run_dpo_training`` takes flat parameters rather than a
    ScriptConfig, and the standalone DPO merge helper runs in a separate
    process, so both receive card metadata through this context object.

    Attributes:
        base_model: Hub identifier of the original base model (frontmatter).
        sft_model_path: Checkpoint DPO actually trained from (provenance).
        train_data: Description of the DPO preference data source.
        license_id: SPDX license identifier, or None to omit from frontmatter.
        hyperparameters: DPO hyperparameters to render on the card.
        hardware_type: GPU/CPU description, or None when unknown.
        developers: 'Developed by' credit, or None to leave unfilled.
        card_authors: 'Model Card Authors' credit, or None to leave unfilled.
        card_contacts: 'Model Card Contact' entries, one per rendered line.
    """

    base_model: str
    sft_model_path: str
    train_data: str
    license_id: str | None = None
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    hardware_type: str | None = None
    developers: str | None = None
    card_authors: str | None = None
    card_contacts: Sequence[str] = ()


def collect_trainer_stats(train_result: Any, trainer: Any) -> dict[str, Any]:
    """Extract serializable training statistics from a trainer run.

    Defensive against mocks and older trainer APIs: anything that is not the
    expected type degrades to an empty value rather than raising.

    Args:
        train_result: Return value of ``trainer.train()`` (TrainOutput-like).
        trainer: The trainer instance after training.

    Returns:
        Dictionary with ``metrics`` (final run metrics) and ``log_history``
        (per-step log entries) keys.
    """
    metrics = getattr(train_result, "metrics", None)
    stats_metrics: dict[str, Any] = dict(metrics) if isinstance(metrics, dict) else {}

    history = getattr(getattr(trainer, "state", None), "log_history", None)
    log_history: list[dict[str, Any]] = (
        [dict(entry) for entry in history if isinstance(entry, dict)]
        if isinstance(history, list)
        else []
    )
    return {"metrics": stats_metrics, "log_history": log_history}


def collect_software_versions() -> dict[str, str]:
    """Return installed versions of the training-stack packages."""
    versions: dict[str, str] = {}
    for package in _SOFTWARE_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def summarize_stats(stats: Mapping[str, Any] | None) -> dict[str, Any]:
    """Condense trainer stats into headline numbers for cards and summaries.

    Args:
        stats: Output of :func:`collect_trainer_stats`, or None.

    Returns:
        Dictionary with any of ``final_train_loss``, ``final_eval_loss``,
        ``total_steps``, ``epochs``, ``train_runtime_seconds``, and
        ``train_samples_per_second`` that could be determined.
    """
    summary: dict[str, Any] = {}
    if not stats:
        return summary

    metrics = stats.get("metrics") or {}
    if isinstance(metrics, Mapping):
        if isinstance(metrics.get("train_loss"), int | float):
            summary["final_train_loss"] = float(metrics["train_loss"])
        if isinstance(metrics.get("train_runtime"), int | float):
            summary["train_runtime_seconds"] = float(metrics["train_runtime"])
        if isinstance(metrics.get("train_samples_per_second"), int | float):
            summary["train_samples_per_second"] = float(
                metrics["train_samples_per_second"]
            )
        if isinstance(metrics.get("epoch"), int | float):
            summary["epochs"] = float(metrics["epoch"])

    log_history = stats.get("log_history") or []
    if isinstance(log_history, list):
        for entry in log_history:
            if not isinstance(entry, Mapping):
                continue
            if isinstance(entry.get("loss"), int | float):
                summary["final_train_loss"] = float(entry["loss"])
            if isinstance(entry.get("eval_loss"), int | float):
                summary["final_eval_loss"] = float(entry["eval_loss"])
            if isinstance(entry.get("step"), int | float):
                summary["total_steps"] = int(entry["step"])

    return summary


def _format_value(value: Any) -> str:
    """Render a hyperparameter or metric value for a markdown table cell."""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value)
    return str(value)


def _markdown_table(rows: Mapping[str, Any], headers: tuple[str, str]) -> str:
    """Render a two-column markdown table from a mapping."""
    lines = [f"| {headers[0]} | {headers[1]} |", "|---|---|"]
    lines.extend(f"| {key} | {_format_value(value)} |" for key, value in rows.items())
    return "\n".join(lines)


def _format_contacts(contacts: Sequence[str]) -> str | None:
    """Render contact entries so each occupies its own line in the card.

    Entries are joined as separate markdown paragraphs rather than with bare
    newlines, which markdown would collapse into a single line.

    Args:
        contacts: Contact strings, e.g. ``"Name - name@example.com"``.

    Returns:
        Rendered markdown block, or None when there are no contacts.
    """
    entries = [contact.strip() for contact in contacts if contact and contact.strip()]
    if not entries:
        return None
    return "\n\n".join(entries)


def _training_regime(precision: str, hyperparameters: Mapping[str, Any]) -> str:
    """Build the 'Training regime' template value with a hyperparameter table."""
    if not hyperparameters:
        return precision
    table = _markdown_table(hyperparameters, ("Hyperparameter", "Value"))
    return f"{precision}\n\nKey hyperparameters:\n\n{table}"


def _speeds_sizes_times(summary: Mapping[str, Any]) -> str | None:
    """Build the 'Speeds, Sizes, Times' template value from a stats summary."""
    rows: dict[str, Any] = {}
    if "train_runtime_seconds" in summary:
        rows["Train runtime (seconds)"] = summary["train_runtime_seconds"]
    if "train_samples_per_second" in summary:
        rows["Train samples / second"] = summary["train_samples_per_second"]
    if "total_steps" in summary:
        rows["Optimizer steps"] = summary["total_steps"]
    if "epochs" in summary:
        rows["Epochs"] = summary["epochs"]
    if not rows:
        return None
    return _markdown_table(rows, ("Measurement", "Value"))


def _results(summary: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Build the 'Results' table and summary sentence from a stats summary."""
    rows: dict[str, Any] = {}
    if "final_train_loss" in summary:
        rows["Final training loss"] = summary["final_train_loss"]
    if "final_eval_loss" in summary:
        rows["Final evaluation loss"] = summary["final_eval_loss"]
    if not rows:
        return None, None
    table = _markdown_table(rows, ("Metric", "Value"))
    sentence = "Loss statistics recorded by the trainer during this stage."
    return table, sentence


def _build_stage_card(
    *,
    stage: str,
    model_id: str,
    base_model: str,
    library_name: str,
    license_id: str | None,
    model_summary: str,
    model_description: str,
    finetuned_from: str,
    training_data: str,
    preprocessing: str | None,
    hyperparameters: Mapping[str, Any],
    stats: Mapping[str, Any] | None,
    hardware_type: str | None,
    developers: str | None = None,
    card_authors: str | None = None,
    card_contacts: Sequence[str] = (),
    precision: str = "bf16 mixed precision with 4-bit (NF4) quantized base weights",
) -> ModelCard:
    """Render one training-stage model card from the official HF template."""
    card_data = ModelCardData(
        base_model=base_model,
        library_name=library_name,
        license=license_id,
        pipeline_tag="text-generation",
        tags=list(_STAGE_TAGS.get(stage, [])),
    )

    summary = summarize_stats(stats)
    results_table, results_summary = _results(summary)

    template_kwargs: dict[str, Any] = {
        "model_id": model_id,
        "model_summary": model_summary,
        "model_description": model_description,
        "model_type": _STAGE_MODEL_TYPE.get(stage, "Causal language model"),
        "finetuned_from": finetuned_from,
        "training_data": training_data,
        "training_regime": _training_regime(precision, hyperparameters),
    }
    if preprocessing:
        template_kwargs["preprocessing"] = preprocessing
    speeds = _speeds_sizes_times(summary)
    if speeds:
        template_kwargs["speeds_sizes_times"] = speeds
    if results_table:
        template_kwargs["results"] = results_table
    if results_summary:
        template_kwargs["results_summary"] = results_summary
    if hardware_type:
        template_kwargs["hardware_type"] = hardware_type
    if developers:
        template_kwargs["developers"] = developers
    if card_authors:
        template_kwargs["model_card_authors"] = card_authors
    contacts = _format_contacts(card_contacts)
    if contacts:
        template_kwargs["model_card_contact"] = contacts
    if "train_runtime_seconds" in summary:
        template_kwargs["hours_used"] = f"{summary['train_runtime_seconds'] / 3600:.2f}"
    versions = collect_software_versions()
    if versions:
        template_kwargs["software"] = ", ".join(
            f"{name} {version}" for name, version in versions.items()
        )

    card: ModelCard = ModelCard.from_template(card_data, **template_kwargs)
    return card


def write_sft_model_card(
    config: ScriptConfig,
    directory: Path,
    stats: Mapping[str, Any] | None,
    *,
    hardware_type: str | None = None,
) -> Path:
    """Write the SFT-stage model card into the merged SFT model directory.

    Args:
        config: Pipeline configuration (source of datasets and hyperparameters).
        directory: Merged SFT model directory to write ``README.md`` into.
        stats: Trainer stats from :func:`collect_trainer_stats`, or None.
        hardware_type: GPU/CPU description for the card, or None.

    Returns:
        Path of the written model card.
    """
    base_model = config.model.name
    hyperparameters: dict[str, Any] = {
        "epochs": config.training.num_train_epochs,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "warmup_ratio": config.training.warmup_ratio,
        "weight_decay": config.training.weight_decay,
        "max_grad_norm": config.training.max_grad_norm,
        "optimizer": config.training.optim,
        "max_sequence_length": config.model.max_length,
        "lora_r": config.lora.r,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "lora_target_modules": config.lora.target_modules,
        "quantization": (
            f"4-bit {config.quantization.quant_type}"
            if config.quantization.enabled
            else "disabled"
        ),
        "seed": config.training.seed,
    }
    training_data = (
        "ICDU-formatted JSONL instruction data: "
        f"train `{config.data.train_file.name}`, "
        f"validation `{config.data.validation_file.name}`."
    )
    card = _build_stage_card(
        stage="sft",
        model_id=directory.name,
        base_model=base_model,
        library_name="transformers",
        license_id=config.model.license,
        model_summary=(
            f"Merged QLoRA supervised fine-tune of {base_model} on "
            "ICDU-formatted instruction data (SFT stage of the ai-factory "
            "pipeline)."
        ),
        model_description=(
            f"Stage 1 (SFT) artifact of the ai-factory QLoRA + DPO pipeline. "
            f"A LoRA adapter was trained on {base_model} with 4-bit "
            "quantization, then merged into the base weights to produce this "
            "standalone checkpoint. The sibling `final_adapter` directory "
            "holds the unmerged adapter; DPO artifacts live in their own "
            "directory and are described by their own cards."
        ),
        finetuned_from=base_model,
        training_data=training_data,
        preprocessing=(
            "ICDU records are rendered to chat format (system context/intent/"
            "persona + user prompt + assistant response) with optional "
            "scenario perturbation; loss is computed on assistant completions "
            "only."
        ),
        hyperparameters=hyperparameters,
        stats=stats,
        hardware_type=hardware_type,
        developers=config.model.developers,
        card_authors=config.model.card_authors,
        card_contacts=config.model.card_contacts,
    )
    return _save_card(card, directory)


def write_dpo_model_card(
    directory: Path,
    context: DPOCardContext,
    stats: Mapping[str, Any] | None,
    *,
    library_name: str,
    model_id: str | None = None,
    artifact_note: str | None = None,
) -> Path:
    """Write a DPO-stage model card into ``directory``.

    Used for DPO checkpoints (``library_name="peft"``), the DPO run directory,
    and the merged DPO model (``library_name="transformers"``).

    Args:
        directory: Directory to write ``README.md`` into.
        context: DPO card metadata.
        stats: Trainer stats from :func:`collect_trainer_stats`, or None.
        library_name: Frontmatter ``library_name`` for this artifact.
        model_id: Card title; defaults to the directory name.
        artifact_note: Optional artifact-specific sentence for the description.

    Returns:
        Path of the written model card.
    """
    description = (
        "Stage 2 (DPO) artifact of the ai-factory QLoRA + DPO pipeline. "
        f"Direct Preference Optimization was run on top of the merged SFT "
        f"checkpoint at `{context.sft_model_path}` (base model "
        f"{context.base_model}) using tool-selection preference pairs."
    )
    if artifact_note:
        description = f"{description} {artifact_note}"

    card = _build_stage_card(
        stage="dpo",
        model_id=model_id or directory.name,
        base_model=context.base_model,
        library_name=library_name,
        license_id=context.license_id,
        model_summary=(
            f"Direct Preference Optimization stage of the ai-factory pipeline, "
            f"trained from the merged SFT checkpoint of {context.base_model}."
        ),
        model_description=description,
        finetuned_from=(
            f"{context.sft_model_path} (merged SFT checkpoint of {context.base_model})"
        ),
        training_data=context.train_data,
        preprocessing=(
            "Chosen/rejected preference pairs contrast correct tool calls "
            "with incorrect or missing tool calls."
        ),
        hyperparameters=context.hyperparameters,
        stats=stats,
        hardware_type=context.hardware_type,
        developers=context.developers,
        card_authors=context.card_authors,
        card_contacts=context.card_contacts,
    )
    return _save_card(card, directory)


def _save_card(card: ModelCard, directory: Path) -> Path:
    """Save a model card as ``README.md`` inside ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    card_path = directory / MODEL_CARD_FILENAME
    card.save(str(card_path))
    logger.info("Wrote model card: %s", card_path)
    return card_path


def read_checkpoint_stats(checkpoint_dir: Path) -> dict[str, Any]:
    """Load trainer stats from a checkpoint's ``trainer_state.json``.

    Args:
        checkpoint_dir: A ``checkpoint-<step>`` directory.

    Returns:
        Stats dictionary compatible with :func:`summarize_stats`; empty when
        the state file is missing or unreadable.
    """
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        return {"metrics": {}, "log_history": []}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", state_path, exc)
        return {"metrics": {}, "log_history": []}
    log_history = state.get("log_history")
    return {
        "metrics": {},
        "log_history": log_history if isinstance(log_history, list) else [],
    }


def _count_jsonl_records(path: Path) -> int | None:
    """Count non-empty lines of a JSONL file; None when unavailable."""
    try:
        with open(path, encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


def _dataset_line(label: str, path: Path) -> str:
    """Render one dataset bullet with an optional record count."""
    count = _count_jsonl_records(path) if path.exists() else None
    suffix = f" ({count} records)" if count is not None else ""
    return f"- **{label}:** `{path.name}`{suffix}"


def _relative_artifact(path: Path, root: Path) -> str:
    """Render an artifact path relative to the run root when possible."""
    try:
        return str(path.relative_to(root)).replace("\\", "/") + "/"
    except ValueError:
        return str(path)


def _phase_section(
    title: str,
    stats: Mapping[str, Any] | None,
    hyperparameters: Mapping[str, Any],
    runtime_seconds: float | None,
) -> list[str]:
    """Build one per-phase section of the run summary."""
    lines = [f"### {title}", ""]
    summary = summarize_stats(stats)
    if runtime_seconds is not None:
        summary = {**summary, "wall_time_seconds": round(runtime_seconds, 1)}
    if summary:
        lines.append(_markdown_table(summary, ("Statistic", "Value")))
    else:
        lines.append("No statistics were recorded for this phase.")
    lines.append("")
    if hyperparameters:
        lines.append("<details><summary>Hyperparameters</summary>")
        lines.append("")
        lines.append(_markdown_table(hyperparameters, ("Hyperparameter", "Value")))
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return lines


def write_pipeline_run_summary(
    config: ScriptConfig,
    *,
    sft_stats: Mapping[str, Any] | None = None,
    dpo_stats: Mapping[str, Any] | None = None,
    phase_runtimes: Mapping[str, float] | None = None,
    hardware: Mapping[str, Any] | None = None,
) -> Path:
    """Write the overall run summary README at the training output root.

    A human-readable report of the full run: datasets, hyperparameters,
    hardware profile, runtime and loss statistics, and pointers to every
    produced artifact directory. Written last, so it replaces any interim
    README a trainer wrote at the output root during checkpointing.

    Args:
        config: Pipeline configuration.
        sft_stats: SFT trainer stats, or None when the phase did not run.
        dpo_stats: DPO trainer stats, or None when the phase did not run.
        phase_runtimes: Wall-clock seconds per phase, keyed by ``sft``,
            ``merge``, and ``dpo``.
        hardware: Hardware profile values (e.g. device name, VRAM, RAM, OS).

    Returns:
        Path of the written summary.
    """
    root = config.training.output_dir
    runtimes = dict(phase_runtimes or {})
    # timezone.utc (not datetime.UTC): the conda training env runs Python 3.10.
    generated = datetime.now(timezone.utc).strftime(  # noqa: UP017
        "%Y-%m-%d %H:%M UTC"
    )

    lines: list[str] = [
        f"# Training run summary — {config.model.name}",
        "",
        f"Generated by the ai-factory pipeline on {generated}.",
        "",
        "Two-stage run: QLoRA supervised fine-tuning (SFT) followed by "
        "Direct Preference Optimization (DPO). Each model artifact below "
        "carries its own Hugging Face model card.",
        "",
        "## Datasets",
        "",
        _dataset_line("SFT training data (ICDU JSONL)", config.data.train_file),
        _dataset_line("SFT validation data (ICDU JSONL)", config.data.validation_file),
    ]
    dpo_train_file = (
        config.dpo.train_file
        if config.dpo is not None and config.dpo.train_file is not None
        else config.data.train_file
    )
    lines.append(_dataset_line("DPO preference source", dpo_train_file))
    lines.append("")

    if hardware:
        lines.extend(
            ["## Hardware", "", _markdown_table(hardware, ("Component", "Value")), ""]
        )

    versions = collect_software_versions()
    if versions:
        lines.extend(
            ["## Software", "", _markdown_table(versions, ("Package", "Version")), ""]
        )

    lines.extend(["## Phases", ""])
    sft_hparams: dict[str, Any] = {
        "epochs": config.training.num_train_epochs,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "optimizer": config.training.optim,
        "max_sequence_length": config.model.max_length,
        "lora_r": config.lora.r,
        "lora_alpha": config.lora.alpha,
        "seed": config.training.seed,
    }
    lines.extend(
        _phase_section(
            "SFT (QLoRA fine-tuning)", sft_stats, sft_hparams, runtimes.get("sft")
        )
    )
    if runtimes.get("merge") is not None:
        lines.extend(
            [
                "### Merge (LoRA adapter into base weights)",
                "",
                f"Wall time: {runtimes['merge']:.1f} seconds.",
                "",
            ]
        )
    dpo_hparams: dict[str, Any] = {}
    if config.dpo is not None:
        dpo_hparams = {
            "max_steps": config.dpo.max_steps,
            "learning_rate": config.dpo.learning_rate,
            "beta": config.dpo.beta,
            "lora_rank": config.dpo.lora_rank,
            "per_device_train_batch_size": config.dpo.per_device_train_batch_size,
            "gradient_accumulation_steps": config.dpo.gradient_accumulation_steps,
            "optimizer": config.dpo.optim,
            "lr_scheduler_type": config.dpo.lr_scheduler_type,
            "warmup_ratio": config.dpo.warmup_ratio,
        }
    lines.extend(
        _phase_section(
            "DPO (Direct Preference Optimization)",
            dpo_stats,
            dpo_hparams,
            runtimes.get("dpo"),
        )
    )

    lines.extend(["## Artifacts", ""])
    artifacts: list[tuple[Path, str]] = [
        (
            config.training.adapter_path,
            "SFT LoRA adapter weights (adapter only; no model card)",
        ),
        (
            config.training.merged_model_path,
            "Merged SFT model with its SFT-stage model card",
        ),
    ]
    dpo_dir = config.dpo_output_dir
    if dpo_dir.exists():
        for checkpoint in sorted(dpo_dir.glob("checkpoint-*")):
            if checkpoint.is_dir():
                artifacts.append(
                    (checkpoint, "DPO training checkpoint with its DPO model card")
                )
    artifacts.append(
        (
            config.dpo_merged_model_path,
            "Merged DPO model with its DPO-stage model card",
        )
    )
    for path, description in artifacts:
        status = "" if path.exists() else " *(not produced by this run)*"
        lines.append(f"- `{_relative_artifact(path, root)}` — {description}{status}")
    lines.append("")

    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / MODEL_CARD_FILENAME
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote run summary: %s", summary_path)
    return summary_path
