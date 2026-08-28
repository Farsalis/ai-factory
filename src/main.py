"""CLI entry point for QLoRA fine-tuning, merging, DPO training, and inference.

This module provides the main command-line interface for the complete training pipeline:
1. QLoRA fine-tuning with quantization
2. Model merging (LoRA weights into base model)
3. DPO (Direct Preference Optimization) training
4. Optional inference with tool-augmented agent loop
5. Model Optimizer: optimize-config to suggest config.yaml from hardware and preset

Usage:
    python -m src.main --config-path config.yaml [--run-inference]
    python -m src.main --config-path config.yaml --inference-only
    python -m src.main optimize-config --config-path config.yaml \\
        --preset fast --output config_optimized.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

import yaml

from src.config import ScriptConfig

# Disable transformers' optional torchvision integration before any later
# torch/transformers imports (including in run_pipeline / inference).
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_IMAGE_TRANSFORMS_DISABLED", "1")

# Configure logging before any pipeline work so CLI failures are always visible.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_QUERIES = [
    "Advise on fitness habits using latest research.",
    "Calculate 5 * (3 + 7) / 2",
    "Add a task to read a book this week.",
]
RESPONSE_PREVIEW_LENGTH = 200
SEPARATOR_LENGTH = 50
PROG_NAME = "python -m src.main"


def load_config_from_yaml(path: Path) -> ScriptConfig:
    """Load and validate configuration from YAML file.

    Resolves relative paths in the configuration relative to the config file's
    parent directory. This allows config files to use relative paths that work
    regardless of where the script is run from.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated ScriptConfig instance.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration is invalid or missing required fields.
    """
    logger.info("Loading configuration from: %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error("Failed to parse YAML configuration: %s", e)
        raise ValueError(f"Invalid YAML in configuration file: {e}") from e

    if not config_dict:
        raise ValueError("Configuration file is empty")

    # Resolve relative paths relative to config file's directory
    base_dir = path.parent.resolve()
    config_dict = _resolve_config_paths(config_dict, base_dir)

    try:
        return ScriptConfig(**config_dict)
    except Exception as e:
        logger.error("Failed to validate configuration: %s", e)
        raise ValueError(f"Invalid configuration: {e}") from e


def _resolve_config_paths(
    config_dict: dict[str, Any], base_dir: Path
) -> dict[str, Any]:
    """Resolve relative paths in configuration dictionary.

    Args:
        config_dict: Configuration dictionary from YAML.
        base_dir: Base directory for resolving relative paths.

    Returns:
        Configuration dictionary with resolved absolute paths.
    """
    # Resolve data file paths
    data_section = config_dict.get("data", {})
    for key in ("train_file", "validation_file"):
        if key in data_section:
            candidate_path = Path(data_section[key])
            if not candidate_path.is_absolute():
                data_section[key] = str((base_dir / candidate_path).resolve())
    config_dict["data"] = data_section

    # Resolve training output directory
    training_section = config_dict.get("training", {})
    if "output_dir" in training_section:
        output_dir_path = Path(training_section["output_dir"])
        if not output_dir_path.is_absolute():
            training_section["output_dir"] = str((base_dir / output_dir_path).resolve())
    config_dict["training"] = training_section

    # Resolve DPO paths if present
    dpo_section = config_dict.get("dpo", {})
    if dpo_section:
        if "output_dir" in dpo_section:
            output_dir_path = Path(dpo_section["output_dir"])
            if not output_dir_path.is_absolute():
                dpo_section["output_dir"] = str((base_dir / output_dir_path).resolve())
        if "train_file" in dpo_section:
            train_path = Path(dpo_section["train_file"])
            if not train_path.is_absolute():
                dpo_section["train_file"] = str((base_dir / train_path).resolve())
        config_dict["dpo"] = dpo_section

    return config_dict


def _find_model_path(config: ScriptConfig) -> Path:
    """Find the best available model path for inference.

    Prefers DPO-trained model over merged model if both exist.

    Args:
        config: Script configuration.

    Returns:
        Path to the model directory.

    Raises:
        FileNotFoundError: If no model is found.
    """
    dpo_path = config.training.output_dir / "dpo_model"
    merged_path = config.training.output_dir / "final_merged_model"

    # Prefer DPO model if available, otherwise use merged model
    model_path = dpo_path if dpo_path.exists() else merged_path

    if not model_path.exists():
        available_paths = [dpo_path, merged_path]
        raise FileNotFoundError(
            f"No model found. Checked paths: {[str(p) for p in available_paths]}"
        )

    return model_path


def run_inference_phase(
    config: ScriptConfig,
    example_queries: list[str],
) -> None:
    """Run inference with tools using the trained model.

    Loads the best available model (DPO-trained preferred) and runs inference
    for each example query using the tool-augmented agent loop.

    Args:
        config: Script configuration.
        example_queries: List of queries to process.

    Raises:
        FileNotFoundError: If no model is found.
        RuntimeError: If model loading or inference fails.
    """
    from src.inference_with_tools import agent_loop, load_model_pipeline

    model_path = _find_model_path(config)
    logger.info("Running inference with model: %s", model_path)

    try:
        model_pipeline = load_model_pipeline(
            str(model_path),
            use_linear_attention_kernels=config.model.use_linear_attention_kernels,
        )
    except Exception as e:
        logger.error("Failed to load model from %s: %s", model_path, e)
        raise RuntimeError(f"Model loading failed: {e}") from e

    logger.info("Processing %d query(ies)", len(example_queries))

    for i, query in enumerate(example_queries, 1):
        logger.info("Query %d/%d: %s", i, len(example_queries), query)

        try:
            response = agent_loop(query, model_pipeline)
            preview = (
                response[:RESPONSE_PREVIEW_LENGTH] + "..."
                if len(response) > RESPONSE_PREVIEW_LENGTH
                else response
            )
            logger.info("Response preview: %s", preview)

            print(f"\nQuery: {query}")
            print(f"Response: {response}")
            print("-" * SEPARATOR_LENGTH)

        except Exception as e:
            error_msg = f"Inference error for query '{query}': {e}"
            logger.error(error_msg, exc_info=True)
            print(f"Error: {error_msg}")


def run_pipeline(
    config: ScriptConfig,
    run_inference: bool = False,
    example_queries: list[str] | None = None,
    torch_compile: bool = False,
) -> None:
    """Orchestrate the complete training, merging, DPO, and inference pipeline.

    Executes the pipeline in the following order:
    1. Training Phase: QLoRA fine-tuning with quantization
    2. Merging Phase: Merge LoRA weights into base model
    3. DPO Phase: Direct Preference Optimization training
    4. Inference Phase (optional): Run inference with tool-augmented agent

    Args:
        config: Script configuration.
        run_inference: Whether to run inference after DPO training.
        example_queries: List of queries for inference (if run_inference is True).
        torch_compile: Enable torch.compile for DPO training (PyTorch 2.0+).

    Raises:
        RuntimeError: If any phase of the pipeline fails.
    """
    import torch

    from src.config import get_default_dpo_config
    from src.dpo import (
        generate_preference_pairs,
        load_jsonl,
        prepare_dpo_dataset,
        run_dpo_training,
    )
    from src.model_setup import load_model, load_tokenizer
    from src.train import merge_and_save_model, run_training
    from src.utils import Environment

    logger.info("Starting training pipeline")
    logger.info("Output directory: %s", config.training.output_dir)

    # Initialize environment and set random seed
    env = Environment()
    env.setup_backends()
    torch.manual_seed(config.training.seed)
    logger.info("Random seed set to: %d", config.training.seed)

    # Training Phase
    logger.info("=" * SEPARATOR_LENGTH)
    logger.info("Training Phase: QLoRA Fine-tuning")
    logger.info("=" * SEPARATOR_LENGTH)

    try:
        tokenizer = load_tokenizer(config.model)
        quantized_model = load_model(config.model, config.quantization, env)
        run_training(config, env, tokenizer, quantized_model)
        logger.info("Training phase completed successfully")
    except Exception as e:
        logger.error("Training phase failed: %s", e, exc_info=True)
        raise RuntimeError(f"Training failed: {e}") from e
    finally:
        # Clean up memory after training
        if "quantized_model" in locals():
            del quantized_model
        if env.cuda_available:
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")

    # Merging Phase
    logger.info("=" * SEPARATOR_LENGTH)
    logger.info("Merging Phase: LoRA Weights into Base Model")
    logger.info("=" * SEPARATOR_LENGTH)

    try:
        merge_and_save_model(config, env)
        logger.info("Merging phase completed successfully")
    except Exception as e:
        logger.error("Merging phase failed: %s", e, exc_info=True)
        raise RuntimeError(f"Merging failed: {e}") from e

    # DPO Phase
    logger.info("=" * SEPARATOR_LENGTH)
    logger.info("DPO Phase: Direct Preference Optimization")
    logger.info("=" * SEPARATOR_LENGTH)

    try:
        if config.dpo is None:
            logger.warning(
                "No DPO configuration found. Using default DPO settings. "
                "Consider adding a 'dpo:' section to your config.yaml"
            )
            dpo_config = get_default_dpo_config()
        else:
            dpo_config = config.dpo

        dpo_train_file = (
            dpo_config.train_file
            if dpo_config.train_file is not None
            else config.data.train_file
        )
        dpo_src = (
            "(dpo.train_file)"
            if dpo_config.train_file is not None
            else "(data.train_file fallback)"
        )
        logger.info(
            "Loading DPO preference data from: %s %s",
            dpo_train_file,
            dpo_src,
        )
        augmented_data = load_jsonl(str(dpo_train_file))
        logger.info("Loaded %d examples", len(augmented_data))

        pref_data = generate_preference_pairs(augmented_data)
        logger.info("Generated %d preference pairs", len(pref_data))

        pref_dataset = prepare_dpo_dataset(pref_data)
        logger.info("Prepared DPO dataset")

        dpo_base_path = config.training.output_dir / "final_merged_model"
        dpo_model_path = (
            str(dpo_base_path) if dpo_base_path.exists() else str(config.model.name)
        )
        logger.info("DPO base model: %s", dpo_model_path)

        dpo_output_dir_raw: Path | str = (
            dpo_config.output_dir
            if dpo_config.output_dir is not None
            else config.training.output_dir / "dpo_model"
        )
        dpo_output_dir_str: str = (
            str(dpo_output_dir_raw)
            if isinstance(dpo_output_dir_raw, Path)
            else str(Path(dpo_output_dir_raw).resolve())
        )

        logger.info("DPO output directory: %s", dpo_output_dir_str)

        run_dpo_training(
            model_path=dpo_model_path,
            dataset=pref_dataset,
            output_dir=dpo_output_dir_str,
            tokenizer=tokenizer,
            lora_rank=dpo_config.lora_rank,
            max_steps=dpo_config.max_steps,
            learning_rate=dpo_config.learning_rate,
            beta=dpo_config.beta,
            per_device_train_batch_size=dpo_config.per_device_train_batch_size,
            gradient_accumulation_steps=dpo_config.gradient_accumulation_steps,
            optim=dpo_config.optim,
            lr_scheduler_type=dpo_config.lr_scheduler_type,
            eval_steps=dpo_config.eval_steps,
            save_steps=dpo_config.save_steps,
            save_total_limit=dpo_config.save_total_limit,
            logging_steps=dpo_config.logging_steps,
            gradient_checkpointing=dpo_config.gradient_checkpointing,
            warmup_ratio=dpo_config.warmup_ratio,
            torch_compile=torch_compile,
            use_linear_attention_kernels=config.model.use_linear_attention_kernels,
            preserve_all_tensors=config.model.preserve_all_tensors,
        )
        logger.info("DPO phase completed successfully")
    except Exception as e:
        logger.error("DPO phase failed: %s", e, exc_info=True)
        raise RuntimeError(f"DPO training failed: {e}") from e

    # Inference Phase (optional)
    if run_inference:
        logger.info("=" * SEPARATOR_LENGTH)
        logger.info("Inference Phase: Tool-Augmented Agent")
        logger.info("=" * SEPARATOR_LENGTH)

        queries = example_queries or DEFAULT_QUERIES
        if not example_queries:
            logger.info("Using default example queries")

        try:
            run_inference_phase(config, queries)
            logger.info("Inference phase completed successfully")
        except Exception as e:
            logger.error("Inference phase failed: %s", e, exc_info=True)
            raise RuntimeError(f"Inference failed: {e}") from e

    logger.info("=" * SEPARATOR_LENGTH)
    logger.info("Pipeline completed successfully")
    logger.info("=" * SEPARATOR_LENGTH)


def _build_pipeline_parser(prog: str) -> argparse.ArgumentParser:
    """Build argparse parser for the training pipeline and inference-only mode."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run QLoRA fine-tuning, merge LoRA weights, DPO training, "
            "and optional inference."
        ),
    )
    parser.add_argument(
        "--config-path",
        required=True,
        type=Path,
        help="Path to configuration YAML file.",
    )
    inference_mode = parser.add_mutually_exclusive_group()
    inference_mode.add_argument(
        "--run-inference",
        action="store_true",
        help="Run inference after DPO training.",
    )
    inference_mode.add_argument(
        "--inference-only",
        action="store_true",
        help=(
            "Skip SFT, merge, and DPO; run inference against an existing "
            "checkpoint (dpo_model, else final_merged_model)."
        ),
    )
    parser.add_argument(
        "--example-queries",
        action="append",
        default=[],
        help="Example queries for inference (repeatable).",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for DPO training (PyTorch 2.0+).",
    )
    return parser


def _build_optimize_config_parser(prog: str) -> argparse.ArgumentParser:
    """Build argparse parser for the optimize-config subcommand."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Suggest config.yaml from hardware and preset "
            "(fast/quality/balanced/low_memory)."
        ),
    )
    parser.add_argument(
        "--config-path",
        required=True,
        type=Path,
        help="Path to base config YAML (required).",
    )
    parser.add_argument(
        "-p",
        "--preset",
        choices=["fast", "quality", "balanced", "low_memory"],
        default="balanced",
        help="Preset (default: balanced).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        help="Write optimized config to this YAML path (default: print only).",
    )
    return parser


def _run_optimize_config(argv: list[str]) -> int:
    """Execute the optimize-config subcommand."""
    from src.hardware import HardwareProfile
    from src.model_optimizer import PRESETS, PresetName, run_optimizer

    parser = _build_optimize_config_parser(f"{PROG_NAME} optimize-config")
    args = parser.parse_args(argv)

    if args.preset not in PRESETS:
        parser.error(
            f"Preset must be one of: {', '.join(PRESETS)}. Got: {args.preset!r}"
        )

    try:
        merged = run_optimizer(
            config_path=str(args.config_path),
            preset=cast(PresetName, args.preset),
            output_path=str(args.output) if args.output else None,
        )
        profile = HardwareProfile()
        vram_gb = profile.vram_bytes // (1024**3)
        ram_gb = profile.system_ram_bytes // (1024**3)
        t = merged.get("training", {})
        logger.info(
            "Suggested for %s preset (%s GB VRAM / %s GB RAM): "
            "batch_size=%s, eval_steps=%s, save_steps=%s",
            args.preset,
            vram_gb,
            ram_gb,
            t.get("per_device_train_batch_size"),
            t.get("eval_steps"),
            t.get("save_steps"),
        )
        if args.output is None:
            print(
                "Optimized config validated. Use --output <path> "
                "to write the new config YAML."
            )
        return 0
    except Exception:
        logger.exception("Optimize-config failed")
        return 1


def _run_pipeline_cli(argv: list[str]) -> int:
    """Execute the training pipeline or inference-only mode from CLI arguments."""
    parser = _build_pipeline_parser(PROG_NAME)
    args = parser.parse_args(argv)

    try:
        config = load_config_from_yaml(args.config_path)
        if args.inference_only:
            queries = args.example_queries or DEFAULT_QUERIES
            run_inference_phase(config, queries)
            return 0
        use_torch_compile = args.torch_compile or (
            config.dpo.torch_compile if config.dpo else False
        )
        run_pipeline(
            config,
            run_inference=args.run_inference,
            example_queries=args.example_queries,
            torch_compile=use_torch_compile,
        )
        return 0
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 1
    except Exception:
        logger.exception("Pipeline failed")
        return 1


def cli(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to pipeline or optimize-config.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 success, non-zero failure).
    """
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _build_pipeline_parser(PROG_NAME).print_help()
        return 2

    if argv[0] == "optimize-config":
        return _run_optimize_config(argv[1:])

    if argv[0] in ("-h", "--help"):
        _build_pipeline_parser(PROG_NAME).print_help()
        return 0

    return _run_pipeline_cli(argv)


if __name__ == "__main__":
    raise SystemExit(cli())
