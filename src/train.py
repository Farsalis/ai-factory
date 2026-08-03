"""Training, merging, and model saving orchestration logic.

This module provides functions for supervised fine-tuning (SFT) using QLoRA,
merging adapters with base models, and orchestrating the complete training pipeline.
"""

import inspect
import logging
from dataclasses import fields as dataclass_fields
from typing import Any

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer  # type: ignore[attr-defined]

from src.config import ScriptConfig
from src.data import VectorizedCompletionOnlyCollator, load_and_prepare_dataset
from src.model_setup import (
    load_model,
    load_tokenizer,
    validate_linear_attention_kernels,
)
from src.utils import Environment

logger = logging.getLogger(__name__)


def _determine_effective_optimizer(config_optim: str, env: Environment) -> str:
    """Determine the effective optimizer based on environment capabilities.

    Args:
        config_optim: Optimizer specified in config.
        env: Environment object with hardware capabilities.

    Returns:
        Effective optimizer name to use.
    """
    if config_optim.startswith("paged_") and not (
        env.cuda_available and env.bnb_available
    ):
        logger.warning(
            f"Paged optimizer '{config_optim}' requires CUDA and bitsandbytes. "
            "Falling back to 'adamw_torch'."
        )
        return "adamw_torch"
    return config_optim


def _prepare_training_arguments(
    config: ScriptConfig, env: Environment
) -> dict[str, Any]:
    """Prepare TrainingArguments dictionary from config and environment.

    Args:
        config: Script configuration.
        env: Environment object with hardware capabilities.

    Returns:
        Dictionary of arguments for TrainingArguments.
    """
    allowed_fields: set[str] = {f.name for f in dataclass_fields(TrainingArguments)}
    args_dict = {
        k: v for k, v in config.training.model_dump().items() if k in allowed_fields
    }

    # Override path fields with string conversions
    if "output_dir" in allowed_fields:
        args_dict["output_dir"] = str(config.training.output_dir)
    if "logging_dir" in allowed_fields:
        args_dict["logging_dir"] = str(config.training.output_dir / "logs")

    # Set precision flags based on environment
    if "bf16" in allowed_fields:
        args_dict["bf16"] = env.bf16_supported
    if "fp16" in allowed_fields:
        args_dict["fp16"] = not env.bf16_supported and env.cuda_available

    # Set optimizer
    effective_optim = _determine_effective_optimizer(config.training.optim, env)
    if "optim" in allowed_fields:
        args_dict["optim"] = effective_optim

    # Handle load_best_model_at_end logic
    _configure_best_model_loading(args_dict, allowed_fields)

    return args_dict


def _configure_best_model_loading(
    args_dict: dict[str, Any], allowed_fields: set[str]
) -> None:
    """Configure load_best_model_at_end based on available fields.

    Args:
        args_dict: Dictionary of training arguments (modified in place).
        allowed_fields: Set of allowed TrainingArguments field names.
    """
    wants_best = args_dict.get("load_best_model_at_end", False)
    if not wants_best:
        return

    has_eval_strategy = (
        "evaluation_strategy" in allowed_fields or "eval_strategy" in allowed_fields
    )

    if not has_eval_strategy:
        logger.warning(
            "Disabling load_best_model_at_end: this transformers version "
            "does not expose evaluation_strategy."
        )
        args_dict["load_best_model_at_end"] = False
        return

    # Sync evaluation strategy with save strategy if not explicitly set
    save_strategy = args_dict.get("save_strategy")
    if save_strategy is not None:
        if (
            "evaluation_strategy" in allowed_fields
            and "evaluation_strategy" not in args_dict
        ):
            args_dict["evaluation_strategy"] = save_strategy
        if "eval_strategy" in allowed_fields and "eval_strategy" not in args_dict:
            args_dict["eval_strategy"] = save_strategy


def _prepare_trainer_kwargs(
    model: PreTrainedModel,
    training_args: TrainingArguments,
    dataset: dict[str, Any],
    lora_config: PeftLoraConfig,
    tokenizer: PreTrainedTokenizer,
    config: ScriptConfig,
) -> dict[str, Any]:
    """Prepare keyword arguments for SFTTrainer initialization.

    Args:
        model: The model to train.
        training_args: TrainingArguments instance.
        dataset: Dictionary with 'train' and 'validation' datasets.
        lora_config: LoRA configuration.
        tokenizer: Tokenizer instance.
        config: Script configuration.

    Returns:
        Dictionary of keyword arguments for SFTTrainer.
    """
    data_collator = VectorizedCompletionOnlyCollator(
        tokenizer=tokenizer,
        response_template="Assistant: ",
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "peft_config": lora_config,
        "data_collator": data_collator,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=3)],
    }

    # Check SFTTrainer signature for optional parameters
    sft_sig = inspect.signature(SFTTrainer.__init__)
    sft_params = set(sft_sig.parameters.keys())

    if "tokenizer" in sft_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in sft_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in sft_params:
        trainer_kwargs["max_seq_length"] = config.model.max_length

    # Handle tokenizer parameter compatibility
    if "tokenizer" not in sft_params:
        logger.warning(
            "Installed TRL SFTTrainer does not accept 'tokenizer'. "
            "Pre-tokenizing dataset."
        )
        trainer_kwargs = _pre_tokenize_datasets(
            trainer_kwargs, tokenizer, config.model.max_length
        )

    return trainer_kwargs


def _pre_tokenize_datasets(
    trainer_kwargs: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> dict[str, Any]:
    """Pre-tokenize datasets when SFTTrainer doesn't accept tokenizer.

    Args:
        trainer_kwargs: Trainer keyword arguments (modified in place).
        tokenizer: Tokenizer to use for tokenization.
        max_length: Maximum sequence length.

    Returns:
        Updated trainer_kwargs with tokenized datasets.
    """

    def tokenize_example(example: dict[str, Any]) -> dict[str, Any]:
        """Tokenize a single example."""
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,  # Explicitly request attention_mask
        )
        # Ensure we have valid tokenized data with both required fields
        input_ids = result.get("input_ids", [])
        attention_mask = result.get("attention_mask", [])

        # If input_ids is empty or missing, create a minimal valid tokenization
        if not input_ids or len(input_ids) == 0:
            logger.warning(
                "Empty tokenization result for example. Text length: %s",
                len(example.get("text", "")),
            )
            # Return a minimal valid tokenization (just pad token)
            pad_token_id = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )
            if pad_token_id is None:
                pad_token_id = 0
            input_ids = [pad_token_id]
            attention_mask = [1]
        # If attention_mask is missing, create it from input_ids
        elif not attention_mask or len(attention_mask) == 0:
            # Create attention_mask: 1 for all non-pad tokens
            pad_token_id = (
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            )
            attention_mask = [
                1 if token_id != pad_token_id else 0 for token_id in input_ids
            ]

        # Ensure lengths match
        if len(input_ids) != len(attention_mask):
            # Truncate or pad attention_mask to match input_ids
            if len(attention_mask) < len(input_ids):
                attention_mask.extend([1] * (len(input_ids) - len(attention_mask)))
            else:
                attention_mask = attention_mask[: len(input_ids)]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    train_dataset = trainer_kwargs["train_dataset"]
    eval_dataset = trainer_kwargs["eval_dataset"]

    tokenized_train = train_dataset.map(
        tokenize_example,
        remove_columns=list(train_dataset.features),
    ).filter(lambda x: len(x.get("input_ids", [])) > 0)

    tokenized_eval = eval_dataset.map(
        tokenize_example,
        remove_columns=list(eval_dataset.features),
    ).filter(lambda x: len(x.get("input_ids", [])) > 0)

    trainer_kwargs.update(
        {
            "train_dataset": tokenized_train,
            "eval_dataset": tokenized_eval,
        }
    )
    trainer_kwargs.pop("dataset_text_field", None)

    return trainer_kwargs


def run_training(
    config: ScriptConfig,
    env: Environment,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
) -> None:
    """Conduct supervised fine-tuning (SFT) using QLoRA.

    Args:
        config: Script configuration.
        env: Environment object with hardware capabilities.
        tokenizer: Pre-trained tokenizer.
        model: Pre-trained model (typically quantized).

    Raises:
        RuntimeError: If training fails.
    """
    logger.info("Loading and preparing dataset...")
    dataset = load_and_prepare_dataset(config.data, tokenizer)

    logger.info("Configuring LoRA adapter...")
    lora_config = PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora.target_modules,
    )

    logger.info("Preparing training arguments...")
    args_dict = _prepare_training_arguments(config, env)
    training_args = TrainingArguments(**args_dict)

    # Configure model cache based on gradient checkpointing
    if training_args.gradient_checkpointing:
        logger.info("Gradient checkpointing enabled. Disabling model cache.")
        model.config.use_cache = False

    logger.info("Preparing trainer configuration...")
    trainer_kwargs = _prepare_trainer_kwargs(
        model=model,
        training_args=training_args,
        dataset=dataset,
        lora_config=lora_config,
        tokenizer=tokenizer,
        config=config,
    )

    trainer = SFTTrainer(**trainer_kwargs)

    logger.info("Starting model training...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise RuntimeError(f"Training error: {e}") from e

    adapter_save_path = config.training.output_dir / "final_adapter"
    logger.info(f"Training complete. Saving final adapter to {adapter_save_path}")
    trainer.save_model(str(adapter_save_path))


def merge_and_save_model(config: ScriptConfig, env: Environment) -> None:
    """Load base model, merge adapter, and save standalone model.

    Args:
        config: Script configuration.
        env: Environment object with hardware capabilities.

    Raises:
        FileNotFoundError: If adapter path doesn't exist.
        RuntimeError: If merging fails.
    """
    logger.info("Starting model merge process...")
    adapter_path = config.training.output_dir / "final_adapter"

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter path not found: {adapter_path}. Run training first."
        )

    logger.info(
        f"Reloading base model '{config.model.name}' in high precision for merging..."
    )

    validate_linear_attention_kernels(config.model.use_linear_attention_kernels)

    try:
        high_precision_model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            trust_remote_code=config.model.trust_remote_code,
            torch_dtype=env.compute_dtype,
            device_map="cpu",
            use_safetensors=True,
        )
    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        raise RuntimeError(f"Model loading error: {e}") from e

    logger.info(f"Loading adapter from {adapter_path} and applying to base model...")
    try:
        peft_model = PeftModel.from_pretrained(high_precision_model, adapter_path)
    except Exception as e:
        logger.error(f"Failed to load adapter: {e}")
        raise RuntimeError(f"Adapter loading error: {e}") from e

    logger.info("Merging adapter weights into base model...")
    try:
        merged_model = peft_model.merge_and_unload()
    except Exception as e:
        logger.error(f"Failed to merge adapter: {e}")
        raise RuntimeError(f"Merging error: {e}") from e

    merged_save_path = config.training.output_dir / "final_merged_model"
    merged_save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to {merged_save_path}...")
    try:
        merged_model.save_pretrained(str(merged_save_path), safe_serialization=True)
    except Exception as e:
        logger.error(f"Failed to save merged model: {e}")
        raise RuntimeError(f"Model saving error: {e}") from e

    logger.info("Saving tokenizer...")
    try:
        tokenizer = load_tokenizer(config.model)
        tokenizer.save_pretrained(str(merged_save_path))
    except Exception as e:
        logger.error(f"Failed to save tokenizer: {e}")
        raise RuntimeError(f"Tokenizer saving error: {e}") from e

    logger.info("Merged model and tokenizer saved successfully.")


def run_pipeline(config: ScriptConfig) -> None:
    """Orchestrate the complete fine-tuning and merging process.

    Args:
        config: Script configuration.

    Raises:
        RuntimeError: If any phase of the pipeline fails.
    """
    logger.info("Starting fine-tuning pipeline...")
    logger.info(f"Configuration:\n{config.model_dump_json(indent=2)}")

    env = Environment()
    env.setup_backends()
    torch.manual_seed(config.training.seed)

    logger.info("--- Entering Training Phase ---")
    try:
        tokenizer = load_tokenizer(config.model)
        quantized_model = load_model(config.model, config.quantization, env)
        run_training(config, env, tokenizer, quantized_model)
    except Exception as e:
        logger.error(f"Training phase failed: {e}")
        raise
    finally:
        # Clean up memory
        if "quantized_model" in locals():
            del quantized_model
        if env.cuda_available:
            torch.cuda.empty_cache()

    logger.info("--- Entering Merging Phase ---")
    try:
        merge_and_save_model(config, env)
    except Exception as e:
        logger.error(f"Merging phase failed: {e}")
        raise

    logger.info("Pipeline finished successfully.")
