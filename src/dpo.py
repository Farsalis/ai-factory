"""DPO training script for optimizing tool selection in Mistral-7B.

This script generates preference datasets from augmented 'Breaking Better' JSONL files
and runs Direct Preference Optimization (DPO) training to improve tool selection.
It creates chosen/rejected pairs (correct vs. incorrect/no tool calls) and fine-tunes
using TRL's DPOTrainer.

Usage:
    python dpo.py --input /path/to/augmented_data.jsonl \\
        --output /path/to/dpo_preferences.jsonl \\
        --model_path mistralai/Mistral-7B-Instruct-v0.3 \\
        --training_output_dir /path/to/dpo_output
"""

import argparse
import inspect
import json
import logging
import random
from collections.abc import Callable
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset  # type: ignore[import-untyped]
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    TrainingArguments,
)

# Ensure compatibility with TRL expecting torch.distributed.fsdp.FSDPModule.
_has_fsdp = hasattr(torch, "distributed") and hasattr(torch.distributed, "fsdp")
if _has_fsdp and not hasattr(torch.distributed.fsdp, "FSDPModule"):
    from torch.distributed.fsdp import FullyShardedDataParallel

    torch.distributed.fsdp.FSDPModule = FullyShardedDataParallel  # type: ignore[attr-defined]

try:
    from trl import (
        DPOConfig,  # type: ignore[attr-defined]  # Newer TRL exposes DPOConfig; TRL 0.8.x does not
    )
except ImportError:
    DPOConfig = None  # type: ignore[assignment,misc]

from trl import (  # noqa: E402  (must follow optional DPOConfig import above)
    DPOTrainer,  # type: ignore[attr-defined]
)

from src.model_setup import (  # noqa: E402  (must follow torch-dependent shim above)
    resolve_model_class,
    validate_linear_attention_kernels,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
KNOWN_TOOLS = [
    "search_web",
    "calc_tool",
    "news_tool",
    "python_repl",
    "read_file",
    "write_file",
    "calendar_tool",
    "task_tracker_tool",
    "job_search_tool",
    "get_current_weather",
    "animal_medical_database",
]

# Per-tool builder for synthetic incorrect-tool arguments. The fallback used
# for any tool not listed here is {"query": <user_msg>}; see
# _generate_incorrect_tool_response.
_WRONG_TOOL_ARG_BUILDERS: dict[str, Callable[[str], dict[str, Any]]] = {
    "search_web": lambda user_msg: {"query": user_msg},
    "news_tool": lambda user_msg: {"query": user_msg},
    "job_search_tool": lambda user_msg: {"query": user_msg},
    "calc_tool": lambda _user_msg: {"query": "2+2"},
    "python_repl": lambda _user_msg: {"code": "print('test')"},
    "read_file": lambda _user_msg: {"filepath": "example.txt"},
    "write_file": lambda _user_msg: {"filepath": "example.txt", "content": "test"},
    "get_current_weather": lambda user_msg: {"location": user_msg},
}

# Training defaults
DEFAULT_LORA_RANK = 16
DEFAULT_MAX_STEPS = 100
DEFAULT_LEARNING_RATE = 5e-6
DEFAULT_LORA_ALPHA_MULTIPLIER = 2
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_BETA = 0.1
DEFAULT_MAX_LENGTH = 1024
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_BATCH_SIZE = 1
DEFAULT_SAVE_STEPS = 50
DEFAULT_SAVE_TOTAL_LIMIT = 2
DEFAULT_LOGGING_STEPS = 10
DEFAULT_EVAL_STEPS = 50

# LLM instruction format
INST_START = "[INST]"
INST_END = "[/INST]"


def _is_triton_available() -> bool:
    """Check if Triton is available (required for torch.compile on GPU).

    Triton is Linux-only; on Windows pip typically cannot install it,
    causing torch.compile to fail with 'Cannot find a working triton installation'.
    """
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries.

    Accepts generic JSONL lines and does not require a specific schema.
    Skips empty lines and non-dict entries with appropriate warnings.

    Args:
        file_path: Path to input JSONL file.

    Returns:
        List of parsed JSON objects (dictionaries).

    Raises:
        ValueError: If the input file does not exist.
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise ValueError(f"Input file not found: {file_path}")

    data: list[dict[str, Any]] = []
    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                if isinstance(example, dict):
                    data.append(example)
                else:
                    logger.warning(
                        f"Skipping non-dict JSON at line {line_num}: {type(example)}"
                    )
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error at line {line_num}: {e}")
                continue

    logger.info(f"Loaded {len(data)} examples from {file_path}")
    return data


def save_jsonl(data: list[dict[str, Any]], file_path: str) -> None:
    """Save a list of dictionaries to a JSONL file.

    Creates parent directories if they don't exist.

    Args:
        data: List of JSON-serializable dictionaries.
        file_path: Path to output JSONL file.
    """
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(data)} examples to {file_path}")


def _has_tool_call(response: str) -> bool:
    """Check if a response contains a tool call.

    Args:
        response: Assistant response text.

    Returns:
        True if response contains a tool call pattern, False otherwise.
    """
    return "tool_call" in response or "Need data:" in response


def _extract_text_without_tool(response: str) -> str:
    """Extract text portion of response without tool calls.

    Args:
        response: Full assistant response.

    Returns:
        Text portion without tool calls, or fallback message if empty.
    """
    # Try to extract text before any tool call markers
    if "Need data:" in response:
        text_part = response.split("Need data:")[0].strip()
    elif "tool_call" in response:
        text_part = response.split("tool_call")[0].strip()
    else:
        text_part = response.strip()

    return text_part or "I'm not sure how to handle that."


def _generate_incorrect_tool_response(user_msg: str, correct_response: str) -> str:
    """Generate a rejected response with an incorrect tool call.

    Args:
        user_msg: Original user message.
        correct_response: The correct assistant response.

    Returns:
        A response string with an incorrect tool call.
    """
    # Find tools not used in the correct response
    available_tools = [t for t in KNOWN_TOOLS if t not in correct_response]
    if not available_tools:
        available_tools = KNOWN_TOOLS

    wrong_tool = random.choice(available_tools)  # noqa: S311

    # Create a safe argument for the tool
    safe_user_msg = (
        user_msg.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")[:64]
    )

    # Generate tool call arguments based on tool type
    builder = _WRONG_TOOL_ARG_BUILDERS.get(
        wrong_tool, lambda user_msg: {"query": user_msg}
    )
    tool_args: dict[str, Any] = builder(safe_user_msg)

    tool_call = {"tool_call": {"name": wrong_tool, "arguments": tool_args}}
    tool_call_json = json.dumps(tool_call, ensure_ascii=False)

    return f"Need data: {tool_call_json}"


def generate_preference_pairs(
    original_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate DPO preference pairs from augmented dataset.

    Creates chosen/rejected pairs where:
    - Chosen: Correct tool call response (from original data)
    - Rejected: Incorrect or no tool call response

    Args:
        original_data: Loaded augmented dataset with messages format.

    Returns:
        List of preference pairs with keys: "prompt", "chosen", "rejected".
    """
    preference_data: list[dict[str, Any]] = []

    for example in original_data:
        if not isinstance(example.get("messages"), list) or not example["messages"]:
            continue

        messages = example["messages"]
        user_msg = next(
            (msg.get("content", "") for msg in messages if msg.get("role") == "user"),
            "",
        )
        assistant_response = next(
            (
                msg.get("content", "")
                for msg in messages
                if msg.get("role") == "assistant"
            ),
            "",
        )

        if not user_msg or not assistant_response:
            continue

        prompt = f"{INST_START}{user_msg}{INST_END}"
        chosen = assistant_response

        # Generate rejected response
        if _has_tool_call(assistant_response):
            # Chosen has tool: rejected is response without tool
            rejected = _extract_text_without_tool(assistant_response)
        else:
            # Chosen has no tool: rejected is response with incorrect tool
            rejected = _generate_incorrect_tool_response(user_msg, assistant_response)

        if prompt and chosen and rejected:
            preference_data.append(
                {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            )

    logger.info(f"Generated {len(preference_data)} preference pairs")
    return preference_data


def prepare_dpo_dataset(preference_data: list[dict[str, Any]]) -> Dataset:
    """Convert preference data list to a Hugging Face Dataset.

    Args:
        preference_data: List of preference pairs with "prompt", "chosen", "rejected".

    Returns:
        A Hugging Face Dataset object ready for DPO training.

    Raises:
        ValueError: If preference_data is empty or missing required keys.
    """
    if not preference_data:
        raise ValueError("Preference data list is empty")

    required_keys = {"prompt", "chosen", "rejected"}
    for idx, pair in enumerate(preference_data):
        missing_keys = required_keys - set(pair.keys())
        if missing_keys:
            raise ValueError(
                f"Preference pair at index {idx} missing required keys: {missing_keys}"
            )

    return Dataset.from_list(preference_data)


def run_dpo_training(
    model_path: str,
    dataset: Dataset,
    output_dir: str,
    tokenizer: PreTrainedTokenizer | None = None,
    lora_rank: int = DEFAULT_LORA_RANK,
    max_steps: int = DEFAULT_MAX_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    beta: float = DEFAULT_BETA,
    per_device_train_batch_size: int = DEFAULT_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    optim: str = "paged_adamw_8bit",
    lr_scheduler_type: str = "cosine",
    eval_steps: int = DEFAULT_EVAL_STEPS,
    save_steps: int = DEFAULT_SAVE_STEPS,
    save_total_limit: int = DEFAULT_SAVE_TOTAL_LIMIT,
    logging_steps: int = DEFAULT_LOGGING_STEPS,
    gradient_checkpointing: bool = True,
    warmup_ratio: float = 0.03,
    torch_compile: bool = False,
    use_linear_attention_kernels: bool = False,
    preserve_all_tensors: bool = True,
) -> None:
    """Run DPO training on the preference dataset.

    Uses QLoRA (4-bit quantization) with LoRA adapters for memory-efficient training.
    Configures the model, tokenizer, and training arguments, then runs DPO training.

    Args:
        model_path: Path to the base model (Hugging Face model ID or local path).
        dataset: DPO preference dataset with prompt/chosen/rejected pairs.
        output_dir: Directory to save the trained DPO model.
        tokenizer: Pre-loaded tokenizer (optional, will load if not provided).
        lora_rank: LoRA rank for QLoRA (lower = less memory, less capacity).
        max_steps: Maximum number of training steps.
        learning_rate: Learning rate for DPO optimizer.
        beta: DPO beta parameter (temperature for reward model).
        per_device_train_batch_size: Batch size per device for training.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        optim: Optimizer name (e.g., 'paged_adamw_8bit').
        lr_scheduler_type: Learning rate scheduler type.
        eval_steps: Number of steps between evaluations.
        save_steps: Number of steps between checkpoints.
        save_total_limit: Maximum number of checkpoints to keep.
        logging_steps: Number of steps between logging.
        gradient_checkpointing: Whether to use gradient checkpointing.
        warmup_ratio: Ratio of warmup steps to total steps.
        torch_compile: Whether to use torch.compile for faster training (PyTorch 2.0+).
        use_linear_attention_kernels: Require causal_conv1d + fla when True.
        preserve_all_tensors: Load the checkpoint's declared architecture so no
            tensors (e.g. a multimodal vision tower) are silently discarded.

    Raises:
        ValueError: If model_path is invalid or dataset is empty.
        RuntimeError: If training fails due to CUDA/memory issues.
    """
    logger.info(f"Starting DPO training with model: {model_path}")

    validate_linear_attention_kernels(use_linear_attention_kernels)

    if len(dataset) == 0:
        raise ValueError("Dataset is empty, cannot train")

    try:
        # Load tokenizer if not provided
        if tokenizer is None:
            logger.info("Loading tokenizer...")
            loaded_tokenizer = cast(
                PreTrainedTokenizer,
                AutoTokenizer.from_pretrained(
                    model_path, use_fast=True, trust_remote_code=True
                ),
            )
            tokenizer = loaded_tokenizer
            if tokenizer is None:
                raise RuntimeError(f"Failed to load tokenizer from {model_path}")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        # QLoRA configuration for memory-efficient training
        logger.info("Configuring QLoRA (4-bit quantization)...")
        bnb_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # With 4-bit quantization, device_map="auto" can leave tensors on "meta",
        # causing "expected device meta but got cuda:0" during backward. Use
        # explicit single-GPU placement when CUDA is available.
        device_map = "cuda:0" if torch.cuda.is_available() else "auto"
        # Load model with quantization
        logger.info("Loading model with 4-bit quantization...")
        model_class = resolve_model_class(
            model_path,
            trust_remote_code=True,
            preserve_all_tensors=preserve_all_tensors,
        )
        model = model_class.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )

        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)  # type: ignore[no-untyped-call]

        # LoRA configuration
        logger.info(f"Configuring LoRA with rank {lora_rank}...")
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * DEFAULT_LORA_ALPHA_MULTIPLIER,
            lora_dropout=DEFAULT_LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )

        # Training arguments
        logger.info("Setting up training arguments...")
        #
        # TRL's DPOTrainer expects a DPOConfig (TrainingArguments-like) when available.
        # In newer TRL versions (including your pinned `trl==0.27.0`), DPO-specific
        # hyperparameters like `beta` live inside DPOConfig, not as a DPOTrainer kwarg.
        if DPOConfig is not None:
            # TRL 0.29+ DPOConfig uses only max_length (no max_prompt_length).
            dpo_kwargs: dict[str, Any] = {
                "output_dir": output_dir,
                "per_device_train_batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "logging_steps": logging_steps,
                "max_steps": max_steps,
                "report_to": "none",
                "save_strategy": "steps",
                "save_steps": save_steps,
                "save_total_limit": save_total_limit,
                "bf16": True,
                "optim": optim,
                "lr_scheduler_type": lr_scheduler_type,
                "warmup_ratio": warmup_ratio,
                "gradient_checkpointing": gradient_checkpointing,
                "beta": beta,
                "max_length": DEFAULT_MAX_LENGTH,
            }
            dpo_fields: set[str]
            try:
                dpo_fields = {field.name for field in dataclass_fields(DPOConfig)}
            except TypeError:
                dpo_fields = set(inspect.signature(DPOConfig.__init__).parameters)
            if "eval_strategy" in dpo_fields:
                dpo_kwargs["eval_strategy"] = "no"
            elif "evaluation_strategy" in dpo_fields:
                dpo_kwargs["evaluation_strategy"] = "no"
            training_args = DPOConfig(**dpo_kwargs)
        else:
            # Very old TRL fallback
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                logging_steps=logging_steps,
                max_steps=max_steps,
                report_to="none",
                save_strategy="steps",
                save_steps=save_steps,
                save_total_limit=save_total_limit,
                bf16=True,
                optim=optim,
                lr_scheduler_type=lr_scheduler_type,
                warmup_ratio=warmup_ratio,
                gradient_checkpointing=gradient_checkpointing,
            )

        # DPO Trainer
        logger.info("Initializing DPO trainer...")
        # TRL's API: newer versions use processing_class= instead of tokenizer=.
        dpo_init_params = set(inspect.signature(DPOTrainer.__init__).parameters.keys())
        trainer_kwargs: dict[str, Any] = {
            "args": training_args,
            "train_dataset": dataset,
            "peft_config": peft_config,
        }
        if "processing_class" in dpo_init_params:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in dpo_init_params:
            trainer_kwargs["tokenizer"] = tokenizer

        # Only pass beta directly if TRL is old and expects it.
        if DPOConfig is None and "beta" in dpo_init_params:
            trainer_kwargs["beta"] = beta

        dpo_trainer = DPOTrainer(model, **trainer_kwargs)

        # Optional torch.compile for faster training (requires Triton; not on Windows)
        if torch_compile:
            if _is_triton_available():
                logger.info("Compiling model with torch.compile...")
                dpo_trainer.model = torch.compile(dpo_trainer.model)  # type: ignore[assignment]
            else:
                logger.warning(
                    "torch_compile requested but Triton is not available "
                    "(e.g. on Windows). Skipping compilation; training runs eagerly."
                )

        # Train
        logger.info(f"Starting training for {max_steps} steps...")
        dpo_trainer.train()

        logger.info(f"DPO training complete. Model saved to {output_dir}")

        # Clean up GPU memory
        del model
        del dpo_trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

    except Exception as e:
        logger.error(f"DPO training error: {e}", exc_info=True)
        raise


def main() -> None:
    """Main entry point for DPO training script.

    Parses command-line arguments, generates preference pairs from input data,
    and runs DPO training on the preference dataset.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate preference data and run DPO training for "
            "tool selection optimization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to input JSONL file with augmented dataset",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path to save DPO preference JSONL file",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to base or fine-tuned Mistral model (Hugging Face ID or local path)",
    )
    parser.add_argument(
        "--training_output_dir",
        required=True,
        type=str,
        help="Directory for DPO model output",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=DEFAULT_LORA_RANK,
        help="LoRA rank for QLoRA (lower = less memory)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for DPO optimizer",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Enable torch.compile on the model (PyTorch 2.0+, faster training)",
    )

    args = parser.parse_args()

    try:
        # Load and generate preference dataset
        logger.info(f"Loading input data from {args.input}...")
        original_data = load_jsonl(args.input)

        logger.info("Generating preference pairs...")
        preference_data = generate_preference_pairs(original_data)

        if not preference_data:
            logger.warning("No preference pairs generated. Check input data format.")
            return

        logger.info(f"Saving preference data to {args.output}...")
        save_jsonl(preference_data, args.output)

        # Prepare dataset and run DPO
        logger.info("Preparing dataset for training...")
        dataset = prepare_dpo_dataset(preference_data)

        run_dpo_training(
            args.model_path,
            dataset,
            args.training_output_dir,
            lora_rank=args.lora_rank,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            torch_compile=args.torch_compile,
        )

        logger.info("Script completed successfully")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Main execution error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
