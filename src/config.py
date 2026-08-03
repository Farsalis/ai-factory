"""Defines the configuration structure using Pydantic for validation and type safety."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Default values as constants
DEFAULT_MAX_LENGTH = 32000
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_WEIGHT_DECAY = 0.001
DEFAULT_MAX_GRAD_NORM = 0.3
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_LORA_RANK = 32
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_EVAL_STEPS = 50
DEFAULT_SAVE_STEPS = 50
DEFAULT_SAVE_TOTAL_LIMIT = 2
DEFAULT_LOGGING_STEPS = 10
DEFAULT_DATALOADER_WORKERS = 2

# Default target modules for LoRA
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class DataConfig(BaseModel):
    """Configuration for data loading and processing.

    Attributes:
        train_file: Path to training data file (JSON/JSONL format).
        validation_file: Path to validation data file (JSON/JSONL format).

    Example:
        ```python
        config = DataConfig(
            train_file=Path("data/train.jsonl"),
            validation_file=Path("data/val.jsonl")
        )
        ```
    """

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    train_file: Path = Field(
        ...,
        description="Path to training data file (JSON or JSONL format)",
    )
    validation_file: Path = Field(
        ...,
        description="Path to validation data file (JSON or JSONL format)",
    )

    @field_validator("train_file", "validation_file")
    @classmethod
    def file_must_exist(cls, v: Path) -> Path:
        """Validate that the specified file exists.

        Args:
            v: File path to validate.

        Returns:
            The validated path.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not v.exists():
            raise FileNotFoundError(f"Data file not found: {v}")
        return v


class ModelConfig(BaseModel):
    """Configuration for the model and tokenizer.

    Attributes:
        name: Model identifier from Hugging Face Hub,
            (e.g. mistralai/Mistral-7B-Instruct-v0.3).
        max_length: Maximum sequence length for tokenization.
        attn_implementation: 'flash_attention_2' for supported hardware,
            'sdpa' for PyTorch 2.0+, 'eager' as fallback.
        trust_remote_code: Whether to trust remote code when loading the model.

    Example:
        ```python
        config = ModelConfig(
            name="mistralai/Mistral-7B-Instruct-v0.3",
            max_length=4096,
            attn_implementation="flash_attention_2"
        )
        ```
    """

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    name: str = Field(
        ...,
        description="Model identifier from Hugging Face Hub",
        examples=["mistralai/Mistral-7B-Instruct-v0.3"],
    )
    max_length: int = Field(
        DEFAULT_MAX_LENGTH,
        description="Maximum sequence length for the tokenizer",
        gt=0,
    )
    attn_implementation: Literal["eager", "flash_attention_2", "sdpa"] | None = Field(
        "flash_attention_2",
        description=(
            "Attention implementation to use. "
            "'flash_attention_2' is recommended for supported hardware, "
            "'sdpa' for PyTorch 2.0+, 'eager' as fallback."
        ),
    )
    trust_remote_code: bool = Field(
        True,
        description="Whether to trust remote code when loading the model",
    )
    use_linear_attention_kernels: bool = Field(
        False,
        description=(
            "Use optimized linear-attention kernels (causal-conv1d + fla) for "
            "Qwen3.5 Gated DeltaNet layers. Requires optional packages; fails "
            "fast at load time when enabled but dependencies are missing."
        ),
    )


class QuantizationConfig(BaseModel):
    """Configuration for BitsAndBytes quantization during training.

    Uses 4-bit quantization to reduce memory usage during training.
    NF4 quantization is recommended for most use cases.

    Attributes:
        enabled: Whether to enable 4-bit quantization.
        quant_type: Quantization type ('nf4' recommended, 'fp4' alternative).
        use_double_quant: Whether to use double quantization for better compression.

    Example:
        ```python
        config = QuantizationConfig(
            enabled=True,
            quant_type="nf4",
            use_double_quant=True
        )
        ```
    """

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    enabled: bool = Field(
        True,
        description="Enable 4-bit quantization for training (reduces memory usage)",
    )
    quant_type: str = Field(
        "nf4",
        description="Quantization type: 'nf4' (recommended) or 'fp4'",
        pattern="^(nf4|fp4)$",
    )
    use_double_quant: bool = Field(
        True,
        description="Use double quantization for better compression",
    )


class LoraConfigModel(BaseModel):
    """Configuration for PEFT LoRA (Low-Rank Adaptation).

    LoRA adds trainable low-rank matrices to model layers, enabling efficient
    fine-tuning with minimal parameter overhead.

    Attributes:
        r: LoRA rank (lower = fewer parameters, less capacity).
        alpha: LoRA alpha scaling factor (typically 2*r for balanced scaling).
        dropout: Dropout rate for LoRA layers.
        target_modules: List of module names to apply LoRA to.

    Example:
        ```python
        config = LoraConfigModel(
            r=16,
            alpha=32,
            dropout=0.05,
            target_modules=["q_proj", "v_proj"]
        )
        ```
    """

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    r: int = Field(
        DEFAULT_LORA_RANK,
        description="LoRA rank (lower = fewer parameters, less capacity)",
        gt=0,
    )
    alpha: int = Field(
        DEFAULT_LORA_ALPHA,
        description="LoRA alpha scaling factor (typically 2*r)",
        gt=0,
    )
    dropout: float = Field(
        DEFAULT_LORA_DROPOUT,
        description="Dropout rate for LoRA layers",
        ge=0.0,
        le=1.0,
    )
    target_modules: list[str] = Field(
        default_factory=lambda: DEFAULT_LORA_TARGET_MODULES.copy(),
        description="List of module names to apply LoRA to",
        min_length=1,
    )


class TrainingConfig(BaseModel):
    """Configuration for Hugging Face TrainingArguments.

    Comprehensive training configuration including optimization, scheduling,
    evaluation, and checkpointing settings.

    Attributes:
        output_dir: Directory to save model checkpoints and outputs.
        seed: Random seed for reproducibility.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per device for training.
        per_device_eval_batch_size: Batch size per device for evaluation.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        optim: Optimizer name (e.g., 'paged_adamw_8bit' for memory efficiency).
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        warmup_ratio: Ratio of warmup steps to total steps.
        lr_scheduler_type: Learning rate scheduler type.
        evaluation_strategy: When to run evaluation ('steps', 'epoch', 'no').
        eval_steps: Number of steps between evaluations.
        save_strategy: When to save checkpoints ('steps', 'epoch').
        save_steps: Number of steps between checkpoints.
        save_total_limit: Maximum number of checkpoints to keep.
        logging_steps: Number of steps between logging.
        group_by_length: Whether to group sequences by length for efficiency.
        gradient_checkpointing: Whether to use gradient checkpointing (saves memory).
        report_to: Where to report metrics ('none', 'tensorboard', 'wandb', etc.).
        load_best_model_at_end: Whether to load best model at end of training.
        metric_for_best_model: Metric to use for selecting best model.
        greater_is_better: Whether higher metric values are better.
        remove_unused_columns: Whether to remove unused columns from dataset.
        dataloader_num_workers: Number of worker processes for data loading.

    Example:
        ```python
        config = TrainingConfig(
            output_dir=Path("./output"),
            learning_rate=2e-4,
            per_device_train_batch_size=4
        )
        ```
    """

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    output_dir: Path = Field(
        ...,
        description="Directory to save model checkpoints and outputs",
    )
    seed: int = Field(
        DEFAULT_SEED,
        description="Random seed for reproducibility",
    )
    num_train_epochs: int = Field(
        1,
        description="Number of training epochs",
        gt=0,
    )
    per_device_train_batch_size: int = Field(
        DEFAULT_BATCH_SIZE,
        description="Batch size per device for training",
        gt=0,
    )
    per_device_eval_batch_size: int = Field(
        DEFAULT_BATCH_SIZE,
        description="Batch size per device for evaluation",
        gt=0,
    )
    gradient_accumulation_steps: int = Field(
        1,
        description="Number of steps to accumulate gradients before updating",
        gt=0,
    )
    optim: str = Field(
        "paged_adamw_8bit",
        description="Optimizer name (memory-efficient: 'paged_adamw_8bit')",
    )
    learning_rate: float = Field(
        DEFAULT_LEARNING_RATE,
        description="Learning rate for optimizer",
        gt=0.0,
    )
    weight_decay: float = Field(
        DEFAULT_WEIGHT_DECAY,
        description="Weight decay coefficient",
        ge=0.0,
    )
    max_grad_norm: float = Field(
        DEFAULT_MAX_GRAD_NORM,
        description="Maximum gradient norm for clipping",
        gt=0.0,
    )
    warmup_ratio: float = Field(
        DEFAULT_WARMUP_RATIO,
        description="Ratio of warmup steps to total training steps",
        ge=0.0,
        le=1.0,
    )
    lr_scheduler_type: str = Field(
        "constant",
        description="Learning rate scheduler type",
        pattern="^(linear|cosine|cosine_with_restarts|polynomial|constant|constant_with_warmup)$",
    )
    evaluation_strategy: str = Field(
        "steps",
        description="When to run evaluation",
        pattern="^(no|steps|epoch)$",
    )
    eval_steps: int = Field(
        DEFAULT_EVAL_STEPS,
        description="Number of steps between evaluations",
        gt=0,
    )
    save_strategy: str = Field(
        "steps",
        description="When to save checkpoints",
        pattern="^(no|steps|epoch)$",
    )
    save_steps: int = Field(
        DEFAULT_SAVE_STEPS,
        description="Number of steps between checkpoints",
        gt=0,
    )
    save_total_limit: int = Field(
        DEFAULT_SAVE_TOTAL_LIMIT,
        description="Maximum number of checkpoints to keep",
        ge=0,
    )
    save_only_model: bool = Field(
        False,
        description=(
            "Save only model weights in \
                checkpoints (no optimizer/scheduler state). "
            "Reduces checkpoint size and memory at save; \
                prevents resuming from checkpoint."
        ),
    )
    logging_steps: int = Field(
        DEFAULT_LOGGING_STEPS,
        description="Number of steps between logging",
        gt=0,
    )
    group_by_length: bool = Field(
        True,
        description="Group sequences by length for efficient batching",
    )
    gradient_checkpointing: bool = Field(
        True,
        description="Use gradient checkpointing to save memory",
    )
    report_to: str = Field(
        "none",
        description="Where to report metrics ('none', 'tensorboard', 'wandb', etc.)",
    )
    load_best_model_at_end: bool = Field(
        True,
        description="Load best model checkpoint at end of training",
    )
    metric_for_best_model: str = Field(
        "eval_loss",
        description="Metric name to use for selecting best model",
    )
    greater_is_better: bool = Field(
        False,
        description="Whether higher metric values indicate better performance",
    )
    remove_unused_columns: bool = Field(
        True,
        description="Remove unused columns from dataset to save memory",
    )
    dataloader_num_workers: int = Field(
        DEFAULT_DATALOADER_WORKERS,
        description="Number of worker processes for data loading",
        ge=0,
    )


class DPOConfig(BaseModel):
    """Configuration for Direct Preference Optimization (DPO) training.

    Parameters for DPO fine-tuning using TRL's DPOTrainer with QLoRA.

    Attributes:
        output_dir: Directory for DPO outputs,
            (optional; defaults to training output_dir).
        train_file: Path to DPO preference JSONL. Optional;
            if unset, uses data.train_file.
        learning_rate: Learning rate for DPO optimizer.
        beta: DPO beta parameter (temperature for reward model).
        max_steps: Maximum number of training steps.
        per_device_train_batch_size: Batch size per device for training.
        per_device_eval_batch_size: Batch size per device for evaluation.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        optim: Optimizer name (e.g. paged_adamw_8bit for memory efficiency).
        lr_scheduler_type: Learning rate scheduler type.
        eval_steps: Number of steps between evaluations.
        save_steps: Number of steps between checkpoints.
        save_total_limit: Maximum number of checkpoints to keep.
        logging_steps: Number of steps between logging.
        gradient_checkpointing: Whether to use gradient checkpointing (saves memory).
        warmup_ratio: Ratio of warmup steps to total steps.
        lora_rank: LoRA rank for DPO (optional; defaults to training lora rank).

    Example:
        ```python
        config = DPOConfig(
            learning_rate=5e-6,
            beta=0.1,
            max_steps=100,
            per_device_train_batch_size=2
        )
        ```
    """

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    output_dir: Path | None = Field(
        None,
        description="DPO output dir (defaults to training output_dir if unset)",
    )
    train_file: Path | None = Field(
        None,
        description=(
            "Path to DPO preference data (messages-format JSONL with tool calls). "
            "If unset, uses data.train_file. Use a separate file when SFT uses ICDU."
        ),
    )
    learning_rate: float = Field(
        5e-6,
        description="Learning rate for DPO optimizer",
        gt=0.0,
    )
    beta: float = Field(
        0.1,
        description="DPO beta parameter (temperature for reward model)",
        gt=0.0,
    )
    max_steps: int = Field(
        100,
        description="Maximum number of training steps",
        gt=0,
    )
    per_device_train_batch_size: int = Field(
        1,
        description="Batch size per device for training",
        gt=0,
    )
    per_device_eval_batch_size: int = Field(
        1,
        description="Batch size per device for evaluation",
        gt=0,
    )
    gradient_accumulation_steps: int = Field(
        4,
        description="Number of steps to accumulate gradients before updating",
        gt=0,
    )
    optim: str = Field(
        "paged_adamw_8bit",
        description="Optimizer name (memory-efficient: 'paged_adamw_8bit')",
    )
    lr_scheduler_type: str = Field(
        "cosine",
        description="Learning rate scheduler type",
        pattern="^(linear|cosine|cosine_with_restarts|polynomial|constant|constant_with_warmup)$",
    )
    eval_steps: int = Field(
        50,
        description="Number of steps between evaluations",
        gt=0,
    )
    save_steps: int = Field(
        50,
        description="Number of steps between checkpoints",
        gt=0,
    )
    save_total_limit: int = Field(
        2,
        description="Maximum number of checkpoints to keep",
        ge=0,
    )
    logging_steps: int = Field(
        10,
        description="Number of steps between logging",
        gt=0,
    )
    gradient_checkpointing: bool = Field(
        True,
        description="Use gradient checkpointing to save memory",
    )
    warmup_ratio: float = Field(
        0.03,
        description="Ratio of warmup steps to total training steps",
        ge=0.0,
        le=1.0,
    )
    lora_rank: int = Field(
        32,
        description="LoRA rank for DPO (lower = fewer parameters, less capacity)",
        gt=0,
    )
    torch_compile: bool = Field(
        False,
        description="Use torch.compile for DPO (PyTorch 2.0+). Enable for fast preset.",
    )

    @field_validator("train_file")
    @classmethod
    def train_file_must_exist_if_set(cls, v: Path | None) -> Path | None:
        """Validate that DPO train file exists when specified (at config load time)."""
        if v is not None:
            p = Path(v)
            if not p.exists():
                raise FileNotFoundError(
                    f"DPO train file not found: {v}. "
                    "Use messages-format JSONL (e.g. augmented chat with tool calls)."
                )
        return v


def get_default_dpo_config() -> DPOConfig:
    """Return a DPOConfig instance with default field values."""
    return DPOConfig()  # type: ignore[call-arg]


class ScriptConfig(BaseModel):
    """Root configuration model for the entire training pipeline.

    Combines all configuration sections into a single validated model.
    This is the top-level configuration that should be loaded from YAML files.

    Attributes:
        data: Data loading and processing configuration.
        model: Model and tokenizer configuration.
        quantization: Quantization settings for memory-efficient training.
        lora: LoRA adapter configuration.
        training: Training arguments and hyperparameters.
        dpo: DPO training configuration (optional).

    Example:
        Load from YAML:
        ```python
        import yaml
        from config import ScriptConfig

        with open("config.yaml") as f:
            config_dict = yaml.safe_load(f)
        config = ScriptConfig(**config_dict)
        ```
    """

    model_config = ConfigDict(validate_assignment=True, frozen=False)

    data: DataConfig = Field(
        ...,
        description="Data loading and processing configuration",
    )
    model: ModelConfig = Field(
        ...,
        description="Model and tokenizer configuration",
    )
    quantization: QuantizationConfig = Field(
        ...,
        description="Quantization settings for memory-efficient training",
    )
    lora: LoraConfigModel = Field(
        ...,
        description="LoRA adapter configuration",
    )
    training: TrainingConfig = Field(
        ...,
        description="Training arguments and hyperparameters",
    )
    dpo: DPOConfig | None = Field(
        None,
        description="DPO training configuration (optional)",
    )
