"""Comprehensive tests for train.py module.

Tests cover helper functions, main training functions, error handling,
and edge cases.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")
pytest.importorskip("pydantic")

from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments

import src.train as train
from src.config import (
    DataConfig,
    LoraConfigModel,
    ModelConfig,
    QuantizationConfig,
    ScriptConfig,
    TrainingConfig,
)
from src.utils import Environment


@pytest.fixture
def tmp_data_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary data files for testing."""
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "validation.jsonl"
    train_file.write_text('{"text": "test"}\n', encoding="utf-8")
    val_file.write_text('{"text": "test"}\n', encoding="utf-8")
    return train_file, val_file


@pytest.fixture
def mock_env_cuda() -> Environment:
    """Create a mock environment with CUDA available."""
    env = MagicMock(spec=Environment)
    env.cuda_available = True
    env.bnb_available = True
    env.bf16_supported = True
    env.compute_dtype = torch.bfloat16
    env.device_name = "CUDA:0"
    return env


@pytest.fixture
def mock_env_cpu() -> Environment:
    """Create a mock environment without CUDA."""
    env = MagicMock(spec=Environment)
    env.cuda_available = False
    env.bnb_available = False
    env.bf16_supported = False
    env.compute_dtype = torch.float16
    env.device_name = "CPU"
    return env


@pytest.fixture
def mock_tokenizer() -> PreTrainedTokenizer:
    """Create a mock tokenizer."""
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "</s>"
    return tokenizer


@pytest.fixture
def mock_model() -> PreTrainedModel:
    """Create a mock model."""
    model = MagicMock(spec=PreTrainedModel)
    model.config = MagicMock()
    model.config.use_cache = True
    return model


@pytest.fixture
def sample_config(tmp_path: Path, tmp_data_files: tuple[Path, Path]) -> ScriptConfig:
    """Create a sample ScriptConfig for testing purposes."""
    train_file, val_file = tmp_data_files
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    return ScriptConfig(
        data=DataConfig(
            train_file=train_file,
            validation_file=val_file,
        ),
        model=ModelConfig(
            name="test-model",
            max_length=256,
            trust_remote_code=False,
        ),
        quantization=QuantizationConfig(
            enabled=False,
            quant_type="nf4",
            use_double_quant=True,
        ),
        lora=LoraConfigModel(
            r=8,
            alpha=16,
            dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ),
        training=TrainingConfig(
            output_dir=output_dir,
            seed=42,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            optim="adamw_torch",
            learning_rate=2e-4,
            evaluation_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            save_steps=10,
            gradient_checkpointing=False,
            load_best_model_at_end=False,
        ),
    )


@pytest.fixture
def mock_dataset() -> dict[str, Dataset]:
    """Create a mock dataset dictionary."""
    train_ds = MagicMock(spec=Dataset)
    train_ds.features = {"text": "string"}
    val_ds = MagicMock(spec=Dataset)
    val_ds.features = {"text": "string"}
    return {"train": train_ds, "validation": val_ds}


# ============================================================================
# Tests for _determine_effective_optimizer
# ============================================================================


@pytest.mark.unit
def test_determine_effective_optimizer_paged_with_cuda(
    mock_env_cuda: Environment,
) -> None:
    """Test that the paged optimizer is kept when CUDA and bitsandbytes available."""
    result = train._determine_effective_optimizer("paged_adamw_8bit", mock_env_cuda)
    assert result == "paged_adamw_8bit"


@pytest.mark.unit
def test_determine_effective_optimizer_paged_without_cuda(
    mock_env_cpu: Environment,
) -> None:
    """Test that the paged optimizer falls back when CUDA/bitsandbytes unavailable."""
    result = train._determine_effective_optimizer("paged_adamw_8bit", mock_env_cpu)
    assert result == "adamw_torch"


@pytest.mark.unit
def test_determine_effective_optimizer_non_paged(
    mock_env_cpu: Environment,
) -> None:
    """Test that the non-paged optimizer is kept regardless of environment."""
    result = train._determine_effective_optimizer("adamw_torch", mock_env_cpu)
    assert result == "adamw_torch"


@pytest.mark.unit
def test_determine_effective_optimizer_paged_without_bnb(
    mock_env_cuda: Environment,
) -> None:
    """Test fallback when bitsandbytes is unavailable."""
    mock_env_cuda.bnb_available = False
    result = train._determine_effective_optimizer("paged_adamw_8bit", mock_env_cuda)
    assert result == "adamw_torch"


# ============================================================================
# Tests for _configure_best_model_loading
# ============================================================================


@pytest.mark.unit
def test_configure_best_model_loading_disabled() -> None:
    """Test when load_best_model_at_end is False."""
    args_dict = {"load_best_model_at_end": False}
    allowed_fields = {"evaluation_strategy", "eval_strategy"}

    train._configure_best_model_loading(args_dict, allowed_fields)

    assert args_dict["load_best_model_at_end"] is False


@pytest.mark.unit
def test_configure_best_model_loading_no_eval_strategy() -> None:
    """Test when evaluation_strategy field is not available."""
    args_dict = {"load_best_model_at_end": True, "save_strategy": "steps"}
    allowed_fields = {"save_strategy"}  # No eval strategy fields

    train._configure_best_model_loading(args_dict, allowed_fields)

    assert args_dict["load_best_model_at_end"] is False


@pytest.mark.unit
def test_configure_best_model_loading_with_evaluation_strategy() -> None:
    """Test when evaluation_strategy field is available."""
    args_dict = {
        "load_best_model_at_end": True,
        "save_strategy": "steps",
    }
    allowed_fields = {"evaluation_strategy", "save_strategy"}

    train._configure_best_model_loading(args_dict, allowed_fields)

    assert args_dict["load_best_model_at_end"] is True
    assert args_dict["evaluation_strategy"] == "steps"


@pytest.mark.unit
def test_configure_best_model_loading_with_eval_strategy() -> None:
    """Test when eval_strategy field is available (alternative name)."""
    args_dict = {
        "load_best_model_at_end": True,
        "save_strategy": "epoch",
    }
    allowed_fields = {"eval_strategy", "save_strategy"}

    train._configure_best_model_loading(args_dict, allowed_fields)

    assert args_dict["load_best_model_at_end"] is True
    assert args_dict["eval_strategy"] == "epoch"


@pytest.mark.unit
def test_configure_best_model_loading_preserves_existing_eval_strategy() -> None:
    """Test that existing evaluation_strategy is not overwritten."""
    args_dict = {
        "load_best_model_at_end": True,
        "save_strategy": "steps",
        "evaluation_strategy": "epoch",  # Already set
    }
    allowed_fields = {"evaluation_strategy", "save_strategy"}

    train._configure_best_model_loading(args_dict, allowed_fields)

    assert args_dict["evaluation_strategy"] == "epoch"  # Not changed


# ============================================================================
# Tests for _prepare_training_arguments
# ============================================================================


@pytest.mark.unit
def test_prepare_training_arguments_basic(
    sample_config: ScriptConfig, mock_env_cuda: Environment
) -> None:
    """Test basic training arguments preparation."""
    args_dict = train._prepare_training_arguments(sample_config, mock_env_cuda)

    assert "output_dir" in args_dict
    assert isinstance(args_dict["output_dir"], str)
    assert args_dict["optim"] == "adamw_torch"
    assert "bf16" in args_dict or "fp16" in args_dict


@pytest.mark.unit
def test_prepare_training_arguments_paths_converted(
    sample_config: ScriptConfig, mock_env_cuda: Environment
) -> None:
    """Test that Path objects are converted to strings."""
    args_dict = train._prepare_training_arguments(sample_config, mock_env_cuda)

    assert isinstance(args_dict["output_dir"], str)
    if "logging_dir" in args_dict:
        assert isinstance(args_dict["logging_dir"], str)


@pytest.mark.unit
def test_prepare_training_arguments_precision_flags(
    sample_config: ScriptConfig, mock_env_cuda: Environment, mock_env_cpu: Environment
) -> None:
    """Test precision flags are set correctly based on environment."""
    # With bf16 support
    args_dict_cuda = train._prepare_training_arguments(sample_config, mock_env_cuda)
    if "bf16" in args_dict_cuda:
        assert args_dict_cuda["bf16"] is True

    # Without bf16 support
    args_dict_cpu = train._prepare_training_arguments(sample_config, mock_env_cpu)
    if "fp16" in args_dict_cpu:
        assert args_dict_cpu["fp16"] == mock_env_cpu.cuda_available


@pytest.mark.unit
def test_prepare_training_arguments_optimizer_fallback(
    sample_config: ScriptConfig, mock_env_cpu: Environment
) -> None:
    """Test optimizer fallback when paged optimizer unavailable."""
    sample_config.training.optim = "paged_adamw_8bit"
    args_dict = train._prepare_training_arguments(sample_config, mock_env_cpu)

    assert args_dict["optim"] == "adamw_torch"


# ============================================================================
# Tests for _pre_tokenize_datasets
# ============================================================================


@pytest.mark.unit
def test_pre_tokenize_datasets(
    mock_tokenizer: PreTrainedTokenizer, mock_dataset: dict[str, Dataset]
) -> None:
    """Test dataset pre-tokenization."""
    trainer_kwargs = {
        "train_dataset": mock_dataset["train"],
        "eval_dataset": mock_dataset["validation"],
        "dataset_text_field": "text",
    }

    # Mock tokenizer return value
    mock_tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    # Mock dataset.map(...).filter(...) chain used by _pre_tokenize_datasets
    after_filter_train = MagicMock(spec=Dataset)
    after_filter_eval = MagicMock(spec=Dataset)
    chain_train = MagicMock()
    chain_train.filter.return_value = after_filter_train
    chain_eval = MagicMock()
    chain_eval.filter.return_value = after_filter_eval
    mock_dataset["train"].map = MagicMock(return_value=chain_train)
    mock_dataset["validation"].map = MagicMock(return_value=chain_eval)

    result = train._pre_tokenize_datasets(trainer_kwargs, mock_tokenizer, 256)

    assert result["train_dataset"] == after_filter_train
    assert result["eval_dataset"] == after_filter_eval
    assert "dataset_text_field" not in result
    mock_dataset["train"].map.assert_called_once()
    mock_dataset["validation"].map.assert_called_once()


# ============================================================================
# Tests for _prepare_trainer_kwargs
# ============================================================================


@pytest.mark.unit
@patch("src.train.inspect.signature")
def test_prepare_trainer_kwargs_with_tokenizer_param(
    mock_signature: MagicMock,
    mock_model: PreTrainedModel,
    mock_tokenizer: PreTrainedTokenizer,
    sample_config: ScriptConfig,
    mock_dataset: dict[str, Dataset],
) -> None:
    """Test trainer kwargs preparation when tokenizer parameter is accepted."""
    # Mock signature to include tokenizer
    mock_sig = MagicMock()
    mock_sig.parameters.keys.return_value = [
        "model",
        "args",
        "train_dataset",
        "eval_dataset",
        "tokenizer",
        "dataset_text_field",
        "max_seq_length",
    ]
    mock_signature.return_value = mock_sig

    training_args = TrainingArguments(output_dir="/tmp", num_train_epochs=1)
    lora_config = PeftLoraConfig(r=8, task_type="CAUSAL_LM")

    result = train._prepare_trainer_kwargs(
        mock_model,
        training_args,
        mock_dataset,
        lora_config,
        mock_tokenizer,
        sample_config,
    )

    assert result["tokenizer"] == mock_tokenizer
    assert result["dataset_text_field"] == "text"
    assert result["max_seq_length"] == sample_config.model.max_length


@pytest.mark.unit
@patch("src.train.inspect.signature")
@patch("src.train._pre_tokenize_datasets")
def test_prepare_trainer_kwargs_without_tokenizer_param(
    mock_pre_tokenize: MagicMock,
    mock_signature: MagicMock,
    mock_model: PreTrainedModel,
    mock_tokenizer: PreTrainedTokenizer,
    sample_config: ScriptConfig,
    mock_dataset: dict[str, Dataset],
) -> None:
    """Test trainer kwargs preparation when tokenizer parameter is not accepted."""
    # Mock signature without tokenizer
    mock_sig = MagicMock()
    mock_sig.parameters.keys.return_value = [
        "model",
        "args",
        "train_dataset",
        "eval_dataset",
    ]
    mock_signature.return_value = mock_sig

    training_args = TrainingArguments(output_dir="/tmp", num_train_epochs=1)
    lora_config = PeftLoraConfig(r=8, task_type="CAUSAL_LM")

    mock_pre_tokenize.return_value = {"train_dataset": mock_dataset["train"]}

    result = train._prepare_trainer_kwargs(
        mock_model,
        training_args,
        mock_dataset,
        lora_config,
        mock_tokenizer,
        sample_config,
    )

    mock_pre_tokenize.assert_called_once()
    assert "tokenizer" not in result


# ============================================================================
# Tests for run_training
# ============================================================================


@pytest.mark.unit
@patch("src.train.SFTTrainer")
@patch("src.train._prepare_trainer_kwargs")
@patch("src.train._prepare_training_arguments")
@patch("src.train.load_and_prepare_dataset")
def test_run_training_success(
    mock_load_dataset: MagicMock,
    mock_prepare_args: MagicMock,
    mock_prepare_kwargs: MagicMock,
    mock_sft_trainer: MagicMock,
    sample_config: ScriptConfig,
    mock_env_cuda: Environment,
    mock_tokenizer: PreTrainedTokenizer,
    mock_model: PreTrainedModel,
    mock_dataset: dict[str, Dataset],
) -> None:
    """Test successful training run."""
    # Setup mocks
    mock_load_dataset.return_value = mock_dataset
    mock_prepare_args.return_value = {"output_dir": "/tmp", "num_train_epochs": 1}
    mock_prepare_kwargs.return_value = {"model": mock_model}

    mock_trainer_instance = MagicMock()
    mock_sft_trainer.return_value = mock_trainer_instance

    # Run training
    train.run_training(sample_config, mock_env_cuda, mock_tokenizer, mock_model)

    # Verify calls
    mock_load_dataset.assert_called_once_with(sample_config.data, mock_tokenizer)
    mock_sft_trainer.assert_called_once()
    mock_trainer_instance.train.assert_called_once()
    mock_trainer_instance.save_model.assert_called_once()


@pytest.mark.unit
@patch("src.train.SFTTrainer")
@patch("src.train._prepare_trainer_kwargs")
@patch("src.train._prepare_training_arguments")
@patch("src.train.load_and_prepare_dataset")
def test_run_training_failure(
    mock_load_dataset: MagicMock,
    mock_prepare_args: MagicMock,
    mock_prepare_kwargs: MagicMock,
    mock_sft_trainer: MagicMock,
    sample_config: ScriptConfig,
    mock_env_cuda: Environment,
    mock_tokenizer: PreTrainedTokenizer,
    mock_model: PreTrainedModel,
    mock_dataset: dict[str, Dataset],
) -> None:
    """Test training failure handling."""
    mock_load_dataset.return_value = mock_dataset
    mock_prepare_args.return_value = {"output_dir": "/tmp", "num_train_epochs": 1}
    mock_prepare_kwargs.return_value = {"model": mock_model}

    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train.side_effect = RuntimeError("Training failed")
    mock_sft_trainer.return_value = mock_trainer_instance

    with pytest.raises(RuntimeError, match="Training error"):
        train.run_training(sample_config, mock_env_cuda, mock_tokenizer, mock_model)


# ============================================================================
# Tests for merge_and_save_model
# ============================================================================


@pytest.mark.unit
@patch("src.train.AutoProcessor")
@patch("src.train.load_tokenizer")
@patch("src.train.resolve_model_class")
@patch("src.train.PeftModel")
def test_merge_and_save_model_success(
    mock_peft_model: MagicMock,
    mock_resolve_model_class: MagicMock,
    mock_load_tokenizer: MagicMock,
    mock_auto_processor: MagicMock,
    sample_config: ScriptConfig,
    mock_env_cuda: Environment,
    tmp_path: Path,
) -> None:
    """Test successful model merging and saving."""
    # Create adapter directory
    adapter_path = sample_config.training.output_dir / "final_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)

    # Setup mocks
    mock_auto_model = mock_resolve_model_class.return_value
    mock_base_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_base_model

    mock_peft_model_instance = MagicMock()
    mock_merged_model = MagicMock()
    mock_peft_model_instance.merge_and_unload.return_value = mock_merged_model
    mock_peft_model.from_pretrained.return_value = mock_peft_model_instance

    mock_tokenizer = MagicMock()
    mock_load_tokenizer.return_value = mock_tokenizer

    # Run merge
    train.merge_and_save_model(sample_config, mock_env_cuda)

    # Verify calls
    mock_auto_model.from_pretrained.assert_called_once()
    mock_peft_model.from_pretrained.assert_called_once()
    mock_peft_model_instance.merge_and_unload.assert_called_once()
    mock_merged_model.save_pretrained.assert_called_once()
    mock_tokenizer.save_pretrained.assert_called_once()
    # The base model is reloaded through its declared architecture, and the
    # processor travels with the merged weights so a preserved vision tower
    # stays usable.
    mock_resolve_model_class.assert_called_once()
    assert mock_resolve_model_class.call_args.kwargs["preserve_all_tensors"] is True
    mock_auto_processor.from_pretrained.return_value.save_pretrained.assert_called_once()


@pytest.mark.unit
def test_merge_and_save_model_adapter_not_found(
    sample_config: ScriptConfig, mock_env_cuda: Environment
) -> None:
    """Test that FileNotFoundError is raised when adapter doesn't exist."""
    # Ensure adapter path doesn't exist
    adapter_path = sample_config.training.output_dir / "final_adapter"
    if adapter_path.exists():
        adapter_path.rmdir()

    with pytest.raises(FileNotFoundError, match="Adapter path not found"):
        train.merge_and_save_model(sample_config, mock_env_cuda)


@pytest.mark.unit
@patch("src.train.resolve_model_class")
def test_merge_and_save_model_load_failure(
    mock_resolve_model_class: MagicMock,
    sample_config: ScriptConfig,
    mock_env_cuda: Environment,
    tmp_path: Path,
) -> None:
    """Test handling of model loading failure."""
    adapter_path = sample_config.training.output_dir / "final_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)

    mock_auto_model = mock_resolve_model_class.return_value
    mock_auto_model.from_pretrained.side_effect = RuntimeError("Load failed")

    with pytest.raises(RuntimeError, match="Model loading error"):
        train.merge_and_save_model(sample_config, mock_env_cuda)


@pytest.mark.unit
@patch("src.train.load_tokenizer")
@patch("src.train.resolve_model_class")
@patch("src.train.PeftModel")
def test_merge_and_save_model_merge_failure(
    mock_peft_model: MagicMock,
    mock_resolve_model_class: MagicMock,
    mock_load_tokenizer: MagicMock,
    sample_config: ScriptConfig,
    mock_env_cuda: Environment,
    tmp_path: Path,
) -> None:
    """Test handling of merge failure."""
    adapter_path = sample_config.training.output_dir / "final_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)

    mock_base_model = MagicMock()
    mock_resolve_model_class.return_value.from_pretrained.return_value = mock_base_model

    mock_peft_model_instance = MagicMock()
    mock_peft_model_instance.merge_and_unload.side_effect = RuntimeError("Merge failed")
    mock_peft_model.from_pretrained.return_value = mock_peft_model_instance

    with pytest.raises(RuntimeError, match="Merging error"):
        train.merge_and_save_model(sample_config, mock_env_cuda)


# ============================================================================
# Tests for run_pipeline
# ============================================================================


@pytest.mark.unit
@patch("src.train.merge_and_save_model")
@patch("src.train.run_training")
@patch("src.train.load_model")
@patch("src.train.load_tokenizer")
@patch("src.train.Environment")
@patch("torch.manual_seed")
def test_run_pipeline_success(
    mock_seed: MagicMock,
    mock_env_class: MagicMock,
    mock_load_tokenizer: MagicMock,
    mock_load_model: MagicMock,
    mock_run_training: MagicMock,
    mock_merge: MagicMock,
    sample_config: ScriptConfig,
) -> None:
    """Test successful pipeline execution."""
    # Setup mocks
    mock_env = MagicMock(spec=Environment)
    mock_env.cuda_available = True
    mock_env.setup_backends = MagicMock()
    mock_env_class.return_value = mock_env

    mock_tokenizer = MagicMock()
    mock_load_tokenizer.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    # Run pipeline
    train.run_pipeline(sample_config)

    # Verify calls
    mock_env_class.assert_called_once()
    mock_env.setup_backends.assert_called_once()
    mock_seed.assert_called_once_with(sample_config.training.seed)
    mock_load_tokenizer.assert_called_once_with(sample_config.model)
    mock_load_model.assert_called_once()
    mock_run_training.assert_called_once()
    mock_merge.assert_called_once()


@pytest.mark.unit
@patch("src.train.merge_and_save_model")
@patch("src.train.run_training")
@patch("src.train.load_model")
@patch("src.train.load_tokenizer")
@patch("src.train.Environment")
@patch("torch.manual_seed")
@patch("torch.cuda.empty_cache")
def test_run_pipeline_memory_cleanup(
    mock_empty_cache: MagicMock,
    mock_seed: MagicMock,
    mock_env_class: MagicMock,
    mock_load_tokenizer: MagicMock,
    mock_load_model: MagicMock,
    mock_run_training: MagicMock,
    mock_merge: MagicMock,
    sample_config: ScriptConfig,
) -> None:
    """Test that memory is cleaned up even if training fails."""
    mock_env = MagicMock(spec=Environment)
    mock_env.cuda_available = True
    mock_env.setup_backends = MagicMock()
    mock_env_class.return_value = mock_env

    mock_tokenizer = MagicMock()
    mock_load_tokenizer.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    mock_run_training.side_effect = RuntimeError("Training failed")

    with pytest.raises(RuntimeError):
        train.run_pipeline(sample_config)

    # Verify cleanup was attempted
    mock_empty_cache.assert_called_once()


@pytest.mark.unit
@patch("src.train.merge_and_save_model")
@patch("src.train.run_training")
@patch("src.train.load_model")
@patch("src.train.load_tokenizer")
@patch("src.train.Environment")
@patch("torch.manual_seed")
def test_run_pipeline_training_failure(
    mock_seed: MagicMock,
    mock_env_class: MagicMock,
    mock_load_tokenizer: MagicMock,
    mock_load_model: MagicMock,
    mock_run_training: MagicMock,
    mock_merge: MagicMock,
    sample_config: ScriptConfig,
) -> None:
    """Test pipeline failure during training phase."""
    mock_env = MagicMock(spec=Environment)
    mock_env.cuda_available = False
    mock_env.setup_backends = MagicMock()
    mock_env_class.return_value = mock_env

    mock_tokenizer = MagicMock()
    mock_load_tokenizer.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    mock_run_training.side_effect = RuntimeError("Training failed")

    with pytest.raises(RuntimeError, match="Training failed"):
        train.run_pipeline(sample_config)

    # Verify merge was not called
    mock_merge.assert_not_called()


@pytest.mark.unit
@patch("src.train.merge_and_save_model")
@patch("src.train.run_training")
@patch("src.train.load_model")
@patch("src.train.load_tokenizer")
@patch("src.train.Environment")
@patch("torch.manual_seed")
def test_run_pipeline_merge_failure(
    mock_seed: MagicMock,
    mock_env_class: MagicMock,
    mock_load_tokenizer: MagicMock,
    mock_load_model: MagicMock,
    mock_run_training: MagicMock,
    mock_merge: MagicMock,
    sample_config: ScriptConfig,
) -> None:
    """Test pipeline failure during merge phase."""
    mock_env = MagicMock(spec=Environment)
    mock_env.cuda_available = False
    mock_env.setup_backends = MagicMock()
    mock_env_class.return_value = mock_env

    mock_tokenizer = MagicMock()
    mock_load_tokenizer.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    mock_merge.side_effect = RuntimeError("Merge failed")

    with pytest.raises(RuntimeError, match="Merge failed"):
        train.run_pipeline(sample_config)

    # Verify training was called
    mock_run_training.assert_called_once()
