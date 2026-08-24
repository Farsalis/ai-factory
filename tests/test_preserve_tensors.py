"""Tests that a base model's tensors survive loading, training, and merging.

``AutoModelForCausalLM`` maps a multimodal checkpoint onto its text-only
submodel and discards the rest without warning, so anything the merge step never
loaded is missing from the exported model. These tests pin the resolution rules
in ``model_setup.resolve_model_class`` and run a miniature Qwen3.5 checkpoint —
built in memory, with a real vision tower — through the merge path end to end.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")
pytest.importorskip("pydantic")

from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from safetensors.torch import load_file

import src.model_setup as model_setup
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

MULTIMODAL_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"

requires_qwen35 = pytest.mark.skipif(
    not hasattr(transformers, MULTIMODAL_ARCHITECTURE),
    reason=f"transformers {transformers.__version__} has no {MULTIMODAL_ARCHITECTURE}",
)


def _patch_declared_architectures(monkeypatch, architectures: list[str] | None) -> None:
    """Make AutoConfig report the given declared architectures."""
    hf_config = SimpleNamespace(architectures=architectures)
    monkeypatch.setattr(
        model_setup.AutoConfig, "from_pretrained", lambda *a, **k: hf_config
    )


# ============================================================================
# Tests for resolve_model_class
# ============================================================================


@pytest.mark.unit
@requires_qwen35
def test_resolve_prefers_declared_multimodal_architecture(monkeypatch) -> None:
    """A multimodal checkpoint is loaded through the class that holds every tensor."""
    _patch_declared_architectures(monkeypatch, [MULTIMODAL_ARCHITECTURE])

    resolved = model_setup.resolve_model_class("multimodal-checkpoint")

    assert resolved is transformers.Qwen3_5ForConditionalGeneration


@pytest.mark.unit
def test_resolve_is_a_no_op_for_text_only_checkpoints(monkeypatch) -> None:
    """A text-only checkpoint resolves to the same class the auto class would build."""
    _patch_declared_architectures(monkeypatch, ["LlamaForCausalLM"])

    resolved = model_setup.resolve_model_class("text-only-checkpoint")

    assert resolved is transformers.LlamaForCausalLM


@pytest.mark.unit
def test_resolve_honours_opt_out(monkeypatch) -> None:
    """preserve_all_tensors=False keeps the previous text-only behavior."""
    calls: list[str] = []

    def tracking_from_pretrained(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append("autoconfig")
        return SimpleNamespace(architectures=[MULTIMODAL_ARCHITECTURE])

    monkeypatch.setattr(
        model_setup.AutoConfig, "from_pretrained", tracking_from_pretrained
    )

    resolved = model_setup.resolve_model_class(
        "multimodal-checkpoint", preserve_all_tensors=False
    )

    assert resolved is model_setup.AutoModelForCausalLM
    assert calls == []


@pytest.mark.unit
def test_resolve_falls_back_for_unavailable_architecture(monkeypatch) -> None:
    """An architecture that the transformers install lacks goes back to auto class."""
    _patch_declared_architectures(monkeypatch, ["SomeVendorForCausalLM"])

    resolved = model_setup.resolve_model_class("remote-code-checkpoint")

    assert resolved is model_setup.AutoModelForCausalLM


@pytest.mark.unit
def test_resolve_falls_back_for_non_generative_architecture(monkeypatch) -> None:
    """A non-generative declared architecture cannot be fine-tuned as a causal LM."""
    _patch_declared_architectures(monkeypatch, ["BertModel"])

    resolved = model_setup.resolve_model_class("encoder-checkpoint")

    assert resolved is model_setup.AutoModelForCausalLM


@pytest.mark.unit
def test_resolve_falls_back_when_config_is_unreadable(monkeypatch) -> None:
    """An unreadable config degrades to the auto class instead of raising."""

    def raising_from_pretrained(*args: object, **kwargs: object) -> None:
        raise OSError("offline")

    monkeypatch.setattr(
        model_setup.AutoConfig, "from_pretrained", raising_from_pretrained
    )

    resolved = model_setup.resolve_model_class("unreachable-checkpoint")

    assert resolved is model_setup.AutoModelForCausalLM


@pytest.mark.unit
def test_resolve_skips_missing_architectures_field(monkeypatch) -> None:
    """A config without an architectures field falls back to the auto class."""
    _patch_declared_architectures(monkeypatch, None)

    resolved = model_setup.resolve_model_class("bare-checkpoint")

    assert resolved is model_setup.AutoModelForCausalLM


@pytest.mark.unit
def test_load_model_loads_through_the_resolved_class(monkeypatch) -> None:
    """load_model builds the model with the resolved class, not the auto class."""
    resolved_class = MagicMock()
    resolved_class.__name__ = MULTIMODAL_ARCHITECTURE
    resolved_class.from_pretrained.return_value = "multimodal-model"
    captured: dict[str, object] = {}

    def fake_resolve(name: str, **kwargs: object) -> MagicMock:
        captured["name"] = name
        captured["kwargs"] = kwargs
        return resolved_class

    def unexpected_from_pretrained(*args: object, **kwargs: object) -> None:
        raise AssertionError("AutoModelForCausalLM must not be used by default")

    monkeypatch.setattr(model_setup, "resolve_model_class", fake_resolve)
    monkeypatch.setattr(
        model_setup.AutoModelForCausalLM, "from_pretrained", unexpected_from_pretrained
    )
    monkeypatch.setattr(model_setup, "_flash_attn_importable", lambda: False)

    model_config = SimpleNamespace(
        name="multimodal-checkpoint",
        trust_remote_code=True,
        attn_implementation="sdpa",
        use_linear_attention_kernels=False,
        preserve_all_tensors=True,
    )
    quant_config = SimpleNamespace(
        enabled=False, quant_type="nf4", use_double_quant=True
    )
    env = SimpleNamespace(
        cuda_available=False, bnb_available=False, compute_dtype=torch.float32
    )

    model = model_setup.load_model(model_config, quant_config, env)

    assert model == "multimodal-model"
    assert captured["name"] == "multimodal-checkpoint"
    assert captured["kwargs"]["preserve_all_tensors"] is True


# ============================================================================
# End-to-end: a miniature multimodal checkpoint through the merge path
# ============================================================================


@pytest.fixture
def tiny_multimodal_checkpoint(tmp_path: Path) -> Path:
    """Save a miniature Qwen3.5 checkpoint that has a real vision tower.

    Same architecture and key layout as Qwen/Qwen3.5-9B (``model.language_model.*``
    plus ``model.visual.*``), at ~180k parameters so it is cheap to round-trip.
    """
    config = transformers.Qwen3_5Config(
        text_config={
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 128,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "eos_token_id": 2,
            "dtype": "float32",
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
            "dtype": "float32",
        },
        dtype="float32",
    )
    torch.manual_seed(0)
    model = transformers.Qwen3_5ForConditionalGeneration(config)

    checkpoint_path = tmp_path / "tiny_base"
    model.save_pretrained(checkpoint_path, safe_serialization=True)
    return checkpoint_path


def _vision_tensors(directory: Path) -> dict[str, torch.Tensor]:
    """Collect every vision-tower tensor written to a saved checkpoint."""
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(directory.glob("*.safetensors")):
        tensors.update(
            {k: v for k, v in load_file(str(shard)).items() if ".visual." in k}
        )
    return tensors


@pytest.mark.integration
@requires_qwen35
def test_auto_causal_lm_drops_the_vision_tower(
    tiny_multimodal_checkpoint: Path,
) -> None:
    """Pins the transformers behaviour this feature exists to work around."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        tiny_multimodal_checkpoint, dtype=torch.float32
    )

    assert type(model).__name__ == "Qwen3_5ForCausalLM"
    assert not any(".visual." in key for key in model.state_dict())


@pytest.mark.integration
@requires_qwen35
def test_resolved_class_loads_the_whole_checkpoint(
    tiny_multimodal_checkpoint: Path,
) -> None:
    """The resolved class keeps the vision tower the auto class would discard."""
    model_class = model_setup.resolve_model_class(str(tiny_multimodal_checkpoint))
    model = model_class.from_pretrained(tiny_multimodal_checkpoint, dtype=torch.float32)

    saved_vision_tensors = _vision_tensors(tiny_multimodal_checkpoint)
    loaded_vision_keys = {k for k in model.state_dict() if ".visual." in k}

    assert model_class is transformers.Qwen3_5ForConditionalGeneration
    assert saved_vision_tensors
    assert loaded_vision_keys == set(saved_vision_tensors)


@pytest.mark.integration
@requires_qwen35
def test_merge_preserves_every_base_model_tensor(
    tiny_multimodal_checkpoint: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The merged artifact keeps every base tensor, not just the text stack."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # A LoRA adapter trained against the full architecture, as run_training produces.
    base_model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        tiny_multimodal_checkpoint, dtype=torch.float32
    )
    peft_model = get_peft_model(
        base_model,
        PeftLoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        ),
    )
    adapted_modules = {
        name.split(".lora_A")[0]
        for name, _ in peft_model.named_parameters()
        if ".lora_A" in name
    }
    # Preserving the vision tower must not widen what LoRA trains.
    assert adapted_modules
    assert not any("visual" in name for name in adapted_modules)
    peft_model.save_pretrained(str(output_dir / "final_adapter"))

    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "validation.jsonl"
    train_file.write_text('{"text": "test"}\n', encoding="utf-8")
    val_file.write_text('{"text": "test"}\n', encoding="utf-8")

    config = ScriptConfig(
        data=DataConfig(train_file=train_file, validation_file=val_file),
        model=ModelConfig(
            name=str(tiny_multimodal_checkpoint),
            max_length=64,
            attn_implementation="eager",
            trust_remote_code=False,
        ),
        quantization=QuantizationConfig(enabled=False),
        lora=LoraConfigModel(r=4, alpha=8, target_modules=["q_proj", "v_proj"]),
        training=TrainingConfig(output_dir=output_dir),
    )
    assert config.model.preserve_all_tensors is True

    env = MagicMock(spec=Environment)
    env.compute_dtype = torch.float32
    # The miniature checkpoint ships no tokenizer files; merging does not need one.
    monkeypatch.setattr(train, "load_tokenizer", lambda model_config: MagicMock())

    train.merge_and_save_model(config, env)

    merged_path = output_dir / "final_merged_model"
    merged_config = json.loads((merged_path / "config.json").read_text())
    assert merged_config["architectures"] == [MULTIMODAL_ARCHITECTURE]
    assert "vision_config" in merged_config

    base_vision_tensors = _vision_tensors(tiny_multimodal_checkpoint)
    merged_vision_tensors = _vision_tensors(merged_path)
    assert set(merged_vision_tensors) == set(base_vision_tensors)
    for key, tensor in base_vision_tensors.items():
        assert torch.equal(merged_vision_tensors[key], tensor), key


@pytest.mark.integration
@requires_qwen35
def test_merge_opt_out_exports_text_only_model(
    tiny_multimodal_checkpoint: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """preserve_all_tensors=False still produces the text-only export on request."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        tiny_multimodal_checkpoint, dtype=torch.float32
    )
    peft_model = get_peft_model(
        base_model,
        PeftLoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        ),
    )
    peft_model.save_pretrained(str(output_dir / "final_adapter"))

    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "validation.jsonl"
    train_file.write_text('{"text": "test"}\n', encoding="utf-8")
    val_file.write_text('{"text": "test"}\n', encoding="utf-8")

    config = ScriptConfig(
        data=DataConfig(train_file=train_file, validation_file=val_file),
        model=ModelConfig(
            name=str(tiny_multimodal_checkpoint),
            max_length=64,
            attn_implementation="eager",
            trust_remote_code=False,
            preserve_all_tensors=False,
        ),
        quantization=QuantizationConfig(enabled=False),
        lora=LoraConfigModel(r=4, alpha=8, target_modules=["q_proj", "v_proj"]),
        training=TrainingConfig(output_dir=output_dir),
    )

    env = MagicMock(spec=Environment)
    env.compute_dtype = torch.float32
    monkeypatch.setattr(train, "load_tokenizer", lambda model_config: MagicMock())

    train.merge_and_save_model(config, env)

    merged_path = output_dir / "final_merged_model"
    merged_config = json.loads((merged_path / "config.json").read_text())
    assert merged_config["architectures"] == ["Qwen3_5ForCausalLM"]
    assert _vision_tensors(merged_path) == {}


@pytest.mark.unit
def test_merge_saves_processor_alongside_weights(tmp_path: Path) -> None:
    """A preserved vision tower is exported with the processor it needs."""
    with patch.object(train, "AutoProcessor") as mock_processor_class:
        train._save_processor("some-model", tmp_path, trust_remote_code=False)

    mock_processor_class.from_pretrained.assert_called_once()
    mock_processor_class.from_pretrained.return_value.save_pretrained.assert_called_once_with(
        str(tmp_path)
    )


@pytest.mark.unit
def test_merge_tolerates_checkpoints_without_a_processor(tmp_path: Path) -> None:
    """Text-only checkpoints have no processor; that must not fail the merge."""
    with patch.object(train, "AutoProcessor") as mock_processor_class:
        mock_processor_class.from_pretrained.side_effect = OSError("no processor")

        train._save_processor("text-only-model", tmp_path, trust_remote_code=False)
