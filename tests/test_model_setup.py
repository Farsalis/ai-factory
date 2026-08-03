from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pydantic")

import src.model_setup as model_setup


@pytest.mark.unit
def test_load_tokenizer_sets_padding(monkeypatch) -> None:
    class DummyTokenizer:
        def __init__(self):
            self.eos_token = "</s>"
            self.pad_token = None
            self.model_max_length = None
            self.padding_side = None

    def fake_from_pretrained(name, trust_remote_code, use_fast):
        return DummyTokenizer()

    monkeypatch.setattr(
        model_setup.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )

    config = SimpleNamespace(name="model", trust_remote_code=True, max_length=128)
    tokenizer = model_setup.load_tokenizer(config)

    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.model_max_length == 128
    assert tokenizer.padding_side == "right"


@pytest.mark.unit
def test_load_model_disables_quantization_without_cuda(monkeypatch) -> None:
    captured = {}

    def fake_from_pretrained(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return "model"

    monkeypatch.setattr(
        model_setup.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(model_setup, "_flash_attn_importable", lambda: False)

    model_config = SimpleNamespace(
        name="model",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_linear_attention_kernels=False,
    )
    quant_config = SimpleNamespace(
        enabled=True, quant_type="nf4", use_double_quant=True
    )
    env = SimpleNamespace(
        cuda_available=False, bnb_available=False, compute_dtype=torch.float16
    )

    model = model_setup.load_model(model_config, quant_config, env)

    assert model == "model"
    assert "quantization_config" not in captured["kwargs"]
    assert captured["kwargs"]["attn_implementation"] == "sdpa"


@pytest.mark.unit
def test_load_model_applies_bitsandbytes_when_cuda_and_bnb(monkeypatch) -> None:
    """Quantization on with CUDA + bitsandbytes available passes QLoRA config."""
    captured: dict[str, object] = {}

    def fake_from_pretrained(name: str, **kwargs: object) -> MagicMock:
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr(
        model_setup.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(model_setup, "_flash_attn_importable", lambda: False)

    model_config = SimpleNamespace(
        name="model-id",
        trust_remote_code=True,
        attn_implementation="sdpa",
        use_linear_attention_kernels=False,
    )
    quant_config = SimpleNamespace(
        enabled=True, quant_type="nf4", use_double_quant=True
    )
    env = SimpleNamespace(
        cuda_available=True,
        bnb_available=True,
        compute_dtype=torch.float16,
    )

    model_setup.load_model(model_config, quant_config, env)

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert "quantization_config" in kwargs
    assert kwargs["device_map"] == "cuda:0"
    assert kwargs["attn_implementation"] == "sdpa"


@pytest.mark.unit
def test_load_model_device_map_auto_without_quantization(monkeypatch) -> None:
    """Without BitsAndBytes config, device_map stays auto."""
    captured: dict[str, object] = {}

    def fake_from_pretrained(name: str, **kwargs: object) -> str:
        captured["kwargs"] = kwargs
        return "model"

    monkeypatch.setattr(
        model_setup.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(model_setup, "_flash_attn_importable", lambda: False)

    model_config = SimpleNamespace(
        name="model",
        trust_remote_code=True,
        attn_implementation="sdpa",
        use_linear_attention_kernels=False,
    )
    quant_config = SimpleNamespace(
        enabled=False, quant_type="nf4", use_double_quant=True
    )
    env = SimpleNamespace(
        cuda_available=True,
        bnb_available=True,
        compute_dtype=torch.bfloat16,
    )

    model = model_setup.load_model(model_config, quant_config, env)

    assert model == "model"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("device_map") == "auto"
    assert "quantization_config" not in kwargs


@pytest.mark.unit
@pytest.mark.parametrize(
    "importable,expected",
    [
        (True, "flash_attention_2"),
        (False, "sdpa"),
    ],
)
def test_resolve_attention_uses_real_import(
    monkeypatch, importable: bool, expected: str
) -> None:
    """flash_attention_2 is kept only when flash_attn actually imports."""
    monkeypatch.setattr(model_setup, "_flash_attn_importable", lambda: importable)

    resolved = model_setup._resolve_attention_implementation("flash_attention_2")

    assert resolved == expected


@pytest.mark.unit
def test_flash_attn_importable_true_with_fake_module(monkeypatch) -> None:
    """A cleanly importable flash_attn module is detected as available."""
    fake_module = ModuleType(model_setup.FLASH_ATTN_MODULE_NAME)
    monkeypatch.setitem(sys.modules, model_setup.FLASH_ATTN_MODULE_NAME, fake_module)

    assert model_setup._flash_attn_importable() is True


@pytest.mark.unit
def test_flash_attn_importable_false_when_import_raises(monkeypatch) -> None:
    """An ABI-broken wheel that raises on import is treated as unavailable."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == model_setup.FLASH_ATTN_MODULE_NAME:
            raise OSError("missing DLL")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, model_setup.FLASH_ATTN_MODULE_NAME, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert model_setup._flash_attn_importable() is False


@pytest.mark.unit
def test_model_head_dim_prefers_explicit_head_dim(monkeypatch) -> None:
    """An explicit head_dim attribute is returned as-is."""
    hf_config = SimpleNamespace(head_dim=288, hidden_size=4096, num_attention_heads=8)
    monkeypatch.setattr(
        model_setup.AutoConfig, "from_pretrained", lambda *a, **k: hf_config
    )

    model_config = SimpleNamespace(name="model", trust_remote_code=True)

    assert model_setup._model_head_dim(model_config) == 288


@pytest.mark.unit
def test_model_head_dim_falls_back_to_hidden_over_heads(monkeypatch) -> None:
    """Without head_dim, derive it from hidden_size // num_attention_heads."""
    hf_config = SimpleNamespace(head_dim=None, hidden_size=4096, num_attention_heads=32)
    monkeypatch.setattr(
        model_setup.AutoConfig, "from_pretrained", lambda *a, **k: hf_config
    )

    model_config = SimpleNamespace(name="model", trust_remote_code=True)

    assert model_setup._model_head_dim(model_config) == 128


@pytest.mark.unit
def test_model_head_dim_returns_none_on_autoconfig_failure(monkeypatch) -> None:
    """A failure to load AutoConfig yields None rather than raising."""

    def raising_from_pretrained(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(
        model_setup.AutoConfig, "from_pretrained", raising_from_pretrained
    )

    model_config = SimpleNamespace(name="model", trust_remote_code=True)

    assert model_setup._model_head_dim(model_config) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "head_dim,expected",
    [
        (257, "sdpa"),
        (256, "flash_attention_2"),
        (None, "flash_attention_2"),
    ],
)
def test_resolve_attention_downgrades_on_oversized_head_dim(
    monkeypatch, head_dim: int | None, expected: str
) -> None:
    """FA2 is downgraded to sdpa only when the head dim exceeds the limit."""
    monkeypatch.setattr(model_setup, "_flash_attn_importable", lambda: True)
    monkeypatch.setattr(model_setup, "_model_head_dim", lambda config: head_dim)

    model_config = SimpleNamespace(name="model", trust_remote_code=True)
    resolved = model_setup._resolve_attention_implementation(
        "flash_attention_2", model_config
    )

    assert resolved == expected


@pytest.mark.unit
def test_load_model_downgrades_fa2_for_oversized_head_dim(monkeypatch) -> None:
    """load_model passes sdpa to from_pretrained when head dim is too large."""
    captured: dict[str, object] = {}

    def fake_from_pretrained(name: str, **kwargs: object) -> str:
        captured["kwargs"] = kwargs
        return "model"

    monkeypatch.setattr(
        model_setup.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(model_setup, "_flash_attn_importable", lambda: True)
    monkeypatch.setattr(model_setup, "_model_head_dim", lambda config: 257)

    model_config = SimpleNamespace(
        name="model",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_linear_attention_kernels=False,
    )
    quant_config = SimpleNamespace(
        enabled=False, quant_type="nf4", use_double_quant=True
    )
    env = SimpleNamespace(
        cuda_available=True,
        bnb_available=True,
        compute_dtype=torch.bfloat16,
    )

    model = model_setup.load_model(model_config, quant_config, env)

    assert model == "model"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["attn_implementation"] == "sdpa"


@pytest.mark.unit
def test_validate_linear_attention_kernels_disabled_is_noop() -> None:
    """Disabled validation does not require optional linear-attention packages."""
    model_setup.validate_linear_attention_kernels(False)


@pytest.mark.unit
def test_validate_linear_attention_kernels_raises_when_missing(monkeypatch) -> None:
    """Enabled validation fails fast when causal_conv1d or fla cannot import."""
    monkeypatch.setattr(model_setup, "_causal_conv1d_importable", lambda: False)
    monkeypatch.setattr(model_setup, "_fla_importable", lambda: False)

    with pytest.raises(RuntimeError, match="Missing linear-attention dependencies"):
        model_setup.validate_linear_attention_kernels(True)


@pytest.mark.unit
def test_validate_linear_attention_kernels_passes_when_mocked(monkeypatch) -> None:
    """Enabled validation succeeds when both optional packages import cleanly."""
    monkeypatch.setattr(model_setup, "_causal_conv1d_importable", lambda: True)
    monkeypatch.setattr(model_setup, "_fla_importable", lambda: True)

    model_setup.validate_linear_attention_kernels(True)


@pytest.mark.unit
def test_causal_conv1d_importable_true_with_fake_module(monkeypatch) -> None:
    """A cleanly importable causal_conv1d module is detected as available."""
    fake_module = ModuleType(model_setup.CAUSAL_CONV1D_MODULE_NAME)
    fake_module.causal_conv1d_fn = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules, model_setup.CAUSAL_CONV1D_MODULE_NAME, fake_module
    )

    assert model_setup._causal_conv1d_importable() is True


@pytest.mark.unit
def test_causal_conv1d_importable_false_when_import_raises(monkeypatch) -> None:
    """An ABI-broken wheel that raises on import is treated as unavailable."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == model_setup.CAUSAL_CONV1D_MODULE_NAME:
            raise OSError("missing DLL")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(
        sys.modules, model_setup.CAUSAL_CONV1D_MODULE_NAME, raising=False
    )
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert model_setup._causal_conv1d_importable() is False


@pytest.mark.unit
def test_fla_importable_true_with_fake_module(monkeypatch) -> None:
    """A cleanly importable fla module is detected as available."""
    fake_module = ModuleType(model_setup.FLA_MODULE_NAME)
    monkeypatch.setitem(sys.modules, model_setup.FLA_MODULE_NAME, fake_module)

    assert model_setup._fla_importable() is True


@pytest.mark.unit
def test_fla_importable_false_when_import_raises(monkeypatch) -> None:
    """An ABI-broken fla install that raises on import is treated as unavailable."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == model_setup.FLA_MODULE_NAME:
            raise OSError("missing DLL")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, model_setup.FLA_MODULE_NAME, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert model_setup._fla_importable() is False


@pytest.mark.unit
def test_load_model_raises_before_from_pretrained_when_linear_kernels_missing(
    monkeypatch,
) -> None:
    """load_model validates linear-attention deps before touching from_pretrained."""
    called = {"from_pretrained": False}

    def fake_from_pretrained(name: str, **kwargs: object) -> str:
        called["from_pretrained"] = True
        return "model"

    monkeypatch.setattr(
        model_setup.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(model_setup, "_causal_conv1d_importable", lambda: False)
    monkeypatch.setattr(model_setup, "_fla_importable", lambda: False)

    model_config = SimpleNamespace(
        name="model",
        trust_remote_code=True,
        attn_implementation="sdpa",
        use_linear_attention_kernels=True,
    )
    quant_config = SimpleNamespace(
        enabled=False, quant_type="nf4", use_double_quant=True
    )
    env = SimpleNamespace(
        cuda_available=True,
        bnb_available=True,
        compute_dtype=torch.bfloat16,
    )

    with pytest.raises(RuntimeError, match="Missing linear-attention dependencies"):
        model_setup.load_model(model_config, quant_config, env)

    assert called["from_pretrained"] is False
