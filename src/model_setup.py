"""Model and tokenizer loading utilities with quantization and attention configuration.

This module provides functions for loading and configuring Hugging Face transformers
models and tokenizers with support for:
- 4-bit quantization via BitsAndBytes
- Flash Attention 2/3 with automatic fallback
- Environment-aware device and dtype selection
- Full-checkpoint loading: the declared architecture is preferred over
  AutoModelForCausalLM so no checkpoint tensors are silently discarded
"""

import logging
import typing

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.config import ModelConfig, QuantizationConfig
from src.utils import Environment

logger = logging.getLogger(__name__)

# Either a concrete model class (e.g. Qwen3_5ForConditionalGeneration) or an
# auto-class; both expose the `from_pretrained` constructor we call.
ModelLoader = type[PreTrainedModel] | type[AutoModelForCausalLM]

# Constants
FLASH_ATTN_MODULE_NAME = "flash_attn"
FLASH_ATTN_IMPLEMENTATIONS = ("flash_attention_2", "flash_attention_3")
FALLBACK_ATTN_IMPLEMENTATION = "sdpa"
DEFAULT_PADDING_SIDE = "right"
# FlashAttention CUDA kernels only support a per-head dimension up to this value;
# larger head dims crash at runtime, so we downgrade to SDPA before loading.
FLASH_ATTN_MAX_HEAD_DIM = 256
CAUSAL_CONV1D_MODULE_NAME = "causal_conv1d"
FLA_MODULE_NAME = "fla"
LINEAR_ATTENTION_INSTALL_HINT = (
    "use_linear_attention_kernels is enabled but required packages are missing. "
    "Install into conda ai-factory (Python 3.10 / torch 2.5.1 / cu124) with "
    "--no-deps so torch is not upgraded:\n"
    '  conda run -n ai-factory pip install --no-deps '
    '"https://github.com/woct0rdho/triton-windows/releases/download/'
    'v3.2.0-windows.post9/triton-3.2.0-cp310-cp310-win_amd64.whl"\n'
    '  conda run -n ai-factory pip install --no-deps '
    '"https://github.com/d8ahazard/AudioLab/releases/download/1.0.0/'
    'causal_conv1d-1.5.0.post8-cp310-cp310-win_amd64.whl"\n'
    '  conda run -n ai-factory pip install --no-deps "flash-linear-attention==0.4.2"\n'
    "Never install flash-linear-attention[cuda] without --no-deps — it upgrades torch. "
    "If causal-conv1d DLL load fails, set use_linear_attention_kernels: false "
    "for the slow PyTorch fallback."
)


def load_tokenizer(config: ModelConfig) -> PreTrainedTokenizer:
    """Load and configure the tokenizer for the specified model.

    Configures the tokenizer with appropriate padding settings, maximum length,
    and uses the EOS token as the padding token if no explicit pad token exists.

    Args:
        config: Model configuration containing tokenizer settings.

    Returns:
        Configured PreTrainedTokenizer instance.

    Raises:
        OSError: If the model cannot be loaded from Hugging Face Hub.
        ValueError: If tokenizer configuration is invalid.
    """
    logger.info(f"Loading tokenizer from: {config.name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.name,
            trust_remote_code=config.trust_remote_code,
            use_fast=True,
        )
    except Exception as e:
        # Retry with use_fast=False when Rust tokenizers fail on ModelWrapper
        # deserialization (e.g. Qwen3-8B).
        if "ModelWrapper" in str(e) or "did not match any variant" in str(e):
            logger.warning(
                "Fast tokenizer failed (%s). Retrying with use_fast=False.",
                type(e).__name__,
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    config.name,
                    trust_remote_code=config.trust_remote_code,
                    use_fast=False,
                )
            except Exception as e2:
                logger.error(f"Failed to load tokenizer from {config.name}: {e2}")
                raise OSError(f"Tokenizer loading failed: {e2}") from e2
        else:
            logger.error(f"Failed to load tokenizer from {config.name}: {e}")
            raise OSError(f"Tokenizer loading failed: {e}") from e

    # Configure tokenizer settings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("Using EOS token as pad token")

    tokenizer.model_max_length = config.max_length
    tokenizer.padding_side = DEFAULT_PADDING_SIDE

    logger.info(
        f"Tokenizer configured: max_length={config.max_length}, "
        f"padding_side={DEFAULT_PADDING_SIDE}"
    )

    return typing.cast(PreTrainedTokenizer, tokenizer)


def _create_bitsandbytes_config(
    quant_config: QuantizationConfig,
    env: Environment,
) -> BitsAndBytesConfig | None:
    """Create BitsAndBytes quantization configuration if conditions are met.

    Args:
        quant_config: Quantization configuration settings.
        env: Environment with hardware capabilities.

    Returns:
        BitsAndBytesConfig if quantization should be enabled, None otherwise.
    """
    if not quant_config.enabled:
        logger.debug("Quantization disabled in configuration")
        return None

    if not (env.cuda_available and env.bnb_available):
        logger.warning(
            "Quantization enabled but CUDA or bitsandbytes unavailable. "
            "Disabling quantization."
        )
        return None

    logger.info("Configuring 4-bit quantization with BitsAndBytes")
    return BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_config.quant_type,
        bnb_4bit_compute_dtype=env.compute_dtype,
        bnb_4bit_use_double_quant=quant_config.use_double_quant,
    )


def _causal_conv1d_importable() -> bool:
    """Check whether causal_conv1d can actually be imported.

    :returns:
        True if causal_conv1d_fn imports cleanly, False otherwise.
    """
    try:
        from causal_conv1d import causal_conv1d_fn  # noqa: F401

        return True
    except Exception as exc:
        logger.warning("causal_conv1d present but failed to import: %s", exc)
        return False


def _fla_importable() -> bool:
    """Check whether flash-linear-attention (fla) can actually be imported.

    :returns:
        True if fla imports cleanly, False otherwise.
    """
    try:
        import fla  # noqa: F401

        return True
    except Exception as exc:
        logger.warning("fla present but failed to import: %s", exc)
        return False


def validate_linear_attention_kernels(enabled: bool) -> None:
    """Fail fast when linear-attention kernels are requested but unavailable.

    :args:
        enabled: Whether the caller requested optimized linear-attention kernels.

    :raises:
        RuntimeError: When enabled is True but causal_conv1d or fla cannot import.
    """
    if not enabled:
        return

    causal_ok = _causal_conv1d_importable()
    fla_ok = _fla_importable()
    if causal_ok and fla_ok:
        logger.info(
            "Linear-attention kernels available (causal_conv1d + fla)"
        )
        return

    missing = []
    if not causal_ok:
        missing.append(CAUSAL_CONV1D_MODULE_NAME)
    if not fla_ok:
        missing.append(FLA_MODULE_NAME)
    raise RuntimeError(
        f"Missing linear-attention dependencies: {', '.join(missing)}. "
        f"{LINEAR_ATTENTION_INSTALL_HINT}"
    )


def _flash_attn_importable() -> bool:
    """Check whether the flash_attn module can actually be imported.

    A presence-only check (e.g. importlib.util.find_spec) is insufficient on
    Windows, where an ABI-broken or partial wheel resolves a spec but then
    crashes on import (missing DLLs, symbol mismatches). Performing a real
    import is the only reliable signal that flash attention will work.

    :returns:
        True if flash_attn imports cleanly, False otherwise.
    """
    try:
        # Intentionally a guarded inline import: we are catching import-time
        # failure, which a top-of-file import cannot express.
        import flash_attn  # noqa: F401

        return True
    except Exception as exc:  # ImportError, OSError (missing DLLs), etc.
        logger.warning("flash_attn present but failed to import: %s", exc)
        return False


def _model_head_dim(model_config: ModelConfig) -> int | None:
    """Return the attention head dimension for a model without loading weights.

    Reads the model's Hugging Face config metadata and prefers an explicit
    ``head_dim`` attribute, falling back to ``hidden_size // num_attention_heads``
    when it is absent.

    :args:
        model_config: Model configuration with the Hub name and trust settings.

    :returns:
        The per-head attention dimension, or None if it cannot be determined.
    """
    try:
        hf_config = AutoConfig.from_pretrained(
            model_config.name,
            trust_remote_code=model_config.trust_remote_code,
        )
    except Exception as exc:
        logger.warning("Could not load AutoConfig to inspect head_dim: %s", exc)
        return None

    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim:
        return int(head_dim)

    hidden = getattr(hf_config, "hidden_size", None)
    heads = getattr(hf_config, "num_attention_heads", None)
    if hidden and heads:
        return int(hidden) // int(heads)

    return None


def _resolve_attention_implementation(
    requested_impl: str | None,
    model_config: ModelConfig | None = None,
) -> str:
    """Resolve the effective attention implementation with fallback logic.

    Applies two layers of fallback for flash_attention_2 / flash_attention_3:
    first an import check (the module may be present but ABI-broken), then a
    head-dimension check (FlashAttention kernels reject head dims above
    FLASH_ATTN_MAX_HEAD_DIM at runtime). Either failing downgrades to sdpa.

    :args:
        requested_impl: Requested attention implementation from config.
        model_config: Optional model configuration used to inspect the head
            dimension. When omitted, the head-dimension guard is skipped.

    :returns:
        Effective attention implementation to use.
    """
    if requested_impl not in FLASH_ATTN_IMPLEMENTATIONS:
        return requested_impl or FALLBACK_ATTN_IMPLEMENTATION

    if not _flash_attn_importable():
        logger.warning(
            "Requested '%s' but '%s' not importable. Falling back to '%s'.",
            requested_impl,
            FLASH_ATTN_MODULE_NAME,
            FALLBACK_ATTN_IMPLEMENTATION,
        )
        return FALLBACK_ATTN_IMPLEMENTATION

    if model_config is not None:
        head_dim = _model_head_dim(model_config)
        if head_dim is not None and head_dim > FLASH_ATTN_MAX_HEAD_DIM:
            logger.warning(
                "Requested '%s' but head_dim %d exceeds FlashAttention limit %d. "
                "Falling back to '%s'.",
                requested_impl,
                head_dim,
                FLASH_ATTN_MAX_HEAD_DIM,
                FALLBACK_ATTN_IMPLEMENTATION,
            )
            return FALLBACK_ATTN_IMPLEMENTATION

    logger.info(f"Using {requested_impl} attention implementation")
    return requested_impl


def _declared_architectures(model_name: str, trust_remote_code: bool) -> list[str]:
    """Return the architecture class names a checkpoint declares in its config.

    :args:
        model_name: Hub identifier or local path of the checkpoint.
        trust_remote_code: Whether remote modeling code may be executed.

    :returns:
        Declared architecture names, or an empty list when the config cannot be
        read or declares none.
    """
    try:
        hf_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        logger.warning(
            "Could not read the config of '%s' to determine its architecture: %s",
            model_name,
            exc,
        )
        return []

    return list(getattr(hf_config, "architectures", None) or [])


def resolve_model_class(
    model_name: str,
    trust_remote_code: bool = False,
    preserve_all_tensors: bool = True,
) -> ModelLoader:
    """Resolve the loader class that can hold every tensor in a checkpoint.

    ``AutoModelForCausalLM`` maps a multimodal checkpoint onto its text-only
    submodel: for Qwen3.5 it yields ``Qwen3_5ForCausalLM``, which declares
    ``_keys_to_ignore_on_load_unexpected = ["^mtp.*", "^model.visual.*"]`` and so
    discards the vision tower with no warning at all. Loading the architecture the
    checkpoint itself declares keeps those tensors, which matters most at merge
    time — whatever the merge step never loaded cannot be written back out.

    Falls back to ``AutoModelForCausalLM`` when the declared architecture is
    unavailable (custom code shipped with the repo) or is not generative, since
    those cases cannot be loaded or trained through the declared class anyway.

    :args:
        model_name: Hub identifier or local path of the checkpoint.
        trust_remote_code: Whether remote modeling code may be executed.
        preserve_all_tensors: When False, always use AutoModelForCausalLM and
            accept that non-text tensors are dropped.

    :returns:
        The class whose ``from_pretrained`` should load this checkpoint.
    """
    if not preserve_all_tensors:
        logger.warning(
            "preserve_all_tensors is disabled: loading '%s' via "
            "AutoModelForCausalLM. Checkpoint tensors outside the causal-LM "
            "submodel (e.g. vision towers) will be dropped and cannot be "
            "recovered by the merge step.",
            model_name,
        )
        return AutoModelForCausalLM

    architectures = _declared_architectures(model_name, trust_remote_code)

    for architecture in architectures:
        candidate = getattr(transformers, architecture, None)
        if not (isinstance(candidate, type) and issubclass(candidate, PreTrainedModel)):
            continue
        # can_generate() is a classmethod, so this needs no weights and avoids
        # importing transformers.generation (and its sklearn/pandas chain) here.
        if not candidate.can_generate():
            logger.warning(
                "Declared architecture '%s' of '%s' is not generative; it cannot "
                "be fine-tuned as a causal LM. Falling back to "
                "AutoModelForCausalLM.",
                architecture,
                model_name,
            )
            continue
        logger.info(
            "Loading '%s' as its declared architecture '%s' to preserve all "
            "checkpoint tensors.",
            model_name,
            architecture,
        )
        return candidate

    if architectures:
        logger.warning(
            "Declared architecture(s) %s of '%s' are not available in this "
            "transformers installation. Falling back to AutoModelForCausalLM; "
            "tensors outside the causal-LM submodel may be dropped.",
            architectures,
            model_name,
        )

    return AutoModelForCausalLM


def load_model(
    model_config: ModelConfig,
    quant_config: QuantizationConfig,
    env: Environment,
) -> PreTrainedModel:
    """Load the model with quantization, attention, and device configuration.

    Loads a Hugging Face causal language model with:
    - The checkpoint's declared architecture, so all of its tensors are kept
      (see resolve_model_class)
    - Optional 4-bit quantization via BitsAndBytes (if enabled and supported)
    - Flash Attention 2/3 with automatic fallback to SDPA
    - Environment-aware device mapping and compute dtype
    - Efficient memory usage for large models

    Args:
        model_config: Model configuration with name and settings.
        quant_config: Quantization configuration.
        env: Environment with detected hardware capabilities.

    Returns:
        Loaded PreTrainedModel instance.

    Raises:
        OSError: If the model cannot be loaded from Hugging Face Hub.
        RuntimeError: If model loading fails due to configuration issues.
    """
    logger.info(f"Loading model: {model_config.name}")

    validate_linear_attention_kernels(
        getattr(model_config, "use_linear_attention_kernels", False)
    )

    model_class = resolve_model_class(
        model_config.name,
        trust_remote_code=model_config.trust_remote_code,
        preserve_all_tensors=getattr(model_config, "preserve_all_tensors", True),
    )

    # Configure quantization
    bnb_config = _create_bitsandbytes_config(quant_config, env)

    # Resolve attention implementation with fallback
    effective_attn_impl = _resolve_attention_implementation(
        model_config.attn_implementation,
        model_config,
    )

    # With quantized models, device_map="auto" can leave some tensors on the "meta"
    # device, causing "expected device meta but got cuda:0" during backward.
    # Use an explicit single-GPU map for quantization to avoid this.
    device_map = "cuda:0" if bnb_config and env.cuda_available else "auto"

    # Build model loading arguments
    model_kwargs: dict[str, typing.Any] = {
        "device_map": device_map,
        "trust_remote_code": model_config.trust_remote_code,
        "use_safetensors": True,
        "attn_implementation": effective_attn_impl,
        "dtype": env.compute_dtype,  # torch_dtype deprecated
        "low_cpu_mem_usage": True,  # Weight streaming with device_map=auto
    }

    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    logger.debug(
        f"Model loading kwargs: loader={model_class.__name__}, "
        f"device_map={device_map!r}, "
        f"dtype={env.compute_dtype}, "
        f"attn={effective_attn_impl}, "
        f"quantization={'enabled' if bnb_config else 'disabled'}"
    )

    try:
        model = model_class.from_pretrained(model_config.name, **model_kwargs)
        logger.info(f"Successfully loaded model: {model_config.name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_config.name}: {e}")
        raise RuntimeError(f"Model loading failed: {e}") from e
