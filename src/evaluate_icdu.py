"""Score an SFT checkpoint on the sealed ICDU test split.

Two signals are collected per record:

* **Reference loss** - teacher-forced negative log-likelihood of the ideal
  response given the exact prompt the trainer used. Comparable to the trainer's
  ``eval_loss`` but measured on the sealed split rather than validation.
* **Rubric** - deterministic checks on a greedy generation: names the governing
  principle's cell, mentions a capability layer, does not end in a question,
  carries no tool scaffolding, and lands in a sane length band.

Prompts come from :func:`build_prompt_messages`, which mirrors
``src.data.format_icdu_to_chat`` minus the assistant turn and the load-time
perturbation, so the model is scored on the format it was trained on.

The rubric is a floor, not a quality judgement: a response can pass every check
and still give poor advice. Treat it as a regression detector alongside the
loss. Human or LLM-judge review of the generations is a separate step.

Usage:
    conda run -n ai-factory python -m src.evaluate_icdu \\
        --config-path src/config.yaml \\
        --model-path training_output/merged_model \\
        --output training_output/icdu_eval_v9_baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.config import ScriptConfig
from src.data import DEFAULT_SYSTEM_PROMPT, SYSTEM_PROMPTS, _format_messages_fallback
from src.data.build_icdu_dataset import (
    ARTIFACT_RESPONSE_MARKERS,
    CAPABILITY_LAYERS,
    load_jsonl,
)
from src.main import load_config_from_yaml
from src.model_setup import load_model, load_tokenizer
from src.utils import Environment

logger = logging.getLogger(__name__)

MIN_RESPONSE_WORDS = 30
MAX_RESPONSE_WORDS = 400
DEFAULT_MAX_NEW_TOKENS = 512

#: Boolean rubric checks, in report order.
RUBRIC_CHECKS: tuple[str, ...] = (
    "non_empty",
    "names_principle",
    "names_capability",
    "no_trailing_question",
    "no_tool_scaffolding",
    "length_in_range",
)

GenerateFn = Callable[[dict[str, Any]], str]
LossFn = Callable[[dict[str, Any]], float]


# ============================================================================
# Prompt construction (must stay in step with src.data.format_icdu_to_chat)
# ============================================================================


def build_prompt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Return the system and user turns the trainer built for ``record``.

    Args:
        record: ICDU record.

    Returns:
        Two-message chat: system (capability prompt + context, intent, persona)
        and user (the application prompt).
    """
    system_content = SYSTEM_PROMPTS.get(
        record["capability_layer"], DEFAULT_SYSTEM_PROMPT
    )
    return [
        {
            "role": "system",
            "content": (
                f"{system_content}\n"
                f"Context: {record['context_summary']}\n"
                f"User Intent: {record['user_intent']}\n"
                f"Persona: {record['persona_archetype']}"
            ),
        },
        {"role": "user", "content": record["application_prompt"]},
    ]


def _has_chat_template(tokenizer: Any) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def render_prompt(messages: list[dict[str, str]], tokenizer: Any) -> str:
    """Render the text the model continues from, ending at the assistant cue."""
    if _has_chat_template(tokenizer):
        return str(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    fallback = _format_messages_fallback(
        [*messages, {"role": "assistant", "content": ""}]
    )
    return fallback.rstrip("\n")


def render_full(messages: list[dict[str, str]], response: str, tokenizer: Any) -> str:
    """Render prompt plus the reference response, as the trainer saw it."""
    full = [*messages, {"role": "assistant", "content": response}]
    if _has_chat_template(tokenizer):
        return str(
            tokenizer.apply_chat_template(
                full, tokenize=False, add_generation_prompt=False
            )
        )
    return _format_messages_fallback(full)


def common_prefix_length(left: list[int], right: list[int]) -> int:
    """Return how many leading token ids two sequences share."""
    count = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        count += 1
    return count


# ============================================================================
# Model-backed measurements
# ============================================================================


def reference_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    record: dict[str, Any],
    max_length: int,
) -> float:
    """Return mean NLL of the ideal response tokens given the prompt.

    Prompt tokens are masked out of the loss. The response span is located by
    the longest shared token prefix between the prompt-only and full renders,
    which tolerates chat templates that add tokens at the generation cue.

    Args:
        model: Causal LM in eval mode.
        tokenizer: Matching tokenizer.
        record: ICDU record.
        max_length: Truncation length for the full sequence.

    Returns:
        Mean per-token NLL over the response, or ``nan`` if nothing survives
        truncation.
    """
    messages = build_prompt_messages(record)
    prompt_ids = tokenizer(
        render_prompt(messages, tokenizer), add_special_tokens=False
    )["input_ids"]
    full_ids = tokenizer(
        render_full(messages, record["ideal_response_final"], tokenizer),
        add_special_tokens=False,
    )["input_ids"][:max_length]
    n_prompt = common_prefix_length(list(prompt_ids), list(full_ids))
    if n_prompt < len(prompt_ids):
        logger.debug(
            "Prompt/full renders diverge %d tokens before the cue for %s",
            len(prompt_ids) - n_prompt,
            record["icdu_id"],
        )
    if n_prompt >= len(full_ids):
        return float("nan")
    input_ids = torch.tensor([full_ids], device=model.device)
    labels = input_ids.clone()
    labels[:, :n_prompt] = -100
    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels)
    return float(output.loss)


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    record: dict[str, Any],
    max_new_tokens: int,
) -> str:
    """Greedy-decode a response to ``record``'s prompt."""
    text = render_prompt(build_prompt_messages(record), tokenizer)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return str(tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()


# ============================================================================
# Rubric and aggregation
# ============================================================================


def _mentions(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE) is not None


def score_response(record: dict[str, Any], generated: str) -> dict[str, Any]:
    """Apply the deterministic rubric to one generation.

    Args:
        record: ICDU record the generation answers.
        generated: Model output.

    Returns:
        One boolean per :data:`RUBRIC_CHECKS` plus ``word_count``.
    """
    text = generated.strip()
    words = text.split()
    cell = record["governing_principle"].split(" > ")[-1]
    return {
        "non_empty": bool(words),
        "names_principle": _mentions(text, cell),
        "names_capability": any(_mentions(text, layer) for layer in CAPABILITY_LAYERS),
        "no_trailing_question": bool(words) and not text.endswith("?"),
        "no_tool_scaffolding": not any(m in text for m in ARTIFACT_RESPONSE_MARKERS),
        "length_in_range": MIN_RESPONSE_WORDS <= len(words) <= MAX_RESPONSE_WORDS,
        "word_count": len(words),
    }


@dataclass
class RecordResult:
    """Everything measured for one test record."""

    icdu_id: str
    capability_layer: str
    governing_principle: str
    generated: str
    scores: dict[str, Any]
    loss: float

    def passes_all(self) -> bool:
        """True when every rubric check passed."""
        return all(bool(self.scores[check]) for check in RUBRIC_CHECKS)


def _rate(values: Iterable[bool]) -> float | None:
    items = list(values)
    return round(sum(1 for v in items if v) / len(items), 4) if items else None


def _mean(values: Iterable[float]) -> float | None:
    items = [v for v in values if not math.isnan(v)]
    return round(sum(items) / len(items), 4) if items else None


def aggregate(results: list[RecordResult]) -> dict[str, Any]:
    """Summarise rubric pass rates and loss, overall and per capability layer."""
    rates = {
        check: _rate(bool(r.scores[check]) for r in results) for check in RUBRIC_CHECKS
    }
    rates["all_checks"] = _rate(r.passes_all() for r in results)
    mean_loss = _mean(r.loss for r in results)
    by_capability: dict[str, Any] = {}
    for layer in CAPABILITY_LAYERS:
        subset = [r for r in results if r.capability_layer == layer]
        if subset:
            by_capability[layer] = {
                "records": len(subset),
                "names_principle": _rate(
                    bool(r.scores["names_principle"]) for r in subset
                ),
                "all_checks": _rate(r.passes_all() for r in subset),
                "mean_reference_loss": _mean(r.loss for r in subset),
            }
    return {
        "records": len(results),
        "rubric_pass_rates": rates,
        "mean_reference_loss": mean_loss,
        "perplexity": round(math.exp(mean_loss), 4) if mean_loss is not None else None,
        "by_capability": by_capability,
    }


def run_evaluation(
    records: list[dict[str, Any]],
    generate_fn: GenerateFn | None,
    loss_fn: LossFn | None,
) -> list[RecordResult]:
    """Measure every record with the supplied generation and loss callables.

    Either callable may be ``None`` to skip that signal.
    """
    results: list[RecordResult] = []
    for index, record in enumerate(records, start=1):
        generated = generate_fn(record) if generate_fn else ""
        loss = loss_fn(record) if loss_fn else float("nan")
        results.append(
            RecordResult(
                icdu_id=record["icdu_id"],
                capability_layer=record["capability_layer"],
                governing_principle=record["governing_principle"],
                generated=generated,
                scores=score_response(record, generated) if generate_fn else {},
                loss=loss,
            )
        )
        if index % 10 == 0 or index == len(records):
            logger.info("Evaluated %d/%d", index, len(records))
    return results


def write_report(
    path: Path,
    *,
    model_path: str,
    test_file: Path,
    results: list[RecordResult],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Write the JSON report and return it."""
    scored = [r for r in results if r.scores]
    report = {
        "model_path": model_path,
        "test_file": str(test_file),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "summary": aggregate(scored) if scored else aggregate(results),
        "records": [asdict(r) for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def derive_test_file(train_file: Path) -> Path:
    """Locate the sealed test split next to the configured training file."""
    candidate = train_file.with_name(train_file.name.replace("training", "test"))
    if candidate == train_file or not candidate.is_file():
        raise FileNotFoundError(
            f"Could not derive a test file from {train_file}; pass --test-file"
        )
    return candidate


# ============================================================================
# CLI
# ============================================================================


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score an SFT checkpoint on the sealed ICDU test split."
    )
    parser.add_argument("--config-path", type=Path, default=Path("src/config.yaml"))
    parser.add_argument(
        "--model-path",
        default=None,
        help="Merged model directory or HF id (default: training.merged_model_path).",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=None,
        help="Sealed test JSONL (default: derived from data.train_file).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--limit", type=int, default=None, help="Score only the first N rows."
    )
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-loss", action="store_true")
    return parser.parse_args(argv)


def _load_for_evaluation(
    config: ScriptConfig, model_path: str
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_config = config.model.model_copy(update={"name": model_path})
    tokenizer = load_tokenizer(model_config)
    model = load_model(model_config, config.quantization, Environment())
    model.config.use_cache = True
    model.eval()  # type: ignore[no-untyped-call]
    return model, tokenizer


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = _parse_args(argv)
    config = load_config_from_yaml(args.config_path)
    model_path = args.model_path or str(config.training.merged_model_path)
    test_file = args.test_file or derive_test_file(Path(config.data.train_file))
    output = args.output or (config.training.output_dir / "icdu_eval_report.json")

    records = load_jsonl(test_file)
    if args.limit is not None:
        records = records[: args.limit]
    logger.info("Scoring %s on %d records from %s", model_path, len(records), test_file)

    model, tokenizer = _load_for_evaluation(config, model_path)

    def _generate(record: dict[str, Any]) -> str:
        return generate_response(model, tokenizer, record, args.max_new_tokens)

    def _loss(record: dict[str, Any]) -> float:
        return reference_loss(model, tokenizer, record, config.model.max_length)

    generate_fn: GenerateFn | None = None if args.skip_generation else _generate
    loss_fn: LossFn | None = None if args.skip_loss else _loss

    started = time.monotonic()
    results = run_evaluation(records, generate_fn, loss_fn)
    report = write_report(
        output,
        model_path=model_path,
        test_file=test_file,
        results=results,
        elapsed_seconds=time.monotonic() - started,
    )
    logger.info("Summary: %s", json.dumps(report["summary"], indent=2))
    logger.info("Report written to %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
