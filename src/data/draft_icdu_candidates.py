"""Draft ICDU candidate records with Claude, aimed at under-covered cells.

The drafter closes the loop the dataset builder opened: it reads the current
version's validation report, works out which persona / principle / capability
labels sit below their floor, and asks Claude for new scenarios in exactly
those cells - using existing rows from the same cell as style anchors. Every
draft is screened with the builder's own rules before it is written, so what
reaches the staging file is already free of the v8 failure modes. Rejects are
kept in a sidecar file with the reason, which is the feedback for prompt
tuning.

Two modes:

* ``new-families`` - brand-new scenarios, allocated to thin cells first.
* ``variants`` - scenario rewrites of existing *training* families. The
  variant keeps its parent's labels and, at build time, its parent's split.

Labels are never chosen by the model. Each request targets one cell and the
returned prompt/response pairs inherit it, so the closed vocabularies hold by
construction.

Requires the ``anthropic`` package (``pip install -e ".[draft]"``) and API
credentials in the environment. ``--dry-run`` prints the plan without calling
the API.

Usage:
    conda run -n ai-factory python -m src.data.draft_icdu_candidates \\
        --mode new-families --count 120 \\
        --output src/data/datasets/staging/icdu_v10_candidates.jsonl

    conda run -n ai-factory python -m src.data.draft_icdu_candidates \\
        --mode variants --count 60 --per-parent 2 \\
        --output src/data/datasets/staging/icdu_v10_variants.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import random
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.data.build_icdu_dataset import (
    ARTIFACT_PROMPT_SUFFIXES,
    ARTIFACT_RESPONSE_MARKERS,
    CAPABILITY_LAYERS,
    GOVERNING_PRINCIPLES,
    INTENT_CONTEXT_TOPICS,
    PERSONA_INTENTS,
    RESPONSE_ATTRIBUTES,
    BuildError,
    canonical_family_hash,
    detect_prompt_artifacts,
    load_jsonl,
    parse_staged_candidate,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_BATCH_SIZE = 5
DEFAULT_EXAMPLES_PER_REQUEST = 4
DEFAULT_PER_PARENT = 2
DEFAULT_SEED = 20260918

DEFAULT_REPORT = Path("src/data/datasets/icdu_v9_validation_report.json")
DEFAULT_CORPUS: tuple[Path, ...] = (
    Path("src/data/datasets/icdu_training_data_v9.jsonl"),
    Path("src/data/datasets/icdu_validation_data_v9.jsonl"),
)
DEFAULT_PARENTS = Path("src/data/datasets/icdu_training_data_v9.jsonl")

#: Minimum training-row count per label before a dimension stops being "thin".
DEFAULT_FLOORS: dict[str, int] = {"personas": 30, "principles": 40, "capabilities": 150}

MIN_PROMPT_WORDS = 8
MAX_PROMPT_WORDS = 90
MIN_RESPONSE_WORDS = 40
MAX_RESPONSE_WORDS = 320
MAX_SIMILARITY = 0.9

MODE_NEW_FAMILIES = "new-families"
MODE_VARIANTS = "variants"

DIMENSION_VOCAB: dict[str, tuple[str, ...]] = {
    "personas": tuple(PERSONA_INTENTS),
    "principles": GOVERNING_PRINCIPLES,
    "capabilities": CAPABILITY_LAYERS,
}

FRAMEWORK_BRIEF = """\
Breaking Better, by Conrad Frerichs, organises personal and professional growth
into a 3x3 framework. The assistant's tone is practical, encouraging and clear,
and its advice is always actionable.

Governing principles (the cell a response leans on):
- Chapter 1 > People, Chapter 1 > Clarity, Chapter 1 > Transparency
- Chapter 2 > People, Chapter 2 > Process, Chapter 2 > Tools
- Chapter 3 > Foundational, Chapter 3 > Transformational, Chapter 3 > Aspirational

Capability layers (how ambitious the user's goal is right now):
- Foundational: build a stable, repeatable base - small habits, routines and
  first steps.
- Transformational: a real identity or role shift that needs a bridge from the
  current base.
- Aspirational: a long-horizon ambition; affirm it, then connect it back to the
  layers beneath.

The example records are the authority on how the framework is voiced. Match them."""

DRAFTING_RULES = """\
Hard rules - a candidate that breaks any of these is discarded automatically:
1. Each scenario must be a genuinely new situation: a different person, context and
   difficulty. Never take an existing prompt and append a sentence to it.
2. The user prompt is first person, specific and natural, 8-35 words. It may end
   with a question. Do not use stock closers such as "What if budget wasn't an
   issue?", "How to track progress?" or "Adjust for my region?".
3. The response is 60-200 words, practical and encouraging. It applies the assigned
   principle and capability layer explicitly but naturally, and ends with a concrete
   next action - never with a question.
4. No tool calls, JSON, "Need data:", "Tool result:" or system-style text in
   the response.
5. Vary the opening of each response; do not begin them all the same way.
6. ideal_response_attributes lists only values from this set:
   Clear, Actionable, Empathetic, Structured, Principle-driven, Encouraging."""


class BuildPlanError(RuntimeError):
    """Raised when a drafting plan cannot be constructed."""


# ============================================================================
# Structured output schema
# ============================================================================


class DraftCandidate(BaseModel):
    """One drafted prompt/response pair as returned by the model."""

    application_prompt: str
    ideal_response_final: str
    ideal_response_attributes: list[str]


class DraftBatch(BaseModel):
    """A batch of drafted candidates."""

    candidates: list[DraftCandidate]


# ============================================================================
# Plan
# ============================================================================


@dataclass(frozen=True)
class Cell:
    """A persona / principle / capability target."""

    persona: str
    principle: str
    capability: str


@dataclass
class DraftRequest:
    """One model call: a cell, style examples, and an optional parent."""

    cell: Cell
    count: int
    examples: list[dict[str, Any]]
    parent: dict[str, Any] | None = None

    @property
    def kind(self) -> str:
        """``variant`` when a parent is set, else ``new_family``."""
        return "variant" if self.parent else "new_family"


def load_split_counts(
    report: dict[str, Any], split: str = "train"
) -> dict[str, dict[str, int]]:
    """Return label counts for ``split`` over the *full* vocabulary.

    Labels absent from the report count as zero - those are the thinnest cells
    and must not be skipped just because the report omitted them.
    """
    try:
        split_block = report["splits"][split]
    except KeyError as err:
        raise BuildPlanError(f"Report has no split {split!r}") from err
    counts: dict[str, dict[str, int]] = {}
    for dimension, vocab in DIMENSION_VOCAB.items():
        present = split_block.get(dimension, {})
        counts[dimension] = {label: int(present.get(label, 0)) for label in vocab}
    return counts


def compute_shortfalls(
    counts: dict[str, dict[str, int]], floors: dict[str, int]
) -> dict[str, dict[str, int]]:
    """Return rows still needed per label to reach each dimension's floor."""
    return {
        dimension: {
            label: max(0, floors.get(dimension, 0) - count)
            for label, count in labels.items()
        }
        for dimension, labels in counts.items()
    }


def _pick_label(
    shortfall: dict[str, int], vocab: Sequence[str], rng: random.Random
) -> str:
    """Largest shortfall wins, ties broken by vocabulary order; else random."""
    best = max(shortfall.values(), default=0)
    if best > 0:
        return next(label for label in vocab if shortfall.get(label, 0) == best)
    return rng.choice(list(vocab))


def plan_new_families(
    shortfalls: dict[str, dict[str, int]],
    count: int,
    batch_size: int,
    seed: int = DEFAULT_SEED,
) -> list[tuple[Cell, int]]:
    """Allocate ``count`` new families to cells, thinnest labels first.

    Each batch takes the label with the largest remaining shortfall on every
    dimension at once, then decrements all three, so one batch of drafts
    reduces persona, principle and capability gaps together. Once every floor
    is met, cells are drawn at random (seeded) to spread coverage.
    """
    if count <= 0:
        return []
    if batch_size <= 0:
        raise BuildPlanError("batch_size must be positive")
    rng = random.Random(seed)  # noqa: S311 - reproducible planning, not security
    remaining = {dim: dict(labels) for dim, labels in shortfalls.items()}
    plan: list[tuple[Cell, int]] = []
    produced = 0
    while produced < count:
        n = min(batch_size, count - produced)
        cell = Cell(
            persona=_pick_label(
                remaining["personas"], DIMENSION_VOCAB["personas"], rng
            ),
            principle=_pick_label(
                remaining["principles"], DIMENSION_VOCAB["principles"], rng
            ),
            capability=_pick_label(
                remaining["capabilities"], DIMENSION_VOCAB["capabilities"], rng
            ),
        )
        for dimension, label in (
            ("personas", cell.persona),
            ("principles", cell.principle),
            ("capabilities", cell.capability),
        ):
            remaining[dimension][label] = max(0, remaining[dimension][label] - n)
        plan.append((cell, n))
        produced += n
    return plan


def plan_variants(
    parents: list[dict[str, Any]],
    count: int,
    per_parent: int,
    seed: int = DEFAULT_SEED,
) -> list[tuple[dict[str, Any], int]]:
    """Choose parents (seeded, without replacement) and how many variants each gets."""
    if count <= 0:
        return []
    if per_parent <= 0:
        raise BuildPlanError("per_parent must be positive")
    if not parents:
        raise BuildPlanError("No parent rows available for variants")
    rng = random.Random(seed)  # noqa: S311 - reproducible planning, not security
    order = list(parents)
    rng.shuffle(order)
    plan: list[tuple[dict[str, Any], int]] = []
    produced = 0
    for parent in order:
        if produced >= count:
            break
        n = min(per_parent, count - produced)
        plan.append((parent, n))
        produced += n
    if produced < count:
        logger.warning(
            "Only %d parents available; planned %d of %d variants",
            len(parents),
            produced,
            count,
        )
    return plan


def cell_of(record: dict[str, Any]) -> Cell:
    """Return the cell an existing record occupies."""
    return Cell(
        persona=record["persona_archetype"],
        principle=record["governing_principle"],
        capability=record["capability_layer"],
    )


# ============================================================================
# Corpus and example selection
# ============================================================================


@dataclass
class CorpusIndex:
    """Existing rows plus the lookups the screen needs."""

    rows: list[dict[str, Any]]
    prompts: set[str] = field(default_factory=set)
    families: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        for row in self.rows:
            prompt = row["application_prompt"]
            self.prompts.add(prompt.strip())
            self.families.add(canonical_family_hash(prompt))

    def __repr__(self) -> str:
        return f"CorpusIndex(rows={len(self.rows)}, families={len(self.families)})"

    @classmethod
    def from_files(cls, paths: Sequence[Path]) -> CorpusIndex:
        """Load and index every row from ``paths``."""
        rows = [row for path in paths for row in load_jsonl(path)]
        return cls(rows=rows)


def select_examples(
    rows: list[dict[str, Any]],
    cell: Cell,
    k: int,
    rng: random.Random,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Pick up to ``k`` style anchors, nearest cell first.

    Tiers: exact cell, same persona, same principle, same capability, anything.
    Within a tier the order is shuffled with ``rng`` so repeated requests for
    the same cell see different anchors.
    """
    excluded = exclude_ids or set()
    tiers: list[Callable[[dict[str, Any]], bool]] = [
        lambda r: cell_of(r) == cell,
        lambda r: r["persona_archetype"] == cell.persona,
        lambda r: r["governing_principle"] == cell.principle,
        lambda r: r["capability_layer"] == cell.capability,
        lambda r: True,
    ]
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for matches in tiers:
        pool = [
            r
            for r in rows
            if matches(r) and r["icdu_id"] not in seen and r["icdu_id"] not in excluded
        ]
        rng.shuffle(pool)
        for row in pool:
            if len(chosen) >= k:
                return chosen
            chosen.append(row)
            seen.add(row["icdu_id"])
    return chosen


# ============================================================================
# Prompts
# ============================================================================


def build_system_prompt(reference_text: str | None = None) -> str:
    """Stable system prompt: framework brief, optional reference text, rules."""
    parts = [
        "You write training examples for an assistant that embodies the book "
        "'Breaking Better'. Each example is a realistic user message and the "
        "ideal assistant reply.",
        FRAMEWORK_BRIEF,
    ]
    if reference_text:
        parts.append("Reference material from the book:\n\n" + reference_text.strip())
    parts.append(DRAFTING_RULES)
    return "\n\n".join(parts)


def _render_example(row: dict[str, Any]) -> str:
    return (
        f"Persona: {row['persona_archetype']} | "
        f"Principle: {row['governing_principle']} | "
        f"Capability: {row['capability_layer']}\n"
        f"User: {row['application_prompt']}\n"
        f"Assistant: {row['ideal_response_final']}"
    )


def build_user_prompt(request: DraftRequest) -> str:
    """Per-request instructions: the target cell, anchors, and the ask."""
    cell = request.cell
    intent = PERSONA_INTENTS[cell.persona]
    topic = INTENT_CONTEXT_TOPICS[intent]
    lines = [
        "Target cell for every candidate in this batch:",
        f"- persona_archetype: {cell.persona}",
        f"- user_intent: {intent} (the user is facing {topic})",
        f"- governing_principle: {cell.principle}",
        f"- capability_layer: {cell.capability}",
        "",
    ]
    if request.examples:
        lines.append("Existing records that show the voice and framework usage:")
        lines.append("")
        for index, row in enumerate(request.examples, start=1):
            lines.append(f"--- Example {index} ---")
            lines.append(_render_example(row))
            lines.append("")
    if request.parent:
        lines += [
            "This batch produces VARIANTS of the parent record below. Each variant "
            "changes the scenario - a different constraint, stake, relationship or "
            "setting - while keeping the same persona, principle and capability. "
            "Write a new prompt from scratch (do not reuse or extend the parent's "
            "wording) and a response written fresh for the new scenario.",
            "",
            "--- Parent ---",
            _render_example(request.parent),
            "",
        ]
    noun = "variants" if request.parent else "new candidates"
    lines.append(
        f"Produce exactly {request.count} {noun} as JSON. Each must be a distinct "
        "scenario from the examples, the parent, and each other."
    )
    return "\n".join(lines)


# ============================================================================
# Screening
# ============================================================================


def _similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(
        None, " ".join(left.lower().split()), " ".join(right.lower().split())
    ).ratio()


@dataclass
class ScreenResult:
    """Candidates that passed, and rejects annotated with a reason."""

    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ScreenResult(accepted={len(self.accepted)}, "
            f"rejected={len(self.rejected)})"
        )


def _reject(reason: str, raw: dict[str, Any], result: ScreenResult) -> None:
    result.rejected.append({**raw, "reason": reason})


def to_staged(raw: dict[str, Any], request: DraftRequest) -> dict[str, Any]:
    """Assemble the staging-contract record from a draft and its target cell."""
    staged = {
        "application_prompt": str(raw.get("application_prompt", "")).strip(),
        "ideal_response_final": str(raw.get("ideal_response_final", "")).strip(),
        "persona_archetype": request.cell.persona,
        "governing_principle": request.cell.principle,
        "capability_layer": request.cell.capability,
        "ideal_response_attributes": list(raw.get("ideal_response_attributes") or []),
    }
    if request.parent:
        staged["parent_application_prompt"] = request.parent["application_prompt"]
    return staged


def screen_candidates(
    raw_candidates: list[dict[str, Any]],
    request: DraftRequest,
    corpus: CorpusIndex,
    seen_families: set[str],
) -> ScreenResult:
    """Apply the builder's rules plus drafting sanity checks to a batch.

    Args:
        raw_candidates: Model output, one dict per candidate.
        request: The request that produced them (supplies labels and parent).
        corpus: Existing rows, for duplicate and artifact checks.
        seen_families: Families accepted earlier in this run; updated in place.

    Returns:
        Accepted staging records and annotated rejects.
    """
    result = ScreenResult()
    batch_families: dict[str, dict[str, Any]] = {}
    for raw in raw_candidates:
        staged = to_staged(raw, request)
        prompt, response = staged["application_prompt"], staged["ideal_response_final"]
        if not prompt or not response:
            _reject("missing_field", staged, result)
            continue
        unknown = sorted(
            set(staged["ideal_response_attributes"]) - set(RESPONSE_ATTRIBUTES)
        )
        if unknown:
            _reject(f"off_vocabulary_attributes:{unknown}", staged, result)
            continue
        if not MIN_PROMPT_WORDS <= len(prompt.split()) <= MAX_PROMPT_WORDS:
            _reject("prompt_length", staged, result)
            continue
        if not MIN_RESPONSE_WORDS <= len(response.split()) <= MAX_RESPONSE_WORDS:
            _reject("response_length", staged, result)
            continue
        if response.rstrip().endswith("?"):
            _reject("trailing_question", staged, result)
            continue
        if any(marker in response for marker in ARTIFACT_RESPONSE_MARKERS):
            _reject("tool_scaffolding", staged, result)
            continue
        if any(prompt.endswith(suffix) for suffix in ARTIFACT_PROMPT_SUFFIXES):
            _reject("mechanical_suffix", staged, result)
            continue
        findings = detect_prompt_artifacts(prompt, corpus.prompts)
        if findings:
            _reject(f"artifact:{findings[0]}", staged, result)
            continue
        family = canonical_family_hash(prompt)
        if family in corpus.families or family in seen_families:
            _reject("duplicate_family", staged, result)
            continue
        if family in batch_families:
            _reject("duplicate_in_batch", staged, result)
            continue
        if any(
            _similarity(response, other["ideal_response_final"]) > MAX_SIMILARITY
            for other in batch_families.values()
        ):
            _reject("near_duplicate_in_batch", staged, result)
            continue
        if request.parent and (
            _similarity(prompt, request.parent["application_prompt"]) > MAX_SIMILARITY
            or _similarity(response, request.parent["ideal_response_final"])
            > MAX_SIMILARITY
        ):
            _reject("too_similar_to_parent", staged, result)
            continue
        try:
            parse_staged_candidate(staged, "draft", len(result.accepted) + 1)
        except BuildError as err:
            _reject(f"invalid:{err}", staged, result)
            continue
        batch_families[family] = staged
        seen_families.add(family)
        result.accepted.append(staged)
    return result


# ============================================================================
# Drafting loop
# ============================================================================

#: ``(system_prompt, user_prompt, count) -> raw candidates``
DraftFn = Callable[[str, str, int], list[dict[str, Any]]]


class AnthropicDrafter:
    """Calls Claude with structured output and returns raw candidate dicts."""

    def __init__(
        self, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> None:
        try:
            import anthropic  # type: ignore[import-not-found, unused-ignore]
        except ImportError as err:  # pragma: no cover - exercised via monkeypatch
            raise BuildPlanError(
                "The anthropic package is required for drafting: "
                'pip install -e ".[draft]"'
            ) from err
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(max_retries=3)
        self.model = model
        self.max_tokens = max_tokens

    def __repr__(self) -> str:
        return f"AnthropicDrafter(model={self.model!r}, max_tokens={self.max_tokens})"

    def __call__(
        self, system_prompt: str, user_prompt: str, count: int
    ) -> list[dict[str, Any]]:
        """Request ``count`` candidates; returns ``[]`` on refusal or API error."""
        api = self._anthropic
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
                output_format=DraftBatch,
            )
        except api.RateLimitError as err:
            logger.error("Rate limited after retries: %s", err)
            return []
        except api.APIStatusError as err:
            logger.error("API error %s: %s", err.status_code, err.message)
            return []
        except api.APIConnectionError as err:
            logger.error("Connection error: %s", err)
            return []
        if response.stop_reason == "refusal":
            logger.warning(
                "Request refused (%s)", getattr(response, "stop_details", None)
            )
            return []
        if response.stop_reason == "max_tokens":
            logger.warning("Response truncated at max_tokens; batch discarded")
            return []
        batch = response.parsed_output
        if batch is None:
            return []
        return [candidate.model_dump() for candidate in batch.candidates]


@dataclass
class DraftSummary:
    """Counts for one drafting run."""

    requested: int = 0
    drafted: int = 0
    accepted: int = 0
    rejected: int = 0
    reasons: Counter[str] = field(default_factory=Counter)

    def __repr__(self) -> str:
        return (
            f"DraftSummary(requested={self.requested}, drafted={self.drafted}, "
            f"accepted={self.accepted}, rejected={self.rejected})"
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "requested": self.requested,
            "drafted": self.drafted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "reject_reasons": dict(self.reasons.most_common()),
        }


def rejected_path_for(output: Path) -> Path:
    """Sidecar file holding rejects for ``output``."""
    return output.with_name(output.stem + ".rejected.jsonl")


def _append_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_requests(
    mode: str,
    *,
    corpus: CorpusIndex,
    count: int,
    batch_size: int,
    examples_per_request: int,
    seed: int,
    shortfalls: dict[str, dict[str, int]] | None = None,
    parents: list[dict[str, Any]] | None = None,
    per_parent: int = DEFAULT_PER_PARENT,
) -> list[DraftRequest]:
    """Turn a plan into concrete requests with style anchors attached."""
    rng = random.Random(seed)  # noqa: S311 - reproducible example selection
    requests: list[DraftRequest] = []
    if mode == MODE_NEW_FAMILIES:
        if shortfalls is None:
            raise BuildPlanError("new-families mode needs shortfalls")
        for cell, n in plan_new_families(shortfalls, count, batch_size, seed):
            examples = select_examples(corpus.rows, cell, examples_per_request, rng)
            requests.append(DraftRequest(cell=cell, count=n, examples=examples))
    elif mode == MODE_VARIANTS:
        for parent, n in plan_variants(parents or [], count, per_parent, seed):
            cell = cell_of(parent)
            examples = select_examples(
                corpus.rows,
                cell,
                examples_per_request,
                rng,
                exclude_ids={parent["icdu_id"]},
            )
            requests.append(
                DraftRequest(cell=cell, count=n, examples=examples, parent=parent)
            )
    else:
        raise BuildPlanError(f"Unknown mode {mode!r}")
    return requests


def run_drafting(
    requests: list[DraftRequest],
    draft_fn: DraftFn,
    *,
    corpus: CorpusIndex,
    output: Path,
    system_prompt: str,
) -> DraftSummary:
    """Execute every request, screen the drafts, and append survivors to ``output``."""
    summary = DraftSummary()
    seen_families: set[str] = set()
    rejected_path = rejected_path_for(output)
    for index, request in enumerate(requests, start=1):
        summary.requested += request.count
        raw = draft_fn(system_prompt, build_user_prompt(request), request.count)
        summary.drafted += len(raw)
        screened = screen_candidates(raw, request, corpus, seen_families)
        _append_jsonl(screened.accepted, output)
        _append_jsonl(screened.rejected, rejected_path)
        summary.accepted += len(screened.accepted)
        summary.rejected += len(screened.rejected)
        summary.reasons.update(r["reason"].split(":")[0] for r in screened.rejected)
        logger.info(
            "[%d/%d] %s | %s > %s > %s | drafted %d, accepted %d",
            index,
            len(requests),
            request.kind,
            request.cell.persona,
            request.cell.principle,
            request.cell.capability,
            len(raw),
            len(screened.accepted),
        )
    return summary


# ============================================================================
# CLI
# ============================================================================


def parse_floors(text: str) -> dict[str, int]:
    """Parse ``personas=30,principles=40,capabilities=150``."""
    floors = dict(DEFAULT_FLOORS)
    if not text:
        return floors
    for item in text.split(","):
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in DIMENSION_VOCAB or not value.strip().isdigit():
            raise BuildPlanError(f"Bad floor spec {item!r}")
        floors[key] = int(value)
    return floors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draft ICDU candidates with Claude, targeted at thin cells."
    )
    parser.add_argument(
        "--mode", choices=(MODE_NEW_FAMILIES, MODE_VARIANTS), default=MODE_NEW_FAMILIES
    )
    parser.add_argument("--count", type=int, required=True, help="Candidates to draft.")
    parser.add_argument(
        "--output", type=Path, required=True, help="Staging JSONL (appended)."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--corpus",
        type=Path,
        action="append",
        default=None,
        help="Existing rows for anchors and dedup; repeatable "
        "(default: v9 train + validation).",
    )
    parser.add_argument(
        "--parents", type=Path, default=DEFAULT_PARENTS, help="Variant parents."
    )
    parser.add_argument(
        "--reference", type=Path, default=None, help="Optional book text."
    )
    parser.add_argument(
        "--floors", default="", help="e.g. personas=30,principles=40,capabilities=150"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--per-parent", type=int, default=DEFAULT_PER_PARENT)
    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES_PER_REQUEST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan; no API calls."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = _parse_args(argv)
    try:
        corpus = CorpusIndex.from_files(args.corpus or list(DEFAULT_CORPUS))
        shortfalls = None
        parents = None
        if args.mode == MODE_NEW_FAMILIES:
            report = json.loads(args.report.read_text(encoding="utf-8"))
            shortfalls = compute_shortfalls(
                load_split_counts(report), parse_floors(args.floors)
            )
        else:
            parents = load_jsonl(args.parents)
        requests = build_requests(
            args.mode,
            corpus=corpus,
            count=args.count,
            batch_size=args.batch_size,
            examples_per_request=args.examples,
            seed=args.seed,
            shortfalls=shortfalls,
            parents=parents,
            per_parent=args.per_parent,
        )
    except (BuildPlanError, BuildError, OSError, ValueError) as err:
        logger.error("Planning failed: %s", err)
        return 1

    logger.info("Planned %d request(s) for %d candidate(s)", len(requests), args.count)
    if args.dry_run:
        for request in requests:
            cell = request.cell
            print(
                f"{request.kind:11s} x{request.count}  {cell.persona} | "
                f"{cell.principle} | {cell.capability}"
                + (
                    f"  <- {request.parent['application_prompt'][:50]}..."
                    if request.parent
                    else ""
                )
            )
        return 0

    reference = args.reference.read_text(encoding="utf-8") if args.reference else None
    try:
        drafter = AnthropicDrafter(model=args.model, max_tokens=args.max_tokens)
    except BuildPlanError as err:
        logger.error("%s", err)
        return 1
    summary = run_drafting(
        requests,
        drafter,
        corpus=corpus,
        output=args.output,
        system_prompt=build_system_prompt(reference),
    )
    logger.info("Summary: %s", json.dumps(summary.to_dict(), indent=2))
    logger.info(
        "Accepted -> %s | rejected -> %s", args.output, rejected_path_for(args.output)
    )
    logger.info(
        "Next: python -m src.data.build_icdu_dataset --base-version 9 --version 10 "
        "--staged %s --dry-run",
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
