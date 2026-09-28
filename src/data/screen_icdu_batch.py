"""Gate one drafted ICDU batch: hard rules plus style-distribution checks.

The dataset builder's gates catch rule violations (artifacts, duplicates,
trailing questions). They cannot see a batch that passes every rule and still
teaches a tic - forty prompts that all open with a biography, or forty
responses that all land at 100 words with the same scaffold. This screen adds
those distribution checks so a generator (human, web chat, or local agent) gets
a pass/fail verdict per batch with reasons it can act on.

Checks:

* Hard rules - every row through ``screen_candidates`` (vocabulary, length
  bands, artifacts, tool text, duplicates against the corpus, earlier batches
  and the batch itself, near-duplicate responses).
* Prompt shape - median length at or below ``--max-prompt-median`` words and
  at least ``--min-short-share`` of prompts under ``SHORT_PROMPT_WORDS``. The
  real v9 corpus has median 14 and p95 26.
* Response spread - the p90-p10 word-count spread must be at least
  ``--min-response-spread``; the corpus runs 32-255 words.
* Phrase reuse - no framing phrase (``"Keep the Foundational ..."``,
  ``"People means ..."``, ``"Today, ..."``) more than ``--max-phrase-repeats``
  times, and no two-word opening more than that either.
* Persona coverage - every persona present unless ``--allow-missing-personas``.

Exit code 0 when the batch passes, 1 when it does not. ``--json`` prints the
full report for tooling.

Usage:
    conda run -n ai-factory python -m src.data.screen_icdu_batch \\
        --batch src/data/datasets/staging/icdu_batch_002.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.build_icdu_dataset import PERSONA_INTENTS, load_jsonl
from src.data.draft_icdu_candidates import (
    DEFAULT_CORPUS,
    Cell,
    CorpusIndex,
    DraftRequest,
    _similarity,
    screen_candidates,
)

logger = logging.getLogger(__name__)

SHORT_PROMPT_WORDS = 18
DEFAULT_MAX_PROMPT_MEDIAN = 22
DEFAULT_MIN_SHORT_SHARE = 0.4
DEFAULT_MIN_RESPONSE_SPREAD = 40
DEFAULT_MAX_PHRASE_REPEATS = 3
DEFAULT_BATCH_GLOB = "icdu_batch_*.jsonl"

FRAMEWORK_WORDS = (
    "People|Clarity|Transparency|Process|Tools|"
    "Foundational|Transformational|Aspirational"
)

#: Framing constructions that become tics when repeated across a batch.
FRAMING_PATTERNS: dict[str, re.Pattern[str]] = {
    "keep_the_layer": re.compile(
        rf"\bKeep (the|this|your) ({FRAMEWORK_WORDS})\b", re.IGNORECASE
    ),
    "your_layer_habit": re.compile(
        rf"\bYour ({FRAMEWORK_WORDS}) (habit|step|goal|practice|change)\b",
        re.IGNORECASE,
    ),
    "principle_verb": re.compile(
        rf"\b({FRAMEWORK_WORDS}) (means|asks|begins with|starts with|includes|"
        r"helps you|gives you|puts)\b"
    ),
    "today_closer": re.compile(r"(^|\. )Today,", re.MULTILINE),
    "biography_open": re.compile(
        r"^I(?:'m| am| have| was)\b[^.?!]{0,80}\b(?:years?|months?|first|"
        r"new parent|student|mid-career|retired)\b",
        re.IGNORECASE,
    ),
}


@dataclass
class ScreenReport:
    """Verdict for one batch."""

    batch: str
    rows: int
    hard_rejects: list[dict[str, Any]] = field(default_factory=list)
    style_failures: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ScreenReport(batch={self.batch!r}, rows={self.rows}, "
            f"rejects={len(self.hard_rejects)}, style={len(self.style_failures)})"
        )

    @property
    def passed(self) -> bool:
        """True when there are no hard rejects and no style failures."""
        return not self.hard_rejects and not self.style_failures

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "batch": self.batch,
            "rows": self.rows,
            "passed": self.passed,
            "hard_rejects": self.hard_rejects,
            "style_failures": self.style_failures,
            "stats": self.stats,
        }


def _quantile(values: Sequence[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def prior_batch_rows(
    batch: Path, pattern: str = DEFAULT_BATCH_GLOB
) -> list[dict[str, Any]]:
    """Rows from sibling batch files that sort before ``batch``."""
    rows: list[dict[str, Any]] = []
    for path in sorted(batch.parent.glob(pattern)):
        if path.resolve() == batch.resolve() or path.name >= batch.name:
            continue
        rows.extend(load_jsonl(path))
    return rows


def hard_screen(
    rows: list[dict[str, Any]], corpus: CorpusIndex
) -> list[dict[str, Any]]:
    """Run the drafter's per-row rules; returns rejects with 1-based row numbers."""
    rejects: list[dict[str, Any]] = []
    seen: set[str] = set()
    accepted_responses: list[str] = []
    for index, row in enumerate(rows, start=1):
        try:
            cell = Cell(
                row["persona_archetype"],
                row["governing_principle"],
                row["capability_layer"],
            )
        except KeyError as err:
            rejects.append({"row": index, "reason": f"missing_field:{err.args[0]}"})
            continue
        request = DraftRequest(cell=cell, count=1, examples=[])
        result = screen_candidates([row], request, corpus, seen)
        for reject in result.rejected:
            rejects.append(
                {
                    "row": index,
                    "reason": reject["reason"],
                    "prompt": row.get("application_prompt", "")[:80],
                }
            )
        if result.accepted:
            response = result.accepted[0]["ideal_response_final"]
            if any(_similarity(response, r) > 0.9 for r in accepted_responses):
                rejects.append(
                    {
                        "row": index,
                        "reason": "near_duplicate_in_batch",
                        "prompt": row.get("application_prompt", "")[:80],
                    }
                )
                continue
            accepted_responses.append(response)
    return rejects


def style_screen(
    rows: list[dict[str, Any]],
    *,
    max_prompt_median: int = DEFAULT_MAX_PROMPT_MEDIAN,
    min_short_share: float = DEFAULT_MIN_SHORT_SHARE,
    min_response_spread: int = DEFAULT_MIN_RESPONSE_SPREAD,
    max_phrase_repeats: int = DEFAULT_MAX_PHRASE_REPEATS,
    require_all_personas: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Distribution checks over the whole batch.

    Returns:
        ``(failures, stats)`` - human-readable failures and the measurements
        behind them.
    """
    if not rows:
        return ["batch is empty"], {}
    prompts = [str(r.get("application_prompt", "")) for r in rows]
    responses = [str(r.get("ideal_response_final", "")) for r in rows]
    prompt_words = [len(p.split()) for p in prompts]
    response_words = [len(r.split()) for r in responses]

    failures: list[str] = []
    prompt_median = _quantile(prompt_words, 0.5)
    short_share = sum(1 for w in prompt_words if w < SHORT_PROMPT_WORDS) / len(rows)
    spread = _quantile(response_words, 0.9) - _quantile(response_words, 0.1)

    if prompt_median > max_prompt_median:
        failures.append(
            f"prompt median {prompt_median} words exceeds {max_prompt_median} "
            "(corpus median is 14)"
        )
    if short_share < min_short_share:
        failures.append(
            f"only {short_share:.0%} of prompts are under {SHORT_PROMPT_WORDS} words "
            f"(need {min_short_share:.0%})"
        )
    if spread < min_response_spread:
        failures.append(
            f"response length p90-p10 spread is {spread} words "
            f"(need at least {min_response_spread}; responses are clustering)"
        )

    phrase_counts: dict[str, int] = {}
    for name, pattern in FRAMING_PATTERNS.items():
        haystack = prompts if name == "biography_open" else responses
        count = sum(1 for text in haystack if pattern.search(text))
        phrase_counts[name] = count
        if count > max_phrase_repeats:
            failures.append(
                f"framing pattern '{name}' appears in {count} rows "
                f"(max {max_phrase_repeats})"
            )

    openings = Counter(" ".join(r.split()[:2]).lower() for r in responses)
    for opening, count in openings.most_common(3):
        if count > max_phrase_repeats:
            failures.append(
                f"{count} responses open with {opening!r} (max {max_phrase_repeats})"
            )

    personas = Counter(str(r.get("persona_archetype", "")) for r in rows)
    missing = sorted(set(PERSONA_INTENTS) - set(personas))
    if require_all_personas and missing:
        failures.append(f"missing personas: {missing}")

    pairwise_prompt = (
        max(_similarity(a, b) for a, b in itertools.combinations(prompts, 2))
        if len(prompts) > 1
        else 0.0
    )
    stats = {
        "prompt_words": {
            "min": min(prompt_words),
            "median": prompt_median,
            "max": max(prompt_words),
            "short_share": round(short_share, 2),
        },
        "response_words": {
            "min": min(response_words),
            "median": _quantile(response_words, 0.5),
            "max": max(response_words),
            "p90_p10_spread": spread,
        },
        "framing_pattern_rows": phrase_counts,
        "top_openings": openings.most_common(3),
        "personas_covered": len(personas),
        "max_pairwise_prompt_similarity": round(pairwise_prompt, 2),
    }
    return failures, stats


def screen_batch(
    batch: Path,
    corpus_files: Sequence[Path],
    *,
    prior_glob: str = DEFAULT_BATCH_GLOB,
    **style_kwargs: Any,
) -> ScreenReport:
    """Screen ``batch`` against the corpus, earlier batches, and style bands."""
    rows = load_jsonl(batch)
    corpus = CorpusIndex(
        rows=[
            *CorpusIndex.from_files(corpus_files).rows,
            *prior_batch_rows(batch, prior_glob),
        ]
    )
    report = ScreenReport(batch=batch.name, rows=len(rows))
    report.hard_rejects = hard_screen(rows, corpus)
    report.style_failures, report.stats = style_screen(rows, **style_kwargs)
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate one drafted ICDU batch.")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", default=None)
    parser.add_argument("--prior-glob", default=DEFAULT_BATCH_GLOB)
    parser.add_argument(
        "--max-prompt-median", type=int, default=DEFAULT_MAX_PROMPT_MEDIAN
    )
    parser.add_argument(
        "--min-short-share", type=float, default=DEFAULT_MIN_SHORT_SHARE
    )
    parser.add_argument(
        "--min-response-spread", type=int, default=DEFAULT_MIN_RESPONSE_SPREAD
    )
    parser.add_argument(
        "--max-phrase-repeats", type=int, default=DEFAULT_MAX_PHRASE_REPEATS
    )
    parser.add_argument("--allow-missing-personas", action="store_true")
    parser.add_argument(
        "--json", action="store_true", help="Print the full report as JSON."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Exit 0 on pass, 1 on fail."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
    args = _parse_args(argv)
    report = screen_batch(
        args.batch,
        args.corpus or list(DEFAULT_CORPUS),
        prior_glob=args.prior_glob,
        max_prompt_median=args.max_prompt_median,
        min_short_share=args.min_short_share,
        min_response_spread=args.min_response_spread,
        max_phrase_repeats=args.max_phrase_repeats,
        require_all_personas=not args.allow_missing_personas,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        verdict = "PASS" if report.passed else "FAIL"
        print(f"{verdict}: {report.batch} ({report.rows} rows)")
        for reject in report.hard_rejects:
            print(
                f"  REJECT row {reject['row']}: {reject['reason']} "
                f":: {reject.get('prompt', '')}"
            )
        for failure in report.style_failures:
            print(f"  STYLE: {failure}")
        stats = report.stats
        if stats:
            print(
                f"  prompts {stats['prompt_words']} | "
                f"responses {stats['response_words']}"
            )
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
