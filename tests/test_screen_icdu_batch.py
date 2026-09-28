"""Tests for the per-batch style/hard-rule screen."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.data.build_icdu_dataset import PERSONA_INTENTS
from src.data.screen_icdu_batch import (
    hard_screen,
    main,
    prior_batch_rows,
    screen_batch,
    style_screen,
)

DATASETS_DIR = Path(__file__).resolve().parents[1] / "src" / "data" / "datasets"
STAGING = DATASETS_DIR / "staging"

SHORT = "I keep saying yes to overtime and then resenting it. What should I change?"
LONG = (
    "Every decision about the new bakery counter turns into a fight with my "
    "brother, and the business cannot keep absorbing the delay while we argue."
)

# Distinct per-row filler so synthetic responses are never near-duplicates.
SENTENCES = [
    "The kettle boils faster when the lid stays on.",
    "Gravel paths drain better than packed clay.",
    "A ledger closes cleanly when receipts are dated.",
    "Wool socks dry slowly on a cold porch.",
    "The bus route changed after the bridge repair.",
    "Tomato seedlings lean toward the kitchen window.",
    "Brass hinges squeak less after a drop of oil.",
    "The choir rehearses on alternate Thursdays.",
    "Paper maps survive a dead phone battery.",
    "Cedar shavings keep the drawer smelling fresh.",
    "The ferry timetable shifts in winter.",
    "Chalk marks wash off the slate in one rain.",
    "A spare key lives under the third flowerpot.",
    "The bakery sells day-old bread at half price.",
    "Fresh snow muffles the traffic on the ring road.",
    "The library extends loans over public holidays.",
    "Copper pans need drying before they are hung.",
]


def _row(prompt: str, response: str, persona: str) -> dict[str, Any]:
    return {
        "application_prompt": prompt,
        "ideal_response_final": response,
        "persona_archetype": persona,
        "governing_principle": "Chapter 1 > Clarity",
        "capability_layer": "Foundational",
        "ideal_response_attributes": ["Clear", "Actionable"],
    }


def _response(words: int, seed: int, opener: str = "Start with") -> str:
    base = SENTENCES[seed % len(SENTENCES)]
    text = f"{opener} " + " ".join([base] * 40)
    return " ".join(text.split()[: words - 1]) + " tonight."


def _varied_batch() -> list[dict[str, Any]]:
    """A batch that satisfies every style band."""
    rows = []
    openers = [
        "Start with",
        "Notice that",
        "One thing",
        "Before anything",
        "Try this",
        "Consider how",
    ]
    for index, persona in enumerate(PERSONA_INTENTS):
        prompt = (
            f"{LONG} Case {index}."
            if index % 3 == 0
            else f"{SHORT[:-1]} number {index}?"
        )
        words = 60 + (index * 9) % 140
        rows.append(_row(prompt, _response(words, index, openers[index % 6]), persona))
    return rows


class TestStyleScreen:
    @pytest.mark.unit
    def test_varied_batch_passes(self) -> None:
        failures, stats = style_screen(_varied_batch())
        assert failures == [], failures
        assert stats["personas_covered"] == 17

    @pytest.mark.unit
    def test_long_uniform_prompts_fail(self) -> None:
        rows = [
            _row(f"{LONG} Case {i}.", _response(60 + i * 8, i), p)
            for i, p in enumerate(PERSONA_INTENTS)
        ]
        failures, _ = style_screen(rows)
        assert any("prompt median" in f for f in failures)
        assert any("under 18 words" in f for f in failures)

    @pytest.mark.unit
    def test_clustered_response_lengths_fail(self) -> None:
        rows = [
            _row(f"{SHORT[:-1]} {i}?", _response(100, i, f"Opener{i} words"), p)
            for i, p in enumerate(PERSONA_INTENTS)
        ]
        failures, _ = style_screen(rows)
        assert any("spread" in f for f in failures)

    @pytest.mark.unit
    def test_repeated_framing_phrase_fails(self) -> None:
        rows = _varied_batch()
        for row in rows[:5]:
            row["ideal_response_final"] += " Keep the Foundational step small."
        failures, stats = style_screen(rows)
        assert any("keep_the_layer" in f for f in failures)
        assert stats["framing_pattern_rows"]["keep_the_layer"] == 5

    @pytest.mark.unit
    def test_repeated_opening_fails(self) -> None:
        rows = _varied_batch()
        for row in rows[:4]:
            row["ideal_response_final"] = "It makes " + row["ideal_response_final"]
        failures, _ = style_screen(rows)
        assert any("open with 'it makes'" in f for f in failures)

    @pytest.mark.unit
    def test_missing_persona_fails_unless_allowed(self) -> None:
        rows = _varied_batch()[:-1]
        failures, _ = style_screen(rows)
        assert any("missing personas" in f for f in failures)
        failures, _ = style_screen(rows, require_all_personas=False)
        assert not any("missing personas" in f for f in failures)

    @pytest.mark.unit
    def test_empty_batch(self) -> None:
        assert style_screen([]) == (["batch is empty"], {})


class TestHardScreen:
    @pytest.mark.unit
    def test_rejects_carry_row_numbers(self, tmp_path: Path) -> None:
        from src.data.draft_icdu_candidates import CorpusIndex

        corpus = CorpusIndex(rows=[])
        rows = _varied_batch()
        rows[2]["ideal_response_final"] += " Does that help?"
        del rows[4]["capability_layer"]
        rejects = hard_screen(rows, corpus)
        assert {r["row"]: r["reason"] for r in rejects} == {
            3: "trailing_question",
            5: "missing_field:capability_layer",
        }


class TestFiles:
    @pytest.mark.unit
    def test_prior_batches_are_deduped_against(self, tmp_path: Path) -> None:
        first = tmp_path / "icdu_batch_001.jsonl"
        second = tmp_path / "icdu_batch_002.jsonl"
        rows = _varied_batch()
        first.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
        second.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        assert len(prior_batch_rows(second)) == 1
        assert prior_batch_rows(first) == []
        report = screen_batch(second, [])
        assert any(
            r["reason"] == "duplicate_family" and r["row"] == 1
            for r in report.hard_rejects
        )

    @pytest.mark.integration
    def test_batch_one_passes_hard_rules_but_fails_style(self) -> None:
        # The accepted GPT batch: clean on every hard rule, but its prompts are
        # uniformly long and its responses cluster - exactly what this screen adds.
        batch = STAGING / "icdu_batch_001.jsonl"
        if not batch.is_file():
            pytest.skip("batch 1 not present")
        report = screen_batch(batch, [DATASETS_DIR / "icdu_training_data_v9.jsonl"])
        assert report.hard_rejects == []
        assert any("prompt median" in f for f in report.style_failures)
        assert report.stats["personas_covered"] == 17

    @pytest.mark.unit
    def test_cli_exit_codes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        good = tmp_path / "icdu_batch_010.jsonl"
        good.write_text(
            "".join(json.dumps(r) + "\n" for r in _varied_batch()), encoding="utf-8"
        )
        bad = tmp_path / "icdu_batch_011.jsonl"
        rows = _varied_batch()
        rows[0]["ideal_response_final"] += " Does that help?"
        bad.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        train = str(DATASETS_DIR / "icdu_training_data_v9.jsonl")
        assert main(["--batch", str(bad), "--corpus", train]) == 1
        assert "REJECT row 1: trailing_question" in capsys.readouterr().out
        assert main(["--batch", str(good), "--corpus", train]) == 0
        assert "PASS" in capsys.readouterr().out
