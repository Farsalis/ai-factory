"""Tests for the sealed-split evaluator's model-independent logic.

Model loading, generation and the forward pass are not exercised here; the
evaluation loop takes injectable callables so everything around them is.
"""

import json
import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.data import format_icdu_to_chat
from src.evaluate_icdu import (
    MIN_RESPONSE_WORDS,
    RUBRIC_CHECKS,
    RecordResult,
    aggregate,
    build_prompt_messages,
    common_prefix_length,
    derive_test_file,
    render_full,
    render_prompt,
    run_evaluation,
    score_response,
    write_report,
)

GOOD_RESPONSE = (
    "Grief has no timeline, and the word should is adding guilt to your pain. "
    "Let's focus on a Foundational Process you can keep: every Sunday morning, "
    "spend ten minutes writing down one favourite memory of her. That gives the "
    "feeling a dedicated place instead of expecting it to disappear on schedule. "
    "Start this coming Sunday and keep the ritual small enough to survive a hard week."
)


def sample_record(**overrides: Any) -> dict[str, Any]:
    """A v9-shaped record."""
    record = {
        "icdu_id": "5cb78141-264e-594a-8a0f-5cc0379c7150",
        "persona_archetype": "Relationship Navigator > Communication Builder",
        "governing_principle": "Chapter 2 > Process",
        "capability_layer": "Foundational",
        "user_intent": "To improve a relationship or communication outcome",
        "context_summary": (
            "The user is seeking practical, encouraging guidance for a relationship "
            "or communication challenge. Apply the Breaking Better 3x3 framework "
            "with emphasis on Chapter 2 > Process."
        ),
        "application_prompt": (
            "My mother passed away a few months ago and I feel like I should be "
            "'over it' by now, but I'm not."
        ),
        "ideal_response_final": GOOD_RESPONSE,
        "ideal_response_attributes": ["Clear", "Actionable"],
        "ideal_response_cot": ["a", "b", "c"],
    }
    record.update(overrides)
    return record


class TemplateTokenizer:
    """Fake tokenizer whose template serialises the messages it was given."""

    chat_template = "fake-template"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        return json.dumps({"messages": messages, "gen": add_generation_prompt})


class NoTemplateTokenizer:
    """Fake tokenizer without a chat template."""

    chat_template = None


class TestPromptConstruction:
    """The evaluator must present exactly the prompt the trainer built."""

    @pytest.mark.unit
    def test_messages_match_trainer_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the trainer's 50% perturbation branch off so the comparison is exact.
        monkeypatch.setattr("src.data.random.random", lambda: 1.0)
        record = sample_record()
        rendered = format_icdu_to_chat(record, TemplateTokenizer())  # type: ignore[arg-type]
        trainer_messages = json.loads(rendered["text"])["messages"]
        assert trainer_messages[:2] == build_prompt_messages(record)
        assert trainer_messages[2] == {
            "role": "assistant",
            "content": record["ideal_response_final"],
        }

    @pytest.mark.unit
    def test_render_prompt_requests_generation_cue(self) -> None:
        payload = json.loads(
            render_prompt(build_prompt_messages(sample_record()), TemplateTokenizer())
        )
        assert payload["gen"] is True
        assert [m["role"] for m in payload["messages"]] == ["system", "user"]

    @pytest.mark.unit
    def test_render_full_appends_reference_response(self) -> None:
        record = sample_record()
        payload = json.loads(
            render_full(
                build_prompt_messages(record),
                record["ideal_response_final"],
                TemplateTokenizer(),
            )
        )
        assert payload["gen"] is False
        assert payload["messages"][-1] == {
            "role": "assistant",
            "content": record["ideal_response_final"],
        }

    @pytest.mark.unit
    def test_fallback_prompt_ends_at_assistant_cue(self) -> None:
        record = sample_record()
        text = render_prompt(build_prompt_messages(record), NoTemplateTokenizer())
        assert text.endswith("Assistant: ")
        assert f"User: {record['application_prompt']}" in text

    @pytest.mark.unit
    def test_fallback_full_contains_response(self) -> None:
        record = sample_record()
        text = render_full(build_prompt_messages(record), "REF", NoTemplateTokenizer())
        assert text.endswith("Assistant: REF\n")

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ([1, 2, 3], [1, 2, 3, 4], 3),
            ([1, 2, 3], [1, 9, 3], 1),
            ([], [1], 0),
            ([5], [5], 1),
        ],
    )
    def test_common_prefix_length(
        self, left: list[int], right: list[int], expected: int
    ) -> None:
        assert common_prefix_length(left, right) == expected


class TestRubric:
    """Deterministic checks on a generation."""

    @pytest.mark.unit
    def test_good_response_passes_every_check(self) -> None:
        scores = score_response(sample_record(), GOOD_RESPONSE)
        assert all(scores[check] for check in RUBRIC_CHECKS)
        assert scores["word_count"] == len(GOOD_RESPONSE.split())

    @pytest.mark.unit
    def test_trailing_question_fails(self) -> None:
        scores = score_response(sample_record(), GOOD_RESPONSE + " What do you think?")
        assert scores["no_trailing_question"] is False

    @pytest.mark.unit
    def test_tool_scaffolding_fails(self) -> None:
        scores = score_response(
            sample_record(), 'Need data: {"tool_call": {}} ' + GOOD_RESPONSE
        )
        assert scores["no_tool_scaffolding"] is False

    @pytest.mark.unit
    def test_empty_generation_fails_non_empty_and_question_check(self) -> None:
        scores = score_response(sample_record(), "   ")
        assert scores["non_empty"] is False
        assert scores["no_trailing_question"] is False
        assert scores["word_count"] == 0

    @pytest.mark.unit
    def test_short_response_fails_length(self) -> None:
        short = " ".join(["word"] * (MIN_RESPONSE_WORDS - 1))
        assert score_response(sample_record(), short)["length_in_range"] is False

    @pytest.mark.unit
    def test_principle_cell_is_matched_case_insensitively(self) -> None:
        record = sample_record(governing_principle="Chapter 1 > Clarity")
        text = "start with CLARITY about what you want. " * 8
        assert score_response(record, text)["names_principle"] is True
        assert (
            score_response(record, "no framework words here " * 8)["names_principle"]
            is False
        )

    @pytest.mark.unit
    def test_capability_mention_detected(self) -> None:
        text = "This is an Aspirational goal, so anchor it first. " * 6
        assert score_response(sample_record(), text)["names_capability"] is True


def _result(layer: str, passes: bool, loss: float) -> RecordResult:
    scores = dict.fromkeys(RUBRIC_CHECKS, passes)
    scores["word_count"] = 100
    return RecordResult(
        icdu_id=f"id-{layer}-{passes}-{loss}",
        capability_layer=layer,
        governing_principle="Chapter 2 > Process",
        generated="x",
        scores=scores,
        loss=loss,
    )


class TestAggregate:
    """Summary statistics."""

    @pytest.mark.unit
    def test_rates_and_loss(self) -> None:
        summary = aggregate(
            [
                _result("Foundational", True, 1.0),
                _result("Foundational", False, 3.0),
                _result("Aspirational", True, float("nan")),
            ]
        )
        assert summary["records"] == 3
        assert summary["rubric_pass_rates"]["all_checks"] == pytest.approx(
            2 / 3, abs=1e-4
        )
        assert summary["mean_reference_loss"] == 2.0  # NaN excluded
        assert summary["perplexity"] == pytest.approx(math.exp(2.0), abs=1e-3)
        assert summary["by_capability"]["Foundational"]["records"] == 2
        assert summary["by_capability"]["Aspirational"]["mean_reference_loss"] is None
        assert "Transformational" not in summary["by_capability"]

    @pytest.mark.unit
    def test_empty_results(self) -> None:
        summary = aggregate([])
        assert summary["records"] == 0
        assert summary["rubric_pass_rates"]["all_checks"] is None
        assert summary["perplexity"] is None


class TestRunAndReport:
    """The loop wires the callables and the report captures everything."""

    @pytest.mark.unit
    def test_run_evaluation_with_fakes(self) -> None:
        records = [sample_record(icdu_id=f"r{i}") for i in range(3)]
        results = run_evaluation(records, lambda r: GOOD_RESPONSE, lambda r: 0.5)
        assert [r.icdu_id for r in results] == ["r0", "r1", "r2"]
        assert all(r.passes_all() for r in results)
        assert all(r.loss == 0.5 for r in results)

    @pytest.mark.unit
    def test_skipping_generation_leaves_scores_empty(self) -> None:
        results = run_evaluation([sample_record()], None, lambda r: 1.5)
        assert results[0].scores == {}
        assert results[0].generated == ""
        assert results[0].loss == 1.5

    @pytest.mark.unit
    def test_skipping_loss_yields_nan(self) -> None:
        results = run_evaluation([sample_record()], lambda r: GOOD_RESPONSE, None)
        assert math.isnan(results[0].loss)

    @pytest.mark.unit
    def test_write_report(self, tmp_path: Path) -> None:
        results = run_evaluation(
            [sample_record()], lambda r: GOOD_RESPONSE, lambda r: 0.25
        )
        out = tmp_path / "nested" / "report.json"
        report = write_report(
            out,
            model_path="m",
            test_file=Path("t.jsonl"),
            results=results,
            elapsed_seconds=1.234,
        )
        on_disk = json.loads(out.read_text(encoding="utf-8"))
        assert on_disk == report
        assert on_disk["summary"]["records"] == 1
        assert on_disk["elapsed_seconds"] == 1.2
        assert on_disk["records"][0]["generated"] == GOOD_RESPONSE


class TestDeriveTestFile:
    """The sealed split is located next to the training file."""

    @pytest.mark.unit
    def test_derives_sibling(self, tmp_path: Path) -> None:
        train = tmp_path / "icdu_training_data_v9.jsonl"
        test = tmp_path / "icdu_test_data_v9.jsonl"
        train.write_text("{}\n", encoding="utf-8")
        test.write_text("{}\n", encoding="utf-8")
        assert derive_test_file(train) == test

    @pytest.mark.unit
    def test_missing_sibling_raises(self, tmp_path: Path) -> None:
        train = tmp_path / "icdu_training_data_v9.jsonl"
        train.write_text("{}\n", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="--test-file"):
            derive_test_file(train)

    @pytest.mark.unit
    def test_unrecognised_name_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            derive_test_file(tmp_path / "data.jsonl")
