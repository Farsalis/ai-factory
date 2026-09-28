"""Tests for the candidate drafter: planning, prompting, screening, and the
SDK boundary. No test calls the Claude API; the drafting loop takes a callable
and the SDK client is exercised through a fake ``anthropic`` module.
"""

import json
import random
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from src.data.build_icdu_dataset import (
    CAPABILITY_LAYERS,
    GOVERNING_PRINCIPLES,
    PERSONA_INTENTS,
)
from src.data.draft_icdu_candidates import (
    DEFAULT_FLOORS,
    MODE_NEW_FAMILIES,
    MODE_VARIANTS,
    AnthropicDrafter,
    BuildPlanError,
    Cell,
    CorpusIndex,
    DraftBatch,
    DraftCandidate,
    DraftRequest,
    build_requests,
    build_system_prompt,
    build_user_prompt,
    cell_of,
    compute_shortfalls,
    load_split_counts,
    main,
    parse_floors,
    plan_new_families,
    plan_variants,
    rejected_path_for,
    run_drafting,
    screen_candidates,
    select_examples,
)

DATASETS_DIR = Path(__file__).resolve().parents[1] / "src" / "data" / "datasets"


def rng() -> random.Random:
    """Seeded RNG for deterministic example selection."""
    return random.Random(0)  # noqa: S311


CELL = Cell(
    persona="Community Participant > Contributor",
    principle="Chapter 1 > Clarity",
    capability="Foundational",
)
CLEAN_PROMPT = (
    "I keep volunteering for extra committee work and then resenting it. "
    "How do I decide what to take on?"
)
CLEAN_RESPONSE = (
    "Resentment is usually a signal that a commitment was made without a boundary. "
    "Start with Clarity: write down, in one sentence, what you want your contribution "
    "to be this year. Then treat every request as a comparison against that sentence "
    "rather than a yes-or-no question. When something does not match, you are not "
    "refusing the person; you are protecting a commitment you already made. Draft "
    "that sentence tonight and keep it where you will see it before the next meeting."
)


@pytest.fixture(scope="module")
def corpus() -> CorpusIndex:
    """Real v9 train + validation rows."""
    return CorpusIndex.from_files(
        [
            DATASETS_DIR / "icdu_training_data_v9.jsonl",
            DATASETS_DIR / "icdu_validation_data_v9.jsonl",
        ]
    )


def raw(
    prompt: str = CLEAN_PROMPT, response: str = CLEAN_RESPONSE, **extra: Any
) -> dict[str, Any]:
    """A raw model candidate."""
    return {
        "application_prompt": prompt,
        "ideal_response_final": response,
        "ideal_response_attributes": ["Clear", "Actionable"],
        **extra,
    }


def request(count: int = 2, parent: dict[str, Any] | None = None) -> DraftRequest:
    return DraftRequest(cell=CELL, count=count, examples=[], parent=parent)


# ============================================================================
# Planning
# ============================================================================


class TestQuotas:
    @pytest.mark.unit
    def test_counts_cover_full_vocabulary_with_zeros(self) -> None:
        report = {
            "splits": {
                "train": {
                    "personas": {"Educator > Teaching Innovator": 1},
                    "principles": {},
                    "capabilities": {"Foundational": 5},
                }
            }
        }
        counts = load_split_counts(report)
        assert len(counts["personas"]) == len(PERSONA_INTENTS)
        assert counts["personas"]["Pet Owner > Responsible Caregiver"] == 0
        assert counts["personas"]["Educator > Teaching Innovator"] == 1
        assert len(counts["principles"]) == len(GOVERNING_PRINCIPLES)
        assert counts["capabilities"]["Aspirational"] == 0

    @pytest.mark.unit
    def test_missing_split_raises(self) -> None:
        with pytest.raises(BuildPlanError, match="no split"):
            load_split_counts({"splits": {}}, "train")

    @pytest.mark.unit
    def test_real_v9_report_loads(self) -> None:
        report = json.loads(
            (DATASETS_DIR / "icdu_v9_validation_report.json").read_text("utf-8")
        )
        counts = load_split_counts(report)
        assert sum(counts["capabilities"].values()) == 240

    @pytest.mark.unit
    def test_shortfalls(self) -> None:
        counts = {"personas": {"a": 1, "b": 50}, "principles": {}, "capabilities": {}}
        short = compute_shortfalls(counts, {"personas": 30})
        assert short["personas"] == {"a": 29, "b": 0}


class TestPlanNewFamilies:
    def _shortfalls(self, **thin: int) -> dict[str, dict[str, int]]:
        short = {
            dim: dict.fromkeys(vocab, 0)
            for dim, vocab in {
                "personas": PERSONA_INTENTS,
                "principles": GOVERNING_PRINCIPLES,
                "capabilities": CAPABILITY_LAYERS,
            }.items()
        }
        short["personas"]["Pet Owner > Responsible Caregiver"] = thin.get("persona", 0)
        short["principles"]["Chapter 3 > Aspirational"] = thin.get("principle", 0)
        short["capabilities"]["Aspirational"] = thin.get("capability", 0)
        return short

    @pytest.mark.unit
    def test_thinnest_labels_come_first_and_counts_sum(self) -> None:
        plan = plan_new_families(
            self._shortfalls(persona=7, principle=7, capability=7), 12, 5
        )
        assert plan[0][0] == Cell(
            "Pet Owner > Responsible Caregiver",
            "Chapter 3 > Aspirational",
            "Aspirational",
        )
        assert [n for _, n in plan] == [5, 5, 2]
        assert sum(n for _, n in plan) == 12

    @pytest.mark.unit
    def test_shortfall_is_consumed(self) -> None:
        # 7 short, batch 5: first batch takes the thin persona, second still does
        # (2 remaining), third is free to move on.
        plan = plan_new_families(self._shortfalls(persona=7), 15, 5)
        assert (
            plan[0][0].persona
            == plan[1][0].persona
            == "Pet Owner > Responsible Caregiver"
        )

    @pytest.mark.unit
    def test_deterministic_for_seed(self) -> None:
        a = plan_new_families(self._shortfalls(), 30, 5, seed=1)
        b = plan_new_families(self._shortfalls(), 30, 5, seed=1)
        assert a == b

    @pytest.mark.unit
    def test_spreads_when_floors_are_met(self) -> None:
        plan = plan_new_families(self._shortfalls(), 50, 5, seed=3)
        assert len({cell.persona for cell, _ in plan}) > 1

    @pytest.mark.unit
    def test_zero_count_and_bad_batch(self) -> None:
        assert plan_new_families(self._shortfalls(), 0, 5) == []
        with pytest.raises(BuildPlanError):
            plan_new_families(self._shortfalls(), 5, 0)


class TestPlanVariants:
    @pytest.mark.unit
    def test_counts_and_determinism(self, corpus: CorpusIndex) -> None:
        parents = corpus.rows[:10]
        plan = plan_variants(parents, 7, 2, seed=5)
        assert [n for _, n in plan] == [2, 2, 2, 1]
        assert plan == plan_variants(parents, 7, 2, seed=5)
        assert len({p["icdu_id"] for p, _ in plan}) == 4

    @pytest.mark.unit
    def test_requires_parents(self) -> None:
        with pytest.raises(BuildPlanError, match="parent"):
            plan_variants([], 3, 2)


# ============================================================================
# Examples and prompts
# ============================================================================


class TestExamples:
    @pytest.mark.unit
    def test_prefers_exact_cell(self, corpus: CorpusIndex) -> None:
        cell = cell_of(corpus.rows[0])
        exact = [r for r in corpus.rows if cell_of(r) == cell]
        chosen = select_examples(corpus.rows, cell, 3, rng())
        assert len(chosen) == 3
        for row in chosen[: min(3, len(exact))]:
            assert cell_of(row) == cell

    @pytest.mark.unit
    def test_excludes_ids_and_never_repeats(self, corpus: CorpusIndex) -> None:
        cell = cell_of(corpus.rows[0])
        chosen = select_examples(
            corpus.rows,
            cell,
            5,
            rng(),
            exclude_ids={corpus.rows[0]["icdu_id"]},
        )
        ids = [r["icdu_id"] for r in chosen]
        assert corpus.rows[0]["icdu_id"] not in ids
        assert len(ids) == len(set(ids))

    @pytest.mark.unit
    def test_falls_back_to_broader_tiers(self, corpus: CorpusIndex) -> None:
        rare = Cell(
            "Pet Owner > Responsible Caregiver",
            "Chapter 3 > Aspirational",
            "Aspirational",
        )
        assert len(select_examples(corpus.rows, rare, 4, rng())) == 4


class TestPrompts:
    @pytest.mark.unit
    def test_user_prompt_names_cell_and_examples(self, corpus: CorpusIndex) -> None:
        req = DraftRequest(cell=CELL, count=3, examples=corpus.rows[:2])
        text = build_user_prompt(req)
        assert (
            CELL.persona in text and CELL.principle in text and CELL.capability in text
        )
        assert PERSONA_INTENTS[CELL.persona] in text
        assert corpus.rows[0]["application_prompt"] in text
        assert "Produce exactly 3 new candidates" in text

    @pytest.mark.unit
    def test_variant_prompt_includes_parent(self, corpus: CorpusIndex) -> None:
        parent = corpus.rows[0]
        text = build_user_prompt(
            DraftRequest(cell=cell_of(parent), count=2, examples=[], parent=parent)
        )
        assert "--- Parent ---" in text
        assert parent["application_prompt"] in text
        assert "Produce exactly 2 variants" in text

    @pytest.mark.unit
    def test_system_prompt_includes_rules_and_reference(self) -> None:
        text = build_system_prompt("CHAPTER ONE TEXT")
        assert "Hard rules" in text
        assert "CHAPTER ONE TEXT" in text
        assert "Reference material" not in build_system_prompt(None)


# ============================================================================
# Screening
# ============================================================================


class TestScreening:
    @pytest.mark.unit
    def test_clean_candidate_is_accepted_with_cell_labels(
        self, corpus: CorpusIndex
    ) -> None:
        seen: set[str] = set()
        result = screen_candidates([raw()], request(), corpus, seen)
        assert not result.rejected
        (record,) = result.accepted
        assert record["persona_archetype"] == CELL.persona
        assert record["governing_principle"] == CELL.principle
        assert record["capability_layer"] == CELL.capability
        assert "parent_application_prompt" not in record
        assert len(seen) == 1

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("candidate", "reason"),
        [
            (raw(response=CLEAN_RESPONSE + " Does that help?"), "trailing_question"),
            (
                raw(response='Need data: {"tool_call": {}} ' + CLEAN_RESPONSE),
                "tool_scaffolding",
            ),
            (
                raw(prompt="I want to get fit. How to track progress?"),
                "mechanical_suffix",
            ),
            (raw(prompt="Help me please?"), "prompt_length"),
            (raw(response="Start small and keep going."), "response_length"),
            (
                raw(ideal_response_attributes=["Clear", "Sassy"]),
                "off_vocabulary_attributes",
            ),
            (raw(response=""), "missing_field"),
        ],
    )
    def test_rejections(
        self, corpus: CorpusIndex, candidate: dict[str, Any], reason: str
    ) -> None:
        result = screen_candidates([candidate], request(), corpus, set())
        assert not result.accepted
        assert result.rejected[0]["reason"].startswith(reason)

    @pytest.mark.unit
    def test_existing_family_is_rejected_case_insensitively(
        self, corpus: CorpusIndex
    ) -> None:
        existing = corpus.rows[0]["application_prompt"]
        result = screen_candidates(
            [raw(prompt=existing.upper())], request(), corpus, set()
        )
        assert result.rejected[0]["reason"] == "duplicate_family"

    @pytest.mark.unit
    def test_prompt_extending_existing_prompt_is_rejected(
        self, corpus: CorpusIndex
    ) -> None:
        existing = corpus.rows[0]["application_prompt"]
        result = screen_candidates(
            [raw(prompt=existing + " Any tips?")], request(), corpus, set()
        )
        assert result.rejected[0]["reason"].startswith("artifact:")

    @pytest.mark.unit
    def test_near_duplicate_within_batch(self, corpus: CorpusIndex) -> None:
        other = (
            "My neighbourhood association keeps asking me to run events "
            "and I say yes too often?"
        )
        result = screen_candidates([raw(), raw(prompt=other)], request(), corpus, set())
        assert len(result.accepted) == 1
        assert result.rejected[0]["reason"] == "near_duplicate_in_batch"

    @pytest.mark.unit
    def test_variant_too_similar_to_parent(self, corpus: CorpusIndex) -> None:
        parent = corpus.rows[0]
        req = DraftRequest(cell=cell_of(parent), count=1, examples=[], parent=parent)
        result = screen_candidates(
            [raw(response=parent["ideal_response_final"])], req, corpus, set()
        )
        assert result.rejected[0]["reason"] == "too_similar_to_parent"

    @pytest.mark.unit
    def test_variant_carries_parent_prompt(self, corpus: CorpusIndex) -> None:
        parent = corpus.rows[0]
        req = DraftRequest(cell=cell_of(parent), count=1, examples=[], parent=parent)
        result = screen_candidates([raw()], req, corpus, set())
        assert (
            result.accepted[0]["parent_application_prompt"]
            == parent["application_prompt"]
        )


# ============================================================================
# Drafting loop and requests
# ============================================================================


class TestRunDrafting:
    @pytest.mark.unit
    def test_writes_accepted_and_rejected(
        self, tmp_path: Path, corpus: CorpusIndex
    ) -> None:
        out = tmp_path / "staged" / "cands.jsonl"

        def fake(system: str, user: str, n: int) -> list[dict[str, Any]]:
            assert "Hard rules" in system
            return [raw(), raw(response=CLEAN_RESPONSE + " Sound right?")]

        summary = run_drafting(
            [request(count=2)],
            fake,
            corpus=corpus,
            output=out,
            system_prompt=build_system_prompt(),
        )
        assert (
            summary.requested,
            summary.drafted,
            summary.accepted,
            summary.rejected,
        ) == (2, 2, 1, 1)
        assert summary.reasons == {"trailing_question": 1}
        assert len(out.read_text("utf-8").splitlines()) == 1
        rejected = json.loads(rejected_path_for(out).read_text("utf-8").splitlines()[0])
        assert rejected["reason"] == "trailing_question"

    @pytest.mark.unit
    def test_dedups_across_requests(self, tmp_path: Path, corpus: CorpusIndex) -> None:
        out = tmp_path / "cands.jsonl"
        summary = run_drafting(
            [request(count=1), request(count=1)],
            lambda s, u, n: [raw()],
            corpus=corpus,
            output=out,
            system_prompt="sys",
        )
        assert summary.accepted == 1
        assert summary.reasons == {"duplicate_family": 1}

    @pytest.mark.unit
    def test_empty_draft_is_tolerated(
        self, tmp_path: Path, corpus: CorpusIndex
    ) -> None:
        summary = run_drafting(
            [request()],
            lambda s, u, n: [],
            corpus=corpus,
            output=tmp_path / "c.jsonl",
            system_prompt="s",
        )
        assert summary.drafted == 0 and summary.accepted == 0
        assert not (tmp_path / "c.jsonl").exists()


class TestBuildRequests:
    @pytest.mark.unit
    def test_new_families(self, corpus: CorpusIndex) -> None:
        shortfalls = compute_shortfalls(
            load_split_counts({"splits": {"train": {}}}), DEFAULT_FLOORS
        )
        requests = build_requests(
            MODE_NEW_FAMILIES,
            corpus=corpus,
            count=12,
            batch_size=5,
            examples_per_request=3,
            seed=1,
            shortfalls=shortfalls,
        )
        assert [r.count for r in requests] == [5, 5, 2]
        assert all(len(r.examples) == 3 and r.parent is None for r in requests)

    @pytest.mark.unit
    def test_variants(self, corpus: CorpusIndex) -> None:
        requests = build_requests(
            MODE_VARIANTS,
            corpus=corpus,
            count=4,
            batch_size=5,
            examples_per_request=2,
            seed=1,
            parents=corpus.rows[:5],
            per_parent=2,
        )
        assert [r.count for r in requests] == [2, 2]
        for r in requests:
            assert r.parent is not None
            assert r.cell == cell_of(r.parent)
            assert r.parent["icdu_id"] not in {e["icdu_id"] for e in r.examples}

    @pytest.mark.unit
    def test_bad_mode_and_missing_shortfalls(self, corpus: CorpusIndex) -> None:
        with pytest.raises(BuildPlanError):
            build_requests(
                "nope",
                corpus=corpus,
                count=1,
                batch_size=1,
                examples_per_request=1,
                seed=0,
            )
        with pytest.raises(BuildPlanError):
            build_requests(
                MODE_NEW_FAMILIES,
                corpus=corpus,
                count=1,
                batch_size=1,
                examples_per_request=1,
                seed=0,
            )


class TestCli:
    @pytest.mark.unit
    def test_parse_floors(self) -> None:
        assert parse_floors("") == DEFAULT_FLOORS
        assert parse_floors("personas=5,capabilities=9")["personas"] == 5
        with pytest.raises(BuildPlanError):
            parse_floors("colours=3")
        with pytest.raises(BuildPlanError):
            parse_floors("personas=lots")

    @pytest.mark.unit
    def test_dry_run_prints_plan_without_api(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(DATASETS_DIR.parents[2])
        code = main(
            ["--count", "7", "--output", str(tmp_path / "x.jsonl"), "--dry-run"]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert out.count("new_family") == 2
        assert not (tmp_path / "x.jsonl").exists()


# ============================================================================
# SDK boundary
# ============================================================================


def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a stand-in ``anthropic`` module; returns it with call/error hooks."""
    module = types.ModuleType("anthropic")
    state: dict[str, Any] = {"calls": [], "response": None, "error": None}

    class APIStatusError(Exception):
        status_code = 500
        message = "boom"

    class RateLimitError(APIStatusError):
        pass

    class APIConnectionError(Exception):
        pass

    class _Messages:
        def parse(self, **kwargs: Any) -> Any:
            state["calls"].append(kwargs)
            if state["error"] is not None:
                raise state["error"]
            return state["response"]

    class Anthropic:
        def __init__(self, **kwargs: Any) -> None:
            self.messages = _Messages()

    module.Anthropic = Anthropic  # type: ignore[attr-defined]
    module.APIStatusError = APIStatusError  # type: ignore[attr-defined]
    module.RateLimitError = RateLimitError  # type: ignore[attr-defined]
    module.APIConnectionError = APIConnectionError  # type: ignore[attr-defined]
    module.state = state  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return module


class TestAnthropicDrafter:
    @pytest.mark.unit
    def test_missing_package_gives_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "anthropic", None)
        with pytest.raises(BuildPlanError, match="pip install"):
            AnthropicDrafter()

    @pytest.mark.unit
    def test_uses_structured_output_and_cached_system(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _install_fake_anthropic(monkeypatch)
        fake.state["response"] = types.SimpleNamespace(
            stop_reason="end_turn",
            parsed_output=DraftBatch(
                candidates=[
                    DraftCandidate(
                        application_prompt="p",
                        ideal_response_final="r",
                        ideal_response_attributes=["Clear"],
                    )
                ]
            ),
        )
        drafter = AnthropicDrafter(model="claude-opus-5", max_tokens=123)
        out = drafter("SYS", "USER", 1)
        assert out == [
            {
                "application_prompt": "p",
                "ideal_response_final": "r",
                "ideal_response_attributes": ["Clear"],
            }
        ]
        (call,) = fake.state["calls"]
        assert call["model"] == "claude-opus-5" and call["max_tokens"] == 123
        assert call["output_format"] is DraftBatch
        assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert call["messages"] == [{"role": "user", "content": "USER"}]

    @pytest.mark.unit
    @pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
    def test_refusal_and_truncation_return_nothing(
        self, monkeypatch: pytest.MonkeyPatch, stop_reason: str
    ) -> None:
        fake = _install_fake_anthropic(monkeypatch)
        fake.state["response"] = types.SimpleNamespace(
            stop_reason=stop_reason, parsed_output=None, stop_details=None
        )
        assert AnthropicDrafter()("s", "u", 1) == []

    @pytest.mark.unit
    def test_api_errors_are_logged_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _install_fake_anthropic(monkeypatch)
        drafter = AnthropicDrafter()
        for exc_class in (
            fake.RateLimitError,
            fake.APIStatusError,
            fake.APIConnectionError,
        ):
            fake.state["error"] = exc_class()
            assert drafter("s", "u", 1) == []
