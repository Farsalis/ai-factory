"""Tests for the deterministic ICDU dataset builder and its validation gates.

The strongest guarantees are pinned against the published v9 artifacts: the
builder must reproduce the three v9 split files byte-for-byte and re-derive
every v9 label from its persona/principle/capability triple. If either breaks,
v9 and later versions were not built by the same rules.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from src.data.build_icdu_dataset import (
    CAPABILITY_LAYERS,
    GOVERNING_PRINCIPLES,
    ICDU_FIELD_ORDER,
    PERSONA_INTENTS,
    RESPONSE_ATTRIBUTES,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    BuildError,
    BuildPaths,
    LedgerEntry,
    SplitLedger,
    allocate_new_family_splits,
    build_dataset,
    build_icdu_record,
    canonical_family_hash,
    content_sha256,
    derived_fields_match,
    detect_prompt_artifacts,
    load_jsonl,
    normalise_attributes,
    parse_staged_candidate,
    sha256_text,
    write_jsonl,
)

DATASETS_DIR = Path(__file__).resolve().parents[1] / "src" / "data" / "datasets"
BASE_VERSION = 9
NEXT_VERSION = 10

V9_SPLIT_FILES = {
    SPLIT_TRAIN: "icdu_training_data_v9.jsonl",
    SPLIT_VALIDATION: "icdu_validation_data_v9.jsonl",
    SPLIT_TEST: "icdu_test_data_v9.jsonl",
}

# A well-formed staged candidate: clean prose, no mechanical suffix, response
# does not end in a question.
CLEAN_PROMPT = (
    "I keep volunteering for extra committee work and then resenting it. "
    "How do I decide what to take on?"
)
CLEAN_RESPONSE = (
    "Resentment is usually a signal that a commitment was made without a "
    "boundary. Start with Clarity: write down what you actually want your "
    "contribution to be this year, in one sentence. Then treat every request "
    "as a comparison against that sentence rather than a yes-or-no question. "
    "When something does not match, you are not refusing the person; you are "
    "protecting the commitment you already made."
)


# ============================================================================
# Fixtures and helpers
# ============================================================================


@pytest.fixture(scope="module")
def v9_records() -> dict[str, list[dict[str, Any]]]:
    """Published v9 records keyed by split."""
    return {
        split: load_jsonl(DATASETS_DIR / name) for split, name in V9_SPLIT_FILES.items()
    }


@pytest.fixture(scope="module")
def v9_lineage() -> list[dict[str, Any]]:
    """Published v9 lineage records."""
    return load_jsonl(DATASETS_DIR / "icdu_v9_source_lineage.jsonl")


@pytest.fixture(scope="module")
def v9_manifest() -> dict[str, Any]:
    """Published v9 manifest."""
    path = DATASETS_DIR / "icdu_v9_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def staged_candidate(
    prompt: str = CLEAN_PROMPT,
    response: str = CLEAN_RESPONSE,
    persona: str = "Community Participant > Contributor",
    principle: str = "Chapter 1 > Clarity",
    capability: str = "Foundational",
    **extra: Any,
) -> dict[str, Any]:
    """Build a staged candidate payload with sensible defaults."""
    return {
        "application_prompt": prompt,
        "ideal_response_final": response,
        "persona_archetype": persona,
        "governing_principle": principle,
        "capability_layer": capability,
        **extra,
    }


def write_staging(tmp_path: Path, candidates: list[dict[str, Any]]) -> Path:
    """Write candidates to a staging JSONL file and return its path."""
    path = tmp_path / "staged.jsonl"
    path.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in candidates),
        encoding="utf-8",
    )
    return path


def run_build(
    tmp_path: Path,
    staged: list[Path] | None = None,
    version: int = NEXT_VERSION,
    **kwargs: Any,
) -> Any:
    """Run a build against the real v9 base, writing into ``tmp_path``."""
    output_dir = tmp_path / f"out-v{version}"
    paths = BuildPaths(
        base_dir=DATASETS_DIR,
        output_dir=output_dir,
        base_version=BASE_VERSION,
        version=version,
    )
    return build_dataset(
        paths=paths,
        staged_files=staged or [],
        build_seed=f"test-seed-v{version}",
        **kwargs,
    )


# ============================================================================
# Hashing and format contract
# ============================================================================


class TestHashingContract:
    """Family/prompt hashing must match what v9 recorded."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("How do I start?", "how do i start?"),
            ("How do I start?", "  How   do I start?  "),
            ("How do I start?", "HOW DO I START?"),
        ],
    )
    def test_family_hash_ignores_case_and_whitespace(
        self, left: str, right: str
    ) -> None:
        assert canonical_family_hash(left) == canonical_family_hash(right)

    @pytest.mark.unit
    def test_family_hash_distinguishes_different_prompts(self) -> None:
        assert canonical_family_hash("How do I start?") != canonical_family_hash(
            "How do I stop?"
        )

    @pytest.mark.unit
    def test_family_hash_matches_published_v9_lineage(
        self,
        v9_records: dict[str, list[dict[str, Any]]],
        v9_lineage: list[dict[str, Any]],
    ) -> None:
        by_id = {row["icdu_id"]: row for row in v9_lineage}
        checked = 0
        for records in v9_records.values():
            for record in records:
                lineage = by_id[record["icdu_id"]]
                assert (
                    canonical_family_hash(record["application_prompt"])
                    == lineage["canonical_family_sha256"]
                )
                checked += 1
        assert checked == 354

    @pytest.mark.unit
    def test_text_hashes_match_published_v9_lineage(
        self,
        v9_records: dict[str, list[dict[str, Any]]],
        v9_lineage: list[dict[str, Any]],
    ) -> None:
        by_id = {row["icdu_id"]: row for row in v9_lineage}
        for records in v9_records.values():
            for record in records:
                lineage = by_id[record["icdu_id"]]
                assert (
                    sha256_text(record["application_prompt"])
                    == (lineage["prompt_sha256"])
                )
                assert (
                    sha256_text(record["ideal_response_final"])
                    == (lineage["response_sha256"])
                )

    @pytest.mark.unit
    def test_content_hash_is_newline_independent(self, tmp_path: Path) -> None:
        crlf, lf = tmp_path / "crlf.jsonl", tmp_path / "lf.jsonl"
        crlf.write_bytes(b'{"a":1}\r\n{"b":2}\r\n')
        lf.write_bytes(b'{"a":1}\n{"b":2}\n')
        assert content_sha256(crlf) == content_sha256(lf)

    @pytest.mark.unit
    def test_content_hash_matches_published_v9_manifest(
        self, v9_manifest: dict[str, Any]
    ) -> None:
        for name, expected in v9_manifest["v9_file_sha256"].items():
            assert content_sha256(DATASETS_DIR / name) == expected, name

    @pytest.mark.unit
    def test_write_jsonl_round_trips_published_bytes(
        self, tmp_path: Path, v9_records: dict[str, list[dict[str, Any]]]
    ) -> None:
        for split, name in V9_SPLIT_FILES.items():
            out = tmp_path / name
            write_jsonl(v9_records[split], out)
            assert out.read_bytes() == (DATASETS_DIR / name).read_bytes(), name


# ============================================================================
# Metadata derivation
# ============================================================================


class TestMetadataDerivation:
    """Derived fields must follow from the three chosen labels, as in v9."""

    @pytest.mark.unit
    def test_derivation_reproduces_every_published_v9_record(
        self, v9_records: dict[str, list[dict[str, Any]]]
    ) -> None:
        for records in v9_records.values():
            for record in records:
                rebuilt = build_icdu_record(
                    prompt=record["application_prompt"],
                    response=record["ideal_response_final"],
                    persona=record["persona_archetype"],
                    principle=record["governing_principle"],
                    capability=record["capability_layer"],
                    attributes=record["ideal_response_attributes"],
                    icdu_id=record["icdu_id"],
                )
                assert rebuilt == record, record["icdu_id"]

    @pytest.mark.unit
    def test_published_v9_records_are_self_consistent(
        self, v9_records: dict[str, list[dict[str, Any]]]
    ) -> None:
        for records in v9_records.values():
            for record in records:
                assert derived_fields_match(record) == []

    @pytest.mark.unit
    def test_record_uses_canonical_field_order(self) -> None:
        record = build_icdu_record(
            prompt=CLEAN_PROMPT,
            response=CLEAN_RESPONSE,
            persona="Learner > Skill Builder",
            principle="Chapter 2 > Process",
            capability="Foundational",
        )
        assert tuple(record.keys()) == ICDU_FIELD_ORDER

    @pytest.mark.unit
    def test_capability_article_agrees_for_aspirational(self) -> None:
        record = build_icdu_record(
            prompt=CLEAN_PROMPT,
            response=CLEAN_RESPONSE,
            persona="Learner > Skill Builder",
            principle="Chapter 2 > Process",
            capability="Aspirational",
        )
        assert (
            "through an aspirational capability lens."
            in record["ideal_response_cot"][1]
        )

    @pytest.mark.unit
    def test_context_summary_embeds_governing_principle(self) -> None:
        for principle in GOVERNING_PRINCIPLES:
            record = build_icdu_record(
                prompt=CLEAN_PROMPT,
                response=CLEAN_RESPONSE,
                persona="Learner > Skill Builder",
                principle=principle,
                capability="Foundational",
            )
            assert principle in record["context_summary"]

    @pytest.mark.unit
    def test_every_persona_maps_to_a_known_context_topic(self) -> None:
        for persona in PERSONA_INTENTS:
            record = build_icdu_record(
                prompt=CLEAN_PROMPT,
                response=CLEAN_RESPONSE,
                persona=persona,
                principle="Chapter 1 > People",
                capability="Foundational",
            )
            assert record["user_intent"] == PERSONA_INTENTS[persona]
            assert record["context_summary"].startswith(
                "The user is seeking practical, encouraging guidance for "
            )

    @pytest.mark.unit
    def test_hand_edited_derived_field_is_detected(self) -> None:
        record = build_icdu_record(
            prompt=CLEAN_PROMPT,
            response=CLEAN_RESPONSE,
            persona="Learner > Skill Builder",
            principle="Chapter 2 > Process",
            capability="Foundational",
        )
        record["context_summary"] = "The user wants something else entirely."
        record["user_intent"] = "To do whatever"
        assert set(derived_fields_match(record)) == {"context_summary", "user_intent"}


class TestClosedVocabularies:
    """Off-vocabulary labels must be rejected at construction time."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "override",
        [
            {"persona": "Watch Buyer > Alex"},
            {"principle": "Chapter 4 > Vibes"},
            {"capability": "Legendary"},
        ],
    )
    def test_unknown_label_is_rejected(self, override: dict[str, str]) -> None:
        kwargs: dict[str, Any] = {
            "prompt": CLEAN_PROMPT,
            "response": CLEAN_RESPONSE,
            "persona": "Learner > Skill Builder",
            "principle": "Chapter 2 > Process",
            "capability": "Foundational",
            **override,
        }
        with pytest.raises(BuildError):
            build_icdu_record(**kwargs)

    @pytest.mark.unit
    def test_unknown_attribute_is_rejected(self) -> None:
        with pytest.raises(BuildError, match="ideal_response_attributes"):
            normalise_attributes(["Clear", "Sassy"])

    @pytest.mark.unit
    def test_attributes_are_canonically_ordered(self) -> None:
        result = normalise_attributes(["Encouraging", "Structured", "Clear"])
        assert result == [a for a in RESPONSE_ATTRIBUTES if a in result]

    @pytest.mark.unit
    def test_required_attributes_are_always_present(self) -> None:
        assert normalise_attributes(["Encouraging"])[:2] == ["Clear", "Actionable"]
        assert normalise_attributes(None) == ["Clear", "Actionable"]


# ============================================================================
# Ledger
# ============================================================================


class TestSplitLedger:
    """The ledger is the authority on which split a family belongs to."""

    @pytest.mark.unit
    def test_reads_published_v9_versioned_split_key(
        self, v9_lineage: list[dict[str, Any]]
    ) -> None:
        ledger = SplitLedger.from_lineage(v9_lineage)
        assert len(ledger) == 354
        sealed = [row for row in v9_lineage if row["v9_split"] == SPLIT_TEST]
        assert len(sealed) == 54
        for row in sealed:
            entry = ledger.get(row["canonical_family_sha256"])
            assert entry is not None
            assert entry.split == SPLIT_TEST

    @pytest.mark.unit
    def test_conflicting_split_for_one_family_is_rejected(self) -> None:
        rows = [
            {
                "icdu_id": "a",
                "canonical_family_sha256": "f" * 64,
                "split": SPLIT_TRAIN,
            },
            {
                "icdu_id": "b",
                "canonical_family_sha256": "f" * 64,
                "split": SPLIT_TEST,
            },
        ]
        with pytest.raises(BuildError, match="Ledger conflict"):
            SplitLedger.from_lineage(rows)

    @pytest.mark.unit
    def test_missing_split_key_is_rejected(self) -> None:
        with pytest.raises(BuildError, match="no split key"):
            SplitLedger.from_lineage(
                [{"icdu_id": "a", "canonical_family_sha256": "f" * 64}]
            )

    @pytest.mark.unit
    def test_readding_a_family_is_rejected(self) -> None:
        ledger = SplitLedger()
        entry = LedgerEntry(split=SPLIT_TRAIN, icdu_id="a", lineage={})
        ledger.add("f" * 64, entry)
        with pytest.raises(BuildError, match="already in the ledger"):
            ledger.add("f" * 64, entry)


# ============================================================================
# Split allocation
# ============================================================================


class TestSplitAllocation:
    """New families are allocated deterministically and stratified by persona."""

    def _candidates(self, count: int, personas: list[str]) -> list[Any]:
        return [
            parse_staged_candidate(
                staged_candidate(
                    prompt=f"Novel scenario number {i} needing guidance?",
                    persona=personas[i % len(personas)],
                ),
                "staged.jsonl",
                i + 1,
            )
            for i in range(count)
        ]

    @pytest.mark.unit
    def test_ratio_is_respected(self) -> None:
        candidates = self._candidates(
            20, ["Learner > Skill Builder", "Organizer > Systems Builder"]
        )
        assignments = allocate_new_family_splits(candidates, 0.25)
        assert sum(1 for s in assignments.values() if s == SPLIT_VALIDATION) == 5

    @pytest.mark.unit
    def test_allocation_is_deterministic(self) -> None:
        candidates = self._candidates(
            17, ["Learner > Skill Builder", "Organizer > Systems Builder"]
        )
        first = allocate_new_family_splits(candidates, 0.3)
        second = allocate_new_family_splits(list(reversed(candidates)), 0.3)
        assert first == second

    @pytest.mark.unit
    def test_no_new_family_is_routed_to_the_sealed_test_split(self) -> None:
        candidates = self._candidates(12, ["Learner > Skill Builder"])
        assignments = allocate_new_family_splits(candidates, 0.5)
        assert SPLIT_TEST not in set(assignments.values())

    @pytest.mark.unit
    def test_each_persona_contributes_to_validation(self) -> None:
        personas = [
            "Learner > Skill Builder",
            "Organizer > Systems Builder",
            "Educator > Teaching Innovator",
        ]
        candidates = self._candidates(30, personas)
        assignments = allocate_new_family_splits(candidates, 0.5)
        by_persona: dict[str, set[str]] = {p: set() for p in personas}
        for candidate in candidates:
            persona = candidate.record["persona_archetype"]
            by_persona[persona].add(assignments[candidate.family_hash])
        for persona, splits in by_persona.items():
            assert SPLIT_VALIDATION in splits, persona

    @pytest.mark.unit
    @pytest.mark.parametrize("ratio", [-0.1, 1.5])
    def test_invalid_ratio_is_rejected(self, ratio: float) -> None:
        with pytest.raises(BuildError, match="validation_ratio"):
            allocate_new_family_splits(
                self._candidates(2, ["Learner > Skill Builder"]), ratio
            )


# ============================================================================
# Staged candidate parsing
# ============================================================================


class TestStagedCandidateParsing:
    """Staging input is validated before it can reach the corpus."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "missing", ["application_prompt", "persona_archetype", "capability_layer"]
    )
    def test_missing_required_field_is_rejected(self, missing: str) -> None:
        payload = staged_candidate()
        del payload[missing]
        with pytest.raises(BuildError, match="missing fields"):
            parse_staged_candidate(payload, "staged.jsonl", 1)

    @pytest.mark.unit
    def test_unknown_field_is_rejected(self) -> None:
        with pytest.raises(BuildError, match="unknown fields"):
            parse_staged_candidate(
                staged_candidate(icdu_id="hand-picked"), "staged.jsonl", 1
            )

    @pytest.mark.unit
    def test_new_family_candidate_has_own_group(self) -> None:
        candidate = parse_staged_candidate(staged_candidate(), "staged.jsonl", 3)
        assert candidate.kind == "new_family"
        assert candidate.group_hash == candidate.family_hash
        assert candidate.parent_family_hash is None
        assert candidate.source_line == 3

    @pytest.mark.unit
    def test_variant_group_is_derived_from_parent_prompt(self) -> None:
        parent = "My team keeps missing deadlines. Where do I start?"
        candidate = parse_staged_candidate(
            staged_candidate(parent_application_prompt=parent), "staged.jsonl", 1
        )
        assert candidate.kind == "variant"
        assert candidate.group_hash == canonical_family_hash(parent)
        assert candidate.group_hash != candidate.family_hash

    @pytest.mark.unit
    def test_variant_identical_to_parent_is_rejected(self) -> None:
        with pytest.raises(BuildError, match="identical to its parent"):
            parse_staged_candidate(
                staged_candidate(parent_application_prompt=CLEAN_PROMPT),
                "staged.jsonl",
                1,
            )

    @pytest.mark.unit
    def test_disagreeing_parent_prompt_and_hash_are_rejected(self) -> None:
        with pytest.raises(BuildError, match="disagree"):
            parse_staged_candidate(
                staged_candidate(
                    parent_application_prompt="Some other parent prompt?",
                    parent_canonical_family_sha256="f" * 64,
                ),
                "staged.jsonl",
                1,
            )


# ============================================================================
# Artifact detection
# ============================================================================


class TestArtifactDetection:
    """The v8 augmentation failure modes must be caught on sight."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "suffix",
        [
            "What if budget wasn't an issue?",
            "How to track progress?",
            "Adjust for my region?",
        ],
    )
    def test_mechanical_suffix_is_flagged(self, suffix: str) -> None:
        findings = detect_prompt_artifacts(f"I want to get fit. {suffix}", set())
        assert any("mechanical v8 suffix" in f for f in findings)

    @pytest.mark.unit
    def test_prompt_extending_an_existing_prompt_is_flagged(self) -> None:
        existing = "I keep missing my workouts."
        findings = detect_prompt_artifacts(
            f"{existing} What should I change first?", {existing}
        )
        assert any("extends existing prompt" in f for f in findings)

    @pytest.mark.unit
    def test_clean_prompt_is_not_flagged(self) -> None:
        assert detect_prompt_artifacts(CLEAN_PROMPT, {"Unrelated prompt."}) == []


# ============================================================================
# End-to-end builds
# ============================================================================


class TestBaseReproduction:
    """Rebuilding the base version must be a no-op on the published bytes."""

    @pytest.mark.integration
    def test_rebuild_reproduces_published_v9_split_files(self, tmp_path: Path) -> None:
        result = run_build(tmp_path, version=BASE_VERSION)
        assert result.report.status == "PASS", result.report.errors
        for name in V9_SPLIT_FILES.values():
            assert (tmp_path / f"out-v{BASE_VERSION}" / name).read_bytes() == (
                DATASETS_DIR / name
            ).read_bytes(), name

    @pytest.mark.integration
    def test_rebuild_reports_published_v9_counts(self, tmp_path: Path) -> None:
        result = run_build(tmp_path, version=BASE_VERSION)
        counts = {split: len(rows) for split, rows in result.records.items()}
        assert counts == {SPLIT_TRAIN: 240, SPLIT_VALIDATION: 60, SPLIT_TEST: 54}
        assert result.report.details["canonical_families"] == 354
        assert result.report.details["exact_prompt_overlap"] == {
            "train_vs_validation": 0,
            "train_vs_test": 0,
            "validation_vs_test": 0,
        }
        assert result.report.details["known_augmentation_artifacts"] == 0

    @pytest.mark.integration
    def test_published_question_ending_response_is_reported_not_failed(
        self, tmp_path: Path
    ) -> None:
        # v9 train contains exactly one response ending in a question. It must
        # be surfaced without failing the build of already-published content.
        result = run_build(tmp_path, version=BASE_VERSION)
        assert result.report.status == "PASS"
        assert (
            result.report.details["splits"][SPLIT_TRAIN]["responses_ending_in_question"]
            == 1
        )

    @pytest.mark.integration
    def test_manifest_hashes_verify_against_written_files(self, tmp_path: Path) -> None:
        result = run_build(tmp_path, version=NEXT_VERSION)
        hashes = result.manifest[f"v{NEXT_VERSION}_file_sha256"]
        assert hashes
        out_dir = tmp_path / f"out-v{NEXT_VERSION}"
        for name, expected in hashes.items():
            assert content_sha256(out_dir / name) == expected, name

    @pytest.mark.integration
    def test_build_is_byte_for_byte_deterministic(self, tmp_path: Path) -> None:
        staged = write_staging(
            tmp_path,
            [
                staged_candidate(),
                staged_candidate(
                    prompt="My side project has stalled for three months. What now?",
                    persona="Creative Practitioner > Blocked Creator",
                    principle="Chapter 2 > Process",
                    response=(
                        "A stall is information, not a verdict. Pick the smallest "
                        "piece of the project that can be finished in one sitting "
                        "and finish only that. Momentum is rebuilt by completion, "
                        "not by planning."
                    ),
                ),
            ],
        )
        first = run_build(tmp_path / "a", [staged])
        second = run_build(tmp_path / "b", [staged])
        assert first.report.status == "PASS", first.report.errors
        assert second.report.status == "PASS", second.report.errors
        by_name = {path.name: path for path in second.written}
        assert {path.name for path in first.written} == set(by_name)
        for path in first.written:
            assert path.read_bytes() == by_name[path.name].read_bytes(), path.name


class TestSplitIntegrityGates:
    """Growth must not move existing rows or leak across splits."""

    @pytest.mark.integration
    def test_existing_families_keep_their_v9_split(self, tmp_path: Path) -> None:
        staged = write_staging(tmp_path, [staged_candidate()])
        result = run_build(tmp_path, [staged])
        assert result.report.status == "PASS", result.report.errors
        published = {
            split: {r["icdu_id"] for r in load_jsonl(DATASETS_DIR / name)}
            for split, name in V9_SPLIT_FILES.items()
        }
        for split, records in result.records.items():
            ids = {r["icdu_id"] for r in records}
            assert published[split] <= ids, split

    @pytest.mark.integration
    def test_sealed_test_split_is_unchanged_by_default(self, tmp_path: Path) -> None:
        staged = write_staging(tmp_path, [staged_candidate()])
        result = run_build(tmp_path, [staged])
        assert result.report.details["sealed_test_unchanged"] is True
        assert len(result.records[SPLIT_TEST]) == 54

    @pytest.mark.integration
    def test_new_family_lands_in_train_or_validation(self, tmp_path: Path) -> None:
        staged = write_staging(tmp_path, [staged_candidate()])
        result = run_build(tmp_path, [staged], validation_ratio=0.0)
        new_ids = {
            row["icdu_id"]
            for row in result.lineage
            if row["record_kind"] == "new_family"
        }
        assert len(new_ids) == 1
        train_ids = {r["icdu_id"] for r in result.records[SPLIT_TRAIN]}
        assert new_ids <= train_ids

    @pytest.mark.integration
    def test_variant_inherits_its_parents_split(self, tmp_path: Path) -> None:
        parent = load_jsonl(DATASETS_DIR / V9_SPLIT_FILES[SPLIT_VALIDATION])[0]
        staged = write_staging(
            tmp_path,
            [
                staged_candidate(
                    prompt=(
                        "A different scenario in the same territory: what is the "
                        "first concrete step when everything feels urgent?"
                    ),
                    response=(
                        "Rank by consequence, not by noise. Write the three items "
                        "whose delay costs the most, then do the cheapest one "
                        "today so the list starts shrinking."
                    ),
                    persona=parent["persona_archetype"],
                    principle=parent["governing_principle"],
                    capability=parent["capability_layer"],
                    parent_application_prompt=parent["application_prompt"],
                )
            ],
        )
        result = run_build(tmp_path, [staged])
        assert result.report.status == "PASS", result.report.errors
        variant = next(row for row in result.lineage if row["record_kind"] == "variant")
        assert variant["split"] == SPLIT_VALIDATION
        assert variant["parent_icdu_id"] == parent["icdu_id"]
        assert variant["family_group_sha256"] == canonical_family_hash(
            parent["application_prompt"]
        )
        validation_ids = {r["icdu_id"] for r in result.records[SPLIT_VALIDATION]}
        assert variant["icdu_id"] in validation_ids

    @pytest.mark.integration
    def test_duplicate_of_an_existing_family_is_rejected(self, tmp_path: Path) -> None:
        existing = load_jsonl(DATASETS_DIR / V9_SPLIT_FILES[SPLIT_TRAIN])[0]
        staged = write_staging(
            tmp_path,
            [staged_candidate(prompt=existing["application_prompt"].upper())],
        )
        result = run_build(tmp_path, [staged])
        assert result.report.status == "FAIL"
        assert any("already in the corpus" in e for e in result.report.errors)

    @pytest.mark.integration
    def test_variant_of_an_unknown_parent_is_rejected(self, tmp_path: Path) -> None:
        staged = write_staging(
            tmp_path,
            [
                staged_candidate(
                    parent_application_prompt="A prompt that is not in the corpus."
                )
            ],
        )
        result = run_build(tmp_path, [staged])
        assert result.report.status == "FAIL"
        assert any("not in the ledger" in e for e in result.report.errors)


class TestQualityGates:
    """Near-duplicates and artifacts must block the build."""

    @pytest.mark.integration
    def test_near_duplicate_variant_is_rejected(self, tmp_path: Path) -> None:
        parent = load_jsonl(DATASETS_DIR / V9_SPLIT_FILES[SPLIT_TRAIN])[0]
        staged = write_staging(
            tmp_path,
            [
                staged_candidate(
                    prompt=parent["application_prompt"] + " Any thoughts on that?",
                    response=parent["ideal_response_final"],
                    persona=parent["persona_archetype"],
                    principle=parent["governing_principle"],
                    capability=parent["capability_layer"],
                    parent_application_prompt=parent["application_prompt"],
                )
            ],
        )
        result = run_build(tmp_path, [staged])
        assert result.report.status == "FAIL"
        assert any("similar to its" in e for e in result.report.errors)

    @pytest.mark.integration
    def test_sibling_variants_that_are_near_copies_are_rejected(
        self, tmp_path: Path
    ) -> None:
        parent = load_jsonl(DATASETS_DIR / V9_SPLIT_FILES[SPLIT_TRAIN])[0]
        shared = (
            "Start with the smallest repeatable action you can do tomorrow, then "
            "let the identity follow the behaviour rather than the reverse."
        )
        staged = write_staging(
            tmp_path,
            [
                staged_candidate(
                    prompt="One fresh angle on this situation, what comes first?",
                    response=shared,
                    parent_application_prompt=parent["application_prompt"],
                ),
                staged_candidate(
                    prompt="Another fresh angle here, where would you begin?",
                    response=shared + " That is the whole trick.",
                    parent_application_prompt=parent["application_prompt"],
                ),
            ],
        )
        result = run_build(tmp_path, [staged])
        assert result.report.status == "FAIL"
        assert any("sibling" in e for e in result.report.errors)

    @pytest.mark.integration
    def test_mechanically_suffixed_new_prompt_is_rejected(self, tmp_path: Path) -> None:
        staged = write_staging(
            tmp_path,
            [staged_candidate(prompt="I want to eat better. How to track progress?")],
        )
        result = run_build(tmp_path, [staged])
        assert result.report.status == "FAIL"
        assert any("augmentation artifact" in e for e in result.report.errors)

    @pytest.mark.integration
    def test_new_response_ending_in_a_question_is_rejected(
        self, tmp_path: Path
    ) -> None:
        staged = write_staging(
            tmp_path,
            [
                staged_candidate(
                    response="Start with one small change. What feels easiest?"
                )
            ],
        )
        result = run_build(tmp_path, [staged])
        assert result.report.status == "FAIL"
        assert any("ends in a question" in e for e in result.report.errors)

    @pytest.mark.integration
    def test_tool_scaffolding_in_a_response_is_rejected(self, tmp_path: Path) -> None:
        staged = write_staging(
            tmp_path,
            [
                staged_candidate(
                    response=(
                        'Need data: {"tool_call": {"name": "task_tracker_tool"}} '
                        "Tool result: Task added. So begin with one small change."
                    )
                )
            ],
        )
        result = run_build(tmp_path, [staged])
        assert result.report.status == "FAIL"
        assert any("tool scaffolding" in e for e in result.report.errors)


class TestFailureHandling:
    """A failed build must leave no artifacts behind."""

    @pytest.mark.integration
    def test_failed_validation_writes_nothing(self, tmp_path: Path) -> None:
        staged = write_staging(
            tmp_path,
            [staged_candidate(prompt="I want to eat better. How to track progress?")],
        )
        result = run_build(tmp_path, [staged])
        assert result.report.status == "FAIL"
        assert result.written == []
        assert not (tmp_path / f"out-v{NEXT_VERSION}").exists()

    @pytest.mark.integration
    def test_dry_run_validates_without_writing(self, tmp_path: Path) -> None:
        staged = write_staging(tmp_path, [staged_candidate()])
        result = run_build(tmp_path, [staged], dry_run=True)
        assert result.report.status == "PASS", result.report.errors
        assert result.written == []
        assert not (tmp_path / f"out-v{NEXT_VERSION}").exists()

    @pytest.mark.integration
    def test_missing_base_file_raises(self, tmp_path: Path) -> None:
        paths = BuildPaths(
            base_dir=tmp_path / "nope",
            output_dir=tmp_path / "out",
            base_version=BASE_VERSION,
            version=NEXT_VERSION,
        )
        with pytest.raises(BuildError, match="not found"):
            build_dataset(paths=paths, staged_files=[], build_seed="x")

    @pytest.mark.integration
    def test_targets_report_shortfall(self, tmp_path: Path) -> None:
        staged = write_staging(tmp_path, [staged_candidate()])
        result = run_build(
            tmp_path,
            [staged],
            targets={SPLIT_TRAIN: 528, SPLIT_VALIDATION: 126, SPLIT_TEST: 54},
        )
        targets = result.report.details["targets"]
        assert targets[SPLIT_TEST]["shortfall"] == 0
        assert targets[SPLIT_TRAIN]["shortfall"] == 528 - len(
            result.records[SPLIT_TRAIN]
        )
        assert targets[SPLIT_VALIDATION]["target"] == 126


@pytest.mark.unit
def test_vocabulary_tables_are_complete_and_consistent() -> None:
    """The closed vocabularies must stay aligned with each other."""
    from src.data.build_icdu_dataset import INTENT_CONTEXT_TOPICS

    assert len(PERSONA_INTENTS) == 17
    assert len(GOVERNING_PRINCIPLES) == 9
    assert len(CAPABILITY_LAYERS) == 3
    assert set(PERSONA_INTENTS.values()) == set(INTENT_CONTEXT_TOPICS)
