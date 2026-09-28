"""Deterministic, reproducible builder for versioned ICDU datasets.

The v9 "clean core" dataset was published without its build script, so the
manifest hashes, split assignments and ``icdu_id`` values could not be
re-derived. This module is that missing builder, generalised so each version is
produced from the previous one plus a file of reviewed candidate records.

Design guarantees
-----------------
* **Split stability.** The previous version's lineage file is the authoritative
  ledger. A prompt family that already has a split keeps it forever, so the
  sealed test set stays byte-identical as the corpus grows.
* **Variant containment.** A variant inherits its parent's split, so paraphrases
  can never leak a training scenario into validation or test.
* **Derived metadata.** Only ``persona_archetype``, ``governing_principle`` and
  ``capability_layer`` are chosen per record. ``user_intent``,
  ``context_summary`` and ``ideal_response_cot`` are derived from them by the
  same rules v9 used, so labels cannot drift between versions.
* **Gated output.** Nothing is written unless every validation gate passes:
  closed-vocabulary labels, zero cross-split family overlap, no known
  augmentation artifacts, and no near-duplicate variants.

Carried-forward records keep the ``icdu_id`` recorded in the ledger. New records
are content-addressed with :data:`ICDU_ID_NAMESPACE`; v9's original id scheme is
not recoverable from the published artifacts, so ids are never recomputed for
records that already have one.

Usage:
    # Reproduce the base version into a scratch directory (integrity check).
    python -m src.data.build_icdu_dataset \\
        --base-version 9 --version 9 --output-dir /tmp/v9-check

    # Build v10 from v9 plus reviewed candidates.
    python -m src.data.build_icdu_dataset \\
        --base-version 9 --version 10 \\
        --staged src/data/datasets/staging/icdu_v10_candidates.jsonl \\
        --validation-ratio 0.314 \\
        --target-train 528 --target-validation 126 --target-test 54
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import logging
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

IcduRecord = dict[str, Any]
LineageRecord = dict[str, Any]

# ============================================================================
# Schema contract (recovered from the published v9 corpus)
# ============================================================================

#: Field order of every ICDU record, byte-for-byte as v9 wrote it.
ICDU_FIELD_ORDER: tuple[str, ...] = (
    "icdu_id",
    "persona_archetype",
    "governing_principle",
    "capability_layer",
    "user_intent",
    "context_summary",
    "application_prompt",
    "ideal_response_final",
    "ideal_response_attributes",
    "ideal_response_cot",
)

#: Persona archetype -> user intent. Strict 1:1; intent is never chosen freely.
PERSONA_INTENTS: dict[str, str] = {
    "AI or Technology User > Intentional Adopter": (
        "To adopt or use technology intentionally"
    ),
    "Career Professional > Workplace Navigator": (
        "To navigate a career or workplace challenge"
    ),
    "Community Participant > Contributor": (
        "To contribute effectively to a community effort"
    ),
    "Creative Practitioner > Blocked Creator": (
        "To make progress on a creative practice"
    ),
    "Educator > Teaching Innovator": "To improve a teaching or education outcome",
    "Entrepreneur > Business Builder": (
        "To make progress on a business-building challenge"
    ),
    "Financial Planner > Money Habit Builder": (
        "To improve a financial decision or money habit"
    ),
    "Framework Learner > Breaking Better Reader": (
        "To understand or apply the Breaking Better framework"
    ),
    "General User > Problem Solver": "To solve a personal-development challenge",
    "Learner > Skill Builder": "To build a skill or sustainable learning habit",
    "Manager or Team Lead > People Leader": (
        "To lead or support a team more effectively"
    ),
    "Organizer > Systems Builder": "To improve organization or time management",
    "Parent or Caregiver > Family Supporter": (
        "To navigate a parenting or caregiving challenge"
    ),
    "Pet Owner > Responsible Caregiver": "To make a responsible pet-care decision",
    "Planner or Host > Event Coordinator": "To plan or host an event successfully",
    "Relationship Navigator > Communication Builder": (
        "To improve a relationship or communication outcome"
    ),
    "Wellness Seeker > Sustainable Habit Builder": (
        "To build a sustainable wellness or health habit"
    ),
}

#: User intent -> the topic phrase (article included) used in ``context_summary``.
INTENT_CONTEXT_TOPICS: dict[str, str] = {
    "To adopt or use technology intentionally": "a technology-adoption decision",
    "To build a skill or sustainable learning habit": (
        "a learning or skill-building challenge"
    ),
    "To build a sustainable wellness or health habit": "a wellness or health habit",
    "To contribute effectively to a community effort": (
        "a community or volunteer challenge"
    ),
    "To improve a financial decision or money habit": (
        "a financial decision or money habit"
    ),
    "To improve a relationship or communication outcome": (
        "a relationship or communication challenge"
    ),
    "To improve a teaching or education outcome": "a teaching or education challenge",
    "To improve organization or time management": (
        "an organization or time-management challenge"
    ),
    "To lead or support a team more effectively": "a team-leadership challenge",
    "To make a responsible pet-care decision": "a pet-care decision",
    "To make progress on a business-building challenge": (
        "a business-building challenge"
    ),
    "To make progress on a creative practice": "a creative-practice challenge",
    "To navigate a career or workplace challenge": "a career or workplace challenge",
    "To navigate a parenting or caregiving challenge": (
        "a parenting or caregiving challenge"
    ),
    "To plan or host an event successfully": "a planning or hosting challenge",
    "To solve a personal-development challenge": "a personal-development challenge",
    "To understand or apply the Breaking Better framework": (
        "a framework-learning question"
    ),
}

#: The nine Breaking Better 3x3 cells usable as a governing principle.
GOVERNING_PRINCIPLES: tuple[str, ...] = (
    "Chapter 1 > Clarity",
    "Chapter 1 > People",
    "Chapter 1 > Transparency",
    "Chapter 2 > People",
    "Chapter 2 > Process",
    "Chapter 2 > Tools",
    "Chapter 3 > Aspirational",
    "Chapter 3 > Foundational",
    "Chapter 3 > Transformational",
)

CAPABILITY_LAYERS: tuple[str, ...] = (
    "Foundational",
    "Transformational",
    "Aspirational",
)

#: Response attributes in canonical order. Emitted order always follows this
#: tuple, so attribute lists are comparable across records and versions.
RESPONSE_ATTRIBUTES: tuple[str, ...] = (
    "Clear",
    "Actionable",
    "Empathetic",
    "Structured",
    "Principle-driven",
    "Encouraging",
)

#: Attributes every record asserts, and the default when none are supplied.
REQUIRED_RESPONSE_ATTRIBUTES: tuple[str, ...] = ("Clear", "Actionable")

CONTEXT_TEMPLATE = (
    "The user is seeking practical, encouraging guidance for {topic}. "
    "Apply the Breaking Better 3x3 framework with emphasis on {principle}."
)
COT_OPENING = (
    "Identify the user's stated goal, constraint, or decision "
    "without inventing context."
)
COT_MIDDLE_TEMPLATE = (
    "Apply {principle} through {article} {capability} capability lens."
)
COT_CLOSING = (
    "Translate the framework into concrete, supportive actions grounded in the prompt."
)

#: Namespace for ids minted by this builder. v9's own scheme is unrecoverable,
#: so ledger-recorded ids always win over recomputation.
ICDU_ID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://github.com/Farsalis/ai-factory/icdu"
)

METADATA_GENERATOR = "deterministic_icdu_v9_rules"

# ============================================================================
# Known augmentation artifacts (measured in the v8 / proactive corpora)
# ============================================================================

#: Mechanical suffixes the v8 pipeline appended across unrelated families.
ARTIFACT_PROMPT_SUFFIXES: tuple[str, ...] = (
    "What if budget wasn't an issue?",
    "I need a solution that is fast and effective.",
    "How can I adapt this for a recent injury?",
    "Resources no issue; best premium plan?",
    "Urgent; quick but effective?",
    "Adjust for my region?",
    "How to track progress?",
    "How to handle moral conflicts?",
)

#: Perturbation labels the publication generator prefixed onto contexts.
ARTIFACT_CONTEXT_LABELS: tuple[str, ...] = (
    "Inverted constraint:",
    "Multi-stakeholder:",
    "Ethical twist:",
    "High-stakes:",
    "Cultural variant:",
    "Outcome-focused:",
)

#: Mock tool-call scaffolding left in v7 / v8 assistant turns.
ARTIFACT_RESPONSE_MARKERS: tuple[str, ...] = (
    '"tool_call"',
    "Tool result:",
    "Need data:",
)

# ============================================================================
# Output format and defaults
# ============================================================================

#: v9 files use CRLF separators with a trailing terminator.
LINE_TERMINATOR = "\r\n"
#: v9 files use compact separators and raw (non-escaped) non-ASCII text.
JSON_SEPARATORS = (",", ":")

DEFAULT_DATASET_DIR = Path("src/data/datasets")
DEFAULT_BASE_VERSION = 9
DEFAULT_VALIDATION_RATIO = 0.2
DEFAULT_MAX_VARIANT_SIMILARITY = 0.9
DEFAULT_MAX_DISTRIBUTION_DRIFT = 0.15
DEFAULT_REPOSITORY = "https://github.com/Farsalis/ai-factory"
DATASET_NAME_TEMPLATE = "ICDU General Dataset v{version}"

RECORD_KIND_CARRIED = "carried_forward"
RECORD_KIND_NEW_FAMILY = "new_family"
RECORD_KIND_VARIANT = "variant"

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
SPLITS: tuple[str, ...] = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)

#: Split -> filename stem fragment, matching the published v9 names.
SPLIT_FILE_STEMS: dict[str, str] = {
    SPLIT_TRAIN: "training",
    SPLIT_VALIDATION: "validation",
    SPLIT_TEST: "test",
}


class BuildError(RuntimeError):
    """Raised when a build cannot proceed or a validation gate fails."""


# ============================================================================
# Hashing and serialisation helpers
# ============================================================================


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_family_hash(prompt: str) -> str:
    """Return the prompt-family hash: SHA-256 of the case- and space-normalised prompt.

    Two prompts belong to the same family when they differ only in letter case
    or whitespace. This is the v9 rule, verified against all 354 published
    lineage records.

    Args:
        prompt: Raw ``application_prompt`` text.

    Returns:
        Hex digest identifying the prompt family.
    """
    return sha256_text(" ".join(prompt.lower().split()))


def content_sha256(path: Path) -> str:
    """Return a newline-independent SHA-256 digest of a file's contents.

    CRLF is normalised to LF before hashing so digests match across platforms
    and git checkout settings. The v9 manifest hashes were computed this way;
    raw-byte digests of the shipped CRLF files do not match it.

    Args:
        path: File to hash.

    Returns:
        Hex digest of the LF-normalised bytes.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def dump_json_line(record: dict[str, Any]) -> str:
    """Serialise one record exactly as the v9 JSONL files did."""
    return json.dumps(record, ensure_ascii=False, separators=JSON_SEPARATORS)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, tolerating either line terminator.

    Args:
        path: File to read.

    Returns:
        One dict per non-blank line.

    Raises:
        BuildError: If the file is missing or a line is not valid JSON.
    """
    if not path.is_file():
        raise BuildError(f"Required input file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise BuildError(
                    f"{path}:{line_number} is not valid JSON: {err}"
                ) from err
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records as JSONL in the v9 wire format (CRLF, compact, UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(dump_json_line(r) + LINE_TERMINATOR for r in records)
    path.write_bytes(body.encode("utf-8"))


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write a pretty-printed JSON document with sorted keys and CRLF endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_bytes(text.replace("\n", LINE_TERMINATOR).encode("utf-8"))


# ============================================================================
# Record construction
# ============================================================================


def _article_for(word: str) -> str:
    """Return ``"an"`` when ``word`` starts with a vowel sound, else ``"a"``."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def derive_user_intent(persona: str) -> str:
    """Return the intent bound to ``persona``.

    Raises:
        BuildError: If the persona is not in the closed vocabulary.
    """
    try:
        return PERSONA_INTENTS[persona]
    except KeyError as err:
        raise BuildError(f"Unknown persona_archetype: {persona!r}") from err


def derive_context_summary(persona: str, principle: str) -> str:
    """Build the ``context_summary`` string for a persona/principle pair."""
    intent = derive_user_intent(persona)
    return CONTEXT_TEMPLATE.format(
        topic=INTENT_CONTEXT_TOPICS[intent], principle=principle
    )


def derive_response_cot(principle: str, capability: str) -> list[str]:
    """Build the three-line public rationale outline for a record."""
    return [
        COT_OPENING,
        COT_MIDDLE_TEMPLATE.format(
            principle=principle,
            article=_article_for(capability),
            capability=capability.lower(),
        ),
        COT_CLOSING,
    ]


def normalise_attributes(attributes: list[str] | None) -> list[str]:
    """Return attributes in canonical order with the required ones guaranteed.

    Args:
        attributes: Requested attributes, or ``None`` for the default pair.

    Returns:
        Canonically ordered attribute list.

    Raises:
        BuildError: If an attribute is outside :data:`RESPONSE_ATTRIBUTES`.
    """
    requested = set(attributes or REQUIRED_RESPONSE_ATTRIBUTES)
    unknown = sorted(requested - set(RESPONSE_ATTRIBUTES))
    if unknown:
        raise BuildError(f"Unknown ideal_response_attributes: {unknown}")
    requested.update(REQUIRED_RESPONSE_ATTRIBUTES)
    return [a for a in RESPONSE_ATTRIBUTES if a in requested]


def mint_icdu_id(family_hash: str) -> str:
    """Return a content-addressed UUIDv5 for a newly introduced record."""
    return str(uuid.uuid5(ICDU_ID_NAMESPACE, family_hash))


def build_icdu_record(
    *,
    prompt: str,
    response: str,
    persona: str,
    principle: str,
    capability: str,
    attributes: list[str] | None = None,
    icdu_id: str | None = None,
) -> IcduRecord:
    """Assemble a complete ICDU record from its three chosen labels.

    Args:
        prompt: Verbatim user prompt.
        response: Verbatim assistant response.
        persona: Persona archetype from :data:`PERSONA_INTENTS`.
        principle: Governing principle from :data:`GOVERNING_PRINCIPLES`.
        capability: Capability layer from :data:`CAPABILITY_LAYERS`.
        attributes: Optional response attributes; defaults to Clear + Actionable.
        icdu_id: Existing id to preserve; a content-addressed id is minted when
            omitted.

    Returns:
        Record with all ten ICDU fields in canonical order.

    Raises:
        BuildError: If any label is outside its closed vocabulary.
    """
    if principle not in GOVERNING_PRINCIPLES:
        raise BuildError(f"Unknown governing_principle: {principle!r}")
    if capability not in CAPABILITY_LAYERS:
        raise BuildError(f"Unknown capability_layer: {capability!r}")
    return {
        "icdu_id": icdu_id or mint_icdu_id(canonical_family_hash(prompt)),
        "persona_archetype": persona,
        "governing_principle": principle,
        "capability_layer": capability,
        "user_intent": derive_user_intent(persona),
        "context_summary": derive_context_summary(persona, principle),
        "application_prompt": prompt,
        "ideal_response_final": response,
        "ideal_response_attributes": normalise_attributes(attributes),
        "ideal_response_cot": derive_response_cot(principle, capability),
    }


def derived_fields_match(record: IcduRecord) -> list[str]:
    """Return the names of derived fields that disagree with the label triple.

    Used as an integrity gate: a record whose ``user_intent``,
    ``context_summary`` or ``ideal_response_cot`` no longer follows from its
    persona/principle/capability was hand-edited or built by different rules.

    Args:
        record: Record to check.

    Returns:
        Names of mismatching fields; empty when the record is self-consistent.
    """
    persona = record["persona_archetype"]
    principle = record["governing_principle"]
    capability = record["capability_layer"]
    expected = {
        "user_intent": derive_user_intent(persona),
        "context_summary": derive_context_summary(persona, principle),
        "ideal_response_cot": derive_response_cot(principle, capability),
        "ideal_response_attributes": normalise_attributes(
            record["ideal_response_attributes"]
        ),
    }
    return [name for name, value in expected.items() if record[name] != value]


# ============================================================================
# Split ledger
# ============================================================================


@dataclass
class LedgerEntry:
    """A family's recorded split assignment and identity from a prior build."""

    split: str
    icdu_id: str
    lineage: LineageRecord


class SplitLedger:
    """Authoritative record of which split each prompt family belongs to.

    Split assignments are append-only: once a family appears in the ledger its
    split never changes, which is what keeps the sealed test set stable and
    prevents reshuffling when new families are added.
    """

    def __init__(self) -> None:
        self._entries: dict[str, LedgerEntry] = {}

    def __repr__(self) -> str:
        counts = Counter(e.split for e in self._entries.values())
        return f"SplitLedger(families={len(self._entries)}, splits={dict(counts)})"

    def __len__(self) -> int:
        return len(self._entries)

    @classmethod
    def from_lineage(cls, lineage: list[LineageRecord]) -> SplitLedger:
        """Build a ledger from a lineage file's records.

        Args:
            lineage: Lineage records from a previous build. Both the v9
                ``v9_split`` key and the current ``split`` key are accepted.

        Returns:
            Populated ledger.

        Raises:
            BuildError: If a record has no split key or a family is assigned to
                two different splits.
        """
        ledger = cls()
        for record in lineage:
            split = _lineage_split(record)
            family = record["canonical_family_sha256"]
            existing = ledger._entries.get(family)
            if existing is not None and existing.split != split:
                raise BuildError(
                    f"Ledger conflict: family {family[:12]} is recorded in both "
                    f"{existing.split!r} and {split!r}"
                )
            ledger._entries[family] = LedgerEntry(
                split=split, icdu_id=record["icdu_id"], lineage=record
            )
        return ledger

    def get(self, family_hash: str) -> LedgerEntry | None:
        """Return the entry for ``family_hash``, or ``None`` if it is new."""
        return self._entries.get(family_hash)

    def add(self, family_hash: str, entry: LedgerEntry) -> None:
        """Record a newly assigned family.

        Raises:
            BuildError: If the family is already present.
        """
        if family_hash in self._entries:
            raise BuildError(f"Family {family_hash[:12]} is already in the ledger")
        self._entries[family_hash] = entry


def _lineage_split(record: LineageRecord) -> str:
    """Extract the split from a lineage record, accepting versioned key names."""
    if "split" in record:
        return str(record["split"])
    for key, value in record.items():
        if key.endswith("_split"):
            return str(value)
    raise BuildError(
        f"Lineage record {record.get('icdu_id', '<no id>')} has no split key"
    )


# ============================================================================
# Staged candidates
# ============================================================================


@dataclass
class StagedCandidate:
    """A reviewed candidate record awaiting a split assignment.

    Attributes:
        record: The assembled ICDU record.
        family_hash: Hash of this record's own prompt.
        group_hash: Split-inheritance key — the parent's family hash for a
            variant, otherwise the record's own family hash.
        kind: ``new_family`` or ``variant``.
        parent_family_hash: Parent family hash for variants, else ``None``.
        source_file: Staging file the candidate came from.
        source_line: 1-based line number within that file.
    """

    record: IcduRecord
    family_hash: str
    group_hash: str
    kind: str
    parent_family_hash: str | None
    source_file: str
    source_line: int


STAGED_REQUIRED_FIELDS: tuple[str, ...] = (
    "application_prompt",
    "ideal_response_final",
    "persona_archetype",
    "governing_principle",
    "capability_layer",
)
STAGED_OPTIONAL_FIELDS: tuple[str, ...] = (
    "ideal_response_attributes",
    "parent_application_prompt",
    "parent_canonical_family_sha256",
)


def parse_staged_candidate(
    raw: dict[str, Any], source_file: str, source_line: int
) -> StagedCandidate:
    """Validate and convert one staged candidate into a :class:`StagedCandidate`.

    Args:
        raw: Candidate as read from the staging JSONL file.
        source_file: Name of the staging file, recorded in lineage.
        source_line: 1-based line number, recorded in lineage.

    Returns:
        Parsed candidate with a fully derived ICDU record.

    Raises:
        BuildError: If required fields are missing, unknown fields are present,
            or any label is outside its closed vocabulary.
    """
    missing = [f for f in STAGED_REQUIRED_FIELDS if not raw.get(f)]
    if missing:
        raise BuildError(f"{source_file}:{source_line} missing fields: {missing}")
    allowed = set(STAGED_REQUIRED_FIELDS) | set(STAGED_OPTIONAL_FIELDS)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise BuildError(f"{source_file}:{source_line} unknown fields: {unknown}")

    record = build_icdu_record(
        prompt=raw["application_prompt"],
        response=raw["ideal_response_final"],
        persona=raw["persona_archetype"],
        principle=raw["governing_principle"],
        capability=raw["capability_layer"],
        attributes=raw.get("ideal_response_attributes"),
    )
    family_hash = canonical_family_hash(raw["application_prompt"])

    parent_hash: str | None = raw.get("parent_canonical_family_sha256")
    if raw.get("parent_application_prompt"):
        derived_parent = canonical_family_hash(raw["parent_application_prompt"])
        if parent_hash and parent_hash != derived_parent:
            raise BuildError(
                f"{source_file}:{source_line} parent prompt and parent hash disagree"
            )
        parent_hash = derived_parent
    if parent_hash == family_hash:
        raise BuildError(
            f"{source_file}:{source_line} variant prompt is identical to its parent"
        )

    return StagedCandidate(
        record=record,
        family_hash=family_hash,
        group_hash=parent_hash or family_hash,
        kind=RECORD_KIND_VARIANT if parent_hash else RECORD_KIND_NEW_FAMILY,
        parent_family_hash=parent_hash,
        source_file=source_file,
        source_line=source_line,
    )


# ============================================================================
# Split allocation
# ============================================================================


def allocate_new_family_splits(
    candidates: list[StagedCandidate], validation_ratio: float
) -> dict[str, str]:
    """Assign splits to new families, stratified by persona.

    Uses deterministic largest-remainder allocation within each persona, ordered
    by family hash, matching the v9 split policy. New families are never routed
    to the sealed test split.

    Args:
        candidates: New-family candidates (variants are not passed here).
        validation_ratio: Target share of new families for validation.

    Returns:
        Mapping of family hash to split.

    Raises:
        BuildError: If ``validation_ratio`` is outside [0, 1].
    """
    if not 0.0 <= validation_ratio <= 1.0:
        raise BuildError(f"validation_ratio must be in [0, 1], got {validation_ratio}")
    if not candidates:
        return {}

    by_persona: dict[str, list[StagedCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_persona[candidate.record["persona_archetype"]].append(candidate)

    target_total = round(len(candidates) * validation_ratio)
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for persona, group in by_persona.items():
        exact = len(group) * validation_ratio
        quotas[persona] = int(exact)
        remainders.append((exact - int(exact), persona))

    leftover = target_total - sum(quotas.values())
    # Largest remainder first; persona name breaks ties so runs are reproducible.
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for _, persona in remainders:
        if leftover <= 0:
            break
        if quotas[persona] < len(by_persona[persona]):
            quotas[persona] += 1
            leftover -= 1

    assignments: dict[str, str] = {}
    for persona, group in by_persona.items():
        ordered = sorted(group, key=lambda c: c.family_hash)
        quota = quotas[persona]
        for index, candidate in enumerate(ordered):
            assignments[candidate.family_hash] = (
                SPLIT_VALIDATION if index < quota else SPLIT_TRAIN
            )
    return assignments


# ============================================================================
# Validation gates
# ============================================================================


@dataclass
class ValidationReport:
    """Accumulated gate results for a build."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ValidationReport(status={self.status}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)})"
        )

    @property
    def status(self) -> str:
        """``"PASS"`` when no gate failed, else ``"FAIL"``."""
        return "FAIL" if self.errors else "PASS"

    def fail(self, message: str) -> None:
        """Record a gate failure."""
        self.errors.append(message)

    def warn(self, message: str) -> None:
        """Record a non-blocking finding."""
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dict."""
        return {
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            **self.details,
        }


def _split_distribution(records: list[IcduRecord]) -> dict[str, Any]:
    """Summarise label distributions and question-ending responses for a split."""
    return {
        "records": len(records),
        "personas": dict(
            sorted(Counter(r["persona_archetype"] for r in records).items())
        ),
        "principles": dict(
            sorted(Counter(r["governing_principle"] for r in records).items())
        ),
        "capabilities": dict(
            sorted(Counter(r["capability_layer"] for r in records).items())
        ),
        "responses_ending_in_question": sum(
            1 for r in records if r["ideal_response_final"].rstrip().endswith("?")
        ),
    }


def _similarity(left: str, right: str) -> float:
    """Return the difflib similarity ratio of two normalised strings."""
    return difflib.SequenceMatcher(
        None, " ".join(left.lower().split()), " ".join(right.lower().split())
    ).ratio()


def detect_prompt_artifacts(prompt: str, known_prompts: set[str]) -> list[str]:
    """Return artifact findings for a candidate prompt.

    Catches the two v8 failure modes: a known mechanical suffix, and a prompt
    that is an existing prompt plus an appended trailing sentence.

    Args:
        prompt: Candidate ``application_prompt``.
        known_prompts: Prompts already in the corpus.

    Returns:
        Human-readable findings; empty when the prompt looks clean.
    """
    findings: list[str] = []
    stripped = prompt.strip()
    for suffix in ARTIFACT_PROMPT_SUFFIXES:
        if stripped.endswith(suffix):
            findings.append(f"mechanical v8 suffix {suffix!r}")
    for known in known_prompts:
        if stripped != known and stripped.startswith(known.strip()):
            findings.append(f"extends existing prompt {known.strip()[:60]!r}")
            break
    return findings


def _check_record_schema(
    record: IcduRecord, label: str, report: ValidationReport
) -> None:
    """Gate a single record's field set, vocabularies and derived fields."""
    if tuple(record.keys()) != ICDU_FIELD_ORDER:
        report.fail(
            f"{label}: field set/order is {list(record.keys())}, "
            f"expected {list(ICDU_FIELD_ORDER)}"
        )
        return
    if record["persona_archetype"] not in PERSONA_INTENTS:
        report.fail(f"{label}: unknown persona {record['persona_archetype']!r}")
    if record["governing_principle"] not in GOVERNING_PRINCIPLES:
        report.fail(f"{label}: unknown principle {record['governing_principle']!r}")
    if record["capability_layer"] not in CAPABILITY_LAYERS:
        report.fail(f"{label}: unknown capability {record['capability_layer']!r}")
    if not record["application_prompt"].strip():
        report.fail(f"{label}: empty application_prompt")
    if not record["ideal_response_final"].strip():
        report.fail(f"{label}: empty ideal_response_final")
    for marker in ARTIFACT_RESPONSE_MARKERS:
        if marker in record["ideal_response_final"]:
            report.fail(f"{label}: response contains tool scaffolding {marker!r}")
    for artifact_label in ARTIFACT_CONTEXT_LABELS:
        if artifact_label in record["context_summary"]:
            report.fail(
                f"{label}: context contains perturbation label {artifact_label!r}"
            )
    try:
        mismatched = derived_fields_match(record)
    except BuildError as err:
        report.fail(f"{label}: {err}")
        return
    if mismatched:
        report.fail(f"{label}: derived fields do not match labels: {mismatched}")


def validate_corpus(
    *,
    assigned: dict[str, list[IcduRecord]],
    group_by_id: dict[str, str],
    new_records: list[tuple[StagedCandidate, str]],
    base_records: dict[str, list[IcduRecord]],
    base_version: int,
    max_variant_similarity: float,
    max_distribution_drift: float,
    allow_test_changes: bool,
    report: ValidationReport,
) -> None:
    """Run every validation gate over the assembled corpus.

    Args:
        assigned: Final records per split.
        group_by_id: ``icdu_id`` -> family-group hash, for leakage checks.
        new_records: Newly added candidates paired with their assigned split.
        base_records: The base version's records per split.
        base_version: Version number the build started from.
        max_variant_similarity: Similarity ceiling for a variant against its
            parent or a sibling.
        max_distribution_drift: Proportion-delta threshold that triggers a
            drift warning.
        allow_test_changes: Whether the sealed test split may differ from base.
        report: Report to append findings to.
    """
    all_records = [r for split in SPLITS for r in assigned[split]]

    for split in SPLITS:
        for record in assigned[split]:
            _check_record_schema(record, f"{split}/{record['icdu_id'][:8]}", report)

    ids = Counter(r["icdu_id"] for r in all_records)
    for icdu_id, count in ids.items():
        if count > 1:
            report.fail(f"duplicate icdu_id {icdu_id} appears {count} times")

    prompts = Counter(r["application_prompt"].strip() for r in all_records)
    for prompt, count in prompts.items():
        if count > 1:
            report.fail(f"duplicate application_prompt ({count}x): {prompt[:70]!r}")

    # Family-group containment: parent and every variant in exactly one split.
    group_splits: dict[str, set[str]] = defaultdict(set)
    for split in SPLITS:
        for record in assigned[split]:
            group_splits[group_by_id[record["icdu_id"]]].add(split)
    overlap_pairs: dict[str, int] = Counter()
    for group, splits in group_splits.items():
        if len(splits) > 1:
            ordered = sorted(splits)
            report.fail(
                f"family group {group[:12]} spans splits {ordered} "
                "(variants must stay in their parent's split)"
            )
            for i, left in enumerate(ordered):
                for right in ordered[i + 1 :]:
                    overlap_pairs[f"{left}_vs_{right}"] += 1

    exact_overlap = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            left_prompts = {r["application_prompt"].strip() for r in assigned[left]}
            right_prompts = {r["application_prompt"].strip() for r in assigned[right]}
            shared = left_prompts & right_prompts
            exact_overlap[f"{left}_vs_{right}"] = len(shared)
            if shared:
                report.fail(
                    f"{len(shared)} prompt(s) appear in both {left} and {right}"
                )

    # Artifact and near-duplicate gates apply to new content only; the base
    # corpus is already published and its one question-ending response is known.
    by_family = {canonical_family_hash(r["application_prompt"]): r for r in all_records}
    base_prompts = {
        r["application_prompt"].strip()
        for split in SPLITS
        for r in base_records.get(split, [])
    }
    artifact_count = 0
    for candidate, split in new_records:
        label = f"{candidate.source_file}:{candidate.source_line}"
        findings = detect_prompt_artifacts(
            candidate.record["application_prompt"], base_prompts
        )
        for finding in findings:
            artifact_count += 1
            report.fail(f"{label}: augmentation artifact - {finding}")
        if candidate.record["ideal_response_final"].rstrip().endswith("?"):
            report.fail(
                f"{label}: response ends in a question (forced follow-up artifact)"
            )
        if candidate.kind != RECORD_KIND_VARIANT:
            continue
        parent_hash = candidate.parent_family_hash or ""
        parent = by_family.get(parent_hash)
        if parent is None:
            report.fail(
                f"{label}: parent family {parent_hash[:12]} is not in the corpus"
            )
            continue
        if split != _split_of(parent["icdu_id"], assigned):
            report.fail(f"{label}: variant split {split!r} differs from its parent's")
        for field_name in ("application_prompt", "ideal_response_final"):
            ratio = _similarity(candidate.record[field_name], parent[field_name])
            if ratio > max_variant_similarity:
                report.fail(
                    f"{label}: variant {field_name} is {ratio:.2f} similar to its "
                    f"parent (ceiling {max_variant_similarity:.2f})"
                )

    _check_sibling_similarity(new_records, max_variant_similarity, report)

    test_unchanged = [dump_json_line(r) for r in assigned[SPLIT_TEST]] == [
        dump_json_line(r) for r in base_records.get(SPLIT_TEST, [])
    ]
    if not test_unchanged and not allow_test_changes:
        report.fail(
            f"sealed test split differs from v{base_version} "
            "(pass --allow-test-changes only for a deliberate reseal)"
        )

    drift = _distribution_drift(assigned, base_records, max_distribution_drift)
    for message in drift["warnings"]:
        report.warn(message)

    report.details.update(
        {
            "records": len(all_records),
            "canonical_families": len(by_family),
            "family_groups": len(group_splits),
            "exact_prompt_overlap": exact_overlap,
            "family_group_overlap": dict(overlap_pairs),
            "known_augmentation_artifacts": artifact_count,
            "sealed_test_unchanged": test_unchanged,
            "splits": {split: _split_distribution(assigned[split]) for split in SPLITS},
            "distribution_drift": {
                "max_proportion_delta": round(drift["max_delta"], 4),
                "threshold": max_distribution_drift,
                "notes": drift["warnings"],
            },
        }
    )


def _split_of(icdu_id: str, assigned: dict[str, list[IcduRecord]]) -> str | None:
    """Return the split containing ``icdu_id``, or ``None``."""
    for split in SPLITS:
        if any(r["icdu_id"] == icdu_id for r in assigned[split]):
            return split
    return None


def _check_sibling_similarity(
    new_records: list[tuple[StagedCandidate, str]],
    ceiling: float,
    report: ValidationReport,
) -> None:
    """Fail when two variants of the same parent are near-copies of each other."""
    siblings: dict[str, list[StagedCandidate]] = defaultdict(list)
    for candidate, _ in new_records:
        if candidate.kind == RECORD_KIND_VARIANT and candidate.parent_family_hash:
            siblings[candidate.parent_family_hash].append(candidate)
    for group in siblings.values():
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                ratio = _similarity(
                    left.record["ideal_response_final"],
                    right.record["ideal_response_final"],
                )
                if ratio > ceiling:
                    report.fail(
                        f"{left.source_file}:{left.source_line} and "
                        f"{right.source_file}:{right.source_line} are sibling "
                        f"variants {ratio:.2f} similar to each other"
                    )


def _distribution_drift(
    assigned: dict[str, list[IcduRecord]],
    base_records: dict[str, list[IcduRecord]],
    threshold: float,
) -> dict[str, Any]:
    """Compare label proportions against the base version.

    Args:
        assigned: Final records per split.
        base_records: Base version records per split.
        threshold: Proportion delta above which a note is emitted.

    Returns:
        Dict with ``max_delta`` and human-readable ``warnings``.
    """
    dimensions = {
        "personas": "persona_archetype",
        "principles": "governing_principle",
        "capabilities": "capability_layer",
    }
    max_delta = 0.0
    warnings: list[str] = []
    for split in (SPLIT_TRAIN, SPLIT_VALIDATION):
        base = base_records.get(split, [])
        if not base or not assigned[split]:
            continue
        for name, field_name in dimensions.items():
            base_counts = Counter(r[field_name] for r in base)
            new_counts = Counter(r[field_name] for r in assigned[split])
            for key in set(base_counts) | set(new_counts):
                delta = abs(
                    new_counts[key] / len(assigned[split])
                    - base_counts[key] / len(base)
                )
                if delta > max_delta:
                    max_delta = delta
                if delta > threshold:
                    warnings.append(
                        f"{split} {name}[{key}] proportion moved {delta:.2f} "
                        "versus the base version"
                    )
    return {"max_delta": max_delta, "warnings": warnings}


# ============================================================================
# Build orchestration
# ============================================================================


@dataclass
class BuildPaths:
    """Resolved input and output paths for a build."""

    base_dir: Path
    output_dir: Path
    base_version: int
    version: int

    def base_split_file(self, split: str) -> Path:
        """Path to the base version's file for ``split``."""
        stem = SPLIT_FILE_STEMS[split]
        return self.base_dir / f"icdu_{stem}_data_v{self.base_version}.jsonl"

    def base_lineage_file(self) -> Path:
        """Path to the base version's lineage file."""
        return self.base_dir / f"icdu_v{self.base_version}_source_lineage.jsonl"

    def split_file(self, split: str) -> Path:
        """Path to the output file for ``split``."""
        stem = SPLIT_FILE_STEMS[split]
        return self.output_dir / f"icdu_{stem}_data_v{self.version}.jsonl"

    def lineage_file(self) -> Path:
        """Path to the output lineage file."""
        return self.output_dir / f"icdu_v{self.version}_source_lineage.jsonl"

    def report_file(self) -> Path:
        """Path to the output validation report."""
        return self.output_dir / f"icdu_v{self.version}_validation_report.json"

    def manifest_file(self) -> Path:
        """Path to the output manifest."""
        return self.output_dir / f"icdu_v{self.version}_manifest.json"


@dataclass
class BuildResult:
    """Outcome of a build attempt."""

    report: ValidationReport
    records: dict[str, list[IcduRecord]]
    lineage: list[LineageRecord]
    manifest: dict[str, Any]
    written: list[Path]

    def __repr__(self) -> str:
        counts = {s: len(r) for s, r in self.records.items()}
        return f"BuildResult(status={self.report.status}, counts={counts})"


def _carried_forward_lineage(
    record: IcduRecord, entry: LedgerEntry, split: str, version: int
) -> LineageRecord:
    """Build the lineage row for a record inherited from the base version."""
    prior = entry.lineage
    return {
        "icdu_id": record["icdu_id"],
        "dataset_version": str(version),
        "record_kind": RECORD_KIND_CARRIED,
        "source_dataset_file": prior.get("source_dataset_file", ""),
        "source_partition": prior.get("source_partition", ""),
        "source_line_number": prior.get("source_line_number", 0),
        "canonical_family_sha256": prior["canonical_family_sha256"],
        "family_group_sha256": prior["canonical_family_sha256"],
        "parent_icdu_id": None,
        "prompt_sha256": prior["prompt_sha256"],
        "response_sha256": prior["response_sha256"],
        "metadata_generator": prior.get("metadata_generator", METADATA_GENERATOR),
        "prompt_preserved_verbatim": True,
        "response_preserved_verbatim": True,
        "split": split,
    }


def _new_lineage(
    candidate: StagedCandidate,
    split: str,
    version: int,
    parent_id: str | None,
) -> LineageRecord:
    """Build the lineage row for a newly staged record."""
    record = candidate.record
    return {
        "icdu_id": record["icdu_id"],
        "dataset_version": str(version),
        "record_kind": candidate.kind,
        "source_dataset_file": candidate.source_file,
        "source_partition": f"v{version}_staged_candidates",
        "source_line_number": candidate.source_line,
        "canonical_family_sha256": candidate.family_hash,
        "family_group_sha256": candidate.group_hash,
        "parent_icdu_id": parent_id,
        "prompt_sha256": sha256_text(record["application_prompt"]),
        "response_sha256": sha256_text(record["ideal_response_final"]),
        "metadata_generator": METADATA_GENERATOR,
        "prompt_preserved_verbatim": True,
        "response_preserved_verbatim": True,
        "split": split,
    }


def build_dataset(
    *,
    paths: BuildPaths,
    staged_files: list[Path],
    build_seed: str,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    max_variant_similarity: float = DEFAULT_MAX_VARIANT_SIMILARITY,
    max_distribution_drift: float = DEFAULT_MAX_DISTRIBUTION_DRIFT,
    allow_test_changes: bool = False,
    targets: dict[str, int] | None = None,
    repository: str = DEFAULT_REPOSITORY,
    source_commit: str | None = None,
    dry_run: bool = False,
) -> BuildResult:
    """Build a new dataset version from the base version plus staged candidates.

    Args:
        paths: Resolved input/output paths.
        staged_files: Reviewed candidate JSONL files; may be empty, which
            reproduces the base corpus under the new version number.
        build_seed: Reproducibility seed recorded in the manifest.
        validation_ratio: Share of new families routed to validation.
        max_variant_similarity: Similarity ceiling for variants.
        max_distribution_drift: Proportion-delta threshold for drift warnings.
        allow_test_changes: Permit the sealed test split to change.
        targets: Optional per-split row targets, reported as progress.
        repository: Repository URL recorded in the manifest.
        source_commit: Optional commit recorded in the manifest.
        dry_run: Validate without writing any files.

    Returns:
        The build result. ``written`` is empty on a dry run or a failed build.

    Raises:
        BuildError: If inputs are missing or malformed. Gate failures are
            reported in the result rather than raised.
    """
    report = ValidationReport()

    base_records = {split: load_jsonl(paths.base_split_file(split)) for split in SPLITS}
    ledger = SplitLedger.from_lineage(load_jsonl(paths.base_lineage_file()))

    assigned: dict[str, list[IcduRecord]] = {split: [] for split in SPLITS}
    lineage: list[LineageRecord] = []
    group_by_id: dict[str, str] = {}
    id_by_family: dict[str, str] = {}

    for split in SPLITS:
        for record in base_records[split]:
            family = canonical_family_hash(record["application_prompt"])
            entry = ledger.get(family)
            if entry is None:
                report.fail(
                    f"{split}/{record['icdu_id'][:8]}: family {family[:12]} is "
                    f"missing from the v{paths.base_version} lineage ledger"
                )
                continue
            if entry.split != split:
                report.fail(
                    f"{record['icdu_id'][:8]}: ledger says {entry.split!r} but the "
                    f"v{paths.base_version} file has it in {split!r}"
                )
            if entry.icdu_id != record["icdu_id"]:
                report.fail(
                    f"{split}/{record['icdu_id'][:8]}: icdu_id disagrees with the "
                    f"ledger ({entry.icdu_id[:8]})"
                )
            if entry.lineage["prompt_sha256"] != sha256_text(
                record["application_prompt"]
            ) or entry.lineage["response_sha256"] != sha256_text(
                record["ideal_response_final"]
            ):
                report.fail(
                    f"{split}/{record['icdu_id'][:8]}: text no longer matches the "
                    "hashes recorded in the ledger"
                )
            assigned[split].append(record)
            lineage.append(
                _carried_forward_lineage(record, entry, split, paths.version)
            )
            group_by_id[record["icdu_id"]] = family
            id_by_family[family] = record["icdu_id"]

    candidates: list[StagedCandidate] = []
    for staged_file in staged_files:
        for line_number, raw in enumerate(load_jsonl(staged_file), start=1):
            candidates.append(
                parse_staged_candidate(raw, staged_file.name, line_number)
            )

    seen_families = set(id_by_family)
    fresh: list[StagedCandidate] = []
    for candidate in candidates:
        if candidate.family_hash in seen_families:
            report.fail(
                f"{candidate.source_file}:{candidate.source_line}: prompt family "
                f"{candidate.family_hash[:12]} is already in the corpus"
            )
            continue
        seen_families.add(candidate.family_hash)
        fresh.append(candidate)

    new_family_splits = allocate_new_family_splits(
        [c for c in fresh if c.kind == RECORD_KIND_NEW_FAMILY], validation_ratio
    )

    # New families are placed first so that a variant of a family introduced in
    # this same build can still resolve its parent through the ledger.
    ordered = [c for c in fresh if c.kind == RECORD_KIND_NEW_FAMILY]
    ordered += [c for c in fresh if c.kind == RECORD_KIND_VARIANT]

    new_records: list[tuple[StagedCandidate, str]] = []
    for candidate in ordered:
        if candidate.kind == RECORD_KIND_NEW_FAMILY:
            split = new_family_splits[candidate.family_hash]
            parent_id = None
        else:
            parent_entry = ledger.get(candidate.group_hash)
            if parent_entry is None:
                report.fail(
                    f"{candidate.source_file}:{candidate.source_line}: parent family "
                    f"{candidate.group_hash[:12]} is not in the ledger"
                )
                continue
            split = parent_entry.split
            parent_id = parent_entry.icdu_id
        assigned[split].append(candidate.record)
        lineage.append(_new_lineage(candidate, split, paths.version, parent_id))
        group_by_id[candidate.record["icdu_id"]] = candidate.group_hash
        new_records.append((candidate, split))
        if candidate.kind == RECORD_KIND_NEW_FAMILY:
            ledger.add(
                candidate.family_hash,
                LedgerEntry(
                    split=split,
                    icdu_id=candidate.record["icdu_id"],
                    lineage=lineage[-1],
                ),
            )

    validate_corpus(
        assigned=assigned,
        group_by_id=group_by_id,
        new_records=new_records,
        base_records=base_records,
        base_version=paths.base_version,
        max_variant_similarity=max_variant_similarity,
        max_distribution_drift=max_distribution_drift,
        allow_test_changes=allow_test_changes,
        report=report,
    )

    report.details["record_kinds"] = dict(
        sorted(Counter(row["record_kind"] for row in lineage).items())
    )
    report.details["dataset_version"] = str(paths.version)
    report.details["base_version"] = str(paths.base_version)
    if targets:
        report.details["targets"] = {
            split: {
                "target": target,
                "actual": len(assigned[split]),
                "shortfall": max(0, target - len(assigned[split])),
            }
            for split, target in targets.items()
        }

    manifest = _build_manifest(
        paths=paths,
        assigned=assigned,
        staged_files=staged_files,
        build_seed=build_seed,
        validation_ratio=validation_ratio,
        repository=repository,
        source_commit=source_commit,
        status=report.status,
    )

    written: list[Path] = []
    if report.errors:
        logger.error(
            "Validation failed with %d error(s); no files written", len(report.errors)
        )
    elif dry_run:
        logger.info("Dry run: validation passed, no files written")
    else:
        written = _write_outputs(paths, assigned, lineage, report, manifest)

    return BuildResult(
        report=report,
        records=assigned,
        lineage=lineage,
        manifest=manifest,
        written=written,
    )


def _build_manifest(
    *,
    paths: BuildPaths,
    assigned: dict[str, list[IcduRecord]],
    staged_files: list[Path],
    build_seed: str,
    validation_ratio: float,
    repository: str,
    source_commit: str | None,
    status: str,
) -> dict[str, Any]:
    """Assemble the manifest document (file hashes are filled in after writing)."""
    counts = {split: len(assigned[split]) for split in SPLITS}
    source_hashes = {
        path.name: content_sha256(path)
        for path in [
            *(paths.base_split_file(s) for s in SPLITS),
            paths.base_lineage_file(),
            *staged_files,
        ]
        if path.is_file()
    }
    return {
        "dataset_name": DATASET_NAME_TEMPLATE.format(version=paths.version),
        "version": f"{paths.version}.0.0",
        "build_seed": build_seed,
        "repository": repository,
        "dataset_build_source_commit": source_commit,
        "base_version": str(paths.base_version),
        "builder": "src/data/build_icdu_dataset.py",
        "record_counts": {**counts, "total": sum(counts.values())},
        "split_policy": {
            "ledger": paths.base_lineage_file().name,
            "existing_family_policy": "split pinned by ledger; never reassigned",
            "new_family_policy": (
                "persona-stratified largest-remainder allocation over train and "
                "validation only"
            ),
            "variant_policy": "inherits the parent family's split",
            "validation_ratio_for_new_families": validation_ratio,
            "test_split": "sealed; unchanged from the base version",
        },
        "source_file_sha256": source_hashes,
        f"v{paths.version}_file_sha256": {},
        "text_policy": "source prompt and assistant response preserved verbatim",
        "metadata_policy": (
            "labels restricted to closed vocabularies; intent, context and "
            "rationale derived deterministically"
        ),
        "hash_policy": "sha256 over LF-normalised file contents",
        "validation_status": status,
    }


def _write_outputs(
    paths: BuildPaths,
    assigned: dict[str, list[IcduRecord]],
    lineage: list[LineageRecord],
    report: ValidationReport,
    manifest: dict[str, Any],
) -> list[Path]:
    """Write all artifacts, hashing them into the manifest as v9 did."""
    written: list[Path] = []
    for split in SPLITS:
        path = paths.split_file(split)
        write_jsonl(assigned[split], path)
        written.append(path)
    write_jsonl(lineage, paths.lineage_file())
    written.append(paths.lineage_file())
    write_json(report.to_dict(), paths.report_file())
    written.append(paths.report_file())

    manifest[f"v{paths.version}_file_sha256"] = {
        path.name: content_sha256(path) for path in written
    }
    write_json(manifest, paths.manifest_file())
    written.append(paths.manifest_file())
    logger.info("Wrote %d artifact(s) to %s", len(written), paths.output_dir)
    return written


# ============================================================================
# CLI
# ============================================================================


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a versioned ICDU dataset from the previous version "
        "plus reviewed candidate records.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory holding the base version's files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write artifacts (default: --base-dir).",
    )
    parser.add_argument(
        "--base-version",
        type=int,
        default=DEFAULT_BASE_VERSION,
        help="Version to build from.",
    )
    parser.add_argument(
        "--version", type=int, required=True, help="Version number to produce."
    )
    parser.add_argument(
        "--staged",
        type=Path,
        action="append",
        default=[],
        help="Reviewed candidate JSONL file; repeatable.",
    )
    parser.add_argument(
        "--build-seed",
        default=None,
        help="Seed recorded in the manifest (default: icdu-general-v<version>).",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=DEFAULT_VALIDATION_RATIO,
        help="Share of new families routed to validation.",
    )
    parser.add_argument(
        "--max-variant-similarity",
        type=float,
        default=DEFAULT_MAX_VARIANT_SIMILARITY,
        help="Similarity ceiling for a variant against its parent or siblings.",
    )
    parser.add_argument(
        "--max-distribution-drift",
        type=float,
        default=DEFAULT_MAX_DISTRIBUTION_DRIFT,
        help="Label-proportion delta that triggers a drift warning.",
    )
    parser.add_argument(
        "--allow-test-changes",
        action="store_true",
        help="Permit the sealed test split to differ from the base version.",
    )
    parser.add_argument("--target-train", type=int, default=None)
    parser.add_argument("--target-validation", type=int, default=None)
    parser.add_argument("--target-test", type=int, default=None)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate without writing files."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success, 1 on validation failure."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = _parse_args(argv)
    paths = BuildPaths(
        base_dir=args.base_dir,
        output_dir=args.output_dir or args.base_dir,
        base_version=args.base_version,
        version=args.version,
    )
    targets = {
        split: value
        for split, value in (
            (SPLIT_TRAIN, args.target_train),
            (SPLIT_VALIDATION, args.target_validation),
            (SPLIT_TEST, args.target_test),
        )
        if value is not None
    }
    try:
        result = build_dataset(
            paths=paths,
            staged_files=list(args.staged),
            build_seed=args.build_seed or f"icdu-general-v{args.version}",
            validation_ratio=args.validation_ratio,
            max_variant_similarity=args.max_variant_similarity,
            max_distribution_drift=args.max_distribution_drift,
            allow_test_changes=args.allow_test_changes,
            targets=targets or None,
            source_commit=args.source_commit,
            dry_run=args.dry_run,
        )
    except BuildError as err:
        logger.error("Build aborted: %s", err)
        return 1

    counts = {split: len(rows) for split, rows in result.records.items()}
    logger.info(
        "Status %s | counts %s | total %d",
        result.report.status,
        counts,
        sum(counts.values()),
    )
    for warning in result.report.warnings:
        logger.warning("%s", warning)
    for error in result.report.errors[:20]:
        logger.error("%s", error)
    if len(result.report.errors) > 20:
        logger.error("... and %d more error(s)", len(result.report.errors) - 20)
    return 1 if result.report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
