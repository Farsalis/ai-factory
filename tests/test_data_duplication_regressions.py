"""Regression tests for content duplication in the dataset generation pipeline.

Two defects duplicated content into generated rows:

* ``generate_dynamic_perturbation`` embedded the base context in every selected
  template, so combining ``depth`` templates repeated the whole context
  ``depth`` times inside a single ``context_summary``.
* ``generate_tool_variant`` shallow-copied the message list, so injecting tool
  content mutated the source example; later variants then wrapped the
  already-injected content and the originals kept in the output were corrupted.
"""

from collections.abc import Callable

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("transformers")

from src.data.augment_dataset import (
    augment_dataset,
    generate_tool_variant,
)
from src.data.augment_validation_dataset import (
    generate_tool_variant as generate_validation_tool_variant,
)
from src.data.generate_icdu_publication_dataset import (
    DEFAULT_PERTURBATION_DEPTH,
    generate_dynamic_perturbation,
    get_augmentations,
)

# The first sentence survives the "inverted constraint" rewrite untouched, so it
# counts body copies whichever template is selected.
BASE_SENTENCE = "User is facing a general challenge requiring practical guidance."
BASE_CONTEXT = (
    f"{BASE_SENTENCE} The user is on a tight budget and needs low-cost solutions."
)

# The single-template wording produced before the fix, which depth=1 must keep.
LEGACY_DEPTH_ONE_OUTPUTS = [
    "Inverted constraint: User is facing a general challenge requiring "
    "practical guidance. The user is on a generous budget and needs "
    "low-cost solutions.",
    f"Multi-stakeholder: {BASE_CONTEXT} involving team or family "
    "consensus and diverse opinions.",
    f"Ethical twist: {BASE_CONTEXT} with moral considerations and fairness principles.",
    f"High-stakes: {BASE_CONTEXT} under pressure from deadlines or high expectations.",
    f"Cultural variant: {BASE_CONTEXT} adapted to regional or cultural nuances.",
    f"Outcome-focused: {BASE_CONTEXT} aiming for measurable success metrics.",
]

TOOL_VARIANT_GENERATORS = [generate_tool_variant, generate_validation_tool_variant]


def _pin_template_selection(
    monkeypatch: pytest.MonkeyPatch, indices: list[int]
) -> None:
    """Force ``random.sample`` to pick specific perturbation templates."""

    def fake_sample(population: list[tuple[str, str]], k: int) -> list[tuple[str, str]]:
        return [population[i] for i in indices[:k]]

    monkeypatch.setattr(
        "src.data.generate_icdu_publication_dataset.random.sample", fake_sample
    )


def _sample_messages() -> list[dict[str, str]]:
    """Build a minimal source example in ``messages`` format."""
    return [
        {"role": "user", "content": "How do I get fit?"},
        {"role": "assistant", "content": "ORIGINAL_ANSWER"},
    ]


class TestGenerateDynamicPerturbation:
    """The base context must appear exactly once, whatever the depth."""

    @pytest.mark.unit
    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5, 6, 10])
    def test_base_context_is_not_repeated(
        self, depth: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Combining templates must not duplicate the context."""
        _pin_template_selection(monkeypatch, [1, 2, 3, 4, 5, 0])
        result = generate_dynamic_perturbation(BASE_CONTEXT, depth)

        assert result.count(BASE_SENTENCE) == 1

    @pytest.mark.unit
    @pytest.mark.parametrize("index", range(6))
    def test_depth_one_wording_is_unchanged(
        self, index: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single template renders exactly as it did before the fix."""
        _pin_template_selection(monkeypatch, [index])
        result = generate_dynamic_perturbation(BASE_CONTEXT, 1)

        assert result == LEGACY_DEPTH_ONE_OUTPUTS[index]

    @pytest.mark.unit
    def test_combined_templates_keep_every_qualifier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Labels and qualifiers of all selected templates survive the merge."""
        _pin_template_selection(monkeypatch, [2, 3])
        result = generate_dynamic_perturbation(BASE_CONTEXT, 2)

        assert result.startswith("Ethical twist, High-stakes: ")
        assert "with moral considerations and fairness principles." in result
        assert "under pressure from deadlines or high expectations." in result
        assert result.count(BASE_CONTEXT) == 1

    @pytest.mark.unit
    def test_inverted_constraint_still_inverts_the_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Selecting the inverted template uses the inverted context as body."""
        _pin_template_selection(monkeypatch, [0, 5])
        result = generate_dynamic_perturbation(BASE_CONTEXT, 2)

        assert "generous budget" in result
        assert "tight budget" not in result
        assert "aiming for measurable success metrics." in result

    @pytest.mark.unit
    def test_augmentation_contexts_are_not_duplicated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No augmentation pair repeats the context at the default depth."""
        _pin_template_selection(monkeypatch, [1, 2, 3, 4, 5, 0])
        augmentations = get_augmentations(
            "Fitness Seeker > Struggling Starter",
            BASE_CONTEXT,
            "How do I get fit?",
            DEFAULT_PERTURBATION_DEPTH,
        )

        for context, _prompt in augmentations:
            assert context.count(BASE_SENTENCE) == 1


class TestGenerateToolVariantIsolation:
    """Variant generation must not write back into the source example."""

    @pytest.mark.unit
    @pytest.mark.parametrize("generator", TOOL_VARIANT_GENERATORS)
    @pytest.mark.parametrize("variant_type", ["single", "multi", "none"])
    def test_original_messages_are_untouched(
        self,
        generator: Callable[..., list[dict[str, str]] | None],
        variant_type: str,
    ) -> None:
        """The source messages keep their original content."""
        messages = _sample_messages()
        generator(messages, ["search_web", "calc_tool"], variant_type)

        assert messages == _sample_messages()

    @pytest.mark.unit
    @pytest.mark.parametrize("generator", TOOL_VARIANT_GENERATORS)
    def test_repeated_variants_do_not_nest(
        self, generator: Callable[..., list[dict[str, str]] | None]
    ) -> None:
        """Each variant wraps the original answer once, never a prior variant."""
        messages = _sample_messages()

        for _ in range(3):
            variant = generator(messages, ["search_web", "calc_tool"], "single")

            assert variant is not None
            content = variant[-1]["content"]
            assert content.count("Integrated advice:") == 1
            assert content.count("ORIGINAL_ANSWER") == 1
            assert content.count("tool_call") == 1

    @pytest.mark.unit
    def test_augment_dataset_keeps_originals_pristine(self) -> None:
        """Originals carried into the output are not rewritten in place."""
        original_data = [{"messages": _sample_messages()}]

        augmented = augment_dataset(original_data, num_variants=3, tools=["search_web"])

        assert augmented[0]["messages"] == _sample_messages()
        assert original_data[0]["messages"] == _sample_messages()
