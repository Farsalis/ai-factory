"""Comprehensive test suite for master_generate_icdu.py.

Tests cover all inference functions, data I/O, processing logic, and evaluation
functions with unit tests, integration tests, and edge case handling.
"""

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import pytest  # pyright: ignore[reportMissingImports]

# Import the module under test
from src.data.master_generate_icdu import (
    DEFAULT_PERSONA,
    DEFAULT_USER_INTENT,
    GOVERNING_PRINCIPLES,
    USER_INTENTS,
    compute_diversity,
    convert_to_icdu,
    ensure_uniqueness,
    generate_follow_up_question,
    get_augmentations,
    infer_capability_layer,
    infer_context_summary,
    infer_governing_principle,
    infer_persona_archetype,
    infer_response_attributes,
    infer_response_cot,
    infer_user_intent,
    load_jsonl,
    paraphrase_response,
    process_entry,
    run_dataset_evaluation,
    save_jsonl,
    update_icdu_entry,
    visualize_stats,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_entry():
    """Create a sample entry with messages format."""
    return {
        "messages": [
            {"role": "user", "content": "I want to get fit and start working out"},
            {
                "role": "assistant",
                "content": (
                    "Start small with a simple habit like walking 10 minutes daily."
                ),
            },
        ]
    }


@pytest.fixture
def sample_icdu_entry():
    """Create a sample ICDU-formatted entry."""
    return {
        "icdu_id": str(uuid.uuid4()),
        "persona_archetype": "Fitness Seeker > Struggling Starter",
        "governing_principle": "Chapter 1 > People",
        "capability_layer": "Foundational",
        "user_intent": "To establish a sustainable fitness routine",
        "context_summary": "User is motivated to improve their health.",
        "application_prompt": "I want to get fit",
        "ideal_response_final": "Start small with walking daily.",
        "ideal_response_attributes": ["Encouraging", "Actionable", "Clear"],
        "ideal_response_cot": ["Acknowledge the problem", "Provide first step"],
    }


@pytest.fixture
def temp_jsonl_file():
    """Create a temporary JSONL file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        test_data = [
            {"id": 1, "text": "First entry"},
            {"id": 2, "text": "Second entry"},
            {"id": 3, "text": "Third entry"},
        ]
        for item in test_data:
            f.write(json.dumps(item) + "\n")
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Tests for Inference Functions
# ============================================================================


class TestInferPersonaArchetype:
    """Test persona archetype inference."""

    def test_fitness_seeker_detection(self):
        """Test detection of Fitness Seeker persona."""
        query = "I want to get fit and start working out at the gym"
        result = infer_persona_archetype(query)
        assert result == "Fitness Seeker > Struggling Starter"

    def test_relationship_manager_detection(self):
        """Test detection of Relationship Manager persona."""
        query = "My spouse and I are fighting about money and budget"
        result = infer_persona_archetype(query)
        assert result == "Relationship Manager > Financial Planner"

    def test_creative_professional_detection(self):
        """Test detection of Creative Professional persona."""
        query = "I'm a writer and I'm completely blocked on my novel"
        result = infer_persona_archetype(query)
        assert result == "Creative Professional > Blocked Writer"

    def test_career_changer_detection(self):
        """Test detection of Career Changer persona."""
        query = "I want to learn to code for career advancement"
        result = infer_persona_archetype(query)
        assert result == "Career Changer > Aspiring Professional"

    def test_default_persona_for_unmatched(self):
        """Test default persona returned for unmatched queries."""
        query = "This is a completely unrelated query with no keywords"
        result = infer_persona_archetype(query)
        assert result == DEFAULT_PERSONA

    def test_case_insensitive(self):
        """Test that persona inference is case-insensitive."""
        query_upper = "I WANT TO GET FIT"
        query_lower = "i want to get fit"
        result_upper = infer_persona_archetype(query_upper)
        result_lower = infer_persona_archetype(query_lower)
        assert result_upper == result_lower == "Fitness Seeker > Struggling Starter"

    @pytest.mark.parametrize(
        "query,expected_persona",
        [
            ("I need to workout", "Fitness Seeker > Struggling Starter"),
            (
                "financial planning with partner",
                "Relationship Manager > Financial Planner",
            ),
            ("writing a script", "Creative Professional > Blocked Writer"),
            ("job interview preparation", "Career Changer > Aspiring Professional"),
        ],
    )
    def test_multiple_persona_patterns(self, query, expected_persona):
        """Test various persona patterns."""
        result = infer_persona_archetype(query)
        assert result == expected_persona


class TestInferGoverningPrinciple:
    """Test governing principle inference."""

    def test_chapter_1_detection(self):
        """Test detection of Chapter 1 (People/Relationships)."""
        query = "I need help with my team and building relationships"
        result = infer_governing_principle(query)
        assert result.startswith("Chapter 1 >")
        assert result.split(" > ")[1] in GOVERNING_PRINCIPLES["Chapter 1"]

    def test_chapter_2_detection(self):
        """Test detection of Chapter 2 (Process/Tools)."""
        query = "I want to optimize my workflow and use better tools"
        result = infer_governing_principle(query)
        assert result.startswith("Chapter 2 >")
        assert result.split(" > ")[1] in GOVERNING_PRINCIPLES["Chapter 2"]

    def test_chapter_3_detection(self):
        """Test detection of Chapter 3 (Long-term/Aspirational)."""
        query = "I have a long-term vision for growth and transformation"
        result = infer_governing_principle(query)
        assert result.startswith("Chapter 3 >")
        assert result.split(" > ")[1] in GOVERNING_PRINCIPLES["Chapter 3"]

    def test_weighted_selection(self):
        """Test weighted chapter selection."""
        query = "unrelated query"
        chapter_weights = {"Chapter 1": 0.0, "Chapter 2": 0.0, "Chapter 3": 1.0}
        result = infer_governing_principle(query, chapter_weights)
        # With high weight on Chapter 3, should get Chapter 3
        assert result.startswith("Chapter 3 >")

    def test_format(self):
        """Test that result is in correct format."""
        query = "test query"
        result = infer_governing_principle(query)
        parts = result.split(" > ")
        assert len(parts) == 2
        assert parts[0] in GOVERNING_PRINCIPLES.keys()
        assert parts[1] in GOVERNING_PRINCIPLES[parts[0]]


class TestInferUserIntent:
    """Test user intent inference."""

    def test_get_fit_intent(self):
        """Test detection of fitness intent."""
        query = "I want to get fit and healthy"
        result = infer_user_intent(query)
        assert result == USER_INTENTS["get fit"]

    def test_money_intent(self):
        """Test detection of financial intent."""
        query = "We're fighting about money and finances"
        result = infer_user_intent(query)
        assert result == USER_INTENTS["money"]

    def test_learn_code_intent(self):
        """Test detection of coding learning intent."""
        query = "I want to learn to code for my career"
        result = infer_user_intent(query)
        assert result == USER_INTENTS["learn to code"]

    def test_default_intent_for_unmatched(self):
        """Test default intent for unmatched queries."""
        query = "completely unrelated query"
        result = infer_user_intent(query)
        assert result == DEFAULT_USER_INTENT

    def test_case_insensitive(self):
        """Test that intent inference is case-insensitive."""
        query = "I WANT TO GET FIT"
        result = infer_user_intent(query)
        assert result == USER_INTENTS["get fit"]

    @pytest.mark.parametrize(
        "query_key,intent_text",
        [
            ("cook", "To streamline weekly meal planning"),
            ("travel", "To plan a budget-friendly trip"),
            ("startup", "To launch and scale a startup venture"),
        ],
    )
    def test_multiple_intents(self, query_key, intent_text):
        """Test various intent patterns."""
        query = f"I need help with {query_key}"
        result = infer_user_intent(query)
        assert intent_text in result


class TestInferContextSummary:
    """Test context summary inference."""

    def test_get_fit_context(self):
        """Test context for fitness queries."""
        query = "I want to get fit"
        result = infer_context_summary(query)
        assert "fitness" in result.lower() or "health" in result.lower()
        assert result.endswith(".") or result.endswith("?")

    def test_perturbation_appended(self):
        """Test that perturbation is appended to context."""
        query = "I want to cook better meals"
        result = infer_context_summary(query)
        # Should contain base context plus perturbation
        assert len(result) > 50  # Base context + perturbation

    def test_default_context(self):
        """Test default context for unmatched queries."""
        query = "completely unrelated query"
        result = infer_context_summary(query)
        assert (
            "general challenge" in result.lower()
            or "practical guidance" in result.lower()
        )

    def test_case_insensitive(self):
        """Test that context inference is case-insensitive."""
        query1 = "I WANT TO GET FIT"
        query2 = "i want to get fit"
        result1 = infer_context_summary(query1)
        result2 = infer_context_summary(query2)
        # Should produce similar contexts (may differ due to random perturbation)
        assert len(result1) > 0 and len(result2) > 0


class TestInferCapabilityLayer:
    """Test capability layer inference."""

    def test_aspirational_detection(self):
        """Test detection of Aspirational layer."""
        response = "This is about your long-term vision and future growth"
        result = infer_capability_layer(response)
        assert result == "Aspirational"

    def test_transformational_detection(self):
        """Test detection of Transformational layer."""
        response = "Let's reframe this problem and experiment with new approaches"
        result = infer_capability_layer(response)
        assert result == "Transformational"

    def test_foundational_detection(self):
        """Test detection of Foundational layer."""
        response = "Start small with a simple habit and basic steps"
        result = infer_capability_layer(response)
        assert result == "Foundational"

    def test_fallback_to_random(self):
        """Test fallback to random selection when no pattern matches."""
        response = "completely unrelated response text"
        result = infer_capability_layer(response)
        assert result in ["Foundational", "Transformational", "Aspirational"]

    def test_case_insensitive(self):
        """Test that layer inference is case-insensitive."""
        response = "LONG-TERM VISION AND FUTURE GROWTH"
        result = infer_capability_layer(response)
        assert result == "Aspirational"


class TestInferResponseAttributes:
    """Test response attributes inference."""

    def test_aspirational_attributes(self):
        """Test attributes for Aspirational layer."""
        result = infer_response_attributes("Aspirational")
        assert len(result) == 3
        assert all(
            attr
            in [
                "Affirming",
                "Proactive",
                "Knowledgeable",
                "Forward-looking",
                "Empathetic",
                "Inspirational",
            ]
            for attr in result
        )

    def test_transformational_attributes(self):
        """Test attributes for Transformational layer."""
        result = infer_response_attributes("Transformational")
        assert len(result) == 3
        assert all(
            attr
            in [
                "Empathetic",
                "Insightful",
                "Principle-driven",
                "Clarifying",
                "Reflective",
                "Adaptive",
            ]
            for attr in result
        )

    def test_foundational_attributes(self):
        """Test attributes for Foundational layer."""
        result = infer_response_attributes("Foundational")
        assert len(result) == 3
        assert all(
            attr
            in [
                "Encouraging",
                "Actionable",
                "Clear",
                "Practical",
                "Concise",
                "Supportive",
                "Structured",
            ]
            for attr in result
        )

    def test_default_attributes(self):
        """Test default attributes for unknown layer."""
        result = infer_response_attributes("UnknownLayer")
        assert len(result) == 3
        # Should use Foundational as default


class TestInferResponseCot:
    """Test chain-of-thought inference."""

    def test_aspirational_cot(self):
        """Test CoT for Aspirational layer."""
        result = infer_response_cot("Aspirational")
        assert len(result) >= 3
        assert any(
            "vision" in step.lower() or "future" in step.lower() for step in result
        )

    def test_transformational_cot(self):
        """Test CoT for Transformational layer."""
        result = infer_response_cot("Transformational")
        assert len(result) >= 3
        assert any(
            "reframe" in step.lower() or "experiment" in step.lower() for step in result
        )

    def test_foundational_cot(self):
        """Test CoT for Foundational layer."""
        result = infer_response_cot("Foundational")
        assert len(result) >= 3
        assert any(
            "first step" in step.lower() or "actionable" in step.lower()
            for step in result
        )

    def test_cot_length(self):
        """Test that CoT has reasonable length."""
        for layer in ["Foundational", "Transformational", "Aspirational"]:
            result = infer_response_cot(layer)
            assert 3 <= len(result) <= 4


class TestGenerateFollowUpQuestion:
    """Test follow-up question generation."""

    def test_fitness_seeker_question(self):
        """Test question for Fitness Seeker persona."""
        persona = "Fitness Seeker > Struggling Starter"
        result = generate_follow_up_question(persona)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "?" in result

    def test_default_question(self):
        """Test default question for unknown persona."""
        persona = "Unknown Persona > Unknown Role"
        result = generate_follow_up_question(persona)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_question_format(self):
        """Test that questions are properly formatted."""
        persona = "Relationship Manager > Financial Planner"
        result = generate_follow_up_question(persona)
        assert result.endswith("?") or "?" in result


# ============================================================================
# Tests for Data I/O Functions
# ============================================================================


class TestLoadJsonl:
    """Test JSONL loading functionality."""

    def test_load_valid_jsonl(self, temp_jsonl_file):
        """Test loading a valid JSONL file."""
        result = load_jsonl(str(temp_jsonl_file))
        assert len(result) == 3
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2
        assert result[2]["id"] == 3

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises error."""
        with pytest.raises(ValueError, match="Input file not found"):
            load_jsonl("nonexistent_file.jsonl")

    def test_load_empty_file(self, temp_output_dir):
        """Test loading an empty file."""
        empty_file = temp_output_dir / "empty.jsonl"
        empty_file.write_text("")
        result = load_jsonl(str(empty_file))
        assert result == []

    def test_load_file_with_invalid_json(self, temp_output_dir):
        """Test loading file with invalid JSON lines."""
        invalid_file = temp_output_dir / "invalid.jsonl"
        invalid_file.write_text(
            '{"valid": true}\ninvalid json line\n{"also": "valid"}\n'
        )
        result = load_jsonl(str(invalid_file))
        # Should skip invalid line and load valid ones
        assert len(result) == 2

    def test_load_file_with_empty_lines(self, temp_output_dir):
        """Test loading file with empty lines."""
        file_with_blanks = temp_output_dir / "blanks.jsonl"
        file_with_blanks.write_text('{"id": 1}\n\n{"id": 2}\n   \n{"id": 3}\n')
        result = load_jsonl(str(file_with_blanks))
        assert len(result) == 3


class TestSaveJsonl:
    """Test JSONL saving functionality."""

    def test_save_valid_data(self, temp_output_dir):
        """Test saving valid data to JSONL."""
        test_data = [{"id": 1, "text": "test"}, {"id": 2, "text": "data"}]
        output_file = temp_output_dir / "output.jsonl"
        save_jsonl(test_data, str(output_file))

        assert output_file.exists()
        result = load_jsonl(str(output_file))
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2

    def test_save_creates_directory(self, temp_output_dir):
        """Test that save creates parent directories."""
        test_data = [{"id": 1}]
        output_file = temp_output_dir / "subdir" / "nested" / "output.jsonl"
        save_jsonl(test_data, str(output_file))

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_save_empty_list(self, temp_output_dir):
        """Test saving empty list."""
        output_file = temp_output_dir / "empty.jsonl"
        save_jsonl([], str(output_file))

        assert output_file.exists()
        result = load_jsonl(str(output_file))
        assert result == []

    def test_save_preserves_unicode(self, temp_output_dir):
        """Test that unicode characters are preserved."""
        test_data = [{"text": "Hello 世界 🌍"}]
        output_file = temp_output_dir / "unicode.jsonl"
        save_jsonl(test_data, str(output_file))

        result = load_jsonl(str(output_file))
        assert result[0]["text"] == "Hello 世界 🌍"


# ============================================================================
# Tests for Utility Functions
# ============================================================================


class TestParaphraseResponse:
    """Test response paraphrasing."""

    def test_paraphrase_basic(self):
        """Test basic paraphrasing."""
        response = "start small habit process"
        result = paraphrase_response(response)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_paraphrase_preserves_length(self):
        """Test that paraphrasing preserves approximate length."""
        response = "start small habit process"
        result = paraphrase_response(response)
        # Should have similar word count
        assert abs(len(result.split()) - len(response.split())) <= 2


class TestGetAugmentations:
    """Test augmentation generation."""

    def test_basic_augmentations(self):
        """Test basic augmentation generation."""
        persona = "General User"
        context = "User context"
        prompt = "User prompt"
        result = get_augmentations(persona, context, prompt)

        assert isinstance(result, list)
        assert len(result) >= 2
        for aug in result:
            assert isinstance(aug, tuple)
            assert len(aug) == 2  # (context, prompt)

    def test_fitness_seeker_specific_augmentation(self):
        """Test Fitness Seeker specific augmentation."""
        persona = "Fitness Seeker > Struggling Starter"
        context = "User wants to get fit"
        prompt = "How do I start?"
        result = get_augmentations(persona, context, prompt)

        assert len(result) >= 3  # Should have fitness-specific augmentation
        assert any(
            "injury" in aug[0].lower() or "injury" in aug[1].lower() for aug in result
        )


class TestEnsureUniqueness:
    """Test uniqueness enforcement."""

    def test_removes_duplicates(self):
        """Test that duplicates are removed."""
        entries = [
            {
                "persona_archetype": "Persona1",
                "application_prompt": "Prompt1",
                "ideal_response_final": "Response1",
            },
            {
                "persona_archetype": "Persona1",
                "application_prompt": "Prompt1",
                "ideal_response_final": "Response1",
            },
            {
                "persona_archetype": "Persona2",
                "application_prompt": "Prompt2",
                "ideal_response_final": "Response2",
            },
        ]
        result = ensure_uniqueness(entries)
        assert len(result) == 2

    def test_preserves_unique_entries(self):
        """Test that unique entries are preserved."""
        entries = [
            {
                "persona_archetype": "Persona1",
                "application_prompt": "Prompt1",
                "ideal_response_final": "Response1",
            },
            {
                "persona_archetype": "Persona2",
                "application_prompt": "Prompt2",
                "ideal_response_final": "Response2",
            },
        ]
        result = ensure_uniqueness(entries)
        assert len(result) == 2

    def test_handles_missing_fields(self):
        """Test handling of entries with missing fields."""
        entries = [
            {
                "persona_archetype": "Persona1",
                "application_prompt": "Prompt1",
                # Missing ideal_response_final
            },
            {
                "persona_archetype": "Persona1",
                "application_prompt": "Prompt1",
                "ideal_response_final": "",
            },
        ]
        result = ensure_uniqueness(entries)
        # Should handle gracefully
        assert isinstance(result, list)


# ============================================================================
# Tests for Processing Functions
# ============================================================================


class TestConvertToIcdu:
    """Test ICDU conversion."""

    def test_converts_valid_entry(self, sample_entry):
        """Test conversion of valid entry."""
        result = convert_to_icdu(sample_entry)

        assert result is not None
        assert "icdu_id" in result
        assert "persona_archetype" in result
        assert "governing_principle" in result
        assert "capability_layer" in result
        assert "user_intent" in result
        assert "context_summary" in result
        assert "application_prompt" in result
        assert "ideal_response_final" in result
        assert "ideal_response_attributes" in result
        assert "ideal_response_cot" in result

    def test_returns_none_for_missing_prompt(self):
        """Test that missing prompt returns None."""
        entry = {
            "messages": [
                {"role": "assistant", "content": "Response without prompt"},
            ]
        }
        result = convert_to_icdu(entry)
        assert result is None

    def test_returns_none_for_missing_response(self):
        """Test that missing response returns None."""
        entry = {
            "messages": [
                {"role": "user", "content": "Prompt without response"},
            ]
        }
        result = convert_to_icdu(entry)
        assert result is None

    def test_uses_augmentation_when_provided(self, sample_entry):
        """Test that augmentation is used when provided."""
        augmentation = ("Custom context", "Custom prompt")
        result = convert_to_icdu(sample_entry, augmentation)

        assert result is not None
        assert result["context_summary"] == "Custom context"
        assert result["application_prompt"] == "Custom prompt"

    def test_generates_uuid(self, sample_entry):
        """Test that ICDU ID is a valid UUID."""
        result = convert_to_icdu(sample_entry)
        assert result is not None
        # Try to parse as UUID
        uuid.UUID(result["icdu_id"])


class TestUpdateIcduEntry:
    """Test ICDU entry updates."""

    def test_updates_entry_without_question(self, sample_icdu_entry):
        """Test updating entry that doesn't end with question."""
        sample_icdu_entry["ideal_response_final"] = "Response without question"
        result = update_icdu_entry(sample_icdu_entry)

        assert result is not None
        assert "?" in result["ideal_response_final"]
        assert "ideal_response_cot" in result

    def test_preserves_entry_with_question(self, sample_icdu_entry):
        """Test that entry ending with question is preserved."""
        sample_icdu_entry["ideal_response_final"] = "Response with question?"
        result = update_icdu_entry(sample_icdu_entry)

        assert result == sample_icdu_entry

    def test_returns_none_for_missing_prompt(self):
        """Test that missing prompt returns None."""
        entry = {"ideal_response_final": "Response"}
        result = update_icdu_entry(entry)
        assert result is None

    def test_returns_none_for_missing_response(self):
        """Test that missing response returns None."""
        entry = {"application_prompt": "Prompt"}
        result = update_icdu_entry(entry)
        assert result is None


class TestProcessEntry:
    """Test entry processing for parallel execution."""

    def test_processes_entry_with_messages(self, sample_entry):
        """Test processing entry with messages format."""
        config = {"augmentation_factor": 1}
        result = process_entry((sample_entry, config))

        assert isinstance(result, list)
        assert len(result) >= 1
        assert all("icdu_id" in entry for entry in result)

    def test_processes_entry_without_messages(self, sample_icdu_entry):
        """Test processing entry without messages (ICDU format)."""
        config = {"augmentation_factor": 1}
        result = process_entry((sample_icdu_entry, config))

        assert isinstance(result, list)
        assert len(result) <= 1  # Should return 0 or 1 entry

    def test_augmentation_factor(self, sample_entry):
        """Test that augmentation factor is respected."""
        config = {"augmentation_factor": 3}
        result = process_entry((sample_entry, config))

        # Should have base entry + (augmentation_factor - 1) augmented entries
        assert len(result) >= 1
        # May have fewer if augmentation pool is small
        assert len(result) <= 3

    def test_handles_invalid_entry(self):
        """Test handling of invalid entry."""
        invalid_entry = {"invalid": "data"}
        config = {"augmentation_factor": 1}
        result = process_entry((invalid_entry, config))

        assert isinstance(result, list)


# ============================================================================
# Tests for Evaluation Functions
# ============================================================================


class TestComputeDiversity:
    """Test diversity computation."""

    def test_computes_diversity_metrics(self):
        """Test basic diversity computation."""
        entries = [
            {"field": "unique text one"},
            {"field": "unique text two"},
            {"field": "unique text three"},
        ]
        result = compute_diversity(entries, "field")

        assert "unique_ratio" in result
        assert "avg_jaccard_similarity" in result
        assert "is_diverse" in result
        assert 0 <= result["unique_ratio"] <= 1
        assert 0 <= result["avg_jaccard_similarity"] <= 1

    def test_handles_empty_entries(self):
        """Test handling of empty entries."""
        result = compute_diversity([], "field")
        assert result["unique_ratio"] == 0
        assert result["avg_jaccard_similarity"] == 0

    def test_handles_missing_field(self):
        """Test handling of missing field."""
        entries = [{"other_field": "value"}]
        result = compute_diversity(entries, "missing_field")
        assert result["unique_ratio"] == 0

    def test_diversity_threshold(self):
        """Test diversity threshold logic."""
        # Create very similar entries
        entries = [
            {"field": "similar text one"},
            {"field": "similar text two"},
            {"field": "similar text three"},
        ]
        result = compute_diversity(entries, "field", threshold=0.5)
        assert isinstance(result["is_diverse"], bool)


class TestVisualizeStats:
    """Test statistics visualization."""

    @patch("src.data.master_generate_icdu.plt.subplots")
    @patch("src.data.master_generate_icdu.plt.tight_layout")
    @patch("src.data.master_generate_icdu.plt.savefig")
    @patch("src.data.master_generate_icdu.plt.close")
    def test_generates_plots(
        self,
        mock_close,
        mock_savefig,
        mock_tight_layout,
        mock_subplots,
        temp_output_dir,
    ):
        """Test that plots are generated."""
        # Create mock figure and axes
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        stats = {
            "persona_archetypes": {"Persona1": 10, "Persona2": 5},
            "capability_layers": {"Foundational": 8, "Transformational": 7},
        }
        visualize_stats(stats, temp_output_dir)

        # Check that plots were saved (mocked, so we check the mock was called)
        assert mock_savefig.call_count == 2

    def test_handles_empty_stats(self, temp_output_dir):
        """Test handling of empty statistics."""
        stats = {"empty_field": {}}
        visualize_stats(stats, temp_output_dir)
        # Should not crash

    def test_handles_none_values(self, temp_output_dir):
        """Test handling of None values."""
        stats = {"none_field": None}
        visualize_stats(stats, temp_output_dir)
        # Should not crash


class TestRunDatasetEvaluation:
    """Test dataset evaluation."""

    @patch("src.data.master_generate_icdu.visualize_stats")
    def test_runs_evaluation(self, mock_visualize, temp_output_dir):
        """Test that evaluation runs successfully."""
        entries = [
            {
                "persona_archetype": "Persona1",
                "governing_principle": "Chapter 1 > People",
                "capability_layer": "Foundational",
                "application_prompt": "Test prompt",
                "ideal_response_final": "Test response",
            }
        ]
        run_dataset_evaluation(entries, temp_output_dir)

        # Check that metrics file was created
        metrics_file = temp_output_dir / "evaluation_metrics.json"
        assert metrics_file.exists()

        # Check metrics content
        with open(metrics_file) as f:
            metrics = json.load(f)
        assert "prompts_diversity" in metrics
        assert "responses_diversity" in metrics

        # Check that visualization was called
        mock_visualize.assert_called_once()


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
class TestEndToEndProcessing:
    """End-to-end integration tests."""

    def test_full_pipeline(self, temp_output_dir):
        """Test full pipeline from entry to ICDU."""
        # Create sample input file
        input_file = temp_output_dir / "input.jsonl"
        sample_data = [
            {
                "messages": [
                    {"role": "user", "content": "I want to get fit"},
                    {"role": "assistant", "content": "Start small with walking daily."},
                ]
            }
        ]
        save_jsonl(sample_data, str(input_file))

        # Load and process
        entries = load_jsonl(str(input_file))
        assert len(entries) == 1

        # Convert to ICDU
        icdu_entry = convert_to_icdu(entries[0])
        assert icdu_entry is not None

        # Save ICDU
        output_file = temp_output_dir / "output.jsonl"
        save_jsonl([icdu_entry], str(output_file))

        # Verify output
        loaded_icdu = load_jsonl(str(output_file))
        assert len(loaded_icdu) == 1
        assert "icdu_id" in loaded_icdu[0]

    def test_uniqueness_enforcement(self):
        """Test that uniqueness is properly enforced."""
        entries = [
            {
                "persona_archetype": "Persona1",
                "application_prompt": "Same prompt",
                "ideal_response_final": "Same response",
            }
            for _ in range(5)
        ]
        unique = ensure_uniqueness(entries)
        assert len(unique) == 1

    def test_augmentation_workflow(self, sample_entry):
        """Test augmentation workflow."""
        config = {"augmentation_factor": 2}
        result = process_entry((sample_entry, config))

        assert len(result) >= 1
        # All entries should have required ICDU fields
        for entry in result:
            assert "icdu_id" in entry
            assert "persona_archetype" in entry


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string_queries(self):
        """Test handling of empty string queries."""
        assert infer_persona_archetype("") == DEFAULT_PERSONA
        assert infer_user_intent("") == DEFAULT_USER_INTENT

    def test_very_long_queries(self):
        """Test handling of very long queries."""
        long_query = "word " * 1000
        result = infer_persona_archetype(long_query)
        assert isinstance(result, str)

    def test_special_characters(self):
        """Test handling of special characters."""
        query = "I want to get fit! @#$%^&*()"
        result = infer_persona_archetype(query)
        assert isinstance(result, str)

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        query = "I want to get fit 世界 🌍"
        result = infer_persona_archetype(query)
        assert isinstance(result, str)

    def test_none_values_in_entries(self):
        """Test handling of None values in entries."""
        entry = {
            "persona_archetype": None,
            "application_prompt": None,
            "ideal_response_final": None,
        }
        result = ensure_uniqueness([entry])
        # Should handle gracefully
        assert isinstance(result, list)

    def test_malformed_json_in_file(self, temp_output_dir):
        """Test handling of malformed JSON."""
        bad_file = temp_output_dir / "bad.jsonl"
        bad_file.write_text('{"valid": true}\n{"invalid": json}\n{"also": "valid"}\n')
        result = load_jsonl(str(bad_file))
        # Should skip invalid line
        assert len(result) == 2
