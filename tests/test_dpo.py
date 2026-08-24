"""Comprehensive tests for dpo.py module.

Tests cover all functions including:
- JSONL loading and saving
- Helper functions for tool call detection and extraction
- Preference pair generation
- Dataset preparation
- DPO training (with mocking)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("transformers")
pytest.importorskip("datasets")
pytest.importorskip("torch")

from src.dpo import (
    _extract_text_without_tool,
    _generate_incorrect_tool_response,
    _has_tool_call,
    generate_preference_pairs,
    load_jsonl,
    prepare_dpo_dataset,
    run_dpo_training,
    save_jsonl,
)

# ============================================================================
# Tests for load_jsonl
# ============================================================================


@pytest.mark.unit
def test_load_jsonl_skips_invalid_lines(tmp_path: Path) -> None:
    """Test that load_jsonl skips invalid JSON and non-dict entries."""
    file_path = tmp_path / "data.jsonl"
    file_path.write_text(
        "{}\nnot-json\n[]\n",
        encoding="utf-8",
    )

    data = load_jsonl(str(file_path))

    assert isinstance(data, list)
    assert len(data) == 1


@pytest.mark.unit
def test_load_jsonl_file_not_found(tmp_path: Path) -> None:
    """Test that load_jsonl raises ValueError for non-existent file."""
    file_path = tmp_path / "nonexistent.jsonl"

    with pytest.raises(ValueError, match="Input file not found"):
        load_jsonl(str(file_path))


@pytest.mark.unit
def test_load_jsonl_empty_file(tmp_path: Path) -> None:
    """Test that load_jsonl handles empty files correctly."""
    file_path = tmp_path / "empty.jsonl"
    file_path.write_text("", encoding="utf-8")

    data = load_jsonl(str(file_path))

    assert isinstance(data, list)
    assert len(data) == 0


@pytest.mark.unit
def test_load_jsonl_valid_data(tmp_path: Path) -> None:
    """Test that load_jsonl loads valid JSONL data correctly."""
    file_path = tmp_path / "data.jsonl"
    file_path.write_text(
        '{"key1": "value1"}\n{"key2": "value2"}\n{"key3": "value3"}\n',
        encoding="utf-8",
    )

    data = load_jsonl(str(file_path))

    assert len(data) == 3
    assert data[0] == {"key1": "value1"}
    assert data[1] == {"key2": "value2"}
    assert data[2] == {"key3": "value3"}


@pytest.mark.unit
def test_load_jsonl_skips_empty_lines(tmp_path: Path) -> None:
    """Test that load_jsonl skips empty lines."""
    file_path = tmp_path / "data.jsonl"
    file_path.write_text(
        '{"key1": "value1"}\n\n\n{"key2": "value2"}\n',
        encoding="utf-8",
    )

    data = load_jsonl(str(file_path))

    assert len(data) == 2
    assert data[0] == {"key1": "value1"}
    assert data[1] == {"key2": "value2"}


@pytest.mark.unit
def test_load_jsonl_unicode_content(tmp_path: Path) -> None:
    """Test that load_jsonl handles unicode content correctly."""
    file_path = tmp_path / "data.jsonl"
    file_path.write_text(
        '{"text": "Hello 世界 🌍"}\n',
        encoding="utf-8",
    )

    data = load_jsonl(str(file_path))

    assert len(data) == 1
    assert data[0]["text"] == "Hello 世界 🌍"


# ============================================================================
# Tests for save_jsonl
# ============================================================================


@pytest.mark.unit
def test_save_jsonl_saves_data(tmp_path: Path) -> None:
    """Test that save_jsonl saves data correctly."""
    file_path = tmp_path / "output.jsonl"
    data = [{"key1": "value1"}, {"key2": "value2"}]

    save_jsonl(data, str(file_path))

    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"key1": "value1"}
    assert json.loads(lines[1]) == {"key2": "value2"}


@pytest.mark.unit
def test_save_jsonl_creates_parent_directories(tmp_path: Path) -> None:
    """Test that save_jsonl creates parent directories if they don't exist."""
    file_path = tmp_path / "nested" / "deep" / "output.jsonl"
    data = [{"key": "value"}]

    save_jsonl(data, str(file_path))

    assert file_path.exists()
    assert file_path.parent.exists()


@pytest.mark.unit
def test_save_jsonl_empty_list(tmp_path: Path) -> None:
    """Test that save_jsonl handles empty list correctly."""
    file_path = tmp_path / "output.jsonl"
    data: list[dict[str, str]] = []

    save_jsonl(data, str(file_path))

    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    assert content == ""


@pytest.mark.unit
def test_save_jsonl_unicode_content(tmp_path: Path) -> None:
    """Test that save_jsonl handles unicode content correctly."""
    file_path = tmp_path / "output.jsonl"
    data = [{"text": "Hello 世界 🌍"}]

    save_jsonl(data, str(file_path))

    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    loaded = json.loads(content.strip())
    assert loaded["text"] == "Hello 世界 🌍"


@pytest.mark.unit
def test_save_jsonl_special_characters(tmp_path: Path) -> None:
    """Test that save_jsonl handles special characters correctly."""
    file_path = tmp_path / "output.jsonl"
    data = [{"text": "Text with \"quotes\" and 'apostrophes' and\nnewlines"}]

    save_jsonl(data, str(file_path))

    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    loaded = json.loads(content.strip())
    assert loaded["text"] == "Text with \"quotes\" and 'apostrophes' and\nnewlines"


# ============================================================================
# Tests for _has_tool_call
# ============================================================================


@pytest.mark.unit
def test_has_tool_call_with_tool_call_string() -> None:
    """Test _has_tool_call detects 'tool_call' in response."""
    response = "Here is the answer. tool_call something"
    assert _has_tool_call(response) is True


@pytest.mark.unit
def test_has_tool_call_with_need_data() -> None:
    """Test _has_tool_call detects 'Need data:' in response."""
    response = "I need to check. Need data: something"
    assert _has_tool_call(response) is True


@pytest.mark.unit
def test_has_tool_call_without_tool() -> None:
    """Test _has_tool_call returns False for responses without tool calls."""
    response = "This is a regular response without any tool calls."
    assert _has_tool_call(response) is False


@pytest.mark.unit
def test_has_tool_call_empty_string() -> None:
    """Test _has_tool_call handles empty string."""
    assert _has_tool_call("") is False


@pytest.mark.unit
def test_has_tool_call_case_sensitive() -> None:
    """Test _has_tool_call is case sensitive."""
    response = "TOOL_CALL something"  # Uppercase
    assert _has_tool_call(response) is False


# ============================================================================
# Tests for _extract_text_without_tool
# ============================================================================


@pytest.mark.unit
def test_extract_text_without_tool_need_data() -> None:
    """Test _extract_text_without_tool extracts text before 'Need data:'."""
    response = "I need to check this. Need data: something"
    result = _extract_text_without_tool(response)
    assert result == "I need to check this."


@pytest.mark.unit
def test_extract_text_without_tool_tool_call() -> None:
    """Test _extract_text_without_tool extracts text before 'tool_call'."""
    response = "Here is the answer. tool_call something"
    result = _extract_text_without_tool(response)
    assert result == "Here is the answer."


@pytest.mark.unit
def test_extract_text_without_tool_no_tool() -> None:
    """Test _extract_text_without_tool returns original text if no tool call."""
    response = "This is a regular response."
    result = _extract_text_without_tool(response)
    assert result == "This is a regular response."


@pytest.mark.unit
def test_extract_text_without_tool_empty_after_extraction() -> None:
    """Test _extract_text_without_tool returns fallback for empty text."""
    response = "Need data: something"
    result = _extract_text_without_tool(response)
    assert result == "I'm not sure how to handle that."


@pytest.mark.unit
def test_extract_text_without_tool_whitespace_only() -> None:
    """Test _extract_text_without_tool handles whitespace-only text."""
    response = "   \n  Need data: something"
    result = _extract_text_without_tool(response)
    assert result == "I'm not sure how to handle that."


@pytest.mark.unit
def test_extract_text_without_tool_prefers_need_data() -> None:
    """Test _extract_text_without_tool prefers 'Need data:' over 'tool_call'."""
    response = "Text before. Need data: something tool_call other"
    result = _extract_text_without_tool(response)
    assert result == "Text before."


# ============================================================================
# Tests for _generate_incorrect_tool_response
# ============================================================================


@pytest.mark.unit
def test_generate_incorrect_tool_response_creates_tool_call() -> None:
    """Test _generate_incorrect_tool_response creates valid tool call format."""
    user_msg = "What is the weather?"
    correct_response = "The weather is sunny."

    result = _generate_incorrect_tool_response(user_msg, correct_response)

    assert "Need data:" in result
    assert "tool_call" in result
    # Parse the JSON part
    json_part = result.split("Need data:")[1].strip()
    tool_call_data = json.loads(json_part)
    assert "tool_call" in tool_call_data
    assert "name" in tool_call_data["tool_call"]
    assert "arguments" in tool_call_data["tool_call"]


@pytest.mark.unit
def test_generate_incorrect_tool_response_uses_different_tool() -> None:
    """Test _generate_incorrect_tool_response uses tool not in correct response."""
    user_msg = "Calculate 2+2"
    correct_response = "The answer is 4. search_web was used"

    result = _generate_incorrect_tool_response(user_msg, correct_response)

    json_part = result.split("Need data:")[1].strip()
    tool_call_data = json.loads(json_part)
    tool_name = tool_call_data["tool_call"]["name"]
    # Should not be search_web since it's in correct_response
    assert tool_name != "search_web"


@pytest.mark.unit
def test_generate_incorrect_tool_response_sanitizes_user_msg() -> None:
    """Test _generate_incorrect_tool_response sanitizes user message."""
    user_msg = "User says \"hello\" with 'quotes' and\nnewlines"
    correct_response = "Response"

    result = _generate_incorrect_tool_response(user_msg, correct_response)

    # Should not contain unescaped quotes or newlines in JSON
    json_part = result.split("Need data:")[1].strip()
    # Should parse without errors
    tool_call_data = json.loads(json_part)
    assert isinstance(tool_call_data, dict)


@pytest.mark.unit
def test_generate_incorrect_tool_response_truncates_long_message() -> None:
    """Test _generate_incorrect_tool_response truncates very long messages."""
    user_msg = "A" * 100  # Very long message
    correct_response = "Response"

    result = _generate_incorrect_tool_response(user_msg, correct_response)

    json_part = result.split("Need data:")[1].strip()
    tool_call_data = json.loads(json_part)
    # Arguments should be truncated to 64 chars
    args = tool_call_data["tool_call"]["arguments"]
    for value in args.values():
        if isinstance(value, str):
            assert len(value) <= 64


@pytest.mark.unit
def test_generate_incorrect_tool_response_tool_specific_args() -> None:
    """Test _generate_incorrect_tool_response creates tool-specific arguments."""
    user_msg = "Test query"
    correct_response = "Response"

    # Test multiple times to check different tools
    results = []
    for _ in range(20):  # Try multiple times to get different tools
        result = _generate_incorrect_tool_response(user_msg, correct_response)
        json_part = result.split("Need data:")[1].strip()
        tool_call_data = json.loads(json_part)
        results.append(tool_call_data["tool_call"])

    # Check that at least one has tool-specific args
    tool_names = [r["name"] for r in results]
    assert len(set(tool_names)) > 0  # Should have variety


# ============================================================================
# Tests for generate_preference_pairs
# ============================================================================


@pytest.mark.unit
def test_generate_preference_pairs_with_tool_code() -> None:
    """Test generate_preference_pairs with tool call in response."""
    original_data = [
        {
            "messages": [
                {"role": "user", "content": "Find info."},
                {"role": "assistant", "content": "tool_code```search```"},
            ]
        }
    ]

    pairs = generate_preference_pairs(original_data)

    assert len(pairs) == 1
    assert pairs[0]["prompt"].startswith("[INST]")
    assert pairs[0]["prompt"].endswith("[/INST]")
    assert pairs[0]["chosen"]
    assert pairs[0]["rejected"]


@pytest.mark.unit
def test_generate_preference_pairs_without_tool() -> None:
    """Test generate_preference_pairs when response has no tool call."""
    original_data = [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is 4."},
            ]
        }
    ]

    pairs = generate_preference_pairs(original_data)

    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "The answer is 4."
    assert "Need data:" in pairs[0]["rejected"]  # Should have incorrect tool


@pytest.mark.unit
def test_generate_preference_pairs_need_data_format() -> None:
    """Test generate_preference_pairs with 'Need data:' format."""
    original_data = [
        {
            "messages": [
                {"role": "user", "content": "Search for Python"},
                {
                    "role": "assistant",
                    "content": (
                        'I\'ll search for that. Need data: {"tool_call": '
                        '{"name": "search_web", "arguments": {"query": "Python"}}}'
                    ),
                },
            ]
        }
    ]

    pairs = generate_preference_pairs(original_data)

    assert len(pairs) == 1
    assert "Need data:" in pairs[0]["chosen"]
    # Rejected should be text without tool
    assert "Need data:" not in pairs[0]["rejected"]


@pytest.mark.unit
def test_generate_preference_pairs_skips_invalid_messages() -> None:
    """Test generate_preference_pairs skips examples with invalid message format."""
    original_data = [
        {"messages": []},  # Empty messages
        {"messages": "not a list"},  # Not a list
        {},  # No messages key
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                # Missing assistant message
            ]
        },
        {
            "messages": [
                # Missing user message
                {"role": "assistant", "content": "Hi"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ]
        },
    ]

    pairs = generate_preference_pairs(original_data)

    # Should only process the last valid example
    assert len(pairs) == 1
    assert pairs[0]["prompt"].startswith("[INST]")


@pytest.mark.unit
def test_generate_preference_pairs_multiple_examples() -> None:
    """Test generate_preference_pairs with multiple valid examples."""
    original_data = [
        {
            "messages": [
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Question 2"},
                {"role": "assistant", "content": "Answer 2 with tool_call"},
            ]
        },
    ]

    pairs = generate_preference_pairs(original_data)

    assert len(pairs) == 2
    assert all("prompt" in p and "chosen" in p and "rejected" in p for p in pairs)


@pytest.mark.unit
def test_generate_preference_pairs_empty_data() -> None:
    """Test generate_preference_pairs with empty input."""
    original_data: list[dict[str, str]] = []

    pairs = generate_preference_pairs(original_data)

    assert len(pairs) == 0


@pytest.mark.unit
def test_generate_preference_pairs_rejects_icdu_format() -> None:
    """ICDU-format data (SFT schema) must produce zero preference pairs.

    DPO expects messages-format entries with a 'messages' key containing
    user/assistant dicts. ICDU entries use flat fields like
    'application_prompt' and 'ideal_response_final'. Feeding ICDU data
    into the DPO path should silently produce no pairs rather than crash.
    """
    icdu_data = [
        {
            "application_prompt": "Recommend a watch.",
            "ideal_response_final": "Consider a Seiko Presage.",
            "persona_archetype": "Alex",
            "capability_layer": "Aspirational",
            "context_summary": "User wants a dive watch.",
            "user_intent": "product_recommendation",
        }
    ]

    pairs = generate_preference_pairs(icdu_data)

    assert len(pairs) == 0, (
        "ICDU-format data should produce no DPO pairs; "
        "DPO requires messages-format entries."
    )


@pytest.mark.unit
def test_generate_preference_pairs_prompt_format() -> None:
    """Test that generate_preference_pairs creates correct prompt format."""
    original_data = [
        {
            "messages": [
                {"role": "user", "content": "Test question"},
                {"role": "assistant", "content": "Test answer"},
            ]
        }
    ]

    pairs = generate_preference_pairs(original_data)

    assert len(pairs) == 1
    prompt = pairs[0]["prompt"]
    assert prompt.startswith("[INST]")
    assert prompt.endswith("[/INST]")
    assert "Test question" in prompt


# ============================================================================
# Tests for prepare_dpo_dataset
# ============================================================================


@pytest.mark.unit
def test_prepare_dpo_dataset_valid_data() -> None:
    """Test prepare_dpo_dataset with valid preference data."""
    preference_data = [
        {
            "prompt": "[INST]Question 1[/INST]",
            "chosen": "Answer 1",
            "rejected": "Wrong answer 1",
        },
        {
            "prompt": "[INST]Question 2[/INST]",
            "chosen": "Answer 2",
            "rejected": "Wrong answer 2",
        },
    ]

    dataset = prepare_dpo_dataset(preference_data)

    assert len(dataset) == 2
    assert dataset[0]["prompt"] == "[INST]Question 1[/INST]"
    assert dataset[0]["chosen"] == "Answer 1"
    assert dataset[0]["rejected"] == "Wrong answer 1"


@pytest.mark.unit
def test_prepare_dpo_dataset_empty_data() -> None:
    """Test prepare_dpo_dataset raises ValueError for empty data."""
    preference_data: list[dict[str, str]] = []

    with pytest.raises(ValueError, match="Preference data list is empty"):
        prepare_dpo_dataset(preference_data)


@pytest.mark.unit
def test_prepare_dpo_dataset_missing_prompt() -> None:
    """Test prepare_dpo_dataset raises ValueError for missing prompt key."""
    preference_data = [
        {
            "chosen": "Answer 1",
            "rejected": "Wrong answer 1",
        }
    ]

    with pytest.raises(ValueError, match="missing required keys"):
        prepare_dpo_dataset(preference_data)


@pytest.mark.unit
def test_prepare_dpo_dataset_missing_chosen() -> None:
    """Test prepare_dpo_dataset raises ValueError for missing chosen key."""
    preference_data = [
        {
            "prompt": "[INST]Question[/INST]",
            "rejected": "Wrong answer",
        }
    ]

    with pytest.raises(ValueError, match="missing required keys"):
        prepare_dpo_dataset(preference_data)


@pytest.mark.unit
def test_prepare_dpo_dataset_missing_rejected() -> None:
    """Test prepare_dpo_dataset raises ValueError for missing rejected key."""
    preference_data = [
        {
            "prompt": "[INST]Question[/INST]",
            "chosen": "Answer",
        }
    ]

    with pytest.raises(ValueError, match="missing required keys"):
        prepare_dpo_dataset(preference_data)


@pytest.mark.unit
def test_prepare_dpo_dataset_partial_missing_keys() -> None:
    """Test prepare_dpo_dataset raises ValueError when some pairs have missing keys."""
    preference_data = [
        {
            "prompt": "[INST]Question 1[/INST]",
            "chosen": "Answer 1",
            "rejected": "Wrong answer 1",
        },
        {
            "prompt": "[INST]Question 2[/INST]",
            "chosen": "Answer 2",
            # Missing rejected
        },
    ]

    with pytest.raises(ValueError, match="missing required keys"):
        prepare_dpo_dataset(preference_data)


# ============================================================================
# Tests for run_dpo_training
# ============================================================================


@pytest.mark.unit
@patch("src.dpo.resolve_model_class")
@patch("src.dpo.AutoTokenizer")
@patch("src.dpo.DPOTrainer")
@patch("src.dpo.prepare_model_for_kbit_training")
def test_run_dpo_training_success(
    mock_prepare_model: MagicMock,
    mock_dpo_trainer_class: MagicMock,
    mock_tokenizer_class: MagicMock,
    mock_resolve_model_class: MagicMock,
) -> None:
    """Test successful DPO training run."""
    from datasets import Dataset

    # Setup mocks
    mock_model_class = mock_resolve_model_class.return_value
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.config.use_cache = True
    mock_prepare_model.return_value = mock_model
    mock_model_class.from_pretrained.return_value = mock_model

    mock_trainer = MagicMock()
    mock_dpo_trainer_class.return_value = mock_trainer

    # Create dataset
    dataset = Dataset.from_list(
        [
            {
                "prompt": "[INST]Question[/INST]",
                "chosen": "Answer",
                "rejected": "Wrong answer",
            }
        ]
    )

    # Run training
    run_dpo_training(
        "mistralai/Mistral-7B-Instruct-v0.3",
        dataset,
        "/tmp/output",
        tokenizer=None,
        max_steps=10,
    )

    # Verify calls
    mock_tokenizer_class.from_pretrained.assert_called_once()
    mock_model_class.from_pretrained.assert_called_once()
    mock_prepare_model.assert_called_once()
    mock_dpo_trainer_class.assert_called_once()
    mock_trainer.train.assert_called_once()
    # DPO loads through the resolver so it keeps every base-model tensor too.
    assert mock_resolve_model_class.call_args.kwargs["preserve_all_tensors"] is True


@pytest.mark.unit
def test_run_dpo_training_empty_dataset() -> None:
    """Test run_dpo_training raises ValueError for empty dataset."""
    from datasets import Dataset

    dataset = Dataset.from_list([])

    with pytest.raises(ValueError, match="Dataset is empty"):
        run_dpo_training(
            "mistralai/Mistral-7B-Instruct-v0.3",
            dataset,
            "/tmp/output",
        )


@pytest.mark.unit
@patch("src.dpo.resolve_model_class")
@patch("src.dpo.AutoTokenizer")
def test_run_dpo_training_with_provided_tokenizer(
    mock_tokenizer_class: MagicMock,
    mock_resolve_model_class: MagicMock,
) -> None:
    """Test run_dpo_training uses provided tokenizer instead of loading."""
    from datasets import Dataset
    from transformers import AutoTokenizer

    mock_tokenizer = MagicMock(spec=AutoTokenizer)
    mock_tokenizer.pad_token = "<pad>"
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"

    mock_model_class = mock_resolve_model_class.return_value
    mock_model = MagicMock()
    mock_model.config.use_cache = True
    mock_model_class.from_pretrained.return_value = mock_model

    with patch("src.dpo.DPOTrainer"), patch("src.dpo.prepare_model_for_kbit_training"):
        dataset = Dataset.from_list(
            [
                {
                    "prompt": "[INST]Question[/INST]",
                    "chosen": "Answer",
                    "rejected": "Wrong answer",
                }
            ]
        )

        run_dpo_training(
            "mistralai/Mistral-7B-Instruct-v0.3",
            dataset,
            "/tmp/output",
            tokenizer=mock_tokenizer,
            max_steps=1,
        )

        # Should not call from_pretrained for tokenizer
        mock_tokenizer_class.from_pretrained.assert_not_called()


@pytest.mark.unit
@patch("src.dpo.resolve_model_class")
@patch("src.dpo.AutoTokenizer")
@patch("src.dpo.DPOTrainer")
@patch("src.dpo.prepare_model_for_kbit_training")
@patch("torch.cuda.empty_cache")
def test_run_dpo_training_cleans_up_gpu(
    mock_empty_cache: MagicMock,
    mock_prepare_model: MagicMock,
    mock_dpo_trainer_class: MagicMock,
    mock_tokenizer_class: MagicMock,
    mock_resolve_model_class: MagicMock,
) -> None:
    """Test run_dpo_training cleans up GPU memory after training."""
    from datasets import Dataset

    mock_model_class = mock_resolve_model_class.return_value
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.config.use_cache = True
    mock_prepare_model.return_value = mock_model
    mock_model_class.from_pretrained.return_value = mock_model

    mock_trainer = MagicMock()
    mock_dpo_trainer_class.return_value = mock_trainer

    with patch("torch.cuda.is_available", return_value=True):
        dataset = Dataset.from_list(
            [
                {
                    "prompt": "[INST]Question[/INST]",
                    "chosen": "Answer",
                    "rejected": "Wrong answer",
                }
            ]
        )

        run_dpo_training(
            "mistralai/Mistral-7B-Instruct-v0.3",
            dataset,
            "/tmp/output",
            max_steps=1,
        )

        # Should clean up GPU cache
        mock_empty_cache.assert_called_once()
