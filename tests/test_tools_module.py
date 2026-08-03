"""Tests for ``src.tools`` (lightweight tool registry and agent loop).

For ``calc_tool`` / full tool suite in ``src.inference_with_tools``, see
``test_inference_with_tools.py`` and ``test_tools.py``.
"""

from unittest.mock import Mock

import pytest

pytest.importorskip("transformers")

from src.tools import (
    TOOL_REGISTRY,
    TOOL_RESULT_PREFIX,
    UNKNOWN_TOOL_MSG,
    _calc_tool_wrapper,
    _mock_search_web,
    _safe_calc,
    agent_loop,
)


@pytest.mark.unit
def test_safe_calc_basic_arithmetic() -> None:
    """``_safe_calc`` evaluates allowed numeric expressions."""
    assert _safe_calc("2 + 3") == "5"
    assert _safe_calc("3 * 4") == "12"


@pytest.mark.unit
def test_safe_calc_invalid_input() -> None:
    """``_safe_calc`` returns error strings for empty or non-string input."""
    assert "Error" in _safe_calc("")
    assert "Error" in _safe_calc(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_mock_search_web() -> None:
    """``_mock_search_web`` returns a mock result or error when query is missing."""
    assert "Mock search" in _mock_search_web({"query": "hello"})
    assert "Error" in _mock_search_web({})


@pytest.mark.unit
def test_calc_tool_wrapper() -> None:
    """``_calc_tool_wrapper`` delegates to ``_safe_calc`` via query key."""
    assert _calc_tool_wrapper({"query": "1+1"}) == "2"


@pytest.mark.unit
def test_tool_registry_keys() -> None:
    """Registry exposes search_web and calc_tool only."""
    assert set(TOOL_REGISTRY.keys()) == {"search_web", "calc_tool"}
    assert TOOL_REGISTRY["search_web"] is _mock_search_web
    assert TOOL_REGISTRY["calc_tool"] is _calc_tool_wrapper


@pytest.mark.unit
def test_agent_loop_rejects_empty_user_query() -> None:
    """``agent_loop`` raises ValueError for empty user_query."""
    mock_pipeline = Mock()
    with pytest.raises(ValueError, match="user_query cannot be empty"):
        agent_loop("", mock_pipeline)
    with pytest.raises(ValueError, match="user_query cannot be empty"):
        agent_loop("   ", mock_pipeline)


@pytest.mark.unit
def test_agent_loop_rejects_invalid_max_iterations() -> None:
    """``agent_loop`` raises ValueError when max_iterations < 1."""
    mock_pipeline = Mock()
    with pytest.raises(ValueError, match="max_iterations must be at least 1"):
        agent_loop("hi", mock_pipeline, max_iterations=0)


@pytest.mark.unit
def test_agent_loop_no_tool_calls_returns_output() -> None:
    """When the model emits no tool JSON, the generated text is returned."""
    mock_pipeline = Mock(return_value=[{"generated_text": "Final answer."}])
    out = agent_loop("Question", mock_pipeline, max_iterations=3)
    assert out == "Final answer."
    mock_pipeline.assert_called_once()


@pytest.mark.unit
def test_agent_loop_executes_calc_tool_and_second_turn() -> None:
    """Tool results are appended with ``TOOL_RESULT_PREFIX`` and the loop continues."""

    def pipeline_side_effect(prompt: str) -> list[dict[str, str]]:
        if TOOL_RESULT_PREFIX in prompt:
            return [{"generated_text": "Done after tool."}]
        return [
            {
                "generated_text": (
                    '{"tool_call": {"name": "calc_tool", '
                    '"arguments": {"query": "2+2"}}}'
                )
            }
        ]

    mock_pipeline = Mock(side_effect=pipeline_side_effect)
    out = agent_loop("Compute", mock_pipeline, max_iterations=3)
    assert out == "Done after tool."
    assert mock_pipeline.call_count == 2


@pytest.mark.unit
def test_agent_loop_unknown_tool_in_output() -> None:
    """Unknown tool names produce ``UNKNOWN_TOOL_MSG`` in the next prompt."""

    def pipeline_side_effect(prompt: str) -> list[dict[str, str]]:
        if UNKNOWN_TOOL_MSG in prompt:
            return [{"generated_text": "Stopped."}]
        return [
            {
                "generated_text": (
                    '{"tool_call": {"name": "missing_tool", "arguments": {}}}'
                )
            }
        ]

    mock_pipeline = Mock(side_effect=pipeline_side_effect)
    out = agent_loop("Q", mock_pipeline, max_iterations=3)
    assert out == "Stopped."


@pytest.mark.unit
def test_agent_loop_pipeline_raises_runtime_error() -> None:
    """Exceptions from the pipeline are wrapped in RuntimeError."""

    def boom(_: str) -> None:
        raise ConnectionError("offline")

    mock_pipeline = Mock(side_effect=boom)
    with pytest.raises(RuntimeError, match="Model pipeline error"):
        agent_loop("x", mock_pipeline)


@pytest.mark.unit
def test_agent_loop_invalid_pipeline_output_format() -> None:
    """Non-list pipeline output raises RuntimeError."""
    mock_pipeline = Mock(return_value=None)
    with pytest.raises(RuntimeError, match="invalid output format"):
        agent_loop("x", mock_pipeline)


@pytest.mark.unit
def test_agent_loop_max_iterations_returns_partial() -> None:
    """When the model always requests tools, partial output is returned at cap."""
    tool_json = '{"tool_call": {"name": "calc_tool", "arguments": {"query": "1+1"}}}'
    mock_pipeline = Mock(return_value=[{"generated_text": tool_json}])
    out = agent_loop("x", mock_pipeline, max_iterations=2)
    assert tool_json in out or out == tool_json
