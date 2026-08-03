"""Comprehensive tests for inference_with_tools module.

This test suite covers:
- All tool functions and their edge cases
- Tool registration and registry
- Tool execution (parallel and sequential)
- Tool call extraction
- Agent loop behavior
- Security configuration
- Error handling and edge cases
"""

import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Skip tests if required dependencies are not available
pytest.importorskip("requests")
pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.inference_with_tools import (
    SINGLE_ARG_TOOLS,
    TOOL_REGISTRY,
    SecurityConfig,
    _execute_tool,
    _execute_tools_parallel,
    _execute_tools_sequential,
    _safe_calc,
    agent_loop,
    animal_medical_database,
    calc_tool,
    calendar_tool,
    extract_tool_calls,
    get_current_weather,
    job_search_tool,
    news_tool,
    python_repl,
    read_file,
    search_web,
    task_tracker_tool,
    write_file,
)

# ============================================================================
# SecurityConfig Tests
# ============================================================================


class TestSecurityConfig:
    """Tests for SecurityConfig class."""

    @pytest.mark.unit
    def test_validate_file_path_read_mode_allowed(self, tmp_path: Path) -> None:
        """Test path validation for read mode with allowed path."""
        allowed_path = tmp_path / "allowed" / "read"
        allowed_path.mkdir(parents=True)
        test_file = allowed_path / "test.txt"
        test_file.write_text("test")

        with patch.object(SecurityConfig, "ALLOWED_READ_PATH", str(allowed_path)):
            assert SecurityConfig.validate_file_path(str(test_file), write_mode=False)

    @pytest.mark.unit
    def test_validate_file_path_read_mode_denied(self, tmp_path: Path) -> None:
        """Test path validation for read mode with denied path."""
        allowed_path = tmp_path / "allowed" / "read"
        allowed_path.mkdir(parents=True)
        denied_path = tmp_path / "denied"
        denied_path.mkdir()
        test_file = denied_path / "test.txt"
        test_file.write_text("test")

        with patch.object(SecurityConfig, "ALLOWED_READ_PATH", str(allowed_path)):
            assert not SecurityConfig.validate_file_path(
                str(test_file), write_mode=False
            )

    @pytest.mark.unit
    def test_validate_file_path_write_mode_allowed(self, tmp_path: Path) -> None:
        """Test path validation for write mode with allowed path."""
        allowed_path = tmp_path / "allowed" / "write"
        allowed_path.mkdir(parents=True)

        with patch.object(SecurityConfig, "ALLOWED_WRITE_PATH", str(allowed_path)):
            test_file = allowed_path / "test.txt"
            assert SecurityConfig.validate_file_path(str(test_file), write_mode=True)

    @pytest.mark.unit
    def test_validate_file_size_within_limit(self, tmp_path: Path) -> None:
        """Test file size validation for file within limit."""
        test_file = tmp_path / "small.txt"
        test_file.write_text("small content")

        with patch.object(SecurityConfig, "MAX_FILE_SIZE", 1000):
            assert SecurityConfig.validate_file_size(str(test_file))

    @pytest.mark.unit
    def test_validate_file_size_exceeds_limit(self, tmp_path: Path) -> None:
        """Test file size validation for file exceeding limit."""
        test_file = tmp_path / "large.txt"
        # Create a file larger than the limit
        test_file.write_text("x" * 2000)

        with patch.object(SecurityConfig, "MAX_FILE_SIZE", 1000):
            assert not SecurityConfig.validate_file_size(str(test_file))

    @pytest.mark.unit
    def test_validate_file_size_nonexistent(self, tmp_path: Path) -> None:
        """Test file size validation for nonexistent file."""
        test_file = tmp_path / "nonexistent.txt"

        assert not SecurityConfig.validate_file_size(str(test_file))


# ============================================================================
# Tool Registry Tests
# ============================================================================


class TestToolRegistry:
    """Tests for tool registration and registry."""

    @pytest.mark.unit
    def test_tool_registry_contains_expected_tools(self) -> None:
        """Test that all expected tools are registered."""
        expected_tools = [
            "search_web",
            "calc_tool",
            "news_tool",
            "python_repl",
            "read_file",
            "write_file",
            "calendar_tool",
            "task_tracker_tool",
            "job_search_tool",
            "get_current_weather",
            "animal_medical_database",
        ]
        for tool_name in expected_tools:
            assert tool_name in TOOL_REGISTRY, f"Tool {tool_name} not registered"

    @pytest.mark.unit
    def test_single_arg_tools_set(self) -> None:
        """Test that SINGLE_ARG_TOOLS contains expected tools."""
        assert "search_web" in SINGLE_ARG_TOOLS
        assert "calc_tool" in SINGLE_ARG_TOOLS
        assert "python_repl" not in SINGLE_ARG_TOOLS  # Multi-arg tool


# ============================================================================
# Safe Calculation Tests
# ============================================================================


class TestSafeCalc:
    """Tests for _safe_calc function."""

    @pytest.mark.unit
    def test_safe_calc_basic_arithmetic(self) -> None:
        """Test basic arithmetic operations."""
        assert _safe_calc("2 + 3") == "5"
        assert _safe_calc("10 - 4") == "6"
        assert _safe_calc("3 * 4") == "12"
        assert _safe_calc("15 / 3") == "5.0"
        assert _safe_calc("2 ** 3") == "8"
        assert _safe_calc("10 % 3") == "1"

    @pytest.mark.unit
    def test_safe_calc_order_of_operations(self) -> None:
        """Test order of operations."""
        assert _safe_calc("2 + 3 * 4") == "14"
        assert _safe_calc("(2 + 3) * 4") == "20"

    @pytest.mark.unit
    def test_safe_calc_unary_operations(self) -> None:
        """Test unary operations."""
        assert _safe_calc("-5") == "-5"
        assert _safe_calc("+10") == "10"

    @pytest.mark.unit
    def test_safe_calc_invalid_input(self) -> None:
        """Test invalid input handling."""
        assert "Error" in _safe_calc("")
        assert "Error" in _safe_calc(None)  # type: ignore
        assert "Error" in _safe_calc("not a number")

    @pytest.mark.unit
    def test_safe_calc_unsafe_operations(self) -> None:
        """Test that unsafe operations are rejected."""
        unsafe_expressions = [
            "__import__('os')",
            "open('/etc/passwd')",
            "eval('malicious')",
            "exec('code')",
        ]
        for expr in unsafe_expressions:
            result = _safe_calc(expr)
            assert "Error" in result, f"Unsafe expression {expr} was not rejected"


# ============================================================================
# Tool Function Tests
# ============================================================================


class TestCalcTool:
    """Tests for calc_tool function."""

    @pytest.mark.unit
    def test_calc_tool_valid_expression(self) -> None:
        """Test calc_tool with valid expressions."""
        assert calc_tool("2 + 3") == "5"
        assert calc_tool("10 * 5") == "50"
        assert calc_tool("100 / 4") == "25.0"

    @pytest.mark.unit
    def test_calc_tool_invalid_input(self) -> None:
        """Test calc_tool with invalid input."""
        with pytest.raises(ValueError, match="Invalid or missing"):
            calc_tool("")
        with pytest.raises(ValueError, match="Invalid or missing"):
            calc_tool(None)  # type: ignore

    @pytest.mark.unit
    def test_calc_tool_caching(self) -> None:
        """Test that calc_tool results are cached."""
        query = "2 + 3"
        result1 = calc_tool(query)
        result2 = calc_tool(query)
        assert result1 == result2
        assert calc_tool.cache_info().hits >= 1


class TestSearchWeb:
    """Tests for search_web function."""

    @pytest.mark.unit
    @patch("src.inference_with_tools.requests.get")
    def test_search_web_success(self, mock_get: Mock) -> None:
        """Test successful web search."""
        mock_response = Mock()
        mock_response.text = (
            '<a class="result-link" href="http://example.com">Example Title</a>'
        )
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = search_web("test query")
        assert "Title" in result
        assert "URL" in result
        mock_get.assert_called_once()

    @pytest.mark.unit
    @patch("src.inference_with_tools.requests.get")
    def test_search_web_no_results(self, mock_get: Mock) -> None:
        """Test web search with no results."""
        # Clear cache to ensure fresh execution
        search_web.cache_clear()
        mock_response = Mock()
        mock_response.text = "<html><body>No results found</body></html>"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = search_web("test query unique no results")
        assert "No search results found" in result

    @pytest.mark.unit
    @patch("src.inference_with_tools.requests.get")
    def test_search_web_request_exception(self, mock_get: Mock) -> None:
        """Test web search with request exception."""
        # Clear cache to ensure fresh execution
        search_web.cache_clear()
        mock_get.side_effect = Exception("Network error")

        result = search_web("test query exception")
        assert "error" in result

    @pytest.mark.unit
    def test_search_web_invalid_input(self) -> None:
        """Test search_web with invalid input."""
        with pytest.raises(ValueError, match="Invalid or missing"):
            search_web("")
        with pytest.raises(ValueError, match="Invalid or missing"):
            search_web(None)  # type: ignore

    @pytest.mark.unit
    def test_search_web_caching(self) -> None:
        """Test that search_web results are cached."""
        # Clear cache to ensure fresh execution
        search_web.cache_clear()
        query = "test query caching unique"
        with patch("src.inference_with_tools.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.text = "<html><body>results</body></html>"
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            result1 = search_web(query)
            result2 = search_web(query)
            assert result1 == result2
            # Should only be called once due to caching
            assert mock_get.call_count == 1


class TestNewsTool:
    """Tests for news_tool function."""

    @pytest.mark.unit
    def test_news_tool_returns_mock_result(self) -> None:
        """Test that news_tool returns mocked result."""
        result = news_tool("test query")
        assert "Mock news" in result
        assert "test query" in result

    @pytest.mark.unit
    def test_news_tool_caching(self) -> None:
        """Test that news_tool results are cached."""
        query = "test"
        result1 = news_tool(query)
        result2 = news_tool(query)
        assert result1 == result2
        assert news_tool.cache_info().hits >= 1


class TestPythonRepl:
    """Tests for python_repl function."""

    @pytest.mark.unit
    def test_python_repl_valid_code(self) -> None:
        """Test python_repl with valid code."""
        result = python_repl("print(2 + 2)")
        assert "4" in result

    @pytest.mark.unit
    def test_python_repl_no_output(self) -> None:
        """Test python_repl with code that produces no output."""
        result = python_repl("x = 5")
        assert "Code executed successfully" in result

    @pytest.mark.unit
    def test_python_repl_error_handling(self) -> None:
        """Test python_repl error handling."""
        result = python_repl("1 / 0")
        assert "Error" in result

    @pytest.mark.unit
    def test_python_repl_invalid_input(self) -> None:
        """Test python_repl with invalid input."""
        with pytest.raises(ValueError, match="Invalid or missing"):
            python_repl("")
        with pytest.raises(ValueError, match="Invalid or missing"):
            python_repl(None)  # type: ignore


class TestFileOperations:
    """Tests for read_file and write_file functions."""

    @pytest.mark.unit
    def test_read_file_success(self, tmp_path: Path) -> None:
        """Test successful file read."""
        allowed_path = tmp_path / "allowed" / "read"
        allowed_path.mkdir(parents=True)
        test_file = allowed_path / "test.txt"
        test_file.write_text("test content")

        with patch.object(SecurityConfig, "validate_file_path", return_value=True):
            with patch.object(SecurityConfig, "validate_file_size", return_value=True):
                result = read_file(str(test_file))
                assert result == "test content"

    @pytest.mark.unit
    def test_read_file_access_denied(self, tmp_path: Path) -> None:
        """Test read_file with access denied."""
        with patch.object(SecurityConfig, "validate_file_path", return_value=False):
            with pytest.raises(ValueError, match="Access denied"):
                read_file("/unauthorized/path.txt")

    @pytest.mark.unit
    def test_read_file_size_exceeded(self, tmp_path: Path) -> None:
        """Test read_file with file size exceeded."""
        with patch.object(SecurityConfig, "validate_file_path", return_value=True):
            with patch.object(SecurityConfig, "validate_file_size", return_value=False):
                with pytest.raises(ValueError, match="exceeds size limit"):
                    read_file("/path/to/large/file.txt")

    @pytest.mark.unit
    def test_write_file_success(self, tmp_path: Path) -> None:
        """Test successful file write."""
        allowed_path = tmp_path / "allowed" / "write"
        allowed_path.mkdir(parents=True)
        test_file = allowed_path / "test.txt"

        with patch.object(SecurityConfig, "validate_file_path", return_value=True):
            result = write_file(str(test_file), "test content")
            assert "Successfully wrote" in result
            assert test_file.read_text() == "test content"

    @pytest.mark.unit
    def test_write_file_access_denied(self) -> None:
        """Test write_file with access denied."""
        with patch.object(SecurityConfig, "validate_file_path", return_value=False):
            with pytest.raises(ValueError, match="Access denied"):
                write_file("/unauthorized/path.txt", "content")

    @pytest.mark.unit
    def test_write_file_invalid_input(self) -> None:
        """Test write_file with invalid input."""
        with pytest.raises(ValueError, match="Invalid or missing"):
            write_file("", "content")
        with pytest.raises(ValueError, match="Invalid or missing"):
            write_file("path", "")


class TestCalendarTool:
    """Tests for calendar_tool function."""

    @pytest.mark.unit
    def test_calendar_tool_returns_mock_result(self) -> None:
        """Test that calendar_tool returns mocked result."""
        result = calendar_tool("create_event")
        assert "Mock calendar action" in result
        assert "create_event" in result

    @pytest.mark.unit
    def test_calendar_tool_caching(self) -> None:
        """Test that calendar_tool results are cached."""
        action = "create_event"
        result1 = calendar_tool(action)
        result2 = calendar_tool(action)
        assert result1 == result2
        assert calendar_tool.cache_info().hits >= 1


class TestTaskTrackerTool:
    """Tests for task_tracker_tool function."""

    @pytest.mark.unit
    def test_task_tracker_tool_adds_task(self, tmp_path: Path) -> None:
        """Test that task_tracker_tool adds a task to database."""
        db_file = tmp_path / "test_tasks.db"
        with patch("src.inference_with_tools.TASK_DB_FILE", str(db_file)):
            result = task_tracker_tool("Test task description")
            assert "Task" in result
            assert "added to tracker" in result
            assert "Test task description" in result

            # Verify task was actually added
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            cursor.execute("SELECT task_details FROM tasks")
            tasks = cursor.fetchall()
            conn.close()
            assert len(tasks) == 1
            assert tasks[0][0] == "Test task description"

    @pytest.mark.unit
    def test_task_tracker_tool_invalid_input(self) -> None:
        """Test task_tracker_tool with invalid input."""
        with pytest.raises(ValueError, match="Invalid or missing"):
            task_tracker_tool("")
        with pytest.raises(ValueError, match="Invalid or missing"):
            task_tracker_tool(None)  # type: ignore


class TestJobSearchTool:
    """Tests for job_search_tool function."""

    @pytest.mark.unit
    def test_job_search_tool_returns_mock_result(self) -> None:
        """Test that job_search_tool returns mocked result."""
        result = job_search_tool("software engineer")
        assert "Mock job listings" in result
        assert "software engineer" in result

    @pytest.mark.unit
    def test_job_search_tool_invalid_input(self) -> None:
        """Test job_search_tool with invalid input."""
        with pytest.raises(ValueError, match="Invalid or missing"):
            job_search_tool("")


class TestGetCurrentWeather:
    """Tests for get_current_weather function."""

    @pytest.mark.unit
    def test_get_current_weather_returns_mock_result(self) -> None:
        """Test that get_current_weather returns mocked result."""
        result = get_current_weather("New York")
        assert "Mock weather" in result
        assert "New York" in result

    @pytest.mark.unit
    def test_get_current_weather_caching(self) -> None:
        """Test that get_current_weather results are cached."""
        location = "London"
        result1 = get_current_weather(location)
        result2 = get_current_weather(location)
        assert result1 == result2
        assert get_current_weather.cache_info().hits >= 1


class TestAnimalMedicalDatabase:
    """Tests for animal_medical_database function."""

    @pytest.mark.unit
    def test_animal_medical_database_returns_mock_result(self) -> None:
        """Test that animal_medical_database returns mocked result."""
        result = animal_medical_database("dog vaccination")
        assert "Mock animal medical info" in result
        assert "dog vaccination" in result

    @pytest.mark.unit
    def test_animal_medical_database_caching(self) -> None:
        """Test that animal_medical_database results are cached."""
        query = "cat health"
        result1 = animal_medical_database(query)
        result2 = animal_medical_database(query)
        assert result1 == result2
        assert animal_medical_database.cache_info().hits >= 1


# ============================================================================
# Tool Call Extraction Tests
# ============================================================================


class TestExtractToolCalls:
    """Tests for extract_tool_calls function."""

    @pytest.mark.unit
    def test_extract_tool_calls_single_valid(self) -> None:
        """Test extracting a single valid tool call."""
        output = '{"tool_call": {"name": "search_web", "arguments": {"query": "test"}}}'
        calls = extract_tool_calls(output)
        assert len(calls) == 1
        assert calls[0]["name"] == "search_web"
        assert calls[0]["arguments"]["query"] == "test"

    @pytest.mark.unit
    def test_extract_tool_calls_multiple_valid(self) -> None:
        """Test extracting multiple valid tool calls."""
        output = (
            'First: {"tool_call": {"name": "calc_tool", "arguments": {"query": "1+1"}}}. '  # noqa: E501
            'Second: {"tool_call": {"name": "search_web", "arguments": {"query": "test"}}}.'  # noqa: E501
        )
        calls = extract_tool_calls(output)
        assert len(calls) == 2
        assert calls[0]["name"] == "calc_tool"
        assert calls[1]["name"] == "search_web"

    @pytest.mark.unit
    def test_extract_tool_calls_no_tools(self) -> None:
        """Test extracting tool calls when none exist."""
        output = "No tools here. Just regular text."
        calls = extract_tool_calls(output)
        assert len(calls) == 0

    @pytest.mark.unit
    def test_extract_tool_calls_invalid_json(self) -> None:
        """Test extracting tool calls with invalid JSON."""
        output = '{"tool_call": {invalid json}}'
        calls = extract_tool_calls(output)
        assert len(calls) == 0

    @pytest.mark.unit
    def test_extract_tool_calls_malformed_structure(self) -> None:
        """Test extracting tool calls with malformed structure."""
        output = '{"tool_call": {"name": "test"}}'  # Missing arguments
        calls = extract_tool_calls(output)
        assert len(calls) == 0

    @pytest.mark.unit
    def test_extract_tool_calls_mixed_valid_invalid(self) -> None:
        """Test extracting tool calls with mix of valid and invalid."""
        output = (
            'Valid: {"tool_call": {"name": "calc_tool", "arguments": {"query": "1+1"}}}. '  # noqa: E501
            'Invalid: {"tool_call": {"name": "test"}}.'
        )
        calls = extract_tool_calls(output)
        assert len(calls) == 1
        assert calls[0]["name"] == "calc_tool"


# ============================================================================
# Tool Execution Tests
# ============================================================================


class TestExecuteTool:
    """Tests for _execute_tool function."""

    @pytest.mark.unit
    def test_execute_tool_single_arg_tool(self) -> None:
        """Test executing a single-arg tool."""
        result_msg, error_msg = _execute_tool("calc_tool", {"query": "2+2"})
        assert error_msg is None
        assert result_msg is not None
        assert "Tool calc_tool result" in result_msg
        assert "4" in result_msg

    @pytest.mark.unit
    def test_execute_tool_multi_arg_tool(self) -> None:
        """Test executing a multi-arg tool."""
        result_msg, error_msg = _execute_tool(
            "write_file", {"filepath": "/tmp/test.txt", "content": "test"}
        )
        # Will fail due to security, but should return error message
        assert error_msg is not None or result_msg is not None

    @pytest.mark.unit
    def test_execute_tool_unknown_tool(self) -> None:
        """Test executing an unknown tool."""
        result_msg, error_msg = _execute_tool("unknown_tool", {})
        assert error_msg is None
        assert result_msg is not None
        assert "Unknown tool" in result_msg

    @pytest.mark.unit
    def test_execute_tool_missing_arg(self) -> None:
        """Test executing a tool with missing required argument."""
        _result_msg, error_msg = _execute_tool("calc_tool", {})
        assert error_msg is not None
        assert "Missing required argument" in error_msg

    @pytest.mark.unit
    def test_execute_tool_validation_error(self) -> None:
        """Test executing a tool with validation error."""
        _result_msg, error_msg = _execute_tool("calc_tool", {"query": ""})
        assert error_msg is not None
        assert "invalid args" in error_msg


class TestExecuteToolsSequential:
    """Tests for _execute_tools_sequential function."""

    @pytest.mark.unit
    def test_execute_tools_sequential_success(self) -> None:
        """Test sequential execution of multiple tools."""
        tool_calls = [
            {"name": "calc_tool", "arguments": {"query": "1+1"}},
            {"name": "calc_tool", "arguments": {"query": "2+2"}},
        ]
        results = _execute_tools_sequential(tool_calls)
        assert len(results) == 2
        assert all("Tool calc_tool result" in r for r in results)

    @pytest.mark.unit
    def test_execute_tools_sequential_missing_name(self) -> None:
        """Test sequential execution with missing tool name."""
        tool_calls = [
            {"name": "calc_tool", "arguments": {"query": "1+1"}},
            {"arguments": {"query": "2+2"}},  # Missing name
        ]
        results = _execute_tools_sequential(tool_calls)
        assert len(results) == 2
        assert "Error: Tool call missing name" in results[1]

    @pytest.mark.unit
    def test_execute_tools_sequential_empty_list(self) -> None:
        """Test sequential execution with empty list."""
        results = _execute_tools_sequential([])
        assert len(results) == 0


class TestExecuteToolsParallel:
    """Tests for _execute_tools_parallel function."""

    @pytest.mark.unit
    def test_execute_tools_parallel_success(self) -> None:
        """Test parallel execution of multiple tools."""
        tool_calls = [
            {"name": "calc_tool", "arguments": {"query": "1+1"}},
            {"name": "calc_tool", "arguments": {"query": "2+2"}},
        ]
        results = _execute_tools_parallel(tool_calls)
        assert len(results) == 2
        assert all("Tool calc_tool result" in r for r in results)

    @pytest.mark.unit
    def test_execute_tools_parallel_missing_name(self) -> None:
        """Test parallel execution with missing tool name."""
        tool_calls = [
            {"name": "calc_tool", "arguments": {"query": "1+1"}},
            {"arguments": {"query": "2+2"}},  # Missing name
        ]
        results = _execute_tools_parallel(tool_calls)
        assert len(results) == 2
        assert "Error: Tool call missing name" in results[1]

    @pytest.mark.unit
    def test_execute_tools_parallel_empty_list(self) -> None:
        """Test parallel execution with empty list."""
        results = _execute_tools_parallel([])
        assert len(results) == 0


# ============================================================================
# Agent Loop Tests
# ============================================================================


class TestAgentLoop:
    """Tests for agent_loop function."""

    @pytest.mark.unit
    def test_agent_loop_no_tools(self) -> None:
        """Test agent loop terminates immediately when no tools are needed."""

        def mock_pipeline(
            prompt: str, max_new_tokens: int = 512, do_sample: bool = False
        ) -> list[dict[str, str]]:
            return [{"generated_text": "Final answer without tools."}]

        result = agent_loop("Test query", mock_pipeline, max_iterations=5)
        assert result == "Final answer without tools."

    @pytest.mark.unit
    def test_agent_loop_single_tool_iteration(self) -> None:
        """Test agent loop with one tool call iteration."""
        call_count = 0

        def mock_pipeline(
            prompt: str, max_new_tokens: int = 512, do_sample: bool = False
        ) -> list[dict[str, str]]:
            nonlocal call_count
            call_count += 1
            if "Tool results" in prompt:
                return [{"generated_text": "Final integrated response."}]
            return [
                {
                    "generated_text": '{"tool_call": {"name": "calc_tool", "arguments": {"query": "1+1"}}}'  # noqa: E501
                }
            ]

        result = agent_loop("Calculate 1+1", mock_pipeline, max_iterations=5)
        assert result == "Final integrated response."
        assert call_count == 2  # Initial + one after tool execution

    @pytest.mark.unit
    def test_agent_loop_multiple_iterations(self) -> None:
        """Test agent loop with multiple tool call iterations."""
        call_count = 0

        def mock_pipeline(
            prompt: str, max_new_tokens: int = 512, do_sample: bool = False
        ) -> list[dict[str, str]]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [
                    {
                        "generated_text": '{"tool_call": {"name": "calc_tool", "arguments": {"query": "1+1"}}}'  # noqa: E501
                    }
                ]
            elif call_count == 2:
                return [
                    {
                        "generated_text": '{"tool_call": {"name": "calc_tool", "arguments": {"query": "2+2"}}}'  # noqa: E501
                    }
                ]
            else:
                return [{"generated_text": "Final response."}]

        result = agent_loop(
            "Calculate multiple things", mock_pipeline, max_iterations=5
        )
        assert result == "Final response."
        assert call_count == 3

    @pytest.mark.unit
    def test_agent_loop_max_iterations_reached(self) -> None:
        """Test agent loop when max iterations is reached."""

        def mock_pipeline(
            prompt: str, max_new_tokens: int = 512, do_sample: bool = False
        ) -> list[dict[str, str]]:
            return [
                {
                    "generated_text": '{"tool_call": {"name": "calc_tool", "arguments": {"query": "1+1"}}}'  # noqa: E501
                }
            ]

        result = agent_loop("Test", mock_pipeline, max_iterations=2)
        assert "calc_tool" in result or "error" in result

    @pytest.mark.unit
    def test_agent_loop_parallel_execution(self) -> None:
        """Test agent loop with parallel tool execution."""
        call_count = 0

        def mock_pipeline(
            prompt: str, max_new_tokens: int = 512, do_sample: bool = False
        ) -> list[dict[str, str]]:
            nonlocal call_count
            call_count += 1
            if "Tool results" in prompt:
                return [{"generated_text": "Final response."}]
            return [
                {
                    "generated_text": (
                        '{"tool_call": {"name": "calc_tool", "arguments": {"query": "1+1"}}}. '  # noqa: E501
                        '{"tool_call": {"name": "calc_tool", "arguments": {"query": "2+2"}}}.'  # noqa: E501
                    )
                }
            ]

        result = agent_loop(
            "Calculate multiple", mock_pipeline, max_iterations=5, tool_parallel=True
        )
        assert result == "Final response."

    @pytest.mark.unit
    def test_agent_loop_sequential_execution(self) -> None:
        """Test agent loop with sequential tool execution."""
        call_count = 0

        def mock_pipeline(
            prompt: str, max_new_tokens: int = 512, do_sample: bool = False
        ) -> list[dict[str, str]]:
            nonlocal call_count
            call_count += 1
            if "Tool results" in prompt:
                return [{"generated_text": "Final response."}]
            return [
                {
                    "generated_text": '{"tool_call": {"name": "calc_tool", "arguments": {"query": "1+1"}}}'  # noqa: E501
                }
            ]

        result = agent_loop(
            "Calculate", mock_pipeline, max_iterations=5, tool_parallel=False
        )
        assert result == "Final response."

    @pytest.mark.unit
    def test_agent_loop_pipeline_error(self) -> None:
        """Test agent loop handles pipeline errors gracefully."""

        def mock_pipeline(
            prompt: str, max_new_tokens: int = 512, do_sample: bool = False
        ) -> list[dict[str, str]]:
            raise RuntimeError("Pipeline error")

        result = agent_loop("Test", mock_pipeline, max_iterations=5)
        assert "Error in model inference" in result


# ============================================================================
# Load Model Pipeline Tests
# ============================================================================


class TestLoadModelPipeline:
    """Tests for load_model_pipeline function."""

    @pytest.mark.unit
    @patch("src.inference_with_tools.AutoTokenizer")
    @patch("src.inference_with_tools.AutoModelForCausalLM")
    def test_load_model_pipeline_success(
        self, mock_model_class: Mock, mock_tokenizer_class: Mock
    ) -> None:
        """Test successful model pipeline loading."""
        mock_tokenizer = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer.decode.return_value = "Generated text"
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}

        with patch(
            "src.inference_with_tools.torch.cuda.is_available", return_value=False
        ):
            from src.inference_with_tools import load_model_pipeline

            pipeline = load_model_pipeline("/fake/path")
            assert callable(pipeline)

    @pytest.mark.unit
    @patch("src.inference_with_tools.AutoTokenizer")
    def test_load_model_pipeline_tokenizer_error(
        self, mock_tokenizer_class: Mock
    ) -> None:
        """Test model pipeline loading with tokenizer error."""
        mock_tokenizer_class.from_pretrained.side_effect = OSError("Token not found")

        with pytest.raises(RuntimeError, match="Failed to load model"):
            from src.inference_with_tools import load_model_pipeline

            load_model_pipeline("/fake/path")

    @pytest.mark.unit
    @patch("src.inference_with_tools.AutoTokenizer")
    @patch("src.inference_with_tools.AutoModelForCausalLM")
    def test_load_model_pipeline_model_error(
        self, mock_model_class: Mock, mock_tokenizer_class: Mock
    ) -> None:
        """Test model pipeline loading with model error."""
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.side_effect = RuntimeError("Model error")

        with pytest.raises(RuntimeError, match="Failed to load model"):
            from src.inference_with_tools import load_model_pipeline

            load_model_pipeline("/fake/path")


# ============================================================================
# Tool integration (pytest; previously unittest — hermetic mocks, no real HTTP/DB)
# ============================================================================


@pytest.mark.unit
def test_tool_integration_extract_tool_calls_valid() -> None:
    """Test parsing valid tool JSON from output."""
    sample_output = (
        'Advice: {"tool_call": {"name": "search_web", "arguments": {"query": "test"}}}. '  # noqa: E501
        "More text."
    )
    calls = extract_tool_calls(sample_output)
    assert len(calls) == 1
    assert calls[0]["name"] == "search_web"
    assert calls[0]["arguments"]["query"] == "test"


@pytest.mark.unit
def test_tool_integration_extract_tool_calls_invalid() -> None:
    """Test handling invalid or missing JSON."""
    sample_output = "No tools here. {invalid json}"
    calls = extract_tool_calls(sample_output)
    assert len(calls) == 0


@pytest.mark.unit
@patch("src.inference_with_tools.requests.get")
def test_tool_integration_search_web_mocked(mock_get: Mock) -> None:
    """search_web parses results when the HTTP response is valid (no real network)."""
    mock_response = Mock()
    mock_response.text = (
        '<a class="result-link" href="http://example.com">Example Title</a>'
    )
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    search_web.cache_clear()
    result = search_web("python programming tool integration unique")
    assert "Title" in result
    assert "Error" not in result


@pytest.mark.unit
def test_tool_integration_search_web_invalid_args() -> None:
    """Test search_web with invalid args."""
    with pytest.raises(ValueError, match="Invalid or missing"):
        search_web("")


@pytest.mark.unit
@patch("src.inference_with_tools.requests.get")
def test_tool_integration_search_web_cache(mock_get: Mock) -> None:
    """LRU cache returns same result and avoids duplicate HTTP for same query."""
    search_web.cache_clear()
    mock_response = Mock()
    mock_response.text = "<html><body>results</body></html>"
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    query = "python programming cache integration unique"
    result1 = search_web(query)
    result2 = search_web(query)
    assert result1 == result2
    assert mock_get.call_count == 1
    assert search_web.cache_info().hits >= 1


@pytest.mark.unit
def test_tool_integration_calc_tool_valid() -> None:
    """Test calc_tool with safe expression."""
    result = calc_tool("2 + 3 * 4")
    assert result == "14"


@pytest.mark.unit
def test_tool_integration_calc_tool_unsafe() -> None:
    """Test calc_tool rejects unsafe code."""
    result = calc_tool("__import__('os').system('ls')")
    assert "Error" in result


@pytest.mark.unit
def test_tool_integration_calc_tool_cache() -> None:
    """Test LRU caching for calc_tool."""
    calc_tool.cache_clear()
    query = "2 + 3 * 4"
    result1 = calc_tool(query)
    result2 = calc_tool(query)
    assert result1 == result2
    assert calc_tool.cache_info().hits >= 1


@pytest.mark.unit
def test_tool_integration_python_repl_valid() -> None:
    """Test python_repl with safe code."""
    result = python_repl("print(2 + 2)")
    assert "4" in result


@pytest.mark.unit
def test_tool_integration_python_repl_unsafe() -> None:
    """Test python_repl rejects unsafe code."""
    result = python_repl("import os; os.system('ls')")
    assert "Error" in result


@pytest.mark.unit
def test_tool_integration_task_tracker_tool(tmp_path: Path) -> None:
    """task_tracker_tool persists to an isolated DB file."""
    db_file = tmp_path / "tasks_tool_integration.db"
    with patch("src.inference_with_tools.TASK_DB_FILE", str(db_file)):
        result = task_tracker_tool("Test task")
    assert "Task" in result
    assert "added to tracker" in result


@pytest.mark.unit
def test_tool_integration_read_file_invalid_path() -> None:
    """Test read_file rejects invalid path."""
    with pytest.raises(ValueError):
        read_file("/unauthorized/file.txt")


@pytest.mark.unit
def test_tool_integration_agent_loop_termination() -> None:
    """Test loop terminates without tools."""

    def mock_pipe(
        prompt: str, max_new_tokens: int = 512, do_sample: bool = False
    ) -> list[dict[str, str]]:
        return [{"generated_text": "No tools needed."}]

    response = agent_loop("Test query", mock_pipe)
    assert response == "No tools needed."


@pytest.mark.unit
def test_tool_integration_agent_loop_with_tool() -> None:
    """Test loop with one tool call and execution."""

    def mock_pipe(
        input_text: str, max_new_tokens: int = 512, do_sample: bool = False
    ) -> list[dict[str, str]]:
        if "Tool results" in input_text:
            return [{"generated_text": "Integrated response."}]
        return [
            {
                "generated_text": (
                    '{"tool_call": {"name": "calc_tool", "arguments": {"query": "1+1"}}}'  # noqa: E501
                )
            }
        ]

    response = agent_loop("Calculate something", mock_pipe)
    assert response == "Integrated response."


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.integration
    def test_full_agent_loop_with_calc_tool(self) -> None:
        """Test complete agent loop with calculation tool."""
        call_count = 0

        def mock_pipeline(
            prompt: str, max_new_tokens: int = 512, do_sample: bool = False
        ) -> list[dict[str, str]]:
            nonlocal call_count
            call_count += 1
            if "Tool results" in prompt:
                return [{"generated_text": "The calculation result is 4."}]
            return [
                {
                    "generated_text": '{"tool_call": {"name": "calc_tool", "arguments": {"query": "2+2"}}}'  # noqa: E501
                }
            ]

        result = agent_loop("What is 2+2?", mock_pipeline, max_iterations=5)
        assert "4" in result or "calculation" in result.lower()
        assert call_count == 2

    @pytest.mark.integration
    def test_multiple_tools_in_sequence(self) -> None:
        """Test agent loop with multiple different tools."""
        call_count = 0

        def mock_pipeline(
            prompt: str, max_new_tokens: int = 512, do_sample: bool = False
        ) -> list[dict[str, str]]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [
                    {
                        "generated_text": '{"tool_call": {"name": "calc_tool", "arguments": {"query": "1+1"}}}'  # noqa: E501
                    }
                ]
            elif call_count == 2:
                return [
                    {
                        "generated_text": '{"tool_call": {"name": "news_tool", "arguments": {"query": "test"}}}'  # noqa: E501
                    }
                ]
            else:
                return [{"generated_text": "All done."}]

        result = agent_loop("Calculate and get news", mock_pipeline, max_iterations=5)
        assert result == "All done."
        assert call_count == 3
