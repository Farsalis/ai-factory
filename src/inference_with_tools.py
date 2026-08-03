"""Inference with tool-augmented agent loop for fine-tuned language models.

This module provides a complete inference system with tool execution capabilities:
- Tool registry with decorator-based registration
- Safe tool execution with caching and parallelization
- Agent loop that iteratively generates responses and executes tools
- Model loading utilities for inference

Usage:
    python inference_with_tools.py --model_path ./model --query "Your query here"
    python inference_with_tools.py --test  # Run unit tests
"""

import argparse
import ast
import concurrent.futures
import contextlib
import io
import json
import logging
import operator
import os
import re
import sqlite3
import sys
from collections.abc import Callable
from functools import lru_cache
from typing import Any, cast
from urllib.parse import quote

import requests
import torch

# Avoid heavy torchvision dependency during text-only generation
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_IMAGE_TRANSFORMS_DISABLED", "1")

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model_setup import validate_linear_attention_kernels

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
CACHE_SIZE = 1000
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MAX_NEW_TOKENS = 512
DUCKDUCKGO_TIMEOUT = 10
MAX_SEARCH_RESULTS = 5
TASK_DB_FILE = "tasks.db"
TOOL_RESULT_PREFIX = "Tool results:"
UNKNOWN_TOOL_MSG = "Unknown tool"

# Tools that take a single string argument (for simplified calling)
SINGLE_ARG_TOOLS = frozenset(
    [
        "search_web",
        "calc_tool",
        "news_tool",
        "calendar_tool",
        "job_search_tool",
        "get_current_weather",
        "animal_medical_database",
    ]
)


class SecurityConfig:
    """Security settings for file access and REPL execution.

    Provides validation methods for file operations and execution environments
    to prevent unauthorized access and resource exhaustion.
    """

    ALLOWED_READ_PATH = os.environ.get("AGENT_ALLOWED_READ_PATH", "/data/allowed/read")
    ALLOWED_WRITE_PATH = os.environ.get(
        "AGENT_ALLOWED_WRITE_PATH", "/data/allowed/write"
    )
    MAX_FILE_SIZE = 1_000_000  # 1MB
    REPL_TIMEOUT = 5  # seconds
    REPL_MAX_MEMORY = 100 * 1024 * 1024  # 100MB

    @classmethod
    def validate_file_path(cls, filepath: str, write_mode: bool = False) -> bool:
        """Validate file path against security restrictions.

        Args:
            filepath: Path to validate.
            write_mode: If True, validate against write path; otherwise read path.

        Returns:
            True if path is allowed, False otherwise.
        """
        abs_path = os.path.abspath(filepath)
        allowed_path = cls.ALLOWED_WRITE_PATH if write_mode else cls.ALLOWED_READ_PATH
        return abs_path.startswith(os.path.abspath(allowed_path))

    @classmethod
    def validate_file_size(cls, filepath: str) -> bool:
        """Check if file size is within limits.

        Args:
            filepath: Path to file to check.

        Returns:
            True if file exists and is within size limit, False otherwise.
        """
        return (
            os.path.exists(filepath) and os.path.getsize(filepath) <= cls.MAX_FILE_SIZE
        )


# Tool registry: Maps tool names to their implementation functions
TOOL_REGISTRY: dict[str, Callable[..., str]] = {}


def register_tool(name: str) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """Decorator for DRY tool registration.

    Args:
        name: Tool name to register.

    Returns:
        Decorator function that registers the tool.
    """

    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        TOOL_REGISTRY[name] = func
        return func

    return decorator


def _safe_calc(expression: str) -> str:
    """Safely evaluate a mathematical expression using AST parsing.

    Only allows basic arithmetic operations and numeric literals.
    Rejects any code execution attempts for security.

    Args:
        expression: Mathematical expression string (e.g., "2 + 3 * 4").

    Returns:
        String representation of the result, or error message if invalid.
    """
    if not expression or not isinstance(expression, str):
        return "Error: Invalid expression input"

    # Allowed operations for safe evaluation
    allowed_operators: dict[type[ast.AST], Callable[..., Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval_node(node: ast.AST) -> Any:
        """Recursively evaluate AST nodes, only allowing safe operations."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op = allowed_operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
            return cast(Callable[[Any, Any], Any], op)(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            op = allowed_operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
            return cast(Callable[[Any], Any], op)(operand)
        else:
            raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except (ValueError, SyntaxError, TypeError) as e:
        logger.warning(f"Invalid calculation expression: {expression} - {e}")
        return f"Error: Invalid expression - {e!s}"
    except Exception as e:
        logger.error(f"Unexpected error in calculation: {e}")
        return f"Error: Calculation failed - {e!s}"


@register_tool("search_web")
@lru_cache(maxsize=CACHE_SIZE)
def search_web(query: str) -> str:
    """Real web search using DuckDuckGo (free, no API key). Cached by query.

    Args:
        query: Search query string.

    Returns:
        Formatted search results or error message.

    Raises:
        ValueError: If query is invalid or missing.
    """
    if not query or not isinstance(query, str):
        raise ValueError("Invalid or missing 'query'")

    try:
        url = f"https://lite.duckduckgo.com/lite/?q={quote(query)}"
        response = requests.get(url, timeout=DUCKDUCKGO_TIMEOUT)
        response.raise_for_status()
        matches = re.findall(
            r'<a class="result-link" href="(.*?)">(.*?)</a>', response.text, re.DOTALL
        )
        results = [
            f"Title: {title.strip()}\nURL: {href.strip()}"
            for href, title in matches[:MAX_SEARCH_RESULTS]
        ]
        return "\n".join(results) if results else "No search results found."
    except requests.RequestException as e:
        logger.error(f"Search web error: {e}")
        return f"Error during web search: {e!s}"
    except Exception as e:
        logger.error(f"Unexpected error in search_web: {e}")
        return f"Unexpected error: {e!s}"


@register_tool("calc_tool")
@lru_cache(maxsize=CACHE_SIZE)
def calc_tool(query: str) -> str:
    """Safe calculation using AST parsing. Cached by query.

    Args:
        query: Math expression string.

    Returns:
        Result as string or error message.

    Raises:
        ValueError: If query is invalid or missing.
    """
    if not query or not isinstance(query, str):
        raise ValueError("Invalid or missing 'query'")

    return _safe_calc(query)


@register_tool("news_tool")
@lru_cache(maxsize=CACHE_SIZE)
def news_tool(query: str) -> str:
    """Stub for news search (requires NewsAPI key; mocked). Cached by query.

    Args:
        query: News search query.

    Returns:
        Mocked news results.
    """
    logger.info(f"News tool called with query: {query}")
    return (
        f"Mock news for '{query}': Breaking news on habits - "
        "consistency key to transformation."
    )


@register_tool("python_repl")
def python_repl(code: str) -> str:
    """Executes Python code safely with restricted environment.

    Not cached due to dynamic output. Uses restricted globals to prevent
    unsafe operations.

    Args:
        code: Python code string.

    Returns:
        Execution result or error message.

    Raises:
        ValueError: If code is invalid or missing.
    """
    if not code or not isinstance(code, str):
        raise ValueError("Invalid or missing 'code'")

    try:
        safe_globals = {"__builtins__": {}, "math": __import__("math"), "print": print}
        output = []
        with contextlib.redirect_stdout(io.StringIO()) as f:
            exec(code, safe_globals, {})  # noqa: S102
            captured_output = f.getvalue()
            if captured_output:
                output.append(captured_output)
        return "\n".join(output) if output else "Code executed successfully."
    except Exception as e:
        logger.error(f"Python REPL error: {e}")
        return f"Error executing code: {e!s}"


@register_tool("read_file")
def read_file(filepath: str) -> str:
    """Reads a file from allowed path with size restrictions.

    Not cached (file content may change). Validates path and file size
    against security restrictions.

    Args:
        filepath: Path to file.

    Returns:
        File contents or error message.

    Raises:
        ValueError: If filepath is invalid, missing, or access is denied.
    """
    if not filepath or not isinstance(filepath, str):
        raise ValueError("Invalid or missing 'filepath'")

    if not SecurityConfig.validate_file_path(filepath):
        raise ValueError(f"Access denied: Filepath {filepath} not in allowed directory")
    if not SecurityConfig.validate_file_size(filepath):
        raise ValueError(f"File {filepath} exceeds size limit or does not exist")

    try:
        with open(filepath, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"File read error: {e}")
        return f"Error reading file: {e!s}"


@register_tool("write_file")
def write_file(filepath: str, content: str) -> str:
    """Writes content to a file in allowed path.

    Not cached (write operation). Validates path against security restrictions.

    Args:
        filepath: Path to file.
        content: Content to write.

    Returns:
        Success message or error message.

    Raises:
        ValueError: If filepath or content is invalid, missing, or access is denied.
    """
    if (
        not filepath
        or not content
        or not isinstance(filepath, str)
        or not isinstance(content, str)
    ):
        raise ValueError("Invalid or missing 'filepath' or 'content'")

    if not SecurityConfig.validate_file_path(filepath, write_mode=True):
        raise ValueError(
            f"Access denied: Filepath {filepath} not in allowed write directory"
        )

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        logger.error(f"File write error: {e}")
        return f"Error writing file: {e!s}"


@register_tool("calendar_tool")
@lru_cache(maxsize=CACHE_SIZE)
def calendar_tool(action: str) -> str:
    """Stub for calendar operations (requires API; mocked). Cached by action.

    Args:
        action: Calendar action (e.g., 'create_event').

    Returns:
        Mocked result.
    """
    logger.info(f"Calendar tool called with action: {action}")
    return f"Mock calendar action '{action}' completed."


@register_tool("task_tracker_tool")
def task_tracker_tool(task_details: str) -> str:
    """Adds a task to a local SQLite database.

    Not cached (database state changes). Creates the database and table
    if they don't exist.

    Args:
        task_details: Task description.

    Returns:
        Success message with task ID or error message.

    Raises:
        ValueError: If task_details is invalid or missing.
    """
    if not task_details or not isinstance(task_details, str):
        raise ValueError("Invalid or missing 'task_details'")

    try:
        conn = sqlite3.connect(TASK_DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_details TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("INSERT INTO tasks (task_details) VALUES (?)", (task_details,))
        conn.commit()
        task_id = cursor.lastrowid
        conn.close()
        return f"Task {task_id} added to tracker: '{task_details}' (Status: pending)"
    except sqlite3.Error as e:
        logger.error(f"SQLite error adding task: {e}")
        return f"Error adding task: {e!s}"
    except Exception as e:
        logger.error(f"Unexpected error adding task: {e}")
        return f"Unexpected error: {e!s}"


@register_tool("job_search_tool")
@lru_cache(maxsize=CACHE_SIZE)
def job_search_tool(query: str) -> str:
    """Job search using a free endpoint (mocked; real impl needs SerpAPI).

    Cached by query.

    Args:
        query: Job search query.

    Returns:
        Mocked job results.

    Raises:
        ValueError: If query is invalid or missing.
    """
    if not query or not isinstance(query, str):
        raise ValueError("Invalid or missing 'query'")

    logger.info(f"Job search for: {query}")
    return (
        f"Mock job listings for '{query}': Software Engineer at "
        "TechCorp, Data Analyst at DataInc."
    )


@register_tool("get_current_weather")
@lru_cache(maxsize=CACHE_SIZE)
def get_current_weather(location: str) -> str:
    """Stub for weather lookup (requires API; mocked). Cached by location.

    Args:
        location: Location for weather.

    Returns:
        Mocked weather data.
    """
    logger.info(f"Weather tool called for location: {location}")
    return f"Mock weather for {location}: Sunny, 25°C."


@register_tool("animal_medical_database")
@lru_cache(maxsize=CACHE_SIZE)
def animal_medical_database(query: str) -> str:
    """Stub for animal medical data lookup (mocked). Cached by query.

    Args:
        query: Medical query.

    Returns:
        Mocked medical info.
    """
    logger.info(f"Animal medical database called with query: {query}")
    return (
        f"Mock animal medical info for '{query}': Consult a "
        "veterinarian for accurate diagnosis."
    )


def extract_tool_calls(model_output: str) -> list[dict[str, Any]]:
    """Extract tool call JSONs from the model's generated text.

    Parses JSON tool calls from model output using regex matching and
    validates the structure.

    Args:
        model_output: Raw string output from the fine-tuned model.

    Returns:
        List of parsed tool call dicts, or empty list if none found.
    """
    tool_calls = []
    # Find all positions where "tool_call" appears in JSON-like structures.
    # Try a simpler approach: find JSON objects containing "tool_call"
    # Look for opening brace followed by "tool_call" and try to parse from there
    start_positions = []
    for match in re.finditer(r'\{"tool_call"', model_output):
        start_positions.append(match.start())

    for start_pos in start_positions:
        # Try to find the matching closing brace
        # Start with brace_count = 1 because we're already inside the opening brace
        brace_count = 1
        end_pos = start_pos
        for i in range(start_pos + 1, len(model_output)):
            if model_output[i] == "{":
                brace_count += 1
            elif model_output[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break

        if end_pos > start_pos:
            try:
                json_str = model_output[start_pos:end_pos]
                tool_call = json.loads(json_str)
                if (
                    "tool_call" in tool_call
                    and "name" in tool_call["tool_call"]
                    and "arguments" in tool_call["tool_call"]
                ):
                    tool_calls.append(tool_call["tool_call"])
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"Invalid tool JSON in output: {json_str[:100]} - Error: {e}"
                )

    if tool_calls:
        logger.info(f"Extracted {len(tool_calls)} tool call(s) from output")

    return tool_calls


def _execute_tool(
    tool_name: str,
    args: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Execute a single tool and return result or error.

    Args:
        tool_name: Name of the tool to execute.
        args: Arguments dictionary for the tool.

    Returns:
        Tuple of (result_message, error_message). One will be None.
    """
    if tool_name not in TOOL_REGISTRY:
        return (f"{UNKNOWN_TOOL_MSG}: {tool_name}", None)

    try:
        logger.debug(f"Executing tool: {tool_name} with args: {args}")

        # Handle single-arg tools vs multi-arg tools
        if tool_name in SINGLE_ARG_TOOLS:
            arg_key = next(iter(args), None)
            if arg_key is None:
                raise ValueError(f"Missing required argument for {tool_name}")
            result = TOOL_REGISTRY[tool_name](args[arg_key])
        else:
            result = TOOL_REGISTRY[tool_name](**args)

        return (f"Tool {tool_name} result: {result}", None)

    except ValueError as ve:
        logger.warning(f"Tool {tool_name} validation error: {ve}")
        return (None, f"Tool {tool_name} invalid args: {ve!s}")
    except Exception as e:
        logger.error(f"Tool {tool_name} execution failed: {e}", exc_info=True)
        return (None, f"Tool {tool_name} error: {e!s}")


def _execute_tools_parallel(
    tool_calls: list[dict[str, Any]],
) -> list[str]:
    """Execute tools in parallel using ThreadPoolExecutor.

    Args:
        tool_calls: List of tool call dictionaries.

    Returns:
        List of result messages.
    """
    if not tool_calls:
        return []

    results: list[str] = []
    # Use at least 1 worker even for empty list (though we already handled that)
    max_workers = max(1, len(tool_calls))
    futures: list[
        tuple[
            str | None,
            concurrent.futures.Future[tuple[str | None, str | None]] | None,
            str | None,
        ]
    ] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for call in tool_calls:
            tool_name = call.get("name")
            args = call.get("arguments", {})

            if not tool_name:
                # Store placeholder for this position to maintain order
                futures.append((None, None, "Error: Tool call missing name"))
                continue

            logger.info(f"Submitting tool {tool_name} for parallel execution")
            future = executor.submit(_execute_tool, tool_name, args)
            futures.append((tool_name, future, None))

        for tool_name, future, error_msg in futures:  # type: ignore[assignment]
            if error_msg:
                # This was a missing name error
                results.append(error_msg)
            else:
                if future is None:
                    raise RuntimeError(
                        f"Tool {tool_name} had no future and no error message"
                    )
                try:
                    result_msg, err_msg = future.result()
                    if err_msg:
                        results.append(err_msg)
                    elif result_msg is not None:
                        results.append(result_msg)
                except Exception as e:
                    logger.error(f"Tool {tool_name} future execution failed: {e}")
                    results.append(f"Tool {tool_name} error: {e!s}")

    return results


def _execute_tools_sequential(
    tool_calls: list[dict[str, Any]],
) -> list[str]:
    """Execute tools sequentially.

    Args:
        tool_calls: List of tool call dictionaries.

    Returns:
        List of result messages.
    """
    results = []
    for call in tool_calls:
        tool_name = call.get("name")
        args = call.get("arguments", {})

        if not tool_name:
            logger.warning("Tool call missing 'name' field")
            results.append("Error: Tool call missing name")
            continue

        result_msg, error_msg = _execute_tool(tool_name, args)
        if error_msg:
            results.append(error_msg)
        elif result_msg is not None:
            results.append(result_msg)

    return results


def agent_loop(
    user_query: str,
    model_pipeline: Callable[..., list[dict[str, str]]],
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tool_parallel: bool = True,
) -> str:
    """Run an agent loop: generate, extract tools, execute, re-prompt until no calls.

    The agent iteratively:
    1. Generates a response using the model pipeline
    2. Extracts tool calls from the response
    3. Executes tools (in parallel or sequentially)
    4. Appends results to input for next iteration
    5. Repeats until no more tool calls or max_iterations reached

    Args:
        user_query: Initial user input.
        model_pipeline: Loaded Hugging Face pipeline function for the fine-tuned model.
        max_iterations: Maximum iterations to prevent infinite loops.
        tool_parallel: If True, execute tools in parallel; otherwise sequentially.

    Returns:
        Final integrated response string.
    """
    current_input = user_query
    iteration = 0

    while iteration < max_iterations:
        logger.info(
            "Agent loop iteration %s/%s: Generating model response",
            iteration + 1,
            max_iterations,
        )

        try:
            output = model_pipeline(
                current_input, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, do_sample=False
            )[0]["generated_text"]
        except Exception as e:
            logger.error(f"Model generation error: {e}", exc_info=True)
            return f"Error in model inference: {e!s}"

        tool_calls = extract_tool_calls(output)
        if not tool_calls:
            logger.info("No more tool calls detected; returning final response")
            return output

        # Execute tools (parallel or sequential)
        if tool_parallel:
            results = _execute_tools_parallel(tool_calls)
        else:
            results = _execute_tools_sequential(tool_calls)

        # Build input for next iteration
        tool_results_str = "\n".join(results)
        current_input = (
            f"{current_input}\n"
            f"Previous output: {output}\n"
            f"{TOOL_RESULT_PREFIX}\n"
            f"{tool_results_str}\n"
            f"Now integrate and continue:"
        )

        iteration += 1

    logger.warning(
        f"Max iterations ({max_iterations}) reached; returning partial response"
    )
    return output


def load_model_pipeline(
    model_path: str,
    use_linear_attention_kernels: bool = False,
) -> Callable[..., list[dict[str, str]]]:
    """Load the fine-tuned model and tokenizer into a pipeline function.

    :args:
        model_path: Path to the merged model directory.
        use_linear_attention_kernels: Require causal_conv1d + fla when True.

    :returns:
        Pipeline function that takes (prompt, max_new_tokens, do_sample) and
        returns list with generated_text dict.

    :raises:
        OSError: If model or tokenizer cannot be loaded.
        RuntimeError: If model loading fails or linear-attention deps are missing.
    """
    logger.info(f"Loading model from: {model_path}")

    validate_linear_attention_kernels(use_linear_attention_kernels)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        def generate_fn(
            prompt: str,
            max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
            do_sample: bool = False,
        ) -> list[dict[str, str]]:
            """Generate text using the loaded model.

            Args:
                prompt: Input prompt text.
                max_new_tokens: Maximum number of tokens to generate.
                do_sample: Whether to use sampling (vs greedy decoding).

            Returns:
                List containing dict with 'generated_text' key.
            """
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
                )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            text_str = text if isinstance(text, str) else str(text)
            return [{"generated_text": text_str}]

        logger.info(f"Successfully loaded model from: {model_path}")
        return generate_fn

    except Exception as e:
        logger.error(f"Model loading error: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e


def main() -> None:
    """CLI entry point for running inference with tools."""
    parser = argparse.ArgumentParser(
        description="Run inference with tools on fine-tuned model."
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to the merged fine-tuned model"
    )
    parser.add_argument("--query", required=True, help="User query to process")

    args = parser.parse_args()

    try:
        model_pipe = load_model_pipeline(args.model_path)
        final_response = agent_loop(args.query, model_pipe)
        print(f"Final Response:\n{final_response}")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Main execution error: {e}", exc_info=True)
        print(f"Error: {e!s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
