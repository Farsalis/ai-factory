"""Tool registry and agent loop for fine-tuned model inference with tool execution.

This module provides a simple agent loop that generates responses using a fine-tuned
model, extracts tool calls from the output, executes them, and iterates until no
more tool calls are needed.
"""

import logging
from collections.abc import Callable
from typing import Any, cast

from transformers import Pipeline

from src.inference_with_tools import _safe_calc, extract_tool_calls

# Configure module-level logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_ITERATIONS = 3
TOOL_RESULT_PREFIX = "Tool results:"
UNKNOWN_TOOL_MSG = "Unknown tool"


def _mock_search_web(args: dict[str, Any]) -> str:
    """Mock web search implementation.

    Args:
        args: Dictionary containing 'query' key with search query string.

    Returns:
        Mock search results string.
    """
    query = args.get("query", "")
    if not query:
        return "Error: Missing 'query' parameter"
    return f"Mock search results for query: {query}"


def _calc_tool_wrapper(args: dict[str, Any]) -> str:
    """Wrapper for calc_tool that extracts query from arguments.

    Args:
        args: Dictionary containing 'query' key with mathematical expression.

    Returns:
        String representation of calculation result or error message.
    """
    query = args.get("query", "0")
    return _safe_calc(str(query))


# Tool registry: Maps tool names to their implementation functions
TOOL_REGISTRY: dict[str, Callable[[dict[str, Any]], str]] = {
    "search_web": _mock_search_web,
    "calc_tool": _calc_tool_wrapper,
}


def agent_loop(
    user_query: str,
    model_pipeline: Pipeline,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> str:
    """Run agent loop: generate, extract tools, execute, re-prompt until no more calls.

    The agent iteratively:
    1. Generates a response using the model pipeline
    2. Extracts tool calls from the response
    3. Executes each tool call
    4. Appends results to the input for the next iteration
    5. Repeats until no more tool calls are detected or max_iterations is reached

    Args:
        user_query: Initial user input query.
        model_pipeline: Loaded Hugging Face pipeline for the fine-tuned model.
        max_iterations: Maximum number of iterations to prevent infinite loops.
            Defaults to 3.

    Returns:
        Final response string after tool integrations, or partial response if
        max_iterations is reached.

    Raises:
        ValueError: If user_query is empty or max_iterations is invalid.
        RuntimeError: If model pipeline fails during generation.
    """
    # Input validation
    if not user_query or not user_query.strip():
        raise ValueError("user_query cannot be empty")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    current_input = user_query.strip()
    output = ""

    for iteration in range(max_iterations):
        logger.info(f"Agent loop iteration {iteration + 1}/{max_iterations}")

        try:
            # Generate response
            pipeline_output = model_pipeline(current_input)
            if not pipeline_output or not isinstance(pipeline_output, list):
                raise RuntimeError("Model pipeline returned invalid output format")

            output = pipeline_output[0].get("generated_text", "")
            if not output:
                logger.warning("Model generated empty response")
                return current_input

        except Exception as e:
            logger.error(f"Model generation failed at iteration {iteration + 1}: {e}")
            raise RuntimeError(f"Model pipeline error: {e}") from e

        # Extract and execute tools
        tool_calls = extract_tool_calls(output)
        if not tool_calls:
            logger.info("No more tool calls detected; returning final response")
            return cast(str, output)

        logger.info(f"Extracted {len(tool_calls)} tool call(s)")

        # Execute tools and collect results
        results: list[str] = []
        for call in tool_calls:
            tool_name = call.get("name")
            if not tool_name:
                logger.warning("Tool call missing 'name' field")
                results.append("Error: Tool call missing name")
                continue

            if tool_name in TOOL_REGISTRY:
                try:
                    tool_args = call.get("arguments", {})
                    if not isinstance(tool_args, dict):
                        logger.warning(
                            f"Tool {tool_name} arguments not a dict: {tool_args}"
                        )
                        tool_args = {}

                    result = TOOL_REGISTRY[tool_name](tool_args)
                    results.append(f"Tool {tool_name} result: {result}")
                    logger.debug(f"Tool {tool_name} executed successfully")

                except Exception as e:
                    error_msg = f"Tool {tool_name} execution failed: {e}"
                    logger.error(error_msg, exc_info=True)
                    results.append(f"Tool {tool_name} error: {e!s}")
            else:
                logger.warning(f"Unknown tool requested: {tool_name}")
                results.append(f"{UNKNOWN_TOOL_MSG}: {tool_name}")

        # Build input for next iteration using efficient string building
        result_lines = "\n".join(results)
        current_input = (
            f"{current_input}\n{output}\n{TOOL_RESULT_PREFIX}\n{result_lines}"
        )

    logger.warning(
        f"Max iterations ({max_iterations}) reached; returning partial response"
    )
    return output
