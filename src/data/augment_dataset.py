"""Augment 'Breaking Better' JSONL with tool-calling examples for Mistral-7B.

This script creates variants by injecting tool calls and responses,
simulating realistic tool usage for fine-tuning a model to use tools effectively.

Usage:
    python augment_dataset.py --input breaking_better_training_data_v6.jsonl \\
        --output augmented_data.jsonl --num_variants 5
"""

import argparse
import json
import logging
import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

# Set up verbose logging for traceability
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Predefined tools aligned with 'Breaking Better' themes (life/work/AI advice)
DEFAULT_TOOLS = [
    "search_web",  # For researching latest advice/data
    "calc_tool",  # For goal metrics, financial calcs
    "news_tool",  # For current events in coaching/business
    "python_repl",  # For AI simulations or computations
    "read_file",  # For accessing stored advice/notes
    "write_file",  # For saving plans or results
    "calendar_tool",  # For scheduling tasks
    "task_tracker_tool",  # For tracking goals
    "job_search_tool",  # For career planning
    "get_current_weather",  # For location-based advice
    "animal_medical_database",  # For pet-related advice
]

# Configuration constants
MAX_TOKEN_LENGTH = 32000  # From config.model.max_length
QUERY_WORDS_LIMIT = 5  # Number of words to extract from query for context

# Load Mistral tokenizer for validation (matches base model)
try:
    TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    raise


def load_jsonl(file_path: Path | str) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries.

    Args:
        file_path: Path to the input JSONL file.

    Returns:
        List of parsed JSON objects with 'messages' field.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file path is invalid.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    data = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                example = json.loads(line)
                if not isinstance(example.get("messages"), list):
                    logger.warning(
                        f"Skipping invalid example at line {line_num}: "
                        "Missing or invalid 'messages' field"
                    )
                    continue
                data.append(example)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error at line {line_num}: {e}")
                continue

    logger.info(f"Loaded {len(data)} valid examples from {file_path}")
    return data


def save_jsonl(data: list[dict[str, Any]], file_path: Path | str) -> None:
    """Save a list of dictionaries to a JSONL file.

    Args:
        data: List of JSON-serializable dictionaries.
        file_path: Path to output JSONL file.

    Raises:
        ValueError: If file_path is empty or invalid.
    """
    output_path = Path(file_path)
    if not str(output_path).strip():
        raise ValueError("Output file path is empty or invalid")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for example in data:
            try:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                saved_count += 1
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to serialize example to JSON: {e}")
                continue

    logger.info(f"Saved {saved_count} augmented examples to {file_path}")


def escape_json_string(s: str) -> str:
    """Escape special characters in a string for valid JSON serialization.

    Args:
        s: Input string to escape.

    Returns:
        Escaped string safe for JSON (without surrounding quotes).
    """
    if not isinstance(s, str):
        s = str(s)
    # json.dumps adds quotes, we remove them to get just the escaped content
    return json.dumps(s)[1:-1]


def _generate_calc_args(user_query: str) -> dict[str, str]:
    """Generate calculator tool arguments from user query.

    Args:
        user_query: User query potentially containing numbers.

    Returns:
        Dictionary with 'query' key containing calculation expression.
    """
    numbers = re.findall(r"\d+\.?\d*", user_query)
    if len(numbers) >= 2:
        calc_expr = f"{numbers[0]} + {numbers[1]}"
    elif numbers:
        calc_expr = f"{numbers[0]} * 2"
    else:
        calc_expr = "1 + 1"
    return {"query": calc_expr}


def generate_tool_arguments(tool_name: str, user_query: str) -> dict[str, Any]:
    """Generate realistic tool arguments based on tool name and user query.

    Args:
        tool_name: Name of the tool to generate arguments for.
        user_query: User query to inform argument generation.

    Returns:
        Dictionary of tool arguments with properly escaped strings.
    """
    query_words = user_query.lower().split()[:QUERY_WORDS_LIMIT]
    query_text = " ".join(query_words)

    # Tool-specific argument generators
    tool_arg_generators: dict[str, dict[str, Any]] = {
        "search_web": {"query": escape_json_string(query_text)},
        "calc_tool": _generate_calc_args(user_query),
        "news_tool": {"query": escape_json_string(query_text)},
        "python_repl": {"code": escape_json_string(f"print({query_text})")},
        "read_file": {"filepath": escape_json_string("/data/allowed/read/notes.txt")},
        "write_file": {
            "filepath": escape_json_string("/data/allowed/write/output.txt"),
            "content": escape_json_string(f"Generated from: {user_query[:50]}"),
        },
        "calendar_tool": {
            "action": "create_event",
            "details": escape_json_string(
                f"Meeting about {query_words[0] if query_words else 'task'}"
            ),
        },
        "task_tracker_tool": {
            "task_details": escape_json_string(f"Task: {user_query[:50]}")
        },
        "job_search_tool": {"query": escape_json_string(query_text)},
        "get_current_weather": {
            "location": escape_json_string(
                query_words[-1] if query_words else "New York"
            )
        },
        "animal_medical_database": {"query": escape_json_string(query_text)},
    }

    return tool_arg_generators.get(tool_name, {})


def _generate_tool_response(tool_name: str, args: dict[str, Any]) -> str:
    """Generate a mock tool response string for a given tool and arguments.

    Args:
        tool_name: Name of the tool that was called.
        args: Dictionary of tool arguments.

    Returns:
        Mock tool response string.
    """
    tool_responses: dict[str, str] = {
        "search_web": f"Tool result: Mock search results for '{args.get('query', '')}'",
        "calc_tool": f"Tool result: Calculated {args.get('query', '')} = [mock result]",
        "news_tool": f"Tool result: Mock news for '{args.get('query', '')}'",
        "python_repl": f"Tool result: Executed code: {args.get('code', '')}",
        "read_file": (
            f"Tool result: Read from {args.get('filepath', '')}: [mock content]"
        ),
        "write_file": f"Tool result: Wrote to {args.get('filepath', '')}",
        "calendar_tool": (
            f"Tool result: Calendar action '{args.get('action', '')}' completed"
        ),
        "task_tracker_tool": f"Tool result: Task added: {args.get('task_details', '')}",
        "job_search_tool": (
            f"Tool result: Mock job listings for '{args.get('query', '')}'"
        ),
        "get_current_weather": (
            f"Tool result: Mock weather for {args.get('location', '')}: Sunny, 25°C"
        ),
        "animal_medical_database": (
            f"Tool result: Mock animal medical info for '{args.get('query', '')}'"
        ),
    }
    return tool_responses.get(tool_name, f"Tool result: {tool_name} executed")


def _inject_tool_call_and_response(
    tool_name: str, user_msg: str, injected_content: list[str]
) -> bool:
    """Inject a tool call and its response into the content list.

    Args:
        tool_name: Name of the tool to call.
        user_msg: User message for context.
        injected_content: List to append tool call and response to.

    Returns:
        True if injection succeeded, False otherwise.
    """
    try:
        args = generate_tool_arguments(tool_name, user_msg)
        tool_call = {"tool_call": {"name": tool_name, "arguments": args}}
        tool_call_json = json.dumps(tool_call, ensure_ascii=False)
        injected_content.append(f"Need data: {tool_call_json}")
        injected_content.append(_generate_tool_response(tool_name, args))
        return True
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to generate valid JSON for tool {tool_name}: {e}")
        return False


def generate_tool_variant(
    original_messages: list[dict[str, str]],
    tools: list[str],
    variant_type: str = "single",
) -> list[dict[str, str]] | None:
    """Generate a variant of messages by injecting tool calls.

    Simulates tool responses tied to 'Breaking Better' themes. Creates variants
    with single tool calls, multiple tool calls, or no tool calls.

    Args:
        original_messages: Original chat messages with 'role' and 'content' keys.
        tools: List of available tool names.
        variant_type: Type of variant - "single" (one tool), "multi" (chain),
            or "none" (direct response).

    Returns:
        New messages list with tool injections, or None if generation fails.
    """
    if not original_messages or not tools:
        logger.warning("Skipping variant generation: Empty messages or tools")
        return None

    new_messages = original_messages.copy()
    user_msg = next(
        (msg.get("content", "") for msg in new_messages if msg.get("role") == "user"),
        "",
    )
    assistant_response = next(
        (
            msg.get("content", "")
            for msg in new_messages
            if msg.get("role") == "assistant"
        ),
        "",
    )

    if not user_msg or not assistant_response:
        logger.warning("Skipping variant: Missing user or assistant message")
        return None

    injected_content: list[str] = []

    if variant_type == "none":
        injected_content.append(assistant_response)
    elif variant_type == "single":
        tool_name = random.choice(tools)  # noqa: S311
        if not _inject_tool_call_and_response(tool_name, user_msg, injected_content):
            return None
        injected_content.append(f"Integrated advice: {assistant_response}")
    elif variant_type == "multi":
        num_tools = min(2, len(tools))
        selected_tools = random.sample(tools, num_tools)
        for tool_name in selected_tools:
            if not _inject_tool_call_and_response(
                tool_name, user_msg, injected_content
            ):
                return None
        injected_content.append(f"Integrated advice: {assistant_response}")
    else:
        logger.warning(f"Unknown variant_type: {variant_type}, using 'none'")
        injected_content.append(assistant_response)

    # Replace assistant's content with injected version (find last assistant message)
    for i in range(len(new_messages) - 1, -1, -1):
        if new_messages[i].get("role") == "assistant":
            new_messages[i]["content"] = "\n".join(injected_content)
            break

    return new_messages


def validate_augmented_example(example: dict[str, Any]) -> bool:
    """Validate an augmented example: check tool JSON validity and token length.

    Args:
        example: Augmented example dictionary with 'messages' field.

    Returns:
        True if example is valid (proper structure, token length, JSON validity),
        False otherwise.
    """
    messages = example.get("messages", [])
    if not messages:
        logger.warning("Validation failed: Empty messages")
        return False

    # Concatenate content for token check
    content_parts = [
        msg.get("content", "")
        for msg in messages
        if isinstance(msg, dict) and "content" in msg
    ]
    full_text = " ".join(content_parts)

    # Check token length
    try:
        tokenized = TOKENIZER(full_text, return_length=True, add_special_tokens=False)
        token_length = tokenized["length"][0]
        if token_length > MAX_TOKEN_LENGTH:
            logger.warning(
                f"Example exceeds max token length: {token_length} > {MAX_TOKEN_LENGTH}"
            )
            return False
    except Exception as e:
        logger.error(f"Tokenizer error during validation: {e}")
        return False

    # Validate any tool JSON in messages
    tool_call_pattern = re.compile(r'\{.*?"tool_call".*?\}', re.DOTALL)
    for msg in messages:
        content = msg.get("content", "")
        if "tool_call" in content:
            json_matches = tool_call_pattern.findall(content)
            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    # Validate structure
                    if not isinstance(parsed, dict) or "tool_call" not in parsed:
                        logger.warning(f"Invalid tool call structure: {match}")
                        return False
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid tool JSON: {match[:100]}... - {e}")
                    return False

    return True


def _create_variant_distribution(num_variants: int) -> list[str]:
    """Create a balanced distribution of variant types.

    Args:
        num_variants: Total number of variants to generate.

    Returns:
        List of variant type strings ("single", "multi", "none").
    """
    # Distribute variants evenly across types
    base_count = num_variants // 3
    remainder = num_variants % 3

    variant_types = (
        ["single"] * (base_count + (1 if remainder > 0 else 0))
        + ["multi"] * (base_count + (1 if remainder > 1 else 0))
        + ["none"] * base_count
    )

    return variant_types


def augment_dataset(
    original_data: list[dict[str, Any]], num_variants: int, tools: list[str]
) -> list[dict[str, Any]]:
    """Augment the dataset by creating variants per original example.

    Creates multiple variants of each original example with different tool
    injection patterns (single tool, multiple tools, or no tools).

    Args:
        original_data: Loaded original examples with 'messages' field.
        num_variants: Number of variants to generate per original example.
        tools: List of tool names to use for injections.

    Returns:
        List of augmented examples (originals + variants). Original examples
        are included at the beginning of the list.
    """
    logger.info(
        f"Augmenting dataset: {len(original_data)} examples, "
        f"{num_variants} variants each"
    )

    augmented = original_data.copy()  # Include originals
    variant_types = _create_variant_distribution(num_variants)

    def generate_variants() -> Iterator[dict[str, Any]]:
        """Generator for memory-efficient variant creation."""
        for orig in original_data:
            # Shuffle variant types for each example to add variety
            shuffled_types = variant_types.copy()
            random.shuffle(shuffled_types)

            for variant_type in shuffled_types:
                variant_messages = generate_tool_variant(
                    orig.get("messages", []), tools, variant_type
                )
                if not variant_messages:
                    continue

                new_example = {"messages": variant_messages}
                if validate_augmented_example(new_example):
                    yield new_example
                else:
                    first_msg = orig.get("messages", [{}])[0]
                    content_preview = first_msg.get("content", "")[:50]
                    logger.debug(f"Discarded invalid variant for: {content_preview}...")

    # Convert generator to list for saving
    variants = list(generate_variants())
    augmented.extend(variants)

    logger.info(
        f"Generated {len(variants)} valid variants "
        f"(target: {len(original_data) * num_variants})"
    )
    return augmented


def main() -> None:
    """CLI entry point for dataset augmentation."""
    parser = argparse.ArgumentParser(
        description="Augment JSONL dataset for tool-calling fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Path to output augmented JSONL file"
    )
    parser.add_argument(
        "--num_variants",
        type=int,
        default=5,
        help="Number of variants to generate per original example",
    )
    parser.add_argument(
        "--tools",
        default=",".join(DEFAULT_TOOLS),
        help="Comma-separated list of tool names to use",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.input == args.output:
        raise ValueError(
            "Output file cannot be the same as input file to prevent overwriting"
        )

    if args.num_variants < 0:
        raise ValueError("Number of variants must be non-negative")

    # Parse tools list
    tools_list = [t.strip() for t in args.tools.split(",") if t.strip()]
    if not tools_list:
        raise ValueError("At least one tool must be specified")

    unrecognized_tools = set(tools_list) - set(DEFAULT_TOOLS)
    if unrecognized_tools:
        logger.warning(
            f"Some tools not in default list: {unrecognized_tools}. "
            f"Proceeding with: {tools_list}"
        )

    try:
        original_data = load_jsonl(args.input)
        logger.info(f"Loaded {len(original_data)} original examples")

        augmented_data = augment_dataset(original_data, args.num_variants, tools_list)
        n_orig = len(original_data)
        n_variants = len(augmented_data) - n_orig
        logger.info(
            "Augmentation complete: %s total examples (%s original + %s variants)",
            len(augmented_data),
            n_orig,
            n_variants,
        )

        save_jsonl(augmented_data, args.output)
        logger.info("Augmentation pipeline completed successfully")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"Augmentation error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during augmentation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
