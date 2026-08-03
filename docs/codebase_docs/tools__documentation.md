# src/tools.py

## 1. Overview

The `tools.py` module provides a lightweight tool registry and agent loop for fine-tuned model inference with tool execution. It implements a simple iterative generate-extract-execute cycle where a language model generates responses, tool calls are extracted from the output, tools are executed, and results are appended to the context for the next iteration.

**Purpose:** Provide a simple agent loop for model inference with basic tool execution capabilities.

**Key Responsibilities:**

*   Safe mathematical expression evaluation via AST parsing
*   Mock web search tool implementation
*   Tool registry mapping tool names to implementation functions
*   Iterative agent loop combining model generation, tool extraction, and tool execution
*   Context management by appending tool results to conversation history

**Connections:**

*   **Not used by main pipeline:** [main](main__documentation.md) calls [inference_with_tools](inference_with_tools__documentation.md). This module is a lightweight alternate agent.
*   **Dependencies:** `src.inference_with_tools.extract_tool_calls`
*   **External:** `transformers.Pipeline`, `ast` (standard library)

**Comparison with **`inference_with_tools.py`**:**

*   Simpler tool set: 2 tools (search\_web, calc\_tool) vs 11 tools
*   Lower iteration limit: max 3 iterations vs 5
*   Mock implementations for search tools vs real API integrations
*   No streaming, step callbacks, or handler protocol support

## 2. Directory/Module Map

```
ai-factory/
├── src/
│   ├── tools.py              # <-- THIS MODULE
│   ├── inference_with_tools.py  # extract_tool_calls function, fuller tool implementation
│   ├── main.py               # Full pipeline uses inference_with_tools, not this module
│   ├── train.py              # Training pipeline
│   ├── config.py             # Configuration management
│   ├── data/               # ICDU package (SFT)
│   ├── model_setup.py        # Model/tokenizer loading
│   └── utils.py              # Environment utilities
├── tests/
│   ├── test_tools.py          # Unit tests for tools.py
│   └── test_tools_module.py   # Additional tools-module tests
└── docs/
    └── codebase_docs/
        └── tools__documentation.md  # This file
```

## 3. Public Interfaces

| Function             | Signature                                                                     | Description                                                                                                                       |
| -------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `agent_loop`         | `(user_query: str, model_pipeline: Pipeline, max_iterations: int = 3) -> str` | Main agent loop: generates responses, extracts tool calls, executes tools, iterates until no more calls or max iterations reached |
| `_safe_calc`         | `(expression: str) -> str`                                                    | Safely evaluate a mathematical expression using AST parsing; only allows basic arithmetic operations                              |
| `_mock_search_web`   | `(args: dict[str, Any]) -> str`                                               | Mock web search implementation returning placeholder results                                                                      |
| `_calc_tool_wrapper` | `(args: dict[str, Any]) -> str`                                               | Wrapper for`_safe_calc`that extracts query from arguments dict                                                                    |


### Constants

| Constant                 | Value             | Description                                      |
| ------------------------ | ----------------- | ------------------------------------------------ |
| `DEFAULT_MAX_ITERATIONS` | `3`               | Default maximum iterations for the agent loop    |
| `TOOL_RESULT_PREFIX`     | `"Tool results:"` | Prefix for tool results appended to context      |
| `UNKNOWN_TOOL_MSG`       | `"Unknown tool"`  | Error message prefix for unrecognized tool names |


### Module-Level Variables

| Variable        | Type                                         | Description                                 |
| --------------- | -------------------------------------------- | ------------------------------------------- |
| `TOOL_REGISTRY` | `dict[str, Callable[[dict[str, Any]], str]]` | Maps tool names to implementation functions |
| `logger`        | `logging.Logger`                             | Module-level logger instance                |


## 4. Execution and Control Flow

### Agent Loop Flow

```
agent_loop(user_query, model_pipeline, max_iterations)
    │
    ├─ Validate inputs (non-empty query, max_iterations >= 1)
    │
    ├─ Initialize current_input = user_query
    │
    └─ FOR iteration in range(max_iterations):
        │
        ├─ Generate response
        │   └─ model_pipeline(current_input) → output
        │
        ├─ Extract tool calls
        │   └─ extract_tool_calls(output) → tool_calls[]
        │
        ├─ IF no tool_calls:
        │   └─ RETURN output (final response)
        │
        ├─ Execute each tool call
        │   ├─ Lookup tool_name in TOOL_REGISTRY
        │   ├─ IF found: TOOL_REGISTRY[tool_name](args) → result
        │   └─ IF not found: append "Unknown tool" message
        │
        └─ Build next iteration input
            └─ current_input = previous_input + output + "Tool results:" + results
    
    └─ RETURN output (partial response after max iterations)
```

### Safe Calculation Flow

```
_safe_calc(expression)
    │
    ├─ Validate input (non-empty string)
    │
    ├─ Parse expression with ast.parse(expression, mode="eval")
    │
    └─ Recursively evaluate AST nodes:
        ├─ ast.Constant → return value
        ├─ ast.BinOp → evaluate left, right, apply operator
        ├─ ast.UnaryOp → evaluate operand, apply operator
        └─ Reject any other node types (security)
```

### Tool Execution Flow

```
For each tool_call in tool_calls:
    │
    ├─ Extract tool_name and arguments
    │
    ├─ IF tool_name in TOOL_REGISTRY:
    │   ├─ Call TOOL_REGISTRY[tool_name](arguments)
    │   ├─ On success: append "Tool {name} result: {result}"
    │   └─ On error: append "Tool {name} error: {error}"
    │
    └─ ELSE:
        └─ append "Unknown tool: {name}"
```

## 5. Data Flow

### Agent Loop Data Flow

```
user_query: str
    │
    ▼
current_input = user_query.strip()
    │
    ├─► model_pipeline(current_input)
    │       │
    │       ▼
    │   output: str (generated text)
    │       │
    │       ├─► extract_tool_calls(output)
    │       │       │
    │       │       ▼
    │       │   tool_calls: list[dict] with "name" and "arguments"
    │       │       │
    │       │       ▼
    │       │   TOOL_REGISTRY[tool_name](arguments)
    │       │       │
    │       │       ▼
    │       │   results: list[str]
    │       │
    │       └─► Build next input:
    │           current_input = f"{current_input}\n{output}\nTool results:\n{results}"
    │
    └─► RETURN output (when no more tool calls)
```

### Tool Argument Extraction

```
tool_call dict:
{
    "name": str,        # Tool name (e.g., "search_web", "calc_tool")
    "arguments": dict   # Tool arguments (e.g., {"query": "2+3"})
}
    │
    ▼
TOOL_REGISTRY[name](arguments)
    │
    ├─ _mock_search_web({"query": "..."})
    │       │
    │       ▼
    │   "Mock search results for query: ..."
    │
    └─ _calc_tool_wrapper({"query": "2+3"})
            │
            ▼
        _safe_calc("2+3")
            │
            ▼
        "5"
```

## 6. Integration Points

### External Dependencies

| Dependency              | Usage                          | Notes                                             |
| ----------------------- | ------------------------------ | ------------------------------------------------- |
| `transformers.Pipeline` | Model inference in agent\_loop | Loaded Hugging Face pipeline for fine-tuned model |
| `ast`                   | Safe expression parsing        | Standard library, used for secure math evaluation |
| `operator`              | Arithmetic operations          | Maps AST operators to Python operators            |


### Internal Module Dependencies

| Module                     | Functions/Classes Used |
| -------------------------- | ---------------------- |
| `src.inference_with_tools` | `extract_tool_calls`   |


### Caller Integration

```
# In src/main.py (optional inference phase)
from src.tools import agent_loop

# Used for the optional inference phase
response = agent_loop(
    user_query=user_query,
    model_pipeline=pipeline,
    max_iterations=3
)
```

### Tool Registry

```
TOOL_REGISTRY: dict[str, Callable[[dict[str, Any]], str]] = {
    "search_web": _mock_search_web,
    "calc_tool": _calc_tool_wrapper,
}
```

## 7. Configuration and Conventions

### Constants

| Constant                 | Value             | Purpose                                                        |
| ------------------------ | ----------------- | -------------------------------------------------------------- |
| `DEFAULT_MAX_ITERATIONS` | `3`               | Prevents infinite loops; default max iterations for agent loop |
| `TOOL_RESULT_PREFIX`     | `"Tool results:"` | Delimiter separating tool results in context                   |
| `UNKNOWN_TOOL_MSG`       | `"Unknown tool"`  | Error message prefix for unrecognized tools                    |


### Safe Calculator Allowed Operations

The `_safe_calc` function uses AST parsing to allow only safe arithmetic:

| AST Node       | Python Operator | Description    |
| -------------- | --------------- | -------------- |
| `ast.Add`      | `+`             | Addition       |
| `ast.Sub`      | `-`             | Subtraction    |
| `ast.Mult`     | `*`             | Multiplication |
| `ast.Div`      | `/`             | True division  |
| `ast.Pow`      | `**`            | Exponentiation |
| `ast.Mod`      | `%`             | Modulo         |
| `ast.FloorDiv` | `//`            | Floor division |
| `ast.USub`     | `-`             | Unary negation |
| `ast.UAdd`     | `+`             | Unary positive |


### Error Handling Conventions

*   Invalid/empty expressions return `"Error: Invalid expression input"` or `"Error: Invalid expression - {details}"`
*   Unknown tools return `"Unknown tool: {tool_name}"`
*   Tool execution failures return `"Tool {name} error: {error}"`
*   Empty model responses return the current input
*   Pipeline errors raise `RuntimeError`

### Logging

*   Module uses `logging.getLogger(__name__)`
*   INFO level: iteration progress, tool calls detected, loop completion
*   WARNING level: empty responses, unknown tools, max iterations reached
*   ERROR level: tool execution failures, pipeline errors
*   DEBUG level: successful tool executions

## 8. Extension and Testing Guidance

### Adding New Tools

1.  Implement tool function with signature `(args: dict[str, Any]) -> str`
2.  Add entry to `TOOL_REGISTRY` dict

<!---->

```
def my_new_tool(args: dict[str, Any]) -> str:
    """Tool description."""
    param = args.get("param", "")
    # Implementation
    return result

TOOL_REGISTRY["my_new_tool"] = my_new_tool
```

### Testing Patterns

The module includes unit tests in `tests/test_tools.py`:

*   **Calculation tests:** Verify `_safe_calc` evaluates expressions correctly
*   **Expression safety tests:** Verify unsafe expressions are rejected
*   **Mock search tests:** Verify `_mock_search_web` returns expected format
*   **Agent loop tests:** Test iteration, tool extraction, and exit conditions

### Safe Calculator Testing

```
# Valid expressions
_safe_calc("2 + 3")      # "5"
_safe_calc("10 * 5 - 3") # "47"
_safe_calc("-2 ** 3")    # "-8"

# Invalid/unsafe expressions
_safe_calc("__import__('os').system('ls')")  # Error
_safe_calc("open('/etc/passwd')")             # Error
_safe_calc("")                                # Error: Invalid expression input
```

## 9. Visualizations

### Simplified Agent-Loop Architecture

```
flowchart TB
    USER["user_query"] --> VALIDATE["Validate query and max_iterations"]
    VALIDATE --> LOOP["agent_loop(..., max_iterations=3)"]
    LOOP --> PIPE["transformers Pipeline"]
    PIPE --> OUT["output = pipeline_output[0].generated_text"]
    OUT --> EXTRACT["src.inference_with_tools.extract_tool_calls"]
    EXTRACT --> CALLS{"Tool calls found?"}
    CALLS -- no --> RET["Return output"]
    CALLS -- yes --> FOR["For each tool call"]
    FOR --> LOOKUP{"tool_name in TOOL_REGISTRY?"}
    LOOKUP -- search_web --> SEARCH["_mock_search_web(arguments)"]
    LOOKUP -- calc_tool --> WRAP["_calc_tool_wrapper(arguments)"]
    WRAP --> CALC["_safe_calc(str(arguments['query']))"]
    LOOKUP -- unknown --> UNKNOWN["Append unknown-tool message"]
    SEARCH --> RESULTS["results[]"]
    CALC --> RESULTS
    UNKNOWN --> RESULTS
    RESULTS --> REPROMPT["current_input = current_input<br/>+ output + tool results"]
    REPROMPT --> PIPE
```

### Safe Calculator Decision Flow

```
flowchart TD
    INPUT["expression: str"] --> VALID{"Non-empty string?"}
    VALID -- no --> ERR1["Return invalid-expression error"]
    VALID -- yes --> PARSE["ast.parse(expression, mode='eval')"]
    PARSE --> NODE{"Node type"}
    NODE -- Constant --> VALUE["Return literal value"]
    NODE -- BinOp --> BIN["Evaluate left/right<br/>apply allowed operator"]
    NODE -- UnaryOp --> UNARY["Evaluate operand<br/>apply allowed unary operator"]
    NODE -- other --> ERR2["Raise ValueError<br/>and return error string"]
    VALUE --> DONE["str(result)"]
    BIN --> DONE
    UNARY --> DONE
```

## 10. Mathematical Framing

### Safe Expression Evaluation

The `_safe_calc` function implements a restricted subset of Python arithmetic:

**Allowed Operations:**

```
E -> number
E -> E + E        (Addition)
E -> E - E        (Subtraction)
E -> E * E        (Multiplication)
E -> E / E        (Division)
E -> E ** E       (Exponentiation)
E -> E % E        (Modulo)
E -> E // E       (Floor Division)
E -> -E           (Unary Negation)
E -> +E           (Unary Positive)
```

**Security Model:**

*   AST parsing with `mode="eval"` ensures only expressions, not statements
*   Whitelist approach: only approved AST node types are evaluated
*   No access to builtins, imports, attribute access, or function calls
*   All string inputs are validated before parsing

### Agent Loop Complexity

**Maximum Iterations:** `DEFAULT_MAX_ITERATIONS = 3`

**Context Growth per Iteration:**

```
L(n) = L(n-1) + len(output) + len(tool_results) + overhead
```

Where:

*   `L(0) = len(user_query)`
*   Each iteration appends the generated output and tool results
*   Context grows linearly with iterations

**Tool Call Extraction:**

*   Calls `extract_tool_calls(output)` from `inference_with_tools`
*   Returns list of dicts with `name` and `arguments` keys
*   Empty list signals loop termination

### Comparison: tools.py vs inference\_with\_tools.py

| Feature               | tools.py                    | inference\_with\_tools.py  |
| --------------------- | --------------------------- | -------------------------- |
| Tools                 | 2 (search\_web, calc\_tool) | 11 (web, code, math, etc.) |
| Max iterations        | 3                           | 5                          |
| Search implementation | Mock                        | Real API calls             |
| Streaming             | No                          | Yes                        |
| Step callbacks        | No                          | Yes                        |
| Handler protocol      | No                          | Yes                        |
| Complexity            | Simple                      | Full-featured              |

***

*Last updated: 2026-08-02.*
