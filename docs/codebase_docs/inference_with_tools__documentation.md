# src/inference\_with\_tools.py

## 1. Overview

The `inference_with_tools.py` module implements a tool-augmented inference agent loop for fine-tuned language models. It provides a complete system for registering tools via decorators, executing them safely (with caching and parallelization), and running an iterative agent loop that generates model responses, extracts tool calls, executes them, and re-prompts until a final answer is produced.

**Purpose:** Enable fine-tuned language models to interact with external tools during inference through a controlled, secure agent loop.

**Key Responsibilities:**

*   Tool registry with decorator-based registration (`@register_tool`)

*   Safe tool execution with LRU caching for idempotent tools and `concurrent.futures` parallelization

*   Agent loop that iteratively generates -> extracts tool calls -> executes -> re-prompts until no more calls

*   Model loading utilities for inference pipelines

*   Security enforcement via `SecurityConfig` for file access and REPL execution

*   JSON-based tool call extraction from model output via regex parsing

**Connections:**

*   **Parent:** Called by `src/main.py` via `agent_loop()` and `load_model_pipeline()`

*   **Siblings:** `src/tools.py` reuses `extract_tool_calls` from this module

*   **External:** `transformers` (AutoModelForCausalLM, AutoTokenizer), `torch`, `requests`, `sqlite3`

## 2. Directory/Module Map

```
ai-factory/
├── src/
│   ├── inference_with_tools.py  # <-- THIS MODULE
│   ├── main.py                  # CLI entry point, imports agent_loop/load_model_pipeline
│   ├── tools.py                 # Reuses extract_tool_calls from this module
│   ├── train.py                 # Training pipeline (separate)
│   ├── config.py                # ScriptConfig
│   ├── data/                    # ICDU package (SFT; not used at inference)
│   ├── model_setup.py           # validate_linear_attention_kernels (shared)
│   └── utils.py                 # Environment class
├── tests/
│   └── test_inference_with_tools.py  # Comprehensive unit tests
└── docs/
    └── codebase_docs/
        └── inference_with_tools__documentation.md  # This file
```

**Pipeline note:** [main](main__documentation.md) imports **this** module for `--run-inference`, not [tools](tools__documentation.md). Standalone: `python -m src.inference_with_tools --model_path PATH --query "..."`.

## 3. Public Interfaces

| Function              | Signature                                                                                                 | Description                                                                                                       |
| --------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `agent_loop`          | `(user_query: str, model_pipeline: Callable, max_iterations: int = 5, tool_parallel: bool = True) -> str` | Main agent loop: generate -> extract tools -> execute -> re-prompt until no more calls or max\_iterations reached |
| `load_model_pipeline` | `(model_path: str, use_linear_attention_kernels: bool = False) -> Callable[..., list[dict[str, str]]]` | Loads model; validates linear kernels when enabled; returns generate_fn |
| `extract_tool_calls`  | `(model_output: str) -> list[dict[str, Any]]`                                                             | Parses JSON tool calls from model output using regex brace-matching                                               |
| `register_tool`       | `(name: str) -> Callable`                                                                                 | Decorator for registering tools into`TOOL_REGISTRY`                                                               |


### Registered Tool Functions

| Tool Name                 | Function                  | Cached | Signature                              | Description                                 |
| ------------------------- | ------------------------- | ------ | -------------------------------------- | ------------------------------------------- |
| `search_web`              | `search_web`              | Yes    | `(query: str) -> str`                  | Real DuckDuckGo web search                  |
| `calc_tool`               | `calc_tool`               | Yes    | `(query: str) -> str`                  | Safe AST-based math evaluation              |
| `news_tool`               | `news_tool`               | Yes    | `(query: str) -> str`                  | Stub/mocked news search                     |
| `python_repl`             | `python_repl`             | No     | `(code: str) -> str`                   | Restricted Python exec with captured stdout |
| `read_file`               | `read_file`               | No     | `(filepath: str) -> str`               | Path-validated file read with size check    |
| `write_file`              | `write_file`              | No     | `(filepath: str, content: str) -> str` | Path-validated file write                   |
| `calendar_tool`           | `calendar_tool`           | Yes    | `(action: str) -> str`                 | Stub/mocked calendar operations             |
| `task_tracker_tool`       | `task_tracker_tool`       | No     | `(task_details: str) -> str`           | SQLite-backed task tracking                 |
| `job_search_tool`         | `job_search_tool`         | Yes    | `(query: str) -> str`                  | Stub/mocked job search                      |
| `get_current_weather`     | `get_current_weather`     | Yes    | `(location: str) -> str`               | Stub/mocked weather lookup                  |
| `animal_medical_database` | `animal_medical_database` | Yes    | `(query: str) -> str`                  | Stub/mocked animal medical info             |


### Internal Functions

| Function                    | Signature                                                         | Description                                                                          |
| --------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `_safe_calc`                | `(expression: str) -> str`                                        | AST-based safe math evaluator; only allows arithmetic operators and numeric literals |
| `_execute_tool`             | `(tool_name: str, args: dict) -> tuple[str \| None, str \| None]` | Execute a single tool; returns (result\_msg, error\_msg)                             |
| `_execute_tools_parallel`   | `(tool_calls: list[dict]) -> list[str]`                           | Execute tools via`ThreadPoolExecutor`                                                |
| `_execute_tools_sequential` | `(tool_calls: list[dict]) -> list[str]`                           | Execute tools one-by-one in order                                                    |


### Classes

| Class            | Description                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- |
| `SecurityConfig` | Security settings for file access and REPL; provides`validate_file_path()`and`validate_file_size()`class methods |


### Module-Level Constants

| Constant                 | Value             | Description                                    |
| ------------------------ | ----------------- | ---------------------------------------------- |
| `CACHE_SIZE`             | 1000              | LRU cache max size for cached tools            |
| `DEFAULT_MAX_ITERATIONS` | 5                 | Default agent loop iteration limit             |
| `DEFAULT_MAX_NEW_TOKENS` | 512               | Default tokens for model generation            |
| `DUCKDUCKGO_TIMEOUT`     | 10                | Timeout in seconds for DuckDuckGo requests     |
| `MAX_SEARCH_RESULTS`     | 5                 | Max results returned from web search           |
| `TASK_DB_FILE`           | `"tasks.db"`      | SQLite database file for task\_tracker\_tool   |
| `TOOL_RESULT_PREFIX`     | `"Tool results:"` | Prefix string for tool results in agent prompt |
| `UNKNOWN_TOOL_MSG`       | `"Unknown tool"`  | Message for unregistered tool names            |


### Module-Level Data Structures

| Structure          | Type                            | Description                                                              |
| ------------------ | ------------------------------- | ------------------------------------------------------------------------ |
| `TOOL_REGISTRY`    | `dict[str, Callable[..., str]]` | Maps tool names to implementation functions                              |
| `SINGLE_ARG_TOOLS` | `frozenset[str]`                | Tools that take a single string argument (simplified calling convention) |


## 4. Execution and Control Flow

### Agent Loop Flow

```
agent_loop(user_query, model_pipeline, max_iterations, tool_parallel)
    │
    ├─ current_input = user_query
    ├─ iteration = 0
    │
    └─ WHILE iteration < max_iterations:
        │
        ├─ model_pipeline(current_input)
        │   └─ output = generated_text
        │
        ├─ extract_tool_calls(output)
        │   ├─ Regex find: \{"tool_call"
        │   ├─ Brace-matching for JSON boundaries
        │   ├─ json.loads() + validate structure
        │   └─ Returns list of {name, arguments} dicts
        │
        ├─ IF no tool_calls:
        │   └─ RETURN output  (final answer)
        │
        ├─ Execute tools:
        │   ├─ IF tool_parallel:
        │   │   └─ _execute_tools_parallel()
        │   │       └─ ThreadPoolExecutor(max_workers=len(tool_calls))
        │   │           └─ _execute_tool() for each call
        │   └─ ELSE:
        │       └─ _execute_tools_sequential()
        │           └─ _execute_tool() in loop
        │
        ├─ Build next prompt:
        │   └─ current_input = previous_input + output + "Tool results:" + results + "Now integrate and continue:"
        │
        └─ iteration += 1
    │
    └─ IF max_iterations reached:
        └─ RETURN output  (partial response with warning)
```

### Tool Execution Flow

```
_execute_tool(tool_name, args)
    │
    ├─ IF tool_name NOT in TOOL_REGISTRY:
    │   └─ RETURN ("Unknown tool: {name}", None)
    │
    ├─ IF tool_name in SINGLE_ARG_TOOLS:
    │   └─ TOOL_REGISTRY[tool_name](args[first_key])
    │
    └─ ELSE:
        └─ TOOL_REGISTRY[tool_name](**args)
```

### Tool Registration Flow

```
@register_tool("tool_name")
@lru_cache(maxsize=CACHE_SIZE)  # optional
def tool_func(arg: str) -> str:
    ...
    │
    └─ TOOL_REGISTRY["tool_name"] = tool_func
```

### Model Loading Flow

```
load_model_pipeline(model_path)
    │
    ├─ AutoTokenizer.from_pretrained(model_path)
    ├─ AutoModelForCausalLM.from_pretrained(
    │       model_path,
    │       torch_dtype=float16 if cuda else float32,
    │       device_map="auto",
    │       low_cpu_mem_usage=True
    │   )
    │
    └─ Returns generate_fn(prompt, max_new_tokens, do_sample)
        ├─ tokenizer(prompt) -> inputs
        ├─ model.generate(**inputs, max_new_tokens, do_sample)
        └─ tokenizer.decode(outputs[0]) -> [{"generated_text": text}]
```

## 5. Data Flow

### Agent Loop Data Path

```
user_query: str
    │
    ▼
current_input ──────────────────────────────────────────┐
    │                                                   │
    ▼                                                   │
model_pipeline(current_input)                           │
    │                                                   │
    ▼                                                   │
output: str                                             │
    │                                                   │
    ├─► extract_tool_calls(output)                      │
    │       │                                           │
    │       ▼                                           │
    │   tool_calls: list[dict]                          │
    │   [{name: str, arguments: dict}, ...]             │
    │       │                                           │
    │       ├─► _execute_tools_parallel(tool_calls)     │
    │       │       or                                  │
    │       └─► _execute_tools_sequential(tool_calls)   │
    │               │                                   │
    │               ▼                                   │
    │           results: list[str]                      │
    │           ["Tool X result: ...", ...]             │
    │               │                                   │
    │               ▼                                   │
    └─────────── current_input ◄────────────────────────┘
                = prev_input + output + "Tool results:" + results + "Now integrate and continue:"
```

### Tool Call Extraction Data Path

```
model_output: str
    │
    ▼
re.finditer(r'\{"tool_call"', model_output)
    │
    ▼
start_positions: list[int]
    │
    ▼  (for each position)
brace-matching scan -> json_str: str
    │
    ▼
json.loads(json_str) -> tool_call: dict
    │
    ▼  (validate: has "tool_call" key with "name" and "arguments")
tool_calls.append(tool_call["tool_call"])
    │
    ▼
result: list[dict[str, Any]]
```

### SecurityConfig Validation Data Path

```
filepath: str
    │
    ├─► validate_file_path(filepath, write_mode=False)
    │       │
    │       ├─ os.path.abspath(filepath)
    │       ├─ allowed = ALLOWED_READ_PATH or ALLOWED_WRITE_PATH
    │       └─ return abs_path.startswith(allowed)
    │
    └─► validate_file_size(filepath)
            │
            ├─ os.path.exists(filepath)
            └─ os.path.getsize(filepath) <= MAX_FILE_SIZE
```

## 6. Integration Points

### External Dependencies

| Dependency                          | Usage                                               | Notes                                           |
| ----------------------------------- | --------------------------------------------------- | ----------------------------------------------- |
| `transformers.AutoModelForCausalLM` | Load causal LM for inference                        | Uses float16 on CUDA, float32 on CPU            |
| `transformers.AutoTokenizer`        | Tokenizer for model input/output                    | Loaded alongside model                          |
| `torch`                             | Tensor operations, CUDA detection, no\_grad context | `torch.cuda.is_available()`for device selection |
| `requests`                          | HTTP requests for DuckDuckGo search                 | 10s timeout, raises for status                  |
| `sqlite3`                           | Local task database for task\_tracker\_tool         | Creates`tasks.db`file                           |
| `ast`                               | Safe math expression parsing                        | Only arithmetic nodes allowed                   |
| `concurrent.futures`                | ThreadPoolExecutor for parallel tool execution      | max\_workers = number of tool calls             |
| `functools.lru_cache`               | Result caching for idempotent tools                 | maxsize = CACHE\_SIZE (1000)                    |


### Internal Module Dependencies

| Module                               | Functions/Classes Used                                  | Direction                      |
| ------------------------------------ | ------------------------------------------------------- | ------------------------------ |
| `src/main.py`                        | `agent_loop`,`load_model_pipeline`                      | Consumer (calls this module)   |
| `src/tools.py`                       | `extract_tool_calls`                                    | Consumer (reuses function)     |
| `tests/test_inference_with_tools.py` | All public functions,`_safe_calc`,`_execute_tool`, etc. | Consumer (tests this module)   |
| `tests/test_tools.py`                | `calc_tool`                                             | Consumer (tests specific tool) |


### Caller Integration

```
# In src/main.py
from src.inference_with_tools import agent_loop, load_model_pipeline

# Load model and run agent
model_pipe = load_model_pipeline(config.model_path)
response = agent_loop(user_query, model_pipe, max_iterations=5)
```

<!---->

```
# In src/tools.py
from src.inference_with_tools import extract_tool_calls

# Reuse extraction logic for alternate tool wiring
calls = extract_tool_calls(model_output)
```

## 7. Configuration and Conventions

### SecurityConfig Settings

| Setting              | Default               | Env Override               | Description                                |
| -------------------- | --------------------- | -------------------------- | ------------------------------------------ |
| `ALLOWED_READ_PATH`  | `/data/allowed/read`  | `AGENT_ALLOWED_READ_PATH`  | Whitelisted directory for file reads       |
| `ALLOWED_WRITE_PATH` | `/data/allowed/write` | `AGENT_ALLOWED_WRITE_PATH` | Whitelisted directory for file writes      |
| `MAX_FILE_SIZE`      | 1,000,000 (1MB)       | -                          | Maximum file size for read operations      |
| `REPL_TIMEOUT`       | 5 seconds             | -                          | Timeout for REPL execution (reserved)      |
| `REPL_MAX_MEMORY`    | 100MB                 | -                          | Memory limit for REPL execution (reserved) |


### Tool Caching Policy

Tools are cached via `@lru_cache(maxsize=CACHE_SIZE)` when their output is deterministic for a given input:

*   **Cached (idempotent):** `search_web`, `calc_tool`, `news_tool`, `calendar_tool`, `job_search_tool`, `get_current_weather`, `animal_medical_database`

*   **Not cached (stateful/dynamic):** `python_repl`, `read_file`, `write_file`, `task_tracker_tool`

### Single-Arg vs Multi-Arg Tools

Tools in `SINGLE_ARG_TOOLS` use simplified calling: the first argument key's value is passed directly. Multi-arg tools use `**kwargs` unpacking.

*   **Single-arg:** `search_web`, `calc_tool`, `news_tool`, `calendar_tool`, `job_search_tool`, `get_current_weather`, `animal_medical_database`

*   **Multi-arg:** `python_repl`, `read_file`, `write_file`, `task_tracker_tool`

### Tool Call JSON Format

The model is expected to generate tool calls in this JSON structure:

```
{"tool_call": {"name": "tool_name", "arguments": {"key": "value"}}}
```

Multiple tool calls can appear in a single output string; each is extracted independently.

### Agent Loop Prompt Construction

When tools are executed, the next prompt is built as:

```
{current_input}
Previous output: {output}
Tool results:
{result_1}
{result_2}
...
Now integrate and continue:
```

## 8. Extension and Testing Guidance

### Adding New Tools

1.  Define the tool function with a `str` return type

2.  Decorate with `@register_tool("tool_name")`

3.  Optionally add `@lru_cache(maxsize=CACHE_SIZE)` for idempotent tools

4.  Add to `SINGLE_ARG_TOOLS` if the tool takes a single string argument

5.  Validate inputs and raise `ValueError` for invalid arguments

<!---->

```
@register_tool("my_new_tool")
@lru_cache(maxsize=CACHE_SIZE)
def my_new_tool(query: str) -> str:
    if not query or not isinstance(query, str):
        raise ValueError("Invalid or missing 'query'")
    # Implementation here
    return f"Result for {query}"
```

### Testing Patterns

The module includes comprehensive tests in `tests/test_inference_with_tools.py`:

*   **SecurityConfig tests:** Path validation, file size limits, read/write mode checks

*   **Tool registry tests:** Verifies all expected tools are registered, SINGLE\_ARG\_TOOLS membership

*   **Safe calc tests:** Basic arithmetic, order of operations, unary ops, unsafe expression rejection

*   **Individual tool tests:** Input validation, caching behavior, error handling

*   **Tool call extraction tests:** Valid JSON, multiple calls, invalid JSON, malformed structure

*   **Tool execution tests:** Single-arg, multi-arg, unknown tools, missing args, parallel/sequential

*   **Agent loop tests:** No-tool termination, single/multiple iterations, max\_iterations, parallel/sequential, pipeline errors

*   **Model pipeline tests:** Successful loading, tokenizer errors, model errors

### Error Handling

*   **Tool validation:** Raises `ValueError` for invalid/missing arguments

*   **Tool execution:** Returns `(None, error_msg)` tuple for runtime errors

*   **Agent loop:** Catches model inference errors, returns error string

*   **Model loading:** Raises `RuntimeError` with chained exception on failure

*   **File operations:** Raises `ValueError` for path/size violations

## 9. Visualizations

### Runtime Architecture

```
flowchart TB
    MAIN["src.main.run_inference_phase"] --> FIND["_find_model_path<br/>prefer dpo_model else final_merged_model"]
    FIND --> LOAD["load_model_pipeline(model_path)"]
    LOAD --> GEN["generate_fn(prompt,<br/>max_new_tokens=512,<b...e)"]

    MAIN --> LOOP["agent_loop(user_query,<br/>model_pipeline,<br/>max_iterations=5,<br/>tool_parallel=True)"]
    LOOP --> GEN
    GEN --> OUT["generated_text"]
    OUT --> EXTRACT["extract_tool_calls(output)"]
    EXTRACT --> CALLS{"Tool calls found?"}
    CALLS -- no --> FINAL["Return model output"]
    CALLS -- yes --> MODE{"tool_parallel?"}

    MODE -- yes --> PAR["_execute_tools_parallel"]
    MODE -- no --> SEQ["_execute_tools_sequential"]
    PAR --> EXEC["_execute_tool(name, arguments)"]
    SEQ --> EXEC

    EXEC --> ARITY{"SINGLE_ARG_TOOLS member?"}
    ARITY -- yes --> ARG1["Pass first argument value"]
    ARITY -- no --> ARGN["Call with keyword arguments"]
    ARG1 --> REG["TOOL_REGISTRY"]
    ARGN --> REG

    REG --> SEARCH["search_web<br/>DuckDuckGo Lite + requests"]
    REG --> CALC["calc_tool / _safe_calc"]
    REG --> REPL["python_repl"]
    REG --> FILEIO["read_file / write_file"]
    REG --> TASKS["task_tracker_tool<br/>sqlite3 tasks.db"]
    REG --> STUBS["news, calendar, jobs,<br/>weather, animal database"]

    EXEC --> PROMPT["Append previous output + tool results<br/>then continue loop"]
    PROMPT --> GEN
```

### Agent Loop Sequence

```
sequenceDiagram
    participant Main as src.main
    participant Loop as agent_loop
    participant Model as generate_fn
    participant Parser as extract_tool_calls
    participant Exec as _execute_tool(s)
    participant Tool as TOOL_REGISTRY entry

    Main->>Loop: agent_loop(query, model_pipeline, max_iterations=5, tool_parallel)

    loop iteration < max_iterations
        Loop->>Model: model_pipeline(current_input, 512, false)
        Model-->>Loop: generated_text payload
        Loop->>Parser: extract_tool_calls(output)
        Parser-->>Loop: tool_calls[]

        alt no tool calls
            Loop-->>Main: final output
        else one or more tool calls
            alt tool_parallel = true
                par each tool call
                    Loop->>Exec: _execute_tool(name, arguments)
                    Exec->>Tool: registry dispatch
                    Tool-->>Exec: result or error
                and remaining tool calls
                    Loop->>Exec: _execute_tool(name, arguments)
                end
            else tool_parallel = false
                loop each tool call in order
                    Loop->>Exec: _execute_tool(name, arguments)
                    Exec->>Tool: registry dispatch
                    Tool-->>Exec: result or error
                end
            end

            Exec-->>Loop: ordered results list
            Loop->>Loop: rebuild current_input with previous output + tool results + continuation cue
        end
    end
```

### Concurrency and Safety Boundaries

```
flowchart LR
    CALLS["tool_calls[]"] --> MODE{"tool_parallel?"}
    MODE -- yes --> POOL["ThreadPoolExecutor(max_workers=len(tool_calls))"]
    POOL --> SUBMIT["Submit _execute_tool in input order"]
    SUBMIT --> JOIN["Consume future.result() in same order"]
    JOIN --> ORDERED["Ordered results[]"]
    MODE -- no --> SEQ["For each call:<br/>_execute_tool"]
    SEQ --> ORDERED
    ORDERED --> NEXTPROMPT["Join results into next prompt"]

    subgraph SAFETY["Checks enforced in source"]
        CALCSEC["_safe_calc"] --> AST["AST whitelist<br/>constants, BinOp, UnaryOp"]
        SEARCHSEC["search_web"] --> TIMEOUT["quote(query) + timeout=10"]
        READSEC["read_file"] --> READPATH["abspath(path).startswith(ALLOWED_READ_PATH)"]
        READPATH --> SIZE["exists and file size is within 1_000_000 bytes"]
        WRITESEC["write_file"] --> WRITEPATH["abspath(path).startswith(ALLOWED_WRITE_PATH)"]
        REPLSEC["python_repl"] --> GLOBALS["__builtins__ empty<br/>math + print only"]
        REPLSEC -. declared but not enforced .-> LIMITS["REPL_TIMEOUT / REPL_MAX_MEMORY"]
        TASKSEC["task_tracker_tool"] --> SQL["Parameterized sqlite INSERT"]
    end
```

## 10. Mathematical Framing

### Safe Calculator AST Evaluation

The `_safe_calc` function parses mathematical expressions into an AST and evaluates only safe arithmetic nodes:

**Allowed Operators:**

```
+, -, *, /, //, %, **
```

**Evaluation:**

```
eval(expr) = walk(AST(expr))
where walk(node) =
    Constant(n)  -> n
    BinOp(l, op, r) -> allowed_ops[op](walk(l), walk(r))
    UnaryOp(op, x) -> allowed_ops[op](walk(x))
```

**Security constraint:** Only `ast.Constant`, `ast.BinOp`, and `ast.UnaryOp` nodes are permitted. All other AST node types (Call, Attribute, Name, etc.) raise `ValueError`.

### Agent Loop Termination

The agent loop terminates when:

1.  `extract_tool_calls(output) == []` (no tool calls in output)

2.  `iteration >= max_iterations` (safety bound)

**Iteration bound:** `O(max_iterations)` with each iteration costing one model inference pass.

### Caching Effectiveness

For cached tools, repeated identical queries return instantly:

```
cache_hit_rate = hits / (hits + misses)
average_latency = (cache_hits * ~0ms + cache_misses * network_compute_time) / total_calls
```

With `CACHE_SIZE = 1000`, the LRU cache evicts least-recently-used entries when full.

### Parallel Execution Speedup

For `n` independent tool calls executed in parallel:

```
T_parallel = max(T_1, T_2, ..., T_n) + overhead
T_sequential = T_1 + T_2 + ... + T_n
speedup = T_sequential / T_parallel  (theoretical max: n)
```

The actual speedup depends on the GIL (for CPU-bound tools) and network latency (for I/O-bound tools like `search_web`).

***

*Last updated: 2026-08-02.*
