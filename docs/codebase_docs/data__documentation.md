# src/data/

## 1. Overview

The `src/data/` package provides data loading, ICDU (Intent-Context-Dynamic-Unified) formatting, vectorized completion-only collation, and comprehensive data generation scripts. It implements the complete data pipeline from raw JSON/JSONL sources through chat-formatted datasets ready for fine-tuning.

**Purpose:** Manage all data processing, generation, and preparation for AI model fine-tuning.

**Key Responsibilities:**

*   ICDU data format conversion and validation
*   Context perturbation for training diversity (Scenario-Perturbation Method)
*   Chat template formatting with system prompts per capability layer
*   Vectorized completion-only collation for efficient fine-tuning
*   Comprehensive data generation pipeline with persona inference
*   Dataset augmentation and tool-calling variant generation
*   Train/validation splitting and deduplication

**Connections:**

*   **Consumer:** [train](train__documentation.md) via `load_and_prepare_dataset()` and `VectorizedCompletionOnlyCollator`
*   **Dependencies:** `src.config.DataConfig`, `transformers.PreTrainedTokenizer`, `torch`, `datasets`
*   **Format split:** SFT requires **ICDU** fields. DPO / augmentation use **messages** chat JSONL ([dpo](dpo__documentation.md)) — `load_and_prepare_dataset` does **not** load plain chat `messages` as SFT data.
*   See also [OVERVIEW](../OVERVIEW.md)

## 2. Directory/Module Map

```
ai-factory/
├── src/
│   ├── data/
│   │   ├── __init__.py                    # <-- MAIN MODULE: Data loading, formatting, collation
│   │   ├── master_generate_icdu.py        # Comprehensive ICDU generation pipeline
│   │   ├── generate_icdu_dataset.py       # Chat-format to ICDU conversion
│   │   ├── augment_dataset.py             # JSONL augmentation with tool-calling variants
│   │   ├── generate_icdu_validation_dataset.py  # Validation-specific ICDU generation
│   │   ├── generate_icdu_publication_dataset.py  # Publication-quality ICDU generation
│   │   ├── generate_icdu_proactive_dataset.py    # Proactive response ICDU generation
│   │   └── augment_validation_dataset.py  # Validation dataset augmentation
│   ├── train.py                           # Uses VectorizedCompletionOnlyCollator
│   ├── config.py                          # DataConfig definitions
│   ├── model_setup.py                     # Model/tokenizer loading
│   └── utils.py                           # Environment utilities
├── tests/
│   ├── test_data.py                       # Unit tests (tests/test_data.py)
│   ├── test_data_scripts_smoke.py
│   └── test_master_generate_icdu.py
└── docs/
    └── codebase_docs/
        └── data__documentation.md         # This file
```

### SFT vs DPO / augmentation formats

| Format | Used by | Required shape |
| ------ | ------- | -------------- |
| **ICDU** | `load_and_prepare_dataset` → SFT | `application_prompt`, `ideal_response_final`, `persona_archetype`, `capability_layer`, `context_summary`, `user_intent` |
| **messages** JSONL | DPO preference gen, `augment_*.py` | `{"messages":[{"role","content"}, ...]}` |

Chat "standard messages" is **not** an alternate SFT loader path in this package.
## 3. Public Interfaces

### Main Module (**init**.py)

| Function/Class                     | Signature                                                                         | Description                                                                                 |
| ---------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `format_icdu_to_chat`              | `(example: dict[str, Any], tokenizer: PreTrainedTokenizer) -> dict[str, str]`     | Converts ICDU example to chat-formatted string with optional perturbation (50% probability) |
| `load_and_prepare_dataset`         | `(config: DataConfig, tokenizer: PreTrainedTokenizer) -> Dataset`                 | Loads JSON files, applies ICDU formatting, filters empty entries, returns HF Dataset        |
| `VectorizedCompletionOnlyCollator` | \`class(tokenizer, response\_template="Assistant: ", label\_pad\_token\_id=\*\*\* | Masks prompt tokens for completion-only fine-tuning using vectorized sliding window         |


### Internal Functions (**init**.py)

| Function                          | Signature                                          | Description                                                                                    |
| --------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `perturb_context`                 | `(context: str, persona: str, intent: str) -> str` | Applies persona-specific perturbations (Carla→budget, Ben→modern design, Alex→limited edition) |
| `_apply_carla_perturbations`      | `(context: str) -> str`                            | Retired hobbyist + budget constraints perturbation                                             |
| `_apply_ben_perturbations`        | `(context: str) -> str`                            | Anniversary gift + modern design perturbation                                                  |
| `_apply_alex_perturbations`       | `(context: str) -> str`                            | Chronograph + limited edition perturbation                                                     |
| `_adjust_prompt_for_perturbation` | `(prompt: str, context: str) -> str`               | Adjusts user prompt to reflect perturbed context                                               |
| `_format_messages_fallback`       | `(messages: list[dict[str, str]]) -> str`          | Manual fallback formatting when chat\_template unavailable                                     |


### Data Generation Scripts

| Script                                 | Purpose                                              | Key Functions                                  |
| -------------------------------------- | ---------------------------------------------------- | ---------------------------------------------- |
| `master_generate_icdu.py`              | Comprehensive ICDU pipeline with parallel processing | `convert_to_icdu()`,`process_entry()`,`main()` |
| `generate_icdu_dataset.py`             | Chat-format to ICDU conversion with query patterns   | Pattern-based persona/intent/context inference |
| `augment_dataset.py`                   | Tool-calling variant generation                      | `augment_example()`, tool injection            |
| `generate_icdu_validation_dataset.py`  | Validation-specific ICDU generation                  | Similar to main with validation focus          |
| `generate_icdu_publication_dataset.py` | Publication-quality dataset generation               | Higher quality, curated examples               |
| `generate_icdu_proactive_dataset.py`   | Proactive response generation                        | Follow-up question integration                 |
| `augment_validation_dataset.py`        | Validation dataset augmentation                      | Validation-specific augmentation               |


## 4. Execution and Control Flow

### Data Loading and Formatting Flow

```
load_and_prepare_dataset(config, tokenizer)
    │
    ├─ Load JSON files from config.train_file, config.validation_file
    │
    ├─ dataset.map(format_icdu_to_chat)
    │   │
    │   ├─ Validate REQUIRED_ICDU_FIELDS
    │   │   (application_prompt, ideal_response_final, persona_archetype,
    │   │    capability_layer, context_summary, user_intent)
    │   │
    │   ├─ Extract fields from example
    │   │
    │   ├─ [50% probability] perturb_context(context, persona, intent)
    │   │   ├─ "carla" → _apply_carla_perturbations()
    │   │   ├─ "ben" → _apply_ben_perturbations()
    │   │   └─ "alex" → _apply_alex_perturbations()
    │   │
    │   ├─ _adjust_prompt_for_perturbation(prompt, context)
    │   │
    │   ├─ Select system prompt from SYSTEM_PROMPTS by capability_layer
    │   │   ├─ Foundational: "factual and transparent assistant..."
    │   │   ├─ Transformational: "empathetic and insightful assistant..."
    │   │   └─ Aspirational: "knowledgeable and proactive partner..."
    │   │
    │   ├─ Construct messages: [system, user, assistant]
    │   │
    │   ├─ Apply chat_template (or fallback to _format_messages_fallback)
    │   │
    │   └─ Return {"text": formatted_string}
    │
    ├─ Filter empty entries (text.strip() != "")
    │
    └─ Return formatted_dataset {train, validation}
```

### Vectorized Collation Flow

```
VectorizedCompletionOnlyCollator.__call__(features)
    │
    ├─ Extract input_ids, attention_mask from features
    │   ├─ Skip invalid/empty entries
    │   ├─ Create attention_mask if missing
    │   └─ Convert to tensors
    │
    ├─ pad_sequence(input_ids, padding_value=pad_token_id)
    ├─ pad_sequence(attention_masks, padding_value=0)
    │
    ├─ _find_template_start(padded_input_ids)
    │   ├─ Create sliding windows of template_len
    │   ├─ Compare windows with response_template_ids
    │   └─ Return start indices [batch_size]
    │
    ├─ _create_label_mask(padded_input_ids, response_starts)
    │   ├─ Mask tokens before response start
    │   └─ Mask padding tokens
    │
    ├─ labels = padded_input_ids.clone()
    ├─ labels[label_mask] = LABEL_PAD_TOKEN_ID (-100)
    │
    └─ Return {input_ids, attention_mask, labels}
```

### Data Generation Pipeline Flow

```
master_generate_icdu.py main()
    │
    ├─ Parse arguments (input_file, output_dir, augmentation_factor, etc.)
    │
    ├─ load_jsonl(input_file)
    │
    ├─ Parallel processing with Pool(num_processes)
    │   └─ process_entry(entry, config)
    │       ├─ If "messages" in entry:
    │       │   ├─ convert_to_icdu(entry)
    │       │   │   ├─ infer_persona_archetype(query) [45 persona patterns]
    │       │   │   ├─ infer_governing_principle(query) [3 chapters × 3 principles]
    │       │   │   ├─ infer_capability_layer(response) [Aspirational/Transformational/Foundational]
    │       │   │   ├─ infer_user_intent(query) [40+ intent mappings]
    │       │   │   ├─ infer_context_summary(query) [context generation]
    │       │   │   ├─ paraphrase_response() [synonym replacement]
    │       │   │   └─ generate_follow_up_question(persona) [persona-aware questions]
    │       │   │
    │       │   └─ [if augmentation_factor > 1]
    │       │       └─ get_augmentations() → convert_to_icdu(entry, aug)
    │       │
    │       └─ If ICDU entry:
    │           └─ update_icdu_entry() [add follow-up question if missing]
    │
    ├─ ensure_uniqueness(all_entries) [deduplication]
    │
    ├─ [if dataset_size] random.sample(unique_entries, dataset_size)
    │
    ├─ [if run_reports] run_dataset_evaluation()
    │   ├─ compute_diversity(prompts, responses)
    │   └─ visualize_stats() [persona, principle, layer distributions]
    │
    ├─ train_test_split(entries, test_size=validation_split)
    │
    └─ save_jsonl(train_data), save_jsonl(val_data)
```

## 5. Data Flow

### ICDU Data Format

```
ICDU Example Structure:
{
    "icdu_id": "uuid",
    "persona_archetype": "Fitness Seeker > Struggling Starter",
    "governing_principle": "Chapter 2 > Process",
    "capability_layer": "Foundational",
    "user_intent": "To establish a sustainable fitness routine",
    "context_summary": "User is motivated to improve their health...",
    "application_prompt": "I want to get fit",
    "ideal_response_final": "Starting small is key... Would you like to explore how to track your progress?",
    "ideal_response_attributes": ["Encouraging", "Actionable", "Clear"],
    "ideal_response_cot": ["Acknowledge the user's immediate problem.", ...]
}
```

### Chat Format Conversion

```
ICDU Example
    │
    ▼
format_icdu_to_chat(example, tokenizer)
    │
    ├─ Validate REQUIRED_ICDU_FIELDS
    │
    ├─ [50%] Apply perturbation
    │   ├─ context = perturb_context(context, persona, intent)
    │   └─ prompt = _adjust_prompt_for_perturbation(prompt, context)
    │
    ├─ Build messages:
    │   [
    │     {"role": "system", "content": "{SYSTEM_PROMPT}\nContext: {context}\nUser Intent: {intent}\nPersona: {persona}"},
    │     {"role": "user", "content": "{prompt}"},
    │     {"role": "assistant", "content": "{response}"}
    │   ]
    │
    ├─ tokenizer.apply_chat_template(messages, tokenize=***
    │
    └─ Return {"text": "formatted_chat_string"}
```

### Training Data Path

```
JSON Files (train.json, validation.json)
    │
    ▼
load_and_prepare_dataset(config, tokenizer)
    │
    ├─ load_dataset("json", data_files)
    │
    ├─ map(format_icdu_to_chat)
    │   └─ Applies perturbations, system prompts, chat formatting
    │
    ├─ filter(text != "")
    │
    ▼
Dataset {train: Dataset, validation: Dataset}
    │
    ▼
SFTTrainer (with VectorizedCompletionOnlyCollator)
    │
    ├─ Collator masks prompt tokens
    └─ Only assistant response contributes to loss
```

## 6. Integration Points

### External Dependencies

| Dependency                                 | Usage                                   | Notes                                         |
| ------------------------------------------ | --------------------------------------- | --------------------------------------------- |
| `transformers.PreTrainedTokenizer`         | Chat template application, tokenization | Requires chat\_template for proper formatting |
| `datasets.load_dataset`                    | JSON file loading                       | Returns HuggingFace Dataset objects           |
| `torch.Tensor`                             | Tensor operations in collation          | pad\_sequence, sliding window matching        |
| `torch.nn.utils.rnn.pad_sequence`          | Batch padding                           | Handles variable-length sequences             |
| `sklearn.model_selection.train_test_split` | Dataset splitting                       | Used in data generation scripts               |
| `matplotlib.pyplot`                        | Visualization                           | Distribution plots for evaluation             |
| `uuid`                                     | ICDU ID generation                      | Unique identifiers for entries                |


### Internal Module Dependencies

| Module       | Functions/Classes Used                                        |
| ------------ | ------------------------------------------------------------- |
| `src.config` | `DataConfig`(train\_file, validation\_file paths)             |
| `src.train`  | `VectorizedCompletionOnlyCollator`,`load_and_prepare_dataset` |


### Caller Integration

```
# In src/train.py
from src.data import VectorizedCompletionOnlyCollator, load_and_prepare_dataset

# Load and format dataset
dataset = load_and_prepare_dataset(config.data, tokenizer)

# Create collator for completion-only fine-tuning
data_collator = VectorizedCompletionOnlyCollator(
    tokenizer=***
    response_template="Assistant: "
)

# Pass to SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    ...
)
```

## 7. Configuration and Conventions

### Constants

| Constant                    | Value         | Description                                  |
| --------------------------- | ------------- | -------------------------------------------- |
| `PERTURBATION_PROBABILITY`  | 0.5           | Probability of applying context perturbation |
| `LABEL_PAD_TOKEN_ID`        | -100          | Token ID for masked labels (ignored in loss) |
| `DEFAULT_RESPONSE_TEMPLATE` | "Assistant: " | Template marking start of assistant response |


### ICDU Required Fields

```
REQUIRED_ICDU_FIELDS = [
    "application_prompt",      # User's query/prompt
    "ideal_response_final",    # Expected assistant response
    "persona_archetype",       # User persona (e.g., "Fitness Seeker > Struggling Starter")
    "capability_layer",        # Foundational/Transformational/Aspirational
    "context_summary",         # Contextual information
    "user_intent",             # User's goal/intent
]
```

### System Prompts by Capability Layer

```
SYSTEM_PROMPTS = {
    "Foundational": (
        "You are a factual and transparent assistant providing direct, "
        "unambiguous answers to user queries. Focus on clarity and practical "
        "information, ensuring responses are concise and helpful."
    ),
    "Transformational": (
        "You are an empathetic and insightful assistant. Guide users through "
        "complex decisions by reframing problems around their preferences and "
        "asking clarifying questions to ensure clarity in decision-making."
    ),
    "Aspirational": (
        "You are a knowledgeable and proactive partner. Affirm user choices, "
        "provide deep insights, and anticipate unstated needs to foster a "
        "long-term collaborative relationship."
    ),
}
```

### Perturbation Patterns

```
PERTURBATION_PATTERNS = {
    "budget": "budget constraints",
    "modern_design": "modern, tech-inspired design",
    "limited_edition": "limited-edition models",
}

PROMPT_ADJUSTMENTS = {
    "budget": " I'm also looking to stay within a reasonable budget.",
    "modern_design": " I prefer something with a modern, techy look.",
    "limited_edition": " I'm curious about limited-edition options.",
}
```

### Data Generation Configuration

| Parameter             | Default      | Description                                |
| --------------------- | ------------ | ------------------------------------------ |
| `augmentation_factor` | 10           | Number of augmentations per source entry   |
| `validation_split`    | 0.1          | Fraction of data for validation set        |
| `num_processes`       | cpu\_count() | CPU cores for parallel processing          |
| `num_variants`        | 5            | Variants per example for tool-calling      |
| `diversity_threshold` | 0.7          | Jaccard similarity threshold for diversity |


## 8. Extension and Testing Guidance

### Adding New Persona Patterns

1.  **Add to PERSONA\_PATTERNS dict:**

<!---->

```
PERSONA_PATTERNS["New Persona > Subtype"] = r"\b(keyword1|keyword2|keyword3)\b"
```

2.  **Add to USER\_INTENTS dict:**

<!---->

```
USER_INTENTS["keyword"] = "To achieve specific goal"
```

3.  **Add context generation in infer\_context\_summary:**

<!---->

```
base_contexts["keyword"] = "User is..."
```

4.  **Add follow-up questions in generate\_follow\_up\_question:**

<!---->

```
question_bank["New Persona"] = ["Question 1?", "Question 2?"]
```

### Adding New Perturbation Types

1.  **Add pattern to PERTURBATION\_PATTERNS:**

<!---->

```
PERTURBATION_PATTERNS["new_pattern"] = "pattern description"
```

2.  **Add prompt adjustment to PROMPT\_ADJUSTMENTS:**

<!---->

```
PROMPT_ADJUSTMENTS["new_pattern"] = " Additional context for prompt."
```

3.  **Create persona-specific perturbation function:**

<!---->

```
def _apply_new_persona_perturbations(context: str) -> str:
    context = context.replace("old_text", "new_text")
    return f"{context} The user is considering {PERTURBATION_PATTERNS['new_pattern']}."
```

4.  **Update perturb\_context() to handle new persona:**

<!---->

```
elif "new_persona" in persona_lower:
    return _apply_new_persona_perturbations(context)
```

### Testing Patterns

The module should include tests for:

*   **ICDU validation tests:** Verify REQUIRED\_ICDU\_FIELDS enforcement

*   **Perturbation tests:** Test persona-specific perturbations at 50% probability

*   **System prompt selection:** Verify correct prompt per capability layer

*   **Chat template fallback:** Test when tokenizer lacks chat\_template

*   **Collator tests:** Test sliding window template matching, label masking

*   **Empty batch handling:** Test error handling for invalid batches

*   **Data generation tests:** Test inference functions (persona, intent, context)

### Error Handling

*   `load_and_prepare_dataset`: Raises `FileNotFoundError` for missing files, `ValueError` if empty after filtering

*   `format_icdu_to_chat`: Returns empty text for invalid examples, logs warnings

*   `VectorizedCompletionOnlyCollator`: Raises `ValueError` for empty batches, empty template encoding

*   Data generation scripts: Graceful handling of malformed JSON, missing fields

## 9. Visualizations

### src.data Package Architecture

```
flowchart LR
    subgraph DATA["src.data package"]
        subgraph INIT["__init__.py"]
            C["Constants<br/>PERTURBATION_PROBABILITY<br/>REQUIRED_ICDU_FIELDS<br/>SYSTEM_PROMPTS<br/>DEFAULT_RESPONSE_TEMPLATE"]
            P["Perturbation helpers<br/>perturb_context<br/>_apply_carla/_ben/_alex<br/>_adjust_prompt_for_perturbation"]
            F["format_icdu_to_chat"]
            L["load_and_prepare_dataset"]
            VC["VectorizedCompletionOnlyCollator<br/>_find_template_start<br/>_create_label_mask<br/>__call__"]
            FB["_format_messages_fallback"]
        end

        subgraph GEN["ICDU generation and conversion"]
            G1["generate_icdu_dataset.py"]
            G2["generate_icdu_validation_dataset.py"]
            GM["master_generate_icdu.py"]
            GP["generate_icdu_publication_dataset.py"]
            GR["generate_icdu_proactive_dataset.py"]
        end

        subgraph AUG["Tool augmentation"]
            A1["augment_dataset.py"]
            A2["augment_validation_dataset.py"]
        end
    end

    CFG["DataConfig"] --> L
    TOK["PreTrainedTokenizer"] --> F
    TOK --> VC
    HF["datasets.load_dataset"] --> L
    TRAIN["src.train / SFTTrainer"] --> L
    TRAIN --> VC

    L --> F
    F --> P
    F --> FB

    G1 --> ICDU["ICDU JSONL outputs"]
    G2 --> ICDU
    GM --> ICDU
    GP --> ICDU
    GR --> ICDU

    A1 --> MSG["Augmented messages JSONL"]
    A2 --> MSG
```

### ICDU Formatting and Dataset Preparation Flow

```
flowchart TD
    A["load_and_prepare_dataset(config, tokenizer)"] --> B["load_dataset('json', train + validation files)"]
    B -->|load error| BX["Log error and re-raise"]
    B --> C["map(format_icdu_to_chat, remove_columns=original_columns)"]

    C --> D{"required ICDU fields present?"}
    D -- no --> E["Return empty text row<br/>and log warning"]
    D -- yes --> F["Extract prompt, response,<br/>persona, layer, context, intent"]

    F --> G{"random() below perturbation probability?"}
    G -- no --> K["Keep original context and prompt"]
    G -- yes --> H["perturb_context(context, persona, intent)"]
    H --> H1{"persona contains Carla, Ben, or Alex?"}
    H1 -- Carla --> H2["Budget-oriented perturbation"]
    H1 -- Ben --> H3["Modern-design perturbation"]
    H1 -- Alex --> H4["Limited-edition perturbation"]
    H1 -- other --> H5["No specialized perturbation"]
    H2 --> I["_adjust_prompt_for_perturbation"]
    H3 --> I
    H4 --> I
    H5 --> I
    I --> K

    K --> L["Select SYSTEM_PROMPTS[layer]<br/>or DEFAULT_SYSTEM_PROMPT"]
    L --> M["Build messages<br/>system + user + assistant"]
    M --> N{"chat_template available and succeeds?"}
    N -- yes --> O["tokenizer.apply_chat_template(..., tokenize=***
    N -- no --> P["_format_messages_fallback(messages)"]
    O --> Q["Mapped row = {'text': text}"]
    P --> Q
    E --> R["Filter rows where text.strip() != ''"]
    Q --> R
    R --> S{"Any rows remain?"}
    S -- no --> T["Raise ValueError<br/>dataset empty after formatting"]
    S -- yes --> U["Return DatasetDict<br/>train + validation"]
```

### Completion-Only Collator Masking Logic

```
flowchart TD
    A["VectorizedCompletionOnlyCollator.__call__(features)"] --> B["Iterate features"]
    B --> C{"feature has non-empty input_ids?"}
    C -- no --> C1["Skip feature"]
    C -- yes --> D{"attention_mask missing or empty?"}
    D -- yes --> E["Synthesize attention_mask from pad_token_id"]
    D -- no --> F["Use provided attention_mask"]
    E --> G{"lengths match?"}
    F --> G
    G -- no --> H["Pad or truncate attention_mask"]
    G -- yes --> I["Tensorize ids + mask"]
    H --> I
    I --> J["Append valid tensors"]
    C1 --> B
    J --> K{"Any valid tensors collected?"}
    K -- no --> K1["Raise ValueError<br/>empty effective batch"]
    K -- yes --> L["pad_sequence(input_ids)<br/>pad_sequence(attention_masks)"]
    L --> M["_find_template_start(padded_input_ids)"]
    M --> N{"response template found?"}
    N -- yes --> N1["response_start = first_match + template_len"]
    N -- no --> N2["response_start = seq_len<br/>mask everything"]
    N1 --> O["_create_label_mask"]
    N2 --> O
    O --> P["labels = clone(input_ids)<br/>labels[mask] = -100"]
    P --> Q["Return input_ids, attention_mask, labels"]
```

### Augmentation and ICDU-Generation Branches

```
flowchart TD
    RAW["Raw messages JSONL"] --> PATH{"Pipeline family"}

    PATH -->|ICDU generation| ICDU1["Line-by-line or parallel conversion<br/>generate_icdu_* / master_generate_icdu"]
    ICDU1 --> ICDU2{"Entry already in ICDU shape?"}
    ICDU2 -- yes --> ICDU3["Pass through or update selected fields"]
    ICDU2 -- no --> ICDU4["Infer persona, principle, layer,<br/>intent, context, prompt, response"]
    ICDU3 --> ICDU5["Write ICDU train/validation outputs"]
    ICDU4 --> ICDU5

    PATH -->|Tool augmentation| AUG1["augment_dataset / augment_validation_dataset"]
    AUG1 --> AUG2["Create variant distribution<br/>none / single / multi"]
    AUG2 --> AUG3{"Variant type"}
    AUG3 -- none --> AUG4["Keep natural assistant response"]
    AUG3 -- single --> AUG5["Inject one Need data tool_call block"]
    AUG3 -- multi --> AUG6["Inject up to two tool_call blocks"]
    AUG4 --> AUG7["validate_augmented_example"]
    AUG5 --> AUG7
    AUG6 --> AUG7
    AUG7 --> AUG8{"Valid JSONL + token length + tool payloads?"}
    AUG8 -- no --> AUG9["Discard variant"]
    AUG8 -- yes --> AUG10["Write augmented messages outputs"]
```

## 10. Mathematical Framing

### Vectorized Template Matching

The collator uses a sliding window approach to find response template positions:

**Given:**

*   `input_ids`: Padded token IDs `[batch_size, seq_len]`

*   `template_ids`: Response template token IDs `[template_len]`

**Sliding Window Unfold:**

```
unfolded = input_ids.unfold(dimension=1, size=template_len, step=1)
# Shape: [batch_size, num_windows, template_len]
# where num_windows = seq_len - template_len + 1
```

**Template Matching:**

```
matches = (unfolded == template_tensor).all(dim=2)
# Shape: [batch_size, num_windows]
# Boolean: True where window matches template exactly
```

**Finding First Match:**

```
match_indices = argmax(matches, dim=1)  # [batch_size]
no_match_mask = ~matches.any(dim=1)
match_indices[no_match_mask] = seq_len  # No match → mask everything

# For matches, add template_len to get position AFTER template
response_starts = match_indices + template_len  # [batch_size]
```

**Label Masking:**

```
mask = arange(seq_len) < response_starts.unsqueeze(1)  # [batch_size, seq_len]
mask |= (input_ids == pad_token_id)  # Also mask padding

labels = input_ids.clone()
labels[mask] = LABEL_PAD_TOKEN_ID  # -100 (ignored in loss)
```

### Perturbation Probability

Context perturbation follows a Bernoulli distribution:

```
P(perturb) = PERTURBATION_PROBABILITY = 0.5
perturb ~ Bernoulli(0.5)
```

When perturbation occurs, the persona determines the perturbation type:

```
P(perturbation_type | persona) = {
    "budget"        if "carla" in persona.lower()
    "modern_design" if "ben" in persona.lower()
    "limited_edition" if "alex" in persona.lower()
    "none"          otherwise
}
```

### Jaccard Similarity for Diversity

Dataset diversity is measured using Jaccard similarity:

```
J(A, B) = |A ∩ B| / |A ∪ B|

where A, B are word sets from different responses/prompts

avg_jaccard = (1/N) Σ J(sample_i, sample_j)
is_diverse = avg_jaccard < threshold (default: 0.7)
```

### Data Augmentation Factor

For each source entry, the augmentation factor determines variant count:

```
num_augmented = augmentation_factor - 1  # -1 for original
total_entries = num_source × augmentation_factor (approximately)
```

After deduplication:

```
unique_entries = deduplicate(all_entries)
final_size = min(dataset_size, len(unique_entries)) if dataset_size specified
```

***

*Last updated: 2026-08-02.*
