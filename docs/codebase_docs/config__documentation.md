# src/config.py

## Table of Contents

1.  [Overview](#overview)
2.  [Directory / Module Map](#directory--module-map)
3.  [Public Interfaces](#public-interfaces)
4.  [Execution and Control Flow](#execution-and-control-flow)
5.  [Data Flow](#data-flow)
6.  [Integration Points](#integration-points)
7.  [Configuration and Conventions](#configuration-and-conventions)
8.  [Extension and Testing Guidance](#extension-and-testing-guidance)
9.  [Visualizations](#visualizations)
10. [Mathematical Framing](#mathematical-framing)

***

## Overview

**Purpose:** The `config.py` module defines the complete configuration structure for the ai-factory training pipeline using Pydantic for validation and type safety. It provides a strongly-typed, validated configuration system that serves as the central contract between all pipeline phases (SFT training, merging, DPO, inference).

**Key Responsibilities:**

*   Define all configuration schemas with validation rules

*   Provide sensible defaults via constants

*   Enforce runtime validation (e.g., file existence checks)

*   Serve as the single source of truth for training hyperparameters

*   Enable YAML-based configuration loading in `main.py`

**Connections in the wider system:**

*   **[main](main__documentation.md):** Loads YAML, instantiates `ScriptConfig`, passes to all pipeline phases

*   **[train](train__documentation.md):** Consumes `config.training`, `config.lora`, `config.data`, `config.model`

*   **[model_setup](model_setup__documentation.md):** Uses `config.model`, `config.quantization`

*   **[dpo](dpo__documentation.md):** Uses `config.dpo` (via main) and falls back to `config.data`, `config.training`

*   **[model_optimizer](model_optimizer__documentation.md):** Modifies config fields based on hardware presets

*   See also [OVERVIEW](../OVERVIEW.md)

***

## Directory / Module Map

```
ai-factory/
├── src/
│   ├── config.py              # <-- THIS MODULE
│   ├── main.py                # Loads YAML → ScriptConfig, runs pipeline
│   ├── train.py               # Uses config.training, config.lora, config.data
│   ├── model_setup.py         # Uses config.model, config.quantization
│   ├── dpo.py                 # Uses config.dpo
│   ├── model_optimizer.py     # Modifies ScriptConfig fields
│   ├── data/                  # Uses config.data
│   └── utils.py               # Environment class
├── tests/
│   ├── test_config.py         # Unit tests for DataConfig validation
│   └── test_main_config.py    # Integration tests for config loading
├── tests/configs/
│   └── test_config.yaml       # Example YAML configuration
└── docs/
    └── codebase_docs/
        └── config__documentation.md  # This file
```

**Grouping by responsibility:**

*   **Configuration Classes:** `DataConfig`, `ModelConfig`, `QuantizationConfig`, `LoraConfigModel`, `TrainingConfig`, `DPOConfig`, `ScriptConfig`

*   **Default Constants:** `DEFAULT_*` values for all hyperparameters

*   **Factory Function:** `get_default_dpo_config()`

***

## Public Interfaces

### Configuration Classes

| Class                | Type      | Purpose                                                                                    |
| -------------------- | --------- | ------------------------------------------------------------------------------------------ |
| `ScriptConfig`       | BaseModel | Root configuration model; composes all sub-configs into a single validated object          |
| `DataConfig`         | BaseModel | Data file paths for training and validation (JSON/JSONL format)                            |
| `ModelConfig`        | BaseModel | HF model id, sequence length, attention backend, `use_linear_attention_kernels`, `preserve_all_tensors` |
| `QuantizationConfig` | BaseModel | BitsAndBytes 4-bit quantization settings (NF4/FP4)                                         |
| `LoraConfigModel`    | BaseModel | LoRA adapter parameters (rank, alpha, dropout, target modules)                             |
| `TrainingConfig`     | BaseModel | Complete Hugging Face TrainingArguments mapping (optimizer, LR, scheduling, checkpointing) |
| `DPOConfig`          | BaseModel | Direct Preference Optimization training parameters (optional phase)                        |


### Factory Functions

| Function                 | Signature         | Description                                                |
| ------------------------ | ----------------- | ---------------------------------------------------------- |
| `get_default_dpo_config` | `() -> DPOConfig` | Returns a DPOConfig instance with all default field values |


### Constants (Defaults)

| Constant                      | Value                                                                           | Description                              |
| ----------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------- |
| `DEFAULT_MAX_LENGTH`          | 32000                                                                           | Maximum sequence length for tokenization |
| `DEFAULT_SEED`                | 42                                                                              | Random seed for reproducibility          |
| `DEFAULT_BATCH_SIZE`          | 4                                                                               | Default batch size (train and eval)      |
| `DEFAULT_LEARNING_RATE`       | 2e-4                                                                            | Default learning rate for SFT optimizer  |
| `DEFAULT_WEIGHT_DECAY`        | 0.001                                                                           | Weight decay coefficient                 |
| `DEFAULT_MAX_GRAD_NORM`       | 0.3                                                                             | Gradient clipping threshold              |
| `DEFAULT_WARMUP_RATIO`        | 0.03                                                                            | Ratio of warmup steps to total steps     |
| `DEFAULT_LORA_RANK`           | 32                                                                              | LoRA adapter rank                        |
| `DEFAULT_LORA_ALPHA`          | 32                                                                              | LoRA alpha scaling factor                |
| `DEFAULT_LORA_DROPOUT`        | 0.05                                                                            | Dropout rate for LoRA layers             |
| `DEFAULT_EVAL_STEPS`          | 50                                                                              | Steps between evaluations                |
| `DEFAULT_SAVE_STEPS`          | 50                                                                              | Steps between checkpoints                |
| `DEFAULT_SAVE_TOTAL_LIMIT`    | 2                                                                               | Maximum checkpoints to keep              |
| `DEFAULT_LOGGING_STEPS`       | 10                                                                              | Steps between log entries                |
| `DEFAULT_DATALOADER_WORKERS`  | 2                                                                               | Data loader worker processes             |
| `DEFAULT_LORA_TARGET_MODULES` | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Default LoRA target modules              |


***

## Execution and Control Flow

### Configuration Instantiation Flow

```
config.yaml (file)
    │
    ▼
yaml.safe_load() → dict
    │
    ▼
ScriptConfig(**dict)
    │
    ├─ data: DataConfig
    │   ├─ train_file: Path      (validated: must exist)
    │   └─ validation_file: Path (validated: must exist)
    │
    ├─ model: ModelConfig
    │   ├─ name: str
    │   ├─ max_length: int (default: 32000; sample yaml often 4096)
    │   ├─ attn_implementation: Literal["eager", "flash_attention_2", "sdpa"] | None
    │   │   (schema; load_model runtime also accepts flash_attention_3)
    │   ├─ trust_remote_code: bool (default: True)
    │   ├─ use_linear_attention_kernels: bool (default: False; Qwen3.5 Gated DeltaNet)
    │   └─ preserve_all_tensors: bool (default: True; load the declared architecture)
    │
    ├─ quantization: QuantizationConfig
    │   ├─ enabled: bool (default: True)
    │   ├─ quant_type: "nf4" | "fp4" (default: "nf4")
    │   └─ use_double_quant: bool (default: True)
    │
    ├─ lora: LoraConfigModel
    │   ├─ r: int (default: 32)
    │   ├─ alpha: int (default: 32)
    │   ├─ dropout: float (default: 0.05)
    │   └─ target_modules: list[str] (default: 7 projection modules)
    │
    ├─ training: TrainingConfig
    │   ├─ output_dir: Path (required)
    │   ├─ seed: int (default: 42)
    │   ├─ num_train_epochs: int (default: 1)
    │   ├─ per_device_train_batch_size: int (default: 4)
    │   ├─ per_device_eval_batch_size: int (default: 4)
    │   ├─ gradient_accumulation_steps: int (default: 1)
    │   ├─ optim: str (default: "paged_adamw_8bit")
    │   ├─ learning_rate: float (default: 2e-4)
    │   ├─ weight_decay: float (default: 0.001)
    │   ├─ max_grad_norm: float (default: 0.3)
    │   ├─ warmup_ratio: float (default: 0.03)
    │   ├─ lr_scheduler_type: str (default: "constant")
    │   ├─ evaluation_strategy: str (default: "steps")
    │   ├─ eval_steps: int (default: 50)
    │   ├─ save_strategy: str (default: "steps")
    │   ├─ save_steps: int (default: 50)
    │   ├─ save_total_limit: int (default: 2)
    │   ├─ save_only_model: bool (default: False)
    │   ├─ logging_steps: int (default: 10)
    │   ├─ group_by_length: bool (default: True)
    │   ├─ gradient_checkpointing: bool (default: True)
    │   ├─ report_to: str (default: "none")
    │   ├─ load_best_model_at_end: bool (default: True)
    │   ├─ metric_for_best_model: str (default: "eval_loss")
    │   ├─ greater_is_better: bool (default: False)
    │   ├─ remove_unused_columns: bool (default: True)
    │   └─ dataloader_num_workers: int (default: 2)
    │
    └─ dpo: DPOConfig | None (optional)
        ├─ output_dir: Path | None
        ├─ train_file: Path | None
        ├─ learning_rate: float (default: 5e-6)
        ├─ beta: float (default: 0.1)
        ├─ max_steps: int (default: 100)
        ├─ per_device_train_batch_size: int (default: 1)
        ├─ per_device_eval_batch_size: int (default: 1)
        ├─ gradient_accumulation_steps: int (default: 4)
        ├─ optim: str (default: "paged_adamw_8bit")
        ├─ lr_scheduler_type: str (default: "cosine")
        ├─ eval_steps: int (default: 50)
        ├─ save_steps: int (default: 50)
        ├─ save_total_limit: int (default: 2)
        ├─ logging_steps: int (default: 10)
        ├─ gradient_checkpointing: bool (default: True)
        ├─ warmup_ratio: float (default: 0.03)
        ├─ lora_rank: int (default: 32)
        └─ torch_compile: bool (default: False)
```

### Validation Flow

```
ScriptConfig instantiation
    │
    ├─ Pydantic type validation (all fields)
    │
    ├─ DataConfig.file_must_exist()
    │   ├─ train_file.exists()?
    │   └─ validation_file.exists()?
    │
    ├─ DPOConfig.train_file_must_exist_if_set()
    │   └─ train_file.exists()? (only if not None)
    │
    ├─ Field constraints (gt, ge, le, pattern)
    │   ├─ quant_type: regex ^(nf4|fp4)$
    │   ├─ lr_scheduler_type: regex ^(linear|cosine|...)$
    │   ├─ evaluation_strategy: regex ^(no|steps|epoch)$
    │   └─ save_strategy: regex ^(no|steps|epoch)$
    │
    └─ Runtime mutation (validate_assignment=True)
        └─ Model Optimizer can modify fields post-init
```

***

## Data Flow

### Configuration Loading Pipeline

```
config.yaml
    │
    ▼
main.py: load_config_from_yaml()
    │
    ├─ _resolve_config_paths()  # Relative → absolute paths
    │
    ▼
ScriptConfig(**yaml_dict)
    │
    ├─ validate_assignment=True  # Allow post-init mutation
    ├─ frozen=False              # Allow field modification
    │
    ▼
config object passed to all pipeline phases
    │
    ├─► train.py: run_training(config, env, tokenizer, model)
    │   ├─ config.data → load_and_prepare_dataset()
    │   ├─ config.lora → PeftLoraConfig()
    │   └─ config.training → TrainingArguments()
    │
    ├─► train.py: merge_and_save_model(config, env)
    │   └─ config.model → resolve_model_class() → from_pretrained()
    │
    ├─► main.py: DPO phase
    │   ├─ config.dpo (or get_default_dpo_config())
    │   └─ config.data.train_file (fallback)
    │
    └─► main.py: Inference phase
        └─ config.model → load_model_pipeline()
```

### Config to TrainingArguments Mapping

```
config.training (TrainingConfig)
    │
    ▼
train.py: _prepare_training_arguments()
    │
    ├─ Filter fields by TrainingArguments dataclass
    │
    ├─ Convert Path → str (output_dir, logging_dir)
    │
    ├─ Set bf16/fp16 based on Environment
    │
    ├─ Resolve optimizer via _determine_effective_optimizer()
    │
    └─ Sync evaluation/save strategies
        │
        ▼
TrainingArguments(**filtered_dict)
```

***

## Integration Points

### External Dependencies

| Dependency                 | Usage                            | Notes                                     |
| -------------------------- | -------------------------------- | ----------------------------------------- |
| `pydantic.BaseModel`       | Base class for all config models | Provides validation, serialization        |
| `pydantic.ConfigDict`      | Model configuration              | `validate_assignment=True`,`frozen=False` |
| `pydantic.Field`           | Field metadata and constraints   | Validation rules, descriptions            |
| `pydantic.field_validator` | Custom validation logic          | File existence checks                     |
| `pathlib.Path`             | File path handling               | Automatic path resolution                 |


### Internal Module Dependencies

| Module                   | Functions/Classes Used                                       |
| ------------------------ | ------------------------------------------------------------ |
| `src/main.py`            | `ScriptConfig`(load from YAML, pass to pipeline)             |
| `src/train.py`           | `ScriptConfig`(extract training/lora/data/model config)      |
| `src/model_setup.py`     | `ModelConfig`,`QuantizationConfig`(load model/tokenizer)     |
| `src/dpo.py`             | `DPOConfig`(DPO training),`DataConfig`(fallback train\_file) |
| `src/model_optimizer.py` | `ScriptConfig`(modify fields based on hardware presets)      |
| `src/utils.py`           | `Environment`(independent; no config dependency)             |


### Caller Integration

```
# In src/main.py
from src.config import ScriptConfig
import yaml

def load_config_from_yaml(config_path: Path) -> ScriptConfig:
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return ScriptConfig(**config_dict)

# In src/model_optimizer.py
def optimize_config(config: ScriptConfig, ...) -> ScriptConfig:
    # Mutate config fields based on hardware profile
    config.training.per_device_train_batch_size = recommended_batch_size
    config.lora.r = recommended_rank
    return config
```

***

## Configuration and Conventions

### Pydantic Model Configuration

All config classes use consistent model configuration:

```
model_config = ConfigDict(
    validate_assignment=True,  # Validate on field mutation
    frozen=False,              # Allow field modification (Model Optimizer needs this)
)
```

### Field Constraints

| Constraint Type | Example                           | Purpose                      |
| --------------- | --------------------------------- | ---------------------------- |
| `gt=0`          | `max_length`,`r`,`learning_rate`  | Positive values only         |
| `ge=0`          | `weight_decay`,`save_total_limit` | Non-negative values          |
| `le=1.0`        | `dropout`,`warmup_ratio`          | Bounded \[0, 1]              |
| `pattern=regex` | `quant_type`,`lr_scheduler_type`  | Enum-like validation         |
| `min_length=1`  | `target_modules`                  | At least one module required |


### File Validation

*   **DataConfig:** Both `train_file` and `validation_file` must exist at config load time

*   **DPOConfig:** `train_file` validation is deferred (only checked if explicitly set)

*   Rationale: DPO is optional; its train\_file may not be present when DPO is disabled

### Attention / kernel notes (`ModelConfig`)

*   **`attn_implementation`:** Pydantic allows `eager` | `flash_attention_2` | `sdpa` | `None`. Runtime `load_model` also understands `flash_attention_3`. Prefer YAML values that validate.
*   **Gemma 4:** Prefer `attn_implementation: sdpa`. The loader’s head-dim guard reads `head_dim` (or `hidden_size // num_attention_heads`), not `global_head_dim` (often 512 on Gemma 4 global layers; FA2 max is 256).
*   **`use_linear_attention_kernels`:** When `true`, `validate_linear_attention_kernels` fail-fasts unless `causal_conv1d` and `fla` (flash-linear-attention) import. Used by SFT load, merge, DPO, and inference. Sample `src/config.yaml` sets this `false`.
*   **`preserve_all_tensors`:** When `true` (default), SFT load, merge, and DPO go through `resolve_model_class` and load the architecture the checkpoint declares, so a multimodal base keeps its vision tower instead of being silently reduced to its text submodel. Set `false` only to deliberately export a text-only model. See [model_setup](model_setup__documentation.md#configuration-and-conventions).

### YAML Structure

Aligned with sample [`src/config.yaml`](../../src/config.yaml) (paths relative to that file’s directory):

```
# Example — see also tests/configs/test_config.yaml for a minimal fixture config
data:
  train_file: ./data/datasets/icdu_training_data_v8.jsonl      # ICDU for SFT
  validation_file: ./data/datasets/icdu_validation_data_v8.jsonl

model:
  name: Qwen/Qwen3.5-9B
  max_length: 4096
  attn_implementation: flash_attention_2
  trust_remote_code: true
  use_linear_attention_kernels: false
  preserve_all_tensors: true

quantization:
  enabled: true
  quant_type: nf4
  use_double_quant: true

lora:
  r: 32
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  output_dir: ./training_output
  seed: 42
  num_train_epochs: 1
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 2
  optim: paged_adamw_8bit
  learning_rate: 0.0002
  weight_decay: 0.001
  max_grad_norm: 0.3
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  evaluation_strategy: steps
  eval_steps: 150
  save_strategy: steps
  save_steps: 300
  save_total_limit: 2
  save_only_model: false
  logging_steps: 10
  group_by_length: true
  gradient_checkpointing: true
  report_to: none
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  remove_unused_columns: true
  dataloader_num_workers: 2

# DPO expects messages-format JSONL (separate from ICDU SFT data)
dpo:
  train_file: ./data/datasets/augmented_datasets/breaking_better/bb_training_data_v7.jsonl
  output_dir: ./training_output
  learning_rate: 0.000005
  beta: 0.1
  max_steps: 100
  lora_rank: 32
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 2
  optim: paged_adamw_8bit
  lr_scheduler_type: cosine
  eval_steps: 50
  save_steps: 50
  save_total_limit: 2
  logging_steps: 10
  gradient_checkpointing: true
  warmup_ratio: 0.03
  torch_compile: false
```

### Path Resolution

*   Paths in YAML can be relative

*   `main.py` resolves them relative to the config file's directory via `_resolve_config_paths()`

*   After resolution, all paths are absolute `Path` objects

***

## Extension and Testing Guidance

### Adding New Configuration Fields

1.  **Add field to appropriate class:**

    ```
    class TrainingConfig(BaseModel):
        new_param: float = Field(
            default_value,
            description="Description of the parameter",
            gt=0.0,
        )
    ```

2.  **Add constant for default (optional):**

    ```
    DEFAULT_NEW_PARAM = 1.0
    ```

3.  **Update YAML example and documentation**

4.  **Field will be automatically available in:**

    *   YAML configuration

    *   `config.training.new_param` access

    *   Serialization via `model_dump()`

### Adding New Configuration Classes

1.  Create the Pydantic model with `model_config = ConfigDict(validate_assignment=True, frozen=False)`

2.  Add to `ScriptConfig` as a field

3.  Update consumers to use the new config section

### Testing Patterns

The module includes tests in `tests/test_config.py`:

*   **File validation tests:** Verify `DataConfig` raises `FileNotFoundError` for missing files

*   **Default value tests:** Verify constants match class defaults

*   **Constraint tests:** Verify field constraints (gt, ge, pattern) reject invalid values

*   **Integration tests:** Verify YAML loading produces valid `ScriptConfig`

### Common Extension Points

| Extension             | Location                           | Impact                            |
| --------------------- | ---------------------------------- | --------------------------------- |
| New optimizer         | `TrainingConfig.optim`             | Auto-passed to`TrainingArguments` |
| New LR scheduler      | `TrainingConfig.lr_scheduler_type` | Add to pattern regex              |
| New LoRA target       | `LoraConfigModel.target_modules`   | Default list modification         |
| New quantization type | `QuantizationConfig.quant_type`    | Add to pattern regex              |
| New training phase    | Add`DPOConfig`-like class          | Add to`ScriptConfig`              |


***

## Visualizations

### Configuration Schema Hierarchy

```
classDiagram
direction TB

class ScriptConfig {
  +data
  +model
  +quantization
  +lora
  +training
  +dpo
  +validate_assignment
  +frozen_false
}

class DataConfig {
  +train_file
  +validation_file
  +file_must_exist()
}

class ModelConfig {
  +name
  +max_length
  +attn_implementation
  +trust_remote_code
  +use_linear_attention_kernels
  +preserve_all_tensors
}

class QuantizationConfig {
  +enabled
  +quant_type
  +use_double_quant
}

class LoraConfigModel {
  +r
  +alpha
  +dropout
  +target_modules
}

class TrainingConfig {
  +output_dir
  +seed
  +per_device_train_batch_size
  +per_device_eval_batch_size
  +gradient_accumulation_steps
  +optim
  +learning_rate
  +evaluation_strategy
  +eval_steps
  +save_strategy
  +save_steps
  +save_only_model
  +gradient_checkpointing
  +dataloader_num_workers
}

class DPOConfig {
  +output_dir
  +train_file
  +learning_rate
  +beta
  +max_steps
  +per_device_train_batch_size
  +gradient_accumulation_steps
  +lora_rank
  +torch_compile
  +train_file_must_exist_if_set()
}

ScriptConfig *-- DataConfig : data
ScriptConfig *-- ModelConfig : model
ScriptConfig *-- QuantizationConfig : quantization
ScriptConfig *-- LoraConfigModel : lora
ScriptConfig *-- TrainingConfig : training
ScriptConfig *-- DPOConfig : dpo
```

### YAML Loading and Validation Pipeline

```
flowchart TD
    A["config.yaml"] --> B["yaml.safe_load()"]
    B --> C["_resolve_config_paths(base_dir)"]
    C --> D["Instantiate ScriptConfig"]
    D --> E["Nested Pydantic parsing"]
    E --> F{"data.train_file exists?"}
    F -- no --> X1["FileNotFoundError"]
    F -- yes --> G{"data.validation_file exists?"}
    G -- no --> X2["FileNotFoundError"]
    G -- yes --> H{"dpo.train_file set?"}
    H -- no --> J["Field constraints<br/>gt, ge, le, min_length, pattern"]
    H -- yes --> I{"dpo.train_file exists?"}
    I -- no --> X3["FileNotFoundError"]
    I -- yes --> J
    J --> K["Validated ScriptConfig"]
    K --> L["Mutable runtime object<br/>validate_assignment=True<br/>frozen=False"]
```

### Configuration Consumers Across the Codebase

```
flowchart LR
    SC["ScriptConfig"] --> MAIN["src.main<br/>load_config_from_yaml<br/>run_pipeline"]
    SC --> TRAIN["src.train<br/>run_training<br/>merge_and_save_model"]
    SC --> SETUP["src.model_setup<br/>load_tokenizer<br/>load_model"]
    SC --> OPT["src.model_optimizer<br/>recommend<br/>merge_config"]
    MAIN --> DPO["src.dpo<br/>run_dpo_training"]
    TRAIN --> DATA["src.data<br/>load_and_prepare_dataset"]
    SETUP --> ENV["Environment"]

    SC -. model + quantization .-> SETUP
    SC -. data + lora + training .-> TRAIN
    SC -. dpo or default dpo fallback .-> DPO
    OPT -. validated merged config .-> SC
```

## Mathematical Framing

### Configuration Defaults as Optimization Hyperparameters

Schema defaults target QLoRA fine-tuning; the sample YAML uses `Qwen/Qwen3.5-9B` with `r=32` and seven target modules:

```
LoRA Effective Parameters (approximate, dense linear layers):
    params ≈ r × (d_in + d_out) × num_target_modules

Generic formula (illustrative for hidden_size ≈ 4096, r=32, 7 modules):
    params ≈ 32 × (4096 + 4096) × 7 ≈ 1.84M trainable LoRA params

Memory Estimation (4-bit NF4):
    base_model_memory ≈ model_params × 0.5 bytes
    LoRA + grads + optimizer states dominate remaining VRAM budget

Total Training Memory ≈ base + lora + gradients + optimizer states
```

### Learning Rate Schedule

```
Effective LR at step s:
    lr(s) = base_lr × schedule_fn(s)

For constant schedule with warmup:
    lr(s) = base_lr × min(s/warmup_steps, 1.0)

Where:
    warmup_steps = warmup_ratio × total_steps
    total_steps = (num_samples / (batch_size × grad_accum)) × num_epochs
```

### DPO Loss Parameters

```
DPO Loss:
    L_DPO = -E[log σ(β × (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

Where:
    β = config.dpo.beta (default: 0.1) — temperature parameter
    Higher β → sharper preference signal
    Lower β → more conservative updates
```

### Gradient Accumulation

```
Effective batch size:
    effective_batch = per_device_batch × num_devices × grad_accum_steps

Sample src/config.yaml:
    effective_batch = 4 × 1 × 2 = 8  (batch 4, grad_accum 2)
```

### Checkpoint Budget

```
Disk usage per checkpoint:
    full_checkpoint = model_size + optimizer_state + scheduler_state
    model_only = model_size (when save_only_model=True)

With save_total_limit=2:
    max_disk = 2 × checkpoint_size
    (oldest checkpoint deleted when limit exceeded)
```

***

*Last updated: 2026-08-23.*
