# src/model\_optimizer.py

## 1. Overview

The `model_optimizer.py` module provides hardware-aware config.yaml recommendation for QLoRA fine-tuning. It inspects the local machine's GPU VRAM, system RAM, and OS via `HardwareProfile`, applies one of four named presets (fast, quality, balanced, low\_memory), and returns a nested dict of config overrides suitable for merging into an existing config.yaml. All recommendations respect TRAINING\_CRASH\_DIAGNOSIS rules: staggered eval/save intervals, `save_only_model=True`, conservative dataloader workers on Windows, etc.

**Purpose:** Automatically recommend optimized config.yaml settings based on detected hardware and a chosen speed/quality preset.

**Key Responsibilities:**

*   Detect hardware capabilities via `HardwareProfile` (VRAM, RAM, OS, GPU count)

*   Apply preset-specific tuning (batch size, LoRA rank, eval/save intervals, gradient checkpointing)

*   Compute VRAM-tier-aware batch size caps and gradient accumulation

*   Determine safe `dataloader_num_workers` per OS and RAM

*   Merge overrides into a base config dict while preserving existing paths and values

*   Validate the merged config via `ScriptConfig`

*   Optionally write the optimized config to a YAML file

**Connections:**

*   **Parent:** Called by CLI (`__main__` block) or programmatically via `run_optimizer()`

*   **Dependencies:** `src.config.ScriptConfig`, `src.hardware.HardwareProfile`

*   **External:** `yaml`, `copy`, `logging`

*   **Downstream:** Output YAML is consumed by `main.py` pipeline


**Dual entrypoints:** `python -m src.main optimize-config --config-path ... --preset balanced -o out.yaml` and `python -m src.model_optimizer config.yaml -p balanced -o out.yaml`. See [main](main__documentation.md) and [OVERVIEW](../OVERVIEW.md).

**Presets:** `fast`, `quality`, `balanced`, `low_memory`. Windows `dataloader_num_workers` is 0 (low_memory) or 2; staggered `save_steps` multiple of `eval_steps`; `save_only_model: true` in overrides.
## 2. Directory/Module Map

```
ai-factory/
├── src/
│   ├── model_optimizer.py      # <-- THIS MODULE
│   ├── config.py               # ScriptConfig for validation
│   ├── hardware.py             # HardwareProfile (VRAM, RAM, OS detection)
│   ├── main.py                 # CLI entry; consumes optimizer output
│   ├── train.py                # Training pipeline
│   └── utils.py                # Environment base class
├── tests/
│   └── test_model_optimizer.py # Unit tests for optimizer functions
└── docs/
    └── codebase_docs/
        └── model_optimizer__documentation.md  # This file
```

## 3. Public Interfaces

| Function        | Signature                                                                                                           | Description                                                                                                                          |
| --------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `recommend`     | `(profile: HardwareProfile, preset: PresetName, base_config_dict: dict[str, Any] \| None = None) -> dict[str, Any]` | Main recommendation entry point. Returns nested overrides dict (training, lora, model, optionally dpo) tuned to hardware and preset. |
| `merge_config`  | `(base_dict: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]`                                          | Deep-merge overrides into a copy of base\_dict, preserving base paths and non-overridden keys.                                       |
| `run_optimizer` | `(config_path: str, preset: PresetName = "balanced", output_path: str \| None = None) -> dict[str, Any]`            | End-to-end: load YAML, detect hardware, recommend, merge, validate with ScriptConfig, optionally write output YAML.                  |


### Internal Functions

| Function                         | Signature                                                           | Description                                                                                |
| -------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `_resolve_config_paths`          | `(config_dict: dict[str, Any], base_dir: Path) -> dict[str, Any]`   | Resolve relative paths in data/training/dpo sections. Mirrors`main._resolve_config_paths`. |
| `_vram_tier`                     | `(vram_bytes: int) -> Literal["low", "medium", "high"]`             | Classify VRAM into tier: low (<8GB), medium (8-12GB), high (>=12GB).                       |
| `_max_batch_for_vram`            | `(vram_bytes: int, preset: PresetName) -> int`                      | Max`per_device_train_batch_size`for VRAM tier and preset.                                  |
| `_dataloader_workers`            | `(profile: HardwareProfile, preset: PresetName) -> int`             | Recommend`dataloader_num_workers`per TRAINING\_CRASH\_DIAGNOSIS rules.                     |
| `_choose_batch_and_accumulation` | `(profile: HardwareProfile, preset: PresetName) -> tuple[int, int]` | Return`(per_device_train_batch_size, gradient_accumulation_steps)`for SFT.                 |
| `_merged_for_validation`         | `(merged: dict[str, Any]) -> dict[str, Any]`                        | Return deep copy of merged config suitable for`ScriptConfig`validation.                    |
| `_to_yaml_friendly`              | `(config: dict[str, Any]) -> dict[str, Any]`                        | Recursively convert Path objects to strings for YAML serialization.                        |


### Module-Level Constants

| Constant                                      | Type      | Description                                                    |
| --------------------------------------------- | --------- | -------------------------------------------------------------- |
| `PresetName`                                  | `Literal` | Type alias:`"fast" \| "quality" \| "balanced" \| "low_memory"` |
| `PRESETS`                                     | `tuple`   | All valid preset names                                         |
| `PRESET_CONFIG`                               | `dict`    | Maps preset name to parameter dict                             |
| `VRAM_6GB / VRAM_8GB / VRAM_12GB / VRAM_24GB` | `int`     | VRAM tier threshold constants in bytes                         |


## 4. Execution and Control Flow

### run\_optimizer Flow

```
run_optimizer(config_path, preset, output_path)
    │
    ├─ Load base YAML (yaml.safe_load)
    │
    ├─ _resolve_config_paths(base_dict, base_dir)
    │   └─ Resolve relative paths in data/, training/, dpo/ sections
    │
    ├─ HardwareProfile()  ← detect VRAM, RAM, OS, GPU count
    │
    ├─ recommend(profile, preset, base_dict)
    │   ├─ _choose_batch_and_accumulation(profile, preset)
    │   │   ├─ _max_batch_for_vram(profile.vram_bytes, preset)
    │   │   │   └─ _vram_tier(vram_bytes) → low | medium | high
    │   │   └─ Compute per_device, gradient_accumulation_steps
    │   ├─ _dataloader_workers(profile, preset)
    │   ├─ Build overrides dict:
    │   │   ├─ training: batch, eval/save, grad_ckpt, workers, epochs, lr_scheduler
    │   │   ├─ lora: r, alpha
    │   │   ├─ model: max_length, attn_implementation
    │   │   └─ [if dpo in base] dpo: batch, grad_ckpt, lora_rank, torch_compile
    │   └─ return overrides
    │
    ├─ merge_config(base_dict, overrides)
    │   └─ Deep copy base → overlay overrides per section
    │
    ├─ ScriptConfig(**merged)  ← validation
    │
    ├─ [if output_path] _to_yaml_friendly(merged) → yaml.dump
    │
    └─ return merged
```

### Batch Size Selection Logic

```
_choose_batch_and_accumulation(profile, preset)
    │
    ├─ Get target range: [target_effective_batch_min, target_effective_batch_max]
    │
    ├─ _max_batch_for_vram(vram_bytes, preset)
    │   ├─ _vram_tier(vram_bytes)
    │   │   ├─ <8GB  → "low"
    │   │   ├─ <12GB → "medium"
    │   │   └─ >=12GB → "high"
    │   │
    │   ├─ "low"    → 2 (low_memory) or 4 (other)
    │   ├─ "medium" → 4
    │   └─ "high"   → 8 (fast) or 4 (other)
    │
    ├─ per_device = min(max_batch, target_hi)
    ├─ effective = per_device * gpu_count
    └─ accum = ceil(target_lo / effective) if effective < target_lo else 1
```

### Dataloader Workers Logic

```
_dataloader_workers(profile, preset)
    │
    ├─ Windows → 0 (low_memory) or 2 (other)
    │
    └─ Linux/macOS:
        ├─ RAM < 16GB → 0
        ├─ RAM < 32GB → 2
        └─ RAM >= 32GB → 4 (normal) or 2 (low_memory)
```

## 5. Data Flow

### Config Override Pipeline

```
config.yaml (on disk)
    │
    ▼
yaml.safe_load() → base_dict: dict[str, Any]
    │
    ├─► _resolve_config_paths(base_dict, base_dir)
    │       └─ Resolve relative paths → absolute paths
    │
    ▼
HardwareProfile()
    │  ├─ vram_bytes
    │  ├─ system_ram_bytes
    │  ├─ gpu_count
    │  ├─ os_name
    │  └─ cuda_available
    │
    ▼
recommend(profile, preset, base_dict)
    │
    ▼
overrides: dict[str, Any]
    │  ├─ training: {per_device_train_batch_size, eval_steps, save_steps, ...}
    │  ├─ lora: {r, alpha}
    │  ├─ model: {max_length, attn_implementation}
    │  └─ [optional] dpo: {per_device_train_batch_size, lora_rank, ...}
    │
    ▼
merge_config(base_dict, overrides) → merged: dict[str, Any]
    │
    ├─► ScriptConfig(**merged)  ← validation
    │
    └─► [optional] yaml.dump(merged, output_path)
```

### Preset Parameter Matrix

| Parameter               | fast  | quality | balanced | low\_memory |
| ----------------------- | ----- | ------- | -------- | ----------- |
| target\_batch\_min      | 16    | 8       | 8        | 4           |
| target\_batch\_max      | 32    | 16      | 16       | 8           |
| eval\_steps             | 200   | 100     | 150      | 150         |
| save\_steps             | 400   | 200     | 300      | 300         |
| lora\_r                 | 16    | 64      | 32       | 16          |
| lora\_alpha             | 32    | 64      | 32       | 32          |
| max\_length             | 2048  | 4096    | 4096     | 2048        |
| num\_train\_epochs      | 1     | 2       | 1        | 1           |
| gradient\_checkpointing | False | True    | True     | True        |
| dpo\_per\_device\_batch | 2     | 1       | 2        | 1           |
| dpo\_torch\_compile     | True  | False   | False    | False       |


## 6. Integration Points

### External Dependencies

| Dependency      | Usage                                 | Notes                                            |
| --------------- | ------------------------------------- | ------------------------------------------------ |
| `yaml`          | Load and dump config YAML files       | `safe_load`for reading,`dump`for writing         |
| `copy.deepcopy` | Deep-merge without mutating base dict | Used in`merge_config`and`_merged_for_validation` |
| `argparse`      | CLI argument parsing                  | `__main__`block entry point                      |
| `logging`       | Info/warning output                   | Module-level logger                              |


### Internal Module Dependencies

| Module         | Functions/Classes Used                                                                    |
| -------------- | ----------------------------------------------------------------------------------------- |
| `src.config`   | `ScriptConfig`(validation only)                                                           |
| `src.hardware` | `HardwareProfile`(vram\_bytes, system\_ram\_bytes, gpu\_count, os\_name, cuda\_available) |


### Caller Integration

```
# Programmatic usage
from src.model_optimizer import recommend, merge_config, run_optimizer
from src.hardware import HardwareProfile

profile = HardwareProfile()
overrides = recommend(profile, "balanced", base_config_dict)
merged = merge_config(base_config_dict, overrides)

# Or end-to-end with validation and file output
merged = run_optimizer("config.yaml", preset="quality", output_path="optimized.yaml")
```

### CLI Usage

```
# Basic usage
python -m src.model_optimizer config.yaml

# With preset and output
python -m src.model_optimizer config.yaml -p fast -o optimized_config.yaml

# All presets: fast, quality, balanced, low_memory
python -m src.model_optimizer config.yaml --preset low_memory
```

## 7. Configuration and Conventions

### TRAINING\_CRASH\_DIAGNOSIS Rules (Always Enforced)

These rules are hard-coded into the recommendation engine and cannot be overridden:

1.  **save\_steps is a multiple of eval\_steps:** `save_steps % eval_steps == 0` (currently `save_steps = 2 * eval_steps`)

2.  **save\_only\_model is always True:** Prevents optimizer state saves that consume disk during crash recovery

3.  **Safe dataloader workers on Windows:** Capped at 0 (low\_memory) or 2 to avoid multiprocessing crashes

4.  **Conservative workers by RAM:** Linux workers scale with system RAM to avoid OOM

### VRAM Tier Thresholds

| Tier   | VRAM Range | Max Batch (fast) | Max Batch (other) |
| ------ | ---------- | ---------------- | ----------------- |
| low    | < 8 GB     | 4                | 4                 |
| medium | 8-12 GB    | 4                | 4                 |
| high   | >= 12 GB   | 8                | 4                 |


### Dataloader Workers by Platform and RAM

| OS      | RAM < 16GB | RAM < 32GB | RAM >= 32GB |
| ------- | ---------- | ---------- | ----------- |
| Windows | 0/2        | 0/2        | 0/2         |
| Linux   | 0          | 2          | 4/2         |


(Values shown as `low_memory`/`normal`)

### Attention Implementation

*   **CUDA available:** `flash_attention_2` (resolved at model load time by `model_setup`)

*   **No CUDA:** `sdpa` (PyTorch Scaled Dot Product Attention)

### DPO Overrides (Conditional)

DPO overrides are only generated when the base config contains a non-empty `dpo` section. The DPO overrides set:

*   Per-device batch size from preset

*   `gradient_accumulation_steps: 2` (fixed)

*   `eval_steps: 50`, `save_steps: 50` (fixed)

*   `gradient_checkpointing: True` (always)

*   `lora_rank` matching the SFT preset

*   `torch_compile: True` only for the `fast` preset

## 8. Extension and Testing Guidance

### Adding a New Preset

1.  Define a `_PRESET_NEW` dict with all required keys (see existing presets for template)

2.  Add to `PRESET_CONFIG` mapping

3.  Add to `PRESETS` tuple and `PresetName` Literal type

4.  Update argparse choices (auto-derived from `PRESETS`)

### Modifying Batch Size Logic

*   Edit `_vram_tier()` to change VRAM threshold boundaries

*   Edit `_max_batch_for_vram()` to change per-tier batch caps

*   Edit `_choose_batch_and_accumulation()` to change accumulation strategy

### Testing Patterns

The module includes comprehensive unit tests in `tests/test_model_optimizer.py`:

*   **VRAM tier tests:** Verify classification at boundary values (0, 6GB, 8GB, 12GB, 24GB)

*   **Max batch tests:** Verify batch caps per tier and preset

*   **Dataloader workers tests:** Verify Windows vs Linux behavior by RAM

*   **Hardware profile tests:** Mocked CUDA detection for both GPU and CPU-only

*   **Recommendation tests:** Verify preset values, save\_steps % eval\_steps == 0, DPO conditional

*   **Merge tests:** Verify base preservation and override application

*   **Integration tests:** Full `run_optimizer` with real test config, YAML output verification

### Error Handling

*   `recommend()`: Raises `ValueError` for unknown presets

*   `run_optimizer()`: Raises `FileNotFoundError` for missing config, `ValueError` for empty/invalid config

*   `merge_config()`: Always succeeds (empty dicts as fallback for missing sections)

*   `ScriptConfig` validation runs after merge, catching schema violations

## 9. Visualizations

### End-to-End Optimizer Flow

```
flowchart TD
    A["run_optimizer(config_path, preset, output_path)"] --> B{"config_path exists?"}
    B -- no --> X1["FileNotFoundError"]
    B -- yes --> C["yaml.safe_load(base config)"]
    C --> D{"Config empty?"}
    D -- yes --> X2["ValueError"]
    D -- no --> E["_resolve_config_paths(base_dict, base_dir)"]
    E --> F["HardwareProfile()"]
    F --> G["recommend(profile, preset, base_dict)"]
    G --> H["merge_config(base_dict, overrides)"]
    H --> I["ScriptConfig(**_merged_for_validation(merged))"]
    I --> J{"output_path provided?"}
    J -- yes --> K["_to_yaml_friendly(merged)<br/>yaml.dump(output_path)"]
    J -- no --> L["Return merged config"]
    K --> L
```

### Recommendation Engine Decision Graph

```
flowchart TD
    A["recommend(profile, preset, base_config_dict)"] --> B{"Preset valid?"}
    B -- no --> X["ValueError"]
    B -- yes --> C["_choose_batch_and_accumulation(profile, preset)"]
    B --> D["_dataloader_workers(profile, preset)"]

    subgraph BATCH["Batch and accumulation policy"]
        C1["_vram_tier(profile.vram_bytes)"] --> C2{"VRAM tier"}
        C2 -- low --> C3["max_batch = 2 or 4"]
        C2 -- medium --> C4["max_batch = 4"]
        C2 -- high --> C5["max_batch = 8 or 4"]
        C3 --> C6["per_device = min(max_batch, target_max)"]
        C4 --> C6
        C5 --> C6
        C6 --> C7["effective = per_device * max(1, gpu_count)"]
        C7 --> C8{"effective batch below target minimum?"}
        C8 -- yes --> C9["grad_accum = ceil(target_min / effective)"]
        C8 -- no --> C10["grad_accum = 1"]
    end

    subgraph WORKERS["Dataloader worker policy"]
        D1{"Windows?"}
        D1 -- yes --> D2["workers = 0 or 2"]
        D1 -- no --> D3{"System RAM tier"}
        D3 -- <16GB --> D4["workers = 0"]
        D3 -- 16-32GB --> D5["workers = 2"]
        D3 -- >=32GB --> D6["workers = 2 or 4"]
    end

    C --> E["Training overrides<br/>batch sizes, grad_accum,<br/>eval/save cadence, epochs,<br/>scheduler, save_only_model"]
    D --> E
    E --> F["LoRA overrides<br/>r and alpha"]
    F --> G{"CUDA available?"}
    G -- yes --> H["model.attn_implementation = flash_attention_2"]
    G -- no --> I["model.attn_implementation = sdpa"]
    H --> J["model.max_length = preset max_length"]
    I --> J
    J --> K{"base config has non-empty dpo section?"}
    K -- yes --> L["Add DPO overrides<br/>batch size, grad_accum,<br/>eval/save steps, lora_rank,<br/>optional torch_compile"]
    K -- no --> M["Return overrides"]
    L --> M
```

## 10. Mathematical Framing

### Effective Batch Size

The effective batch size determines gradient update frequency:

```
effective_batch = per_device_batch_size * gpu_count
accumulation_steps = ceil(target_min_batch / effective_batch)
total_effective_batch = per_device_batch_size * gpu_count * accumulation_steps
```

The optimizer prefers larger `per_device_batch_size` (up to the VRAM cap) and uses accumulation only when the GPU count is insufficient to reach the target minimum.

### VRAM Tier Classification

VRAM is classified into three tiers using hard thresholds:

```
tier(vram) = {
    "low"    if vram < 8 * 1024^3
    "medium" if vram < 12 * 1024^3
    "high"   if vram >= 12 * 1024^3
}
```

### LoRA Rank and Alpha Scaling

Each preset defines `lora_r` (rank) and `lora_alpha`. The effective LoRA scaling factor applied to the adapter output is:

```
scaling = alpha / r
```

| Preset      | r  | alpha | scaling |
| ----------- | -- | ----- | ------- |
| fast        | 16 | 32    | 2.0     |
| quality     | 64 | 64    | 1.0     |
| balanced    | 32 | 32    | 1.0     |
| low\_memory | 16 | 32    | 2.0     |


Higher rank increases adapter expressiveness but costs more VRAM. The fast and low\_memory presets use rank 16 with double-alpha scaling (2.0x), while quality and balanced use matched rank/alpha (1.0x scaling) with higher rank for quality.

### Save/Eval Interval Relationship

The module enforces a fixed ratio between save and eval intervals:

```
save_steps = 2 * eval_steps
```

This ensures that every save checkpoint has a recent evaluation score, supporting crash recovery with known model quality.

### Dataloader Worker Safety Function

Workers are computed as a function of OS and system RAM:

```
workers(profile, preset) = {
    0  if os == "Windows" and preset == "low_memory"
    2  if os == "Windows"
    0  if ram_gb < 16
    2  if ram_gb < 32
    2  if ram_gb >= 32 and preset == "low_memory"
    4  if ram_gb >= 32
}
```

This follows TRAINING\_CRASH\_DIAGNOSIS rules to avoid multiprocessing crashes on Windows and OOM on low-RAM systems.

***

*Last updated: 2026-08-02.*
