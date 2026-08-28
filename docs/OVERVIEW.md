# Application Overview

This document describes how the **ai-factory** application works: its pipeline, data flow, components, and configuration.

For run instructions and environment setup, see [`README.md`](../README.md) (Installation, Usage, Troubleshooting). On Windows, prefer the conda `ai-factory` stack (`conda run -n ai-factory ...`) and set `$env:KMP_DUPLICATE_LIB_OK = 'TRUE'` before training.

---

## Documentation index

| Document | Covers |
|----------|--------|
| [main](codebase_docs/main__documentation.md) | CLI (`argparse`), pipeline orchestration |
| [config](codebase_docs/config__documentation.md) | Pydantic `ScriptConfig` / YAML schema |
| [train](codebase_docs/train__documentation.md) | QLoRA SFT and merge |
| [dpo](codebase_docs/dpo__documentation.md) | Preference pairs and DPO training |
| [model_setup](codebase_docs/model_setup__documentation.md) | Tokenizer/model load, attention, linear kernels |
| [data](codebase_docs/data__documentation.md) | ICDU datasets and generation scripts |
| [inference_with_tools](codebase_docs/inference_with_tools__documentation.md) | Tool-augmented agent loop |
| [tools](codebase_docs/tools__documentation.md) | Lightweight alternate tool agent |
| [model_optimizer](codebase_docs/model_optimizer__documentation.md) | Hardware-aware config presets |
| [utils_and_hardware](codebase_docs/utils_and_hardware__documentation.md) | `Environment` and `HardwareProfile` |
| [combined](codebase_docs/combined__documentation.md) | Root-level architecture summary |

---

## 1. High-Level Flow

The application is a **local LLM training and inference suite** that:

1. **Fine-tunes** a base causal LM with QLoRA (4-bit quantization + LoRA) using ICDU-formatted data.
2. **Merges** the LoRA adapter into the base model and saves a standalone model.
3. **Runs DPO** (Direct Preference Optimization) on preference pairs derived from messages-format JSONL (often a separate Breaking Better file) to improve tool selection.
4. **Optionally runs inference** with a tool-augmented agent loop using the best available model (DPO model preferred over merged model).

The sample config ([`src/config.yaml`](../src/config.yaml)) defaults to `Qwen/Qwen3.5-9B` with `max_length: 4096`. Attention backends and optional Qwen linear-attention kernels are controlled under `model:` (see §8).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION BOUNDARY                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   config.yaml ──► main.py (argparse CLI) ──► run_pipeline()                      │
│                        │                                                         │
│                        ▼                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ 1. TRAINING PHASE     load_tokenizer, load_model (QLoRA), run_training  │   │
│   │    ICDU JSONL ──► load_and_prepare_dataset ──► SFTTrainer ──► final_adapter │ │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                        │                                                         │
│                        ▼                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ 2. MERGING PHASE     merge_and_save_model                                │   │
│   │    base model + final_adapter ──► merge_and_unload ──► final_merged_model │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                        │                                                         │
│                        ▼                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ 3. DPO PHASE         generate_preference_pairs, run_dpo_training          │   │
│   │    messages JSONL ──► preference pairs ──► DPOTrainer ──► dpo_model        │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                        │                                                         │
│                        ▼                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ 4. INFERENCE (opt)   load_model_pipeline, agent_loop                      │   │
│   │    query ──► generate ──► extract_tool_calls ──► execute tools ──► reply  │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Pipeline Phases (Sequential)

```
                    ┌──────────────┐
                    │ config.yaml  │
                    │ (data, model,│
                    │  lora, etc.) │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  main.py     │  load_config_from_yaml → ScriptConfig
                    │  (argparse)  │  Environment.setup_backends(), seed
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ PHASE 1        │ │ PHASE 2        │ │ PHASE 3        │
│ Training       │ │ Merging        │ │ DPO            │
│ (QLoRA SFT)    │ │ (LoRA→base)    │ │ (preference)   │
└───────┬────────┘ └───────┬────────┘ └───────┬────────┘
        │                  │                  │
        ▼                  ▼                  ▼
  final_adapter     final_merged_model    dpo_model
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ PHASE 4      │  Optional: --run-inference
                    │ Inference    │  Prefer dpo_model else final_merged_model
                    │ (agent loop) │
                    └──────────────┘
```

**CLI (argparse):**

```bash
python -m src.main --config-path src/config.yaml
python -m src.main --config-path src/config.yaml --run-inference
python -m src.main --config-path src/config.yaml --run-inference \
  --example-queries "Calculate 2+2" --torch-compile
python -m src.main --config-path src/config.yaml --inference-only
python -m src.main optimize-config --config-path src/config.yaml \
  --preset balanced -o optimized_config.yaml
```

Empty argv prints help and exits `2`. `-h` / `--help` prints pipeline help and exits `0`.

---

## 3. Phase 1: Training (QLoRA SFT)

**Entry:** `run_training(config, env, tokenizer, quantized_model)` in `train.py`  
**Inputs:** `config.data.train_file`, `config.data.validation_file` (**ICDU** JSONL), base model name, quantization/LoRA/training config  
**Outputs:** `{output_dir}/final_adapter/` (LoRA weights only)

```
config.data (train_file, validation_file)  # ICDU fields required
        │
        ▼
load_and_prepare_dataset(config.data, tokenizer)
        │  load_dataset("json", data_files) → Dataset
        │  map(format_icdu_to_chat) → system/user/assistant with optional perturbation
        │  filter(empty) → formatted_dataset
        ▼
VectorizedCompletionOnlyCollator(tokenizer, response_template="Assistant: ")
        │  Pads sequences, masks prompt tokens; only assistant response used for loss
        ▼
PeftLoraConfig + SFTTrainer (TRL)
        │  EarlyStoppingCallback(patience=3)
        │  TrainingArguments from config.training (bf16/fp16 from env)
        ▼
trainer.train() → trainer.save_model(final_adapter)
```

**Key modules:**

- **`src/data/__init__.py`**: `load_and_prepare_dataset`, `format_icdu_to_chat`, `VectorizedCompletionOnlyCollator`
- **`src/model_setup.py`**: `load_tokenizer`, `load_model` (4-bit BnB; Flash Attention 2/3 or SDPA with import + head-dim guards; optional linear-attention kernel validation)
- **`src/train.py`**: `run_training`, `_prepare_training_arguments`, `_prepare_trainer_kwargs`

Note: `train.run_pipeline()` runs SFT + merge only. The full four-phase pipeline lives in `main.run_pipeline()`.

---

## 4. Phase 2: Merging

**Entry:** `merge_and_save_model(config, env)` in `train.py`  
**Inputs:** Base model (HF name), `{output_dir}/final_adapter/`  
**Outputs:** `{output_dir}/final_merged_model/` (full model + tokenizer)

```
config.model.name (base model)
        │
        ▼
validate_linear_attention_kernels (if enabled)
        │
        ▼
resolve_model_class(...) → declared architecture (keeps every checkpoint tensor)
        │
        ▼
<resolved class>.from_pretrained(..., device_map="cpu", high precision)
        │
        ▼
PeftModel.from_pretrained(base_model, final_adapter)
        │
        ▼
peft_model.merge_and_unload() → merged_model.save_pretrained(final_merged_model)
        │
        ▼
load_tokenizer(config.model) → tokenizer.save_pretrained(final_merged_model)
```

Merge reloads the base on CPU and does not re-apply `attn_implementation` the same way as `load_model`.

The base is reloaded through the architecture its `config.json` declares (`model.preserve_all_tensors`, default `true`), so a multimodal base keeps its vision tower in `final_merged_model` — the merge can only write out what it loaded. Its processor is saved alongside the tokenizer.

---

## 5. Phase 3: DPO (Direct Preference Optimization)

**Entry:** DPO block in `run_pipeline()` in `main.py`  
**Inputs:** `dpo.train_file` if set, else `data.train_file` — must be **messages**-format JSONL (not ICDU). Base = `final_merged_model` if present, else `model.name`.  
**Outputs:** `dpo.output_dir` or `{training.output_dir}/dpo_model/`

```
dpo.train_file or data.train_file (JSONL with "messages": [...])
        │
        ▼
load_jsonl(train_file) → list of dicts
        │
        ▼
generate_preference_pairs(data)
        │  For each example: prompt = [INST]user_msg[/INST]
        │  chosen = assistant_response
        │  rejected = text_without_tool OR incorrect_tool_response (random wrong tool)
        ▼
prepare_dpo_dataset(preference_data) → Dataset(prompt, chosen, rejected)
        │
        ▼
run_dpo_training(model_path=final_merged_model, dataset, output_dir, tokenizer, ...)
        │  Loads model directly (4-bit BnB, LoRA q_proj/v_proj) — does not use load_model's
        │  attention resolver; FA2-incompatible models need care (e.g. explicit SDPA elsewhere)
        ▼
DPOTrainer.train() → save to dpo_model
```

**Key module:** `src/dpo.py` — `load_jsonl`, `generate_preference_pairs`, `prepare_dpo_dataset`, `run_dpo_training`  
Also runnable standalone: `python -m src.dpo ...`

---

## 6. Phase 4: Inference (Tool-Augmented Agent)

**Entry:** `run_inference_phase(config, example_queries)` in `main.py` when `--run-inference`  
**Model choice:** `_find_model_path(config)` → `dpo_model` if present, else `final_merged_model`  
**Flow:** `load_model_pipeline` → for each query, `agent_loop(query, model_pipeline)`.

Default queries (when `--run-inference` with no `--example-queries`): fitness advice, a calculation, and an add-task string (`DEFAULT_QUERIES` in `main.py`).

**Agent loop** (in `inference_with_tools.py`):

```
user_query (or current_input with prior output + tool results)
        │
        ▼
model_pipeline(current_input, max_new_tokens, do_sample=False) → generated text
        │
        ▼
extract_tool_calls(output)  →  list of { "name", "arguments" }
        │
        ├── if no tool calls  →  return output (final response)
        │
        └── if tool calls:
                    _execute_tools_parallel(tool_calls) or _execute_tools_sequential(tool_calls)
                    │  TOOL_REGISTRY[name](**args) → "Tool X result: ..."
                    ▼
                current_input = previous_input + "Previous output: ..." + "Tool results:" + results + "Now integrate and continue:"
                    │
                    ▼
                next iteration (until no tool calls or max_iterations)
```

The pipeline imports **`inference_with_tools`**, not `tools.py`. Tools such as `search_web`, `calc_tool`, `news_tool`, `python_repl`, `read_file`, `write_file`, `calendar_tool`, `task_tracker_tool`, `job_search_tool`, `get_current_weather`, `animal_medical_database` are registered in `inference_with_tools.py`. `tools.py` is a lighter sibling that reuses `extract_tool_calls`.

---

## 7. Data Flow Summary

| Stage         | Input(s)                                      | Output(s)                              |
|---------------|-----------------------------------------------|----------------------------------------|
| Config        | `config.yaml`                                 | `ScriptConfig` (Pydantic)              |
| Dataset load  | ICDU `train_file`, `validation_file`          | Hugging Face `Dataset` (train/val)     |
| SFT training  | Dataset, base model, LoRA config              | `final_adapter/`                       |
| Merge         | Base model, `final_adapter`                   | `final_merged_model/`                  |
| DPO           | messages JSONL, merged model                  | Preference `Dataset`, then `dpo_model/`|
| Inference     | User query, `dpo_model` or merged             | Text response (optional tool use)     |

---

## 8. Configuration Structure

Defined in `src/config.py` (Pydantic):

- **ScriptConfig**
  - **data:** `DataConfig` — `train_file`, `validation_file` (paths; must exist at load)
  - **model:** `ModelConfig` — `name`, `max_length`, `attn_implementation` (`eager` / `flash_attention_2` / `sdpa` / `None`), `trust_remote_code`, **`use_linear_attention_kernels`**
  - **quantization:** `QuantizationConfig` — `enabled`, `quant_type` (nf4/fp4), `use_double_quant`
  - **lora:** `LoraConfigModel` — `r`, `alpha`, `dropout`, `target_modules`
  - **training:** `TrainingConfig` — `output_dir`, seed, batch sizes, optimizer, LR, eval/save/log steps, **`save_only_model`**, etc.
  - **dpo:** `DPOConfig` (optional) — `train_file`, `output_dir`, `learning_rate`, `beta`, `max_steps`, `lora_rank`, `torch_compile`, etc.

Paths in YAML can be relative; they are resolved relative to the config file’s directory in `main.py` (`_resolve_config_paths`).

### Attention and kernels

- **Flash Attention 2:** default in schema; `load_model` verifies a real `flash_attn` import and falls back to SDPA if import fails or head dim &gt; 256.
- **Gemma 4 / large global heads:** `_model_head_dim` does **not** read `global_head_dim`. Models such as `google/gemma-4-12B` should set `attn_implementation: sdpa` explicitly (FA2 max head dim 256).
- **Qwen3.5 linear attention:** set `use_linear_attention_kernels: true` only when `causal_conv1d` and `flash-linear-attention` (`fla`) are installed; otherwise load fail-fasts. On Windows use `--no-deps` installs into conda `ai-factory` so torch is not upgraded.
- **Loader vs schema:** runtime `load_model` also accepts `flash_attention_3`; the Pydantic `Literal` does not — prefer `flash_attention_2` or `sdpa` in YAML.
- **DPO path:** loads models itself and does not use `load_model`’s attention resolver; FA2-incompatible models need explicit care.

### Model Optimizer

The **Model Optimizer** (`src/model_optimizer.py`) suggests `config.yaml` settings from your machine (GPU VRAM, system RAM, OS) and a preset (**fast**, **quality**, **balanced**, **low_memory**). It applies stability steps like staggered eval/save steps, `save_only_model: true`, and safe `dataloader_num_workers` (e.g. 0 or 2 on Windows). Paths and model name are preserved; training/LoRA/model/DPO knobs are overridden.

```bash
python -m src.main optimize-config --config-path src/config.yaml --preset balanced -o out.yaml
python -m src.model_optimizer src/config.yaml -p balanced -o out.yaml
```

See [`README.md`](../README.md) and [model_optimizer docs](codebase_docs/model_optimizer__documentation.md).

---

## 9. Key Files and Roles

| Path                            | Role |
|---------------------------------|------|
| `src/main.py`                   | argparse CLI; load config; full pipeline (SFT → merge → DPO → optional inference). |
| `src/config.py`                 | Pydantic config models (`ScriptConfig`, …). |
| `src/config.yaml`               | Example YAML (`Qwen/Qwen3.5-9B`, ICDU + DPO messages paths). |
| `src/data/`                     | ICDU load/format/collate; generation and augmentation scripts. |
| `src/model_setup.py`            | Tokenizer/model load; BnB; attention resolution; linear-kernel validation. |
| `src/train.py`                  | SFT with SFTTrainer; merge adapter into base. |
| `src/dpo.py`                    | Preference pairs; DPOTrainer; standalone CLI. |
| `src/inference_with_tools.py`   | Pipeline inference: tool registry, agent loop. |
| `src/tools.py`                  | Lightweight alternate agent; reuses `extract_tool_calls`. |
| `src/utils.py`                  | `Environment` (CUDA, bitsandbytes, dtype, device). |
| `src/hardware.py`               | `HardwareProfile` for the Model Optimizer. |
| `src/model_optimizer.py`        | Presets and recommendation engine. |
| `src/helper_scripts/`           | Helpers (e.g. DPO merge base, HF upload). |

---

## 10. Artifact Layout (Typical)

After a full run, `config.training.output_dir` (e.g. `./training_output`) typically contains:

- `final_adapter/`        — LoRA weights from SFT
- `final_merged_model/`   — Full model + tokenizer after merge
- `dpo_model/`            — DPO-trained model (when `dpo.output_dir` is unset, or points here)
- `checkpoint-*/`         — Training checkpoints (if save_steps / save_strategy keep them)
- `logs/`                 — Training logs (if `report_to` / logging_dir used)

Inference uses **dpo_model** if present, otherwise **final_merged_model**.

---

*Last updated: 2026-08-23*
