# AGENTS.md — AI Factory

Instructions for coding agents working in this repository. Prefer this file over guessing. Point to docs instead of pasting them.

## What this repo is

Local LLM training/inference suite: QLoRA SFT → LoRA merge → DPO → optional tool-augmented inference. Config is YAML + Pydantic (`src/config.py` / `src/config.yaml`). CLI is **argparse** in `src/main.py` (not Typer).

Pipeline entry:

```bash
conda run -n ai-factory python -m src.main --config-path src/config.yaml
conda run -n ai-factory python -m src.main optimize-config --config-path src/config.yaml --preset fast --output config_optimized.yaml
```

GPU Docker (Windows Docker Desktop WSL2 or Linux NVIDIA; same Compose file):

```bash
docker compose run --rm gpu-check
docker compose run --rm train
docker compose --profile infer run --rm infer
```

See README **Running with Docker (GPU)** for volumes, `HF_TOKEN`, and optional flash-attn builds. Compose bind-mounts `src/` so Python/config changes apply without `docker compose build`.

## Layout (start here)

| Path | Role |
|------|------|
| `src/main.py` | CLI + pipeline orchestration |
| `src/config.py`, `src/config.yaml` | Schema + defaults |
| `src/train.py` | QLoRA SFT + merge |
| `src/dpo.py` | Preference pairs + DPO (loads models **directly**, not via `load_model`) |
| `src/model_setup.py` | Tokenizer/model load, attention + linear-kernel guards |
| `src/inference_with_tools.py`, `src/tools.py` | Tool agent loop |
| `src/model_optimizer.py`, `src/hardware.py`, `src/utils.py` | Hardware presets / env helpers |
| `src/data/` | ICDU datasets and generation scripts |
| `tests/` | pytest suite (run from repo root) |
| `Dockerfile`, `docker-compose.yml`, `requirements.docker.txt` | Linux CUDA 12.4 GPU image + Compose (`train`, `gpu-check`, `infer` profile) |
| `docs/OVERVIEW.md` | Architecture + doc index |
| `docs/codebase_docs/` | Per-module docs |
| `.cursor/rules/python-lang-styling.mdc` | Python style / tests / design |

## Environment (Windows)

Canonical stack: **conda `ai-factory`** (Python 3.10, torch 2.5.1, cu124 — see `environment.yml`). Do **not** mix it with the repo `venv` (often a different Python/torch).

| Task | Command |
|------|---------|
| Installs / train / pytest (preferred) | `conda run -n ai-factory ...` |
| GPU Docker (Windows Desktop or Linux) | `docker compose run --rm gpu-check` then `train` / `--profile infer` |
| Pytest (venv, if intentional) | `.\venv\Scripts\python.exe -m pytest` |
| OpenMP abort before training | `$env:KMP_DUPLICATE_LIB_OK = 'TRUE'` |
| PATH on PowerShell | `$env:PATH = 'new_segment;' + $env:PATH` (never cmd-style `...;%PATH%`) |

Optional wheels (conda `ai-factory` only; exact match cp310 / torch2.5.1 / cu124):

- **flash-attn-2**: prebuilt wheel with `cxx11abiFALSE`. Do **not** pin flash-attn unconditionally in `requirements.txt`.
- **Qwen3.5 linear kernels** (`model.use_linear_attention_kernels: true`): install `triton-windows`, then `causal-conv1d` + `flash-linear-attention==0.4.2` with `--no-deps`. Never `flash-linear-attention[cuda]` without `--no-deps` (upgrades torch). Set the flag `false` for PyTorch fallback if wheels/DLLs fail.

## Agent working rules

- Follow `.cursor/rules/python-lang-styling.mdc` for Python style, tests, and design.
- Ask before guessing when requirements are ambiguous or conflict with prior constraints.
- When implementing an attached plan: do **not** edit the plan file; use existing todos only (mark in progress, then complete each).
- Prefer small, focused edits. Do not expand scope into unrelated refactors or drive-by docs.
- Only commit when the user explicitly asks. Never update git config or force-push.
- For deep architecture questions, read `docs/OVERVIEW.md` and the matching file under `docs/codebase_docs/` before inventing structure.

## Model / attention gotchas (do not “fix” casually)

- **`google/gemma-4-12B`**: use `attn_implementation: sdpa`. FA2 fails on global layers (`global_head_dim=512` > FA2 max 256). SFT’s `_model_head_dim` guard only reads `head_dim` (256), not `global_head_dim` — keep explicit `sdpa` in config.
- **DPO** (`src/dpo.py`): does not use `load_model`’s attention guard. FA2-incompatible models need explicit SDPA in DPO config/paths too.
- **Qwen3.5**: `validate_linear_attention_kernels` fail-fasts at load (SFT/DPO/merge/inference) when kernels are enabled but packages are missing.

## Verification

```bash
# From repo root — prefer conda stack for training-related changes
conda run -n ai-factory python -m pytest

# Narrow when touching one area
conda run -n ai-factory python -m pytest tests/test_config.py tests/test_main_cli.py
```

Smoke-check CLI help after entrypoint changes: `conda run -n ai-factory python -m src.main --help`.

## Boundaries

**Do**

- Change configs and code together when a model/backend requirement changes (SFT and DPO paths).
- Keep optional CUDA packages out of unconditional `requirements.txt` pins.
- Update the relevant `docs/codebase_docs/*` doc when behavior or public APIs change meaningfully.

**Ask first**

- Switching default models, attention backends, or kernel flags in sample configs.
- Adding heavy dependencies or changing `environment.yml` / torch CUDA pins.
- Large refactors across train/DPO/inference load paths.

**Never**

- Install `flash-linear-attention[cuda]` without `--no-deps` into the training env.
- Assume `venv` and conda `ai-factory` are interchangeable.
- Treat DPO model loading as covered by `model_setup.load_model` guards.
