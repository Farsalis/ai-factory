# AI Factory - Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Data Formats](#data-formats)
5. [Step-by-Step Usage Guide](#step-by-step-usage-guide)
6. [Configuration Reference](#configuration-reference)
7. [Module Documentation](#module-documentation)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## Overview

**AI Factory** is a comprehensive pipeline for fine-tuning, evaluating, and optimizing large language models (LLMs) using advanced techniques like QLoRA (4-bit quantization) and Direct Preference Optimization (DPO). The system is designed for efficient training on consumer GPUs (e.g., RTX 4070 with 8GB VRAM) and supports the complete workflow from data generation to model deployment.

### Key Features

- **QLoRA Fine-tuning**: Memory-efficient 4-bit quantization fine-tuning using BitsAndBytes
- **DPO Training**: Direct Preference Optimization for model alignment via preference data
- **ICDU Dataset Format**: Structured dataset format with persona archetypes, capability layers, and governing principles
- **Tool-Augmented Inference**: Agent loop with tool execution capabilities (web search, calculations, file operations, etc.)
- **Model Merging**: Automatic merging of LoRA adapters into base models
- **Modular Configuration**: YAML-based configuration system with Pydantic validation
- **Comprehensive Testing**: Unit and integration tests with pytest

### Target Use Cases

- Fine-tuning Mistral-7B and similar models for specific domains
- Training agentic AI systems with tool-calling capabilities
- Research and prototyping of alignment techniques (DPO)
- Custom dataset generation and augmentation

---

## Architecture

### System Components

```
ai-factory/
├── src/
│   ├── main.py                 # Main CLI entry point
│   ├── config.py              # Configuration schema (Pydantic)
│   ├── config.yaml            # Default configuration file
│   ├── train.py               # QLoRA training and model merging
│   ├── model_optimizer.py     # Hardware-aware config recommendation
│   ├── dpo.py                 # DPO training and preference generation
│   ├── model_setup.py         # Model/tokenizer loading utilities
│   ├── inference_with_tools.py # Tool-augmented inference agent loop
│   ├── tools.py               # Tool registry and execution
│   ├── utils.py               # Environment detection and utilities
│   ├── data/
│   │   ├── __init__.py        # Data loading and ICDU formatting
│   │   ├── master_generate_icdu.py  # ICDU dataset generation
│   │   ├── generate_icdu_dataset.py   # Dataset conversion
│   │   ├── augment_dataset.py        # Data augmentation
│   │   └── ...
│   └── upload/
│       └── upload_to_hf.py    # Hugging Face model upload
├── tests/                     # Test suite
├── data/                      # Training/validation data (JSONL)
└── requirements.txt           # Python dependencies
```

### Pipeline Flow

1. **Data Generation** → Generate ICDU-formatted datasets from raw data
2. **Training** → QLoRA fine-tuning with 4-bit quantization
3. **Merging** → Merge LoRA adapters into base model
4. **DPO Training** → Direct Preference Optimization for alignment
5. **Inference** → Tool-augmented agent loop for interactive use

---

## Installation & Setup

### Prerequisites

- **Python**: 3.10+ (3.13.3 recommended per `pyproject.toml` (venv) and 3.12 per `environment.yml` (conda))
- **GPU**: NVIDIA GPU with 8GB+ VRAM (tested on RTX 4070 and A5500 Laptop GPU)
- **CUDA**: 12.1+ (installed via conda)
- **Operating System**: Windows, Linux, or macOS (Windows and Linux tested)

### Creating Conda/Miniconda Environment

This training setup uses a combination of torchvision, torchaudio, torch, pytorch, and pytorch-cuda in a conda env for 
higher efficiency training. Because of pytorch-cuda versioning for max CUDA 12.4 toolkit, all other installed packages 
must have the same CUDA version which requires the use of python 3.10 instead of newer versions.
Most packages are installed through the conda package manager for easier compilation and resolution, however,
for certain parts of the codebase to work the `torch` module must be installed through pip 
(which must also use CUDA 12.4 to avoid conflicts). 

For Windows, to compile binaries for packages like `bitsandbytes` and `flash-attn` you will need to have 
compatible versions of the CUDA toolkit and MSVC build tools. Otherwise, building binaries will typically fail. 
If this does happen, you can attempt to download pre-built wheels from the respective package repositories or websites.

#### Step 1: Install Miniconda/Anaconda

If you don't have Conda installed, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended for minimal footprint).

#### Step 2: Create Conda Environment

Check current anaconda installation info. Make sure to use only the python venv or conda venv at a time, not both.

```bash
conda init  # To start working in conda env

conda info # Check installation info and python version

conda clear --all  # Clear past conda downloads, unused packages, etc. if necessary
```

Create conda environment using `environment.yml` (conda default). Modify for your local CUDA version and such.

```bash
conda update -n [ENV_NAME] --all  # Update conda packages if necessary

conda env create --file environment.yml 
conda activate ai-factory
```

##### Step 2.1: Determine CUDA GPU Configuration

In terminal, check which CUDA version your GPU runs (e.g. 12.X). As long as GPU driver CUDA version number is >= conda pytorch-cuda package, then they are compatible. See troubleshooting below for more details.  

```bash
nvidia-smi
```

##### Step 2.2: Install PyTorch with CUDA

```bash
conda search pytorch-cuda -c pytorch  # Look into pytorch channel for different CUDA versions

# Look into pytorch and nvidia channels within anaconda registry; then, install each together. 
# Conda package manager will automatically resolve versions and dependencies.
# (Optionally) Use --dry-run flag to see how install will work. Use -y to skip confirmation prompt.
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia # Whichever cuda version compatible with your GPU

conda list pytorch  # Check for pytorch install

pip show bitsandbytes  # Make sure bitsandbytes installed
pip show torch  # Make sure python code can access torch module
```

This ensures you get the correct CUDA runtime without requiring a full system CUDA installation. 

If for some reason you are unable to install PyTorch + CUDA packages, install all dependencies and packages through the 
python project venv:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

##### Step 2.3: Install Python Project Dependencies

Sometimes if there are package resolutions / dependency errors, pip will not install required python packages. 
If this is the case, you will have to run the pip install separately.

```bash
pip install -r requirements.txt
```

In the case you need to install older package versions:

```bash
pip install -r requirements.txt --ignore-requires-python
```

If you encounter compatibility issues with `bitsandbytes`, install the binary files with:

```bash
pip install bitsandbytes --prefer-binary
```

#### Step 3: Verify Installation

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Expected output:

```
True
12.X
```

#### Environment Maintenance

Conda will update packages based on their package and version dependencies. 

```bash
conda activate ai-factory  # If not activated already

conda update --all --dry-run  # Check how update will be handled

conda update --all
```

#### Troubleshooting Installation

- **Driver Issues**: Verify NVIDIA drivers are up to date (NVIDIA driver 530+ for CUDA 12.1)
- **CUDA Version Check**:
  ```bash
    nvcc --version
    nvidia-smi
  ```
- **bitsandbytes Issues**: On Windows, ensure you're using a compatible version. Try:
  ```bash
    pip uninstall bitsandbytes
    pip install bitsandbytes --prefer-binary --no-cache-dir
  ```
- **flash-attn** 
First, check your conda's CUDA toolkit version and MSVC build tools version. If they are not, proceed: 
  ```bash
    pip cache purge  # Clear pip cache to avoid old conflicts
    pip show flash-attn  # Make sure flash-attn not installed  

    pip install flash-attn --no-build-isolation 
  ```
  On Windows, building from source usually fails (no CUDA Toolkit / MSVC `cl`). Prefer an
  exact-match **prebuilt wheel** instead — a prebuilt wheel needs no CUDA Toolkit or MSVC.
  The canonical env is conda `ai-factory` (Windows / Python 3.10 / torch 2.5.1 / cu124 / Ampere):

  ```bash
    conda run -n ai-factory pip install "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4+cu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl"
  ```

  Fallback source (kingbri1 mirror, same exact tags):

  ```bash
    conda run -n ai-factory pip install "https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl"
  ```

  The wheel must match your **python + torch + cuda + abi** exactly (here `cp310`, `torch2.5.1`,
  `cu124`, `cxx11abiFALSE`); a mismatched wheel may import but crash at model load.

### Running with Docker (GPU)

The same Compose file runs on **Windows Docker Desktop** (WSL2 + NVIDIA GPU) and **Linux NVIDIA** hosts. The image is Linux CUDA 12.4 / Python 3.10 / torch 2.5.1 (pip), not conda. Host Windows conda remains the native workflow.

**Prerequisites**

- Windows: Docker Desktop with the WSL2 backend and NVIDIA GPU support enabled; a recent NVIDIA driver.
- Linux: NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- Copy `.env.example` to `.env` and set `HF_TOKEN` if you load gated Hugging Face models.
- Datasets referenced in `src/config.yaml` must exist under `src/data/` (they are bind-mounted, not baked into the image). Sample config paths such as `icdu_training_data_v8.jsonl` are not in-tree; point the config at files you actually have.

**Commands** (from the repo root)

```bash
cp .env.example .env   # then edit HF_TOKEN if needed

# Smoke: CUDA visible to PyTorch inside the container
docker compose run --rm gpu-check

# Full pipeline: QLoRA SFT → merge → DPO
docker compose run --rm train

# DPO torch.compile (Linux containers; ignored/unreliable on native Windows)
docker compose run --rm train python -m src.main --config-path src/config.yaml --torch-compile

# Inference only (does not retrain). Requires checkpoints under src/training_output/
# (dpo_model preferred, else final_merged_model).
docker compose --profile infer run --rm infer
docker compose --profile infer run --rm infer python -m src.main \
  --config-path src/config.yaml --inference-only \
  --example-queries "Calculate 2+2"
```

Compose bind-mounts `./src` into the container (Python, `config.yaml`, datasets, `training_output`). **You do not need to rebuild the image** after changing training code or YAML. Rebuild only for Dockerfile / `requirements.docker.txt` / flash-attn.

Compose sets `shm_size: 16gb` so DataLoader workers do not fail on `/dev/shm`. Hugging Face downloads persist in the `hf-cache` named volume. Checkpoints persist in `src/training_output/`. Tool-agent file tools use `data/allowed/read` and `data/allowed/write`.

Flash Attention 2 is **off** in the default image; `model_setup` falls back to SDPA if `flash_attn` is missing. To build FA2 into the image (needs the CUDA **devel** base):

```bash
docker compose build \
  --build-arg INSTALL_FLASH_ATTN=true \
  --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

Do not change `src/config.yaml` attention defaults unless you intend to. Linear-attention kernels stay disabled unless you install those packages yourself (`model.use_linear_attention_kernels`).

A full SFT run is not a smoke test (hours / VRAM). After `gpu-check`, `docker compose run --rm train python -m src.main --help` is enough to confirm the CLI inside the image.

---

## Data Formats

### ICDU Dataset Format

The codebase uses the **ICDU (Intent Conscious Data Unit)** format for training data. Each line in a JSONL file represents one training example with the following structure:

```json
{
  "icdu_id": "unique-uuid-identifier",
  "persona_archetype": "Fitness Seeker > Struggling Starter",
  "governing_principle": "Empowerment through incremental progress",
  "capability_layer": "Foundational",
  "user_intent": "Seek guidance on starting a fitness routine",
  "context_summary": "User is a busy professional looking to start exercising",
  "application_prompt": "How do I start working out?",
  "ideal_response_final": "Starting a fitness routine begins with...",
  "ideal_response_attributes": ["clear", "actionable", "encouraging"],
  "ideal_response_cot": "Step 1: Assess current fitness level..."
}
```

#### Required ICDU Fields

- `application_prompt`: User's input query
- `ideal_response_final`: Desired assistant response
- `persona_archetype`: User persona classification
- `capability_layer`: Response capability level ("Foundational", "Transformational", "Aspirational")
- `context_summary`: Contextual information about the user
- `user_intent`: Classification of user's intent

#### Capability Layers

- **Foundational**: Direct, factual answers with clarity
- **Transformational**: Empathetic guidance with problem reframing
- **Aspirational**: Proactive, insightful responses with anticipation

### Standard Chat Format (Alternative)

For non-ICDU datasets, use the standard chat format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."}
  ]
}
```

### DPO Data Format

```json
{
  "messages": [
    {"role": "system", "content": "system-level directive"},
    {"role": "user", "content": "user prompt"},
    {"role": "assistant", "content": "assistant directives w/ option tooling details"}
  ]
}
```

### Data Generation Overview

#### Generate ICDU Dataset (w/ DPO Data) from non-ICDU (Regular Chat) Data

#### Files:

- `generate_icdu_dataset.py` - Only Converts the Provided Dataset
- `generate_icdu_validation_dataset.py` - Only Converts the Validation Dataset 
- `generate_icdu_publication_dataset.py` - Converts and Heavily Augments the Given Dataset

non-ICDU Formatted Dataset -> Run One of the Files Above -> Output is ICDU-Formatted Dataset w/ DPO Data

#### Augment DPO Datasets

#### Files:

- `augment_dataset.py` - Augments the Given DPO Dataset
- `augment_validation_dataset` - Augments the Given Validation DPO Dataset

DPO Dataset -> Generate New Augment Data w/ Inserted Tool Calling -> Output is New Perturbed DPO Dataset

#### Generate/Augment ICDU w/o DPO Datasets

#### Files:

- `master_generate_icdu.py` - Either Generates or Augments (adds more to) Given Dataset (either Chat or ICDU Data)
- `generate_icdu_proactive_dataset.py` - Generates or Augments Given Dataset (either Chat or ICDU Data) with Follow-up Questions

Chat or ICDU Dataset -> Run One of the Above Files -> ICDU Dataset (w/ or w/o Augmentation) or ICDU Dataset + Inserted Proactive Follow-up Questions

---

## Model Optimizer (recommended config from hardware)

The Model Optimizer suggests `config.yaml` settings based on your machine (GPU VRAM, system RAM, OS) and a preset. It follows the stability rules in **TRAINING_CRASH_DIAGNOSIS.md** (staggered eval/save, `save_only_model`, safe `dataloader_num_workers`).

**Presets:**

- **fast** — Throughput: larger batch, fewer eval/save steps, smaller LoRA rank, shorter sequences; DPO `torch_compile` when supported.
- **quality** — Convergence: more epochs, higher LoRA rank, longer sequences, more frequent eval/save, gradient checkpointing on.
- **balanced** — Default: mix of speed and stability (e.g. eval_steps 150, save_steps 300).
- **low_memory** — Minimal VRAM/RAM: smallest batch sizes, gradient checkpointing, minimal workers.

**Commands:**

```bash
# Suggest settings and write optimized config (requires an existing config file)
python -m src.main optimize-config --config-path src/config.yaml --preset fast --output src/config_optimized.yaml

# Or via the optimizer module
python -m src.model_optimizer tests/configs/test_config.yaml -p quality -o config_optimized_quality.yaml
```

Paths and model name are taken from your base config; only training/LoRA/model/DPO knobs are tuned. Use the generated file as your new `config.yaml` or merge the suggested values by hand.

**YOU STILL HAVE TO SPECIFY THE LOCATIONS OF DATASETS**

---

## Step-by-Step Usage Guide

### Complete Training Pipeline

The main.py file entry point orchestrates the entire pipeline: training → merging → DPO → inference.

#### Step 1: Prepare Your Data

Ensure you have ICDU-formatted JSONL files for training and validation:

```bash
# Example: Generate ICDU dataset from raw data
# Make sure DPO Dataset is Included or Generated at Some Point (optional)
python -m src.data.master_generate_icdu \
    --input-file /path/to/data/raw_data.jsonl \
    --output-dir /path/to/data/generated_data \
    --augmentation-factor 10 \
    --validation-split 0.1
```

This will create:

- `./datasets/icdu_training_data_vX.jsonl`
- `./datasets/icdu_validation_data_vX.jsonl`

##### Step 1.2: Generate DPO Preferences Data

Part of the training pipeline automatically create DPO preference pairs, for specific training on correct tool calling. ICDU datasets will not have these user-response message by default. You will have to either augment existing DPO preference data in this directory via master_generate_icdu.py or create your own. DPO Preference training can also be done separately after ICDU-SFT training. 

Once you have the DPO Data, you may either put them in the ICDU dataset or train using them separately. 

#### Step 2: Configure Training

Edit `src/config.yaml` to match your setup:

```yaml
data:
  train_file: "./data/datasets/icdu_training_data_vX.jsonl"
  validation_file: "./data/datasets/icdu_validation_data_vX.jsonl"

model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"  # or "Qwen3/Qwen3-4B-Instruct-2057"
  max_length: 8192
  attn_implementation: "flash_attention_2"

training:
  output_dir: "./output/my-model"
  per_device_train_batch_size: 2
  learning_rate: 0.0002
  num_train_epochs: 1
  # ... other settings
```

**Important Path Resolution**:

- Paths in `config.yaml` are resolved **relative to the config file's parent directory** (not the current working directory)
- If `config.yaml` is in `src/`, then `./data/file.jsonl` resolves to `src/data/file.jsonl`
- Use relative paths (starting with `./`) for portability
- Absolute paths are also supported
- IF USING LOCAL MODEL, DIRECTORY MUST BE IN HUGGINF FACE FORMAT. OTHERWISE, USE HUGGING FACE MODEL ID in `config.yaml`.

#### Step 3: Run Training Pipeline

**Basic training (training + merging + DPO):**

From the project root directory:

```bash
python -m src.main --config-path src/config.yaml
```

Or if running from the `src/` directory:

```bash
python main.py --config-path config.yaml
```

**With inference after training:**

```bash
python -m src.main \
    --config-path src/config.yaml \
    --run-inference \
    --example-queries "Advise on fitness habits using latest research." \
    --example-queries "Calculate 5 * (3 + 7) / 2"
```

**With torch.compile optimization (PyTorch 2.0+):**

```bash
python -m src.main \
    --config-path src/config.yaml \
    --torch-compile
```

#### Step 4: Monitor Training

Training logs will show:

- Training loss and evaluation metrics
- Checkpoint saves (every `save_steps` steps)
- Best model selection based on `metric_for_best_model`

Output directory structure:

```
output/my-model/
├── final_adapter/          # LoRA adapter weights
├── final_merged_model/     # Merged model (base + adapter)
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
└── dpo_model/             # DPO-trained model
    ├── adapter_config.json
    └── adapter_model.safetensors
```

#### Step 5: Run Inference

**Using the trained model with tools:**

```python
from src.inference_with_tools import load_model_pipeline, agent_loop

# Load the best available model (prefers DPO model)
model_pipeline = load_model_pipeline("./output/my-model/dpo_model")

# Run agent loop with tool execution
response = agent_loop(
    "Search for recent AI research papers and summarize findings",
    model_pipeline,
    max_iterations=5
)
print(response)
```

**Command-line inference:**

```bash
# Using the final_merged_model works too.
python -m src.inference_with_tools \
    --model_path ./output/my-model/dpo_model \  
    --query "Your query here"
```

### Individual Component Usage

#### Generate ICDU Dataset via Complete Build Pipeline (will work for most of your needs)

```bash
python -m src.data.master_generate_icdu \
    --input-file data/raw_data.jsonl \
    --output-dir data/generated_data \
    --augmentation-factor 20 \
    --validation-split 0.1 \
    --num-processes 8 \
    --run-reports
```

Options:

- `--augmentation-factor`: Number of augmented variants per example (default: 10)
- `--validation-split`: Fraction of data for validation (default: 0.1)
- `--num-processes`: Parallel processing workers (default: CPU count)
- `--run-reports`: Generate distribution plots and metrics

#### Run DPO Training Separately

```bash
python -m src.dpo \
    --input data/augmented_data.jsonl \
    --output data/dpo_preferences.jsonl \
    --model_path ./output/my-model/final_merged_model \
    --training_output_dir ./output/my-model/dpo_model \
    --lora_rank 32 \
    --max_steps 100 \
    --learning_rate 5e-6
```

---

## Configuration Reference

### Configuration File Structure

The `config.yaml` file uses a hierarchical structure validated by Pydantic models in `src/config.py`.

#### Data Configuration

```yaml
data:
  train_file: "./data/icdu_training_data_vX.jsonl"      # Required: Training data path
  validation_file: "./data/icdu_validation_data_vX.jsonl"  # Required: Validation data path
```

Paths are resolved relative to the config file's directory.

#### Model Configuration

```yaml
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"  # Hugging Face model ID
  max_length: 8192                              # Maximum sequence length
  attn_implementation: "flash_attention_2"      # "flash_attention_2", "sdpa", or "eager"
  trust_remote_code: true                        # Allow custom model code
  preserve_all_tensors: true                     # Keep every base-model tensor
```

**Attention Implementation**:

- `flash_attention_2`: Fastest, requires `flash-attn` package
- `sdpa`: PyTorch 2.0+ scaled dot product attention (fallback)
- `eager`: Standard attention (slowest, most compatible)

**Preserve All Tensors** (default `true`):

The pipeline loads the architecture the checkpoint declares in its `config.json`
rather than `AutoModelForCausalLM`. This matters for natively multimodal
checkpoints — Qwen3.5 among them: `AutoModelForCausalLM` maps them onto their
text-only submodel (`Qwen3_5ForCausalLM`), which lists `^model.visual.*` in
`_keys_to_ignore_on_load_unexpected` and therefore discards the vision tower
without printing a warning. Since the merge step can only write out what it
loaded, the exported model would silently lose those weights.

With `preserve_all_tensors: true`, the vision tower is loaded, carried through
the merge, and written to `final_merged_model` alongside the base model's
processor. LoRA is unaffected: the configured `target_modules` (`q_proj`, …) do
not match the vision tower's module names (`qkv`, `proj`, `linear_fc1/2`), so the
same text-stack modules are trained as before.

Set it to `false` only to deliberately export a text-only model. Note that
Qwen3.5's multi-token-prediction head (`mtp.*`, ~0.24B parameters) is dropped
either way — transformers implements no class that holds it.

#### Quantization Configuration

```yaml
quantization:
  enabled: true              # Enable 4-bit quantization
  quant_type: "nf4"          # "nf4" (recommended) or "fp4"
  use_double_quant: true     # Double quantization for better compression
```

**Note**: Quantization requires CUDA and bitsandbytes. Automatically disabled if unavailable.

#### LoRA Configuration

```yaml
lora:
  r: 32                      # LoRA rank (lower = fewer parameters)
  alpha: 32                  # LoRA alpha (typically 2*r)
  dropout: 0.05             # Dropout rate for LoRA layers
  target_modules:            # Modules to apply LoRA to
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

**LoRA Rank Guidelines**:

- `r=8-16`: Very memory-efficient, lower capacity
- `r=32`: Balanced (default)
- `r=64+`: Higher capacity, more memory

#### Training Configuration

```yaml
training:
  output_dir: "./output/my-model"           # Output directory (required)
  seed: 42                                   # Random seed
  num_train_epochs: 1                        # Training epochs
  per_device_train_batch_size: 2            # Batch size per device
  per_device_eval_batch_size: 2              # Eval batch size
  gradient_accumulation_steps: 2             # Gradient accumulation
  optim: "paged_adamw_8bit"                  # Optimizer (memory-efficient)
  learning_rate: 0.0002                      # Learning rate
  weight_decay: 0.001                        # Weight decay
  max_grad_norm: 0.3                         # Gradient clipping
  warmup_ratio: 0.03                         # Warmup steps ratio
  lr_scheduler_type: "cosine"                # LR scheduler
  evaluation_strategy: "steps"                # When to evaluate
  eval_steps: 50                              # Steps between evals
  save_strategy: "steps"                       # When to save
  save_steps: 50                              # Steps between saves
  save_total_limit: 2                          # Max checkpoints to keep
  logging_steps: 10                           # Steps between logs
  group_by_length: true                       # Group sequences by length
  gradient_checkpointing: true                 # Memory-saving technique
  report_to: "none"                           # "wandb", "tensorboard", or "none"
  load_best_model_at_end: true                 # Load best checkpoint
  metric_for_best_model: "eval_loss"          # Metric for best model
  greater_is_better: false                    # Lower is better for loss
  remove_unused_columns: true                  # Remove unused dataset columns
  dataloader_num_workers: 4                   # Data loading workers
```

**Memory Optimization Tips**:

- Reduce `per_device_train_batch_size` if OOM errors occur
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable `gradient_checkpointing` for memory savings
- Use `paged_adamw_8bit` optimizer for lower memory usage

---

## Module Documentation

### `src/main.py`

Main CLI entry point that orchestrates the complete pipeline.

**Functions**:

- `cli()`: CLI entry point (argparse; `python -m src.main --config-path ...`)
- `run_pipeline()`: Executes training → merging → DPO → inference
- `load_config_from_yaml()`: Loads and validates YAML configuration
- `run_inference_phase()`: Runs inference with tool-augmented agent

**Usage**:

```bash
# From project root
python -m src.main --config-path src/config.yaml [--run-inference] [--example-queries ...]

# Or from src/ directory
python main.py --config-path config.yaml [--run-inference] [--example-queries ...]
```

### `src/train.py`

QLoRA fine-tuning and model merging logic.

**Functions**:

- `run_training()`: Conducts supervised fine-tuning with QLoRA
- `merge_and_save_model()`: Merges LoRA adapters into base model
- `_prepare_training_arguments()`: Converts config to TrainingArguments
- `_prepare_trainer_kwargs()`: Configures SFTTrainer

**Key Features**:

- Automatic optimizer fallback (paged optimizers → adamw_torch if CUDA unavailable)
- Environment-aware precision (bf16/fp16)
- Early stopping callback
- Completion-only loss masking (via `VectorizedCompletionOnlyCollator`)

### `src/dpo.py`

Direct Preference Optimization training and preference pair generation.

**Functions**:

- `generate_preference_pairs()`: Creates chosen/rejected pairs from training data
- `prepare_dpo_dataset()`: Converts preference pairs to Hugging Face Dataset
- `run_dpo_training()`: Executes DPO training with QLoRA
- `load_jsonl()` / `save_jsonl()`: JSONL file utilities

**Preference Pair Generation**:

- If response has tool call → chosen = with tool, rejected = without tool
- If response has no tool → chosen = without tool, rejected = incorrect tool

**Usage**:

```bash
python -m src.dpo --input data.jsonl --output prefs.jsonl --model_path ... --training_output_dir ...
```

### `src/data/__init__.py`

Data loading and ICDU formatting utilities.

**Functions**:

- `load_and_prepare_dataset()`: Loads JSONL files and applies ICDU formatting
- `format_icdu_to_chat()`: Converts ICDU examples to chat format with perturbations
- `perturb_context()`: Applies scenario-perturbation method for data augmentation
- `VectorizedCompletionOnlyCollator`: Data collator that masks prompt tokens

**ICDU Formatting Process**:

1. Validates required ICDU fields
2. Applies context perturbations (50% probability)
3. Constructs system prompts based on capability layer
4. Formats using tokenizer's chat template (or fallback)

### `src/inference_with_tools.py`

Tool-augmented inference agent loop.

**Tool Registry**:

- `search_web`: Real web search via DuckDuckGo (cached)
- `calc_tool`: Safe mathematical calculations (AST-based)
- `news_tool`: News search (mocked)
- `python_repl`: Safe Python code execution
- `read_file` / `write_file`: File operations (with security restrictions)
- `calendar_tool`: Calendar operations (mocked)
- `task_tracker_tool`: SQLite-based task tracking
- `job_search_tool`: Job search (mocked)
- `get_current_weather`: Weather lookup (mocked)
- `animal_medical_database`: Medical database (mocked)

**Agent Loop**:

1. Generates response using model
2. Extracts tool calls from JSON in output
3. Executes tools (parallel or sequential)
4. Appends results to input for next iteration
5. Repeats until no tool calls or max iterations

**Security**:

- File operations restricted to allowed paths (`AGENT_ALLOWED_READ_PATH`, `AGENT_ALLOWED_WRITE_PATH`)
- Python REPL uses restricted globals
- Calculation tool uses AST parsing (no code execution)

**Usage**:

```python
from src.inference_with_tools import load_model_pipeline, agent_loop

pipeline = load_model_pipeline("./model_path")
response = agent_loop("Query with tool needs", pipeline, max_iterations=5)
```

### `src/model_setup.py`

Model and tokenizer loading with quantization and attention configuration.

**Functions**:

- `load_model()`: Loads model with quantization, attention, and device config
- `load_tokenizer()`: Loads and configures tokenizer
- `_resolve_attention_implementation()`: Handles flash attention fallback
- `_create_bitsandbytes_config()`: Creates quantization config

**Features**:

- Automatic flash attention fallback to SDPA
- Environment-aware dtype selection (bf16/fp16)
- Efficient device mapping with `device_map="auto"`

### `src/utils.py`

Environment detection and hardware configuration.

**Classes**:

- `Environment`: Detects CUDA, bitsandbytes, bf16 support, and optimal compute dtype

**Methods**:

- `setup_backends()`: Enables TF32 for Ampere+ GPUs
- `is_bitsandbytes_available()`: Checks for bitsandbytes library

### `src/data/master_generate_icdu.py`

Comprehensive ICDU dataset generation pipeline.

**Features**:

- Persona archetype inference from patterns
- Governing principle assignment
- Capability layer classification
- Context-aware response generation
- Data augmentation with deduplication
- Distribution reports and metrics

**Usage**:

```bash
python -m src.data.master_generate_icdu \
    --input-file data.jsonl \
    --output-dir output/ \
    --augmentation-factor 20
```

---

## Testing

### Running Tests

**Install dev dependencies** (formatting, linting, type-checking, tests):

```bash
pip install -e ".[dev]"
# or, if using a requirements file: pip install -r requirements.txt
```

**Install pre-commit hooks** (optional; runs Ruff, mypy, bandit on commit):

```bash
pre-commit install
```

Then run `pre-commit run --all-files` once to verify.

**Run all tests**:

```bash
pytest
```

**Run fast tests only** (exclude slow tests):

```bash
pytest -m "not slow"
```

**Run slow tests** (may require GPU/network):

```bash
pytest -m slow
```

**Run specific test file**:

```bash
pytest tests/test_train.py
```

**Run with verbose output**:

```bash
pytest -v
```

### Test Structure

Tests are organized in `tests/` directory:

- `test_config.py`: Configuration validation tests
- `test_data.py`: Data loading and formatting tests
- `test_train.py`: Training pipeline tests
- `test_dpo.py`: DPO training tests
- `test_inference_with_tools.py`: Inference and tool execution tests
- `test_model_setup.py`: Model loading tests
- `test_utils.py`: Utility function tests

### Test Markers

- `@pytest.mark.unit`: Fast unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Slow tests (GPU/network required)

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Errors

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:

- Reduce `per_device_train_batch_size` in config (e.g., from 2 to 1)
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable `gradient_checkpointing: true`
- Reduce `max_length` in model config
- Use lower LoRA rank (`r: 16` instead of `r: 32`)

#### 2. bitsandbytes Not Found

**Symptoms**: `ModuleNotFoundError: No module named 'bitsandbytes'`

**Solutions**:

```bash
pip uninstall bitsandbytes
pip install bitsandbytes --prefer-binary --no-cache-dir
```

On Windows, ensure you're using a compatible Python version (3.10-3.11 recommended).

#### 3. Flash Attention Not Available

**Symptoms**: Warning about flash_attention_2 falling back to sdpa

**Solutions**:

- Install flash-attn (if supported on your system):
  ```bash
  pip install flash-attn --no-build-isolation
  ```
- On Windows, use an exact-match prebuilt wheel (no CUDA Toolkit / MSVC required). For the
  canonical conda `ai-factory` env (Python 3.10 / torch 2.5.1 / cu124 / Ampere):
  ```bash
  conda run -n ai-factory pip install "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4+cu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl"
  ```
  Fallback (kingbri1 mirror):
  ```bash
  conda run -n ai-factory pip install "https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl"
  ```
  The wheel must match your python + torch + cuda + abi exactly.
- Or change `attn_implementation` to `"sdpa"` in config (works on PyTorch 2.0+)

#### 4. Linear Attention Kernels Not Available (Qwen3.5)

**Symptoms**: `RuntimeError: Missing linear-attention dependencies` when
`use_linear_attention_kernels: true` in config

**Context**: Qwen3.5 models use Gated DeltaNet linear-attention layers. When
`use_linear_attention_kernels` is enabled, training/inference expects optional
`causal-conv1d` and `flash-linear-attention` (`fla`) packages for fast CUDA
kernels. These are **not** pinned in `requirements.txt` or `environment.yml`.

**Solutions**:

- Set `use_linear_attention_kernels: false` in your config to use the slow
  PyTorch fallback (works without extra packages).
- Install into conda `ai-factory` (Python 3.10 / torch 2.5.1 / cu124). Use
  `--no-deps` on every step so torch is not upgraded:

  ```powershell
  conda run -n ai-factory pip install --no-deps "https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post9/triton-3.2.0-cp310-cp310-win_amd64.whl"
  conda run -n ai-factory pip install --no-deps "https://github.com/d8ahazard/AudioLab/releases/download/1.0.0/causal_conv1d-1.5.0.post8-cp310-cp310-win_amd64.whl"
  conda run -n ai-factory pip install --no-deps "flash-linear-attention==0.4.2"
  ```

  **Never** run `pip install flash-linear-attention[cuda]` without `--no-deps` —
  it pulls a newer torch and breaks this stack.

- Verify imports:

  ```powershell
  conda run -n ai-factory python -c "from causal_conv1d import causal_conv1d_fn; import fla; print('OK')"
  ```

- If `causal-conv1d` fails with a DLL load error on Windows, either disable the
  flag (fallback) or build `causal-conv1d` from source with MSVC + CUDA toolkit.

#### 5. OpenMP Duplicate Library (Windows)

**Symptoms**: `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
already initialized` — process may abort or appear to do nothing.

**Solutions**:

```powershell
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
```

Set before `python -m src.main` in each PowerShell session, or add to conda
`activate.d` scripts under the `ai-factory` env. Use only the conda env for
training (do not mix with the repo `venv`).

#### 6. Configuration File Not Found

**Symptoms**: `FileNotFoundError: Configuration file not found`

**Solutions**:

- Use absolute path or path relative to current working directory
- Ensure config file exists at specified location
- Check that paths in config use forward slashes (/) or escaped backslashes ()

#### 7. Dataset Empty After Formatting

**Symptoms**: `ValueError: Dataset is empty after formatting and filtering`

**Solutions**:

- Verify ICDU format has all required fields
- Check that `application_prompt` and `ideal_response_final` are non-empty
- Review logs for warnings about missing fields
- Ensure JSONL file is valid (one JSON object per line)

#### 8. DPO Training Fails

**Symptoms**: DPO training errors or preference pairs not generated

**Solutions**:

- Ensure training data has been generated first
- Check that `final_merged_model` exists (or base model path is correct)
- Verify preference pair generation succeeded (check logs)
- Reduce `max_steps` or `per_device_train_batch_size` if OOM

#### 9. Tool Execution Errors

**Symptoms**: Tools fail during inference

**Solutions**:

- Check tool registry has required tools registered
- Verify file paths for `read_file`/`write_file` are in allowed directories
- Check network connectivity for `search_web`
- Review security config (`SecurityConfig` in `inference_with_tools.py`)

#### 10. Model Loading Fails

**Symptoms**: `OSError` or `RuntimeError` during model loading

**Solutions**:

- Verify model name/path is correct
- Check Hugging Face Hub connectivity (for remote models)
- Ensure sufficient disk space for model download
- Try loading with `trust_remote_code: false` if security concerns

### Debugging Tips

1. **Enable Verbose Logging**:
  ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
  ```
2. **Check Environment**:
  ```python
   from src.utils import Environment
   env = Environment()
   print(env)  # Shows CUDA, bitsandbytes, dtype, device
  ```
3. **Verify Data Format**:
  ```python
   import json
   with open("data/train.jsonl") as f:
       example = json.loads(f.readline())
       print(example.keys())  # Check required fields
  ```
4. **Test Model Loading**:
  ```python
   from src.model_setup import load_model, load_tokenizer
   from src.config import ModelConfig, QuantizationConfig
   from src.utils import Environment

   model_config = ModelConfig(name="mistralai/Mistral-7B-Instruct-v0.3")
   quant_config = QuantizationConfig(enabled=True)
   env = Environment()

   tokenizer = load_tokenizer(model_config)
   model = load_model(model_config, quant_config, env)
  ```

---

## Advanced Usage

### Custom Tool Registration

Add custom tools to the inference agent:

```python
from src.inference_with_tools import register_tool, TOOL_REGISTRY

@register_tool("my_custom_tool")
def my_custom_tool(query: str) -> str:
    """Custom tool implementation."""
    # Your logic here
    return "Result"

# Tool is now available in agent_loop
```

### Custom Data Collator

Implement a custom data collator for different training objectives:

```python
from torch.nn.utils.rnn import pad_sequence

class CustomCollator:
    def __call__(self, features):
        # Custom collation logic
        return {"input_ids": ..., "labels": ...}
```

### Multi-GPU Training

The codebase uses `device_map="auto"` which automatically handles multi-GPU. For explicit control:

```python
# In model_setup.py, modify device_map
model_kwargs["device_map"] = {
    0: [0, 1, 2, ...],  # GPU 0 layers
    1: [3, 4, 5, ...],  # GPU 1 layers
}
```

### Hugging Face Model Upload

Use `src/upload/upload_to_hf.py` to upload trained models:

```python
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="./output/my-model/final_merged_model",
    path_in_repo=".",
    repo_id="your-username/your-model",
    repo_type="model",
    token="your-hf-token"
)
```

### Custom Persona Archetypes

Extend persona inference in `master_generate_icdu.py`:

```python
PERSONA_PATTERNS["My Custom Persona"] = r"\b(pattern|keywords)\b"
```

### Batch Inference

Process multiple queries efficiently:

```python
from src.inference_with_tools import load_model_pipeline, agent_loop

pipeline = load_model_pipeline("./model_path")
queries = ["Query 1", "Query 2", "Query 3"]

for query in queries:
    response = agent_loop(query, pipeline)
    print(f"Query: {query}\nResponse: {response}\n")
```

### Monitoring Training with Weights & Biases

Enable W&B logging in config:

```yaml
training:
  report_to: "wandb"
```

Set environment variable:

```bash
export WANDB_API_KEY="your-api-key"
```

### Export to ONNX or Other Formats

After training, convert merged model:

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("./output/my-model/final_merged_model")
dummy_input = torch.randint(0, 1000, (1, 10))

# Export to ONNX (example)
torch.onnx.export(model, dummy_input, "model.onnx")
```

---

## Additional Resources

### Key Dependencies

- **transformers**: Hugging Face Transformers library
- **peft**: Parameter-Efficient Fine-Tuning (LoRA)
- **trl**: Transformer Reinforcement Learning (DPO)
- **bitsandbytes**: 4-bit quantization
- **datasets**: Hugging Face Datasets
- **accelerate**: Multi-GPU/multi-node training

### Related Documentation

- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [DPO Paper](https://arxiv.org/abs/2305.18290)

### Contributing

When contributing:

1. Follow the existing code style (see `.cursor/skills/styling-guide/SKILL.md`)
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass: `pytest`

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

## Credits

- Built using Hugging Face Transformers, PEFT, TRL, and bitsandbytes
- Inspired by recent advances in low-rank adaptation and AI alignment research
- Designed for efficient training on consumer GPUs

---

**Last Updated**: 08/02/2026
**Version**: 0.1.0 (per pyproject.toml)