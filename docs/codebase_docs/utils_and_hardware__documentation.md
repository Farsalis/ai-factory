# src/utils.py and src/hardware.py

## 1. Overview

See also [OVERVIEW](../OVERVIEW.md), [main](main__documentation.md), [model_optimizer](model_optimizer__documentation.md).

The `utils.py` and `hardware.py` modules provide environment detection, hardware profiling, and runtime configuration for the ai-factory pipeline. Together they form the foundation layer that determines available compute capabilities and configures PyTorch backends for optimal performance.

**Purpose:** Detect hardware/software environment and provide configuration recommendations for model training and optimization.

**Key Responsibilities:**

*   Detect CUDA availability and GPU device information
*   Check for bitsandbytes library presence (required for 4-bit quantization)
*   Determine optimal compute dtype (float16 vs bfloat16) based on platform
*   Configure PyTorch backends for Ampere+ GPU acceleration (TF32)
*   Profile hardware resources (VRAM, RAM, GPU count, compute capability)
*   Provide data structures consumed by Model Optimizer for config recommendations

**Connections:**

*   **Parent:** Used by `main.py`, `train.py`, `model_setup.py`, `model_optimizer.py`

*   **Dependencies:** `torch`, `psutil` (optional), `platform`

*   **External:** `bitsandbytes` (optional, detected at runtime)

## 2. Directory/Module Map

```
ai-factory/
├── src/
│   ├── utils.py              # <-- Environment detection (THIS MODULE)
│   ├── hardware.py           # <-- Hardware profiling (THIS MODULE)
│   ├── main.py               # CLI entry point, creates Environment/HardwareProfile
│   ├── train.py              # Uses Environment for bf16/optimizer decisions
│   ├── model_setup.py        # Uses Environment for model loading config
│   ├── model_optimizer.py    # Uses HardwareProfile for config recommendations
│   ├── config.py             # ScriptConfig, training args validation
│   ├── data/                 # ICDU package (SFT datasets)
│   └── dpo.py                # DPO training pipeline
├── tests/
│   ├── test_utils.py         # Unit tests for utils.py functions
│   └── test_model_optimizer.py  # Tests using HardwareProfile
└── docs/
    └── codebase_docs/
        └── utils_and_hardware__documentation.md  # This file
```

## 3. Public Interfaces

### src/utils.py

| Function/Class               | Signature           | Description                                                |
| ---------------------------- | ------------------- | ---------------------------------------------------------- |
| `is_bitsandbytes_available`  | `() -> bool`        | Checks if bitsandbytes library is installed via importlib  |
| `Environment`                | `class Environment` | Auto-detects CUDA, bnb, bf16, compute\_dtype, device\_name |
| `Environment.setup_backends` | `(self) -> None`    | Enables TF32 for Ampere+ GPUs (compute capability >= 8.0)  |


### src/hardware.py

| Function/Class            | Signature                            | Description                                                |
| ------------------------- | ------------------------------------ | ---------------------------------------------------------- |
| `HardwareProfile`         | `class HardwareProfile(Environment)` | Extends Environment with VRAM, RAM, OS, GPU details        |
| `_get_vram_bytes`         | `() -> int`                          | Gets total GPU VRAM via torch.cuda.get\_device\_properties |
| `_get_compute_capability` | `() -> tuple[int, int] \| None`      | Gets (major, minor) compute capability for device 0        |
| `_get_system_ram_bytes`   | `() -> int`                          | Gets total system RAM via psutil                           |


### Environment Attributes

| Attribute        | Type          | Description                                     |
| ---------------- | ------------- | ----------------------------------------------- |
| `cuda_available` | `bool`        | Whether CUDA is available                       |
| `bnb_available`  | `bool`        | Whether bitsandbytes library is installed       |
| `bf16_supported` | `bool`        | Whether bfloat16 is supported (requires CUDA)   |
| `compute_dtype`  | `torch.dtype` | Recommended compute dtype (float16 or bfloat16) |
| `device_name`    | `str`         | Name of CUDA device or "CPU"                    |


### HardwareProfile Attributes (extends Environment)

| Attribute            | Type                      | Description                                  |
| -------------------- | ------------------------- | -------------------------------------------- |
| `vram_bytes`         | `int`                     | Total GPU VRAM in bytes (0 if no CUDA)       |
| `system_ram_bytes`   | `int`                     | Total system RAM in bytes                    |
| `gpu_count`          | `int`                     | Number of CUDA devices                       |
| `compute_capability` | `tuple[int, int] \| None` | (major, minor) for device 0, or None         |
| `os_name`            | `str`                     | platform.system() (e.g., "Windows", "Linux") |


## 4. Execution and Control Flow

### Environment Detection Flow

```
Environment.__init__()
    │
    ├─ torch.cuda.is_available() → cuda_available
    │
    ├─ is_bitsandbytes_available() → bnb_available
    │   └─ importlib.util.find_spec("bitsandbytes")
    │
    ├─ cuda_available and torch.cuda.is_bf16_supported() → bf16_supported
    │
    ├─ Determine compute_dtype:
    │   ├─ If Windows: torch.float16
    │   └─ Else: torch.bfloat16 if bf16_supported else torch.float16
    │
    └─ _get_device_name() → device_name
        ├─ If not cuda_available: return "CPU"
        ├─ If torch.cuda.device_count() > 0: return torch.cuda.get_device_name(0)
        └─ On error: log warning, return "CPU"
```

### HardwareProfile Initialization Flow

```
HardwareProfile.__init__()
    │
    ├─ super().__init__()  # Environment detection
    │
    ├─ _get_vram_bytes() → vram_bytes
    │   ├─ If !cuda_available: return 0
    │   ├─ If device_count > 0: return get_device_properties(0).total_memory
    │   └─ On error: log warning, return 0
    │
    ├─ _get_system_ram_bytes() → system_ram_bytes
    │   ├─ Import psutil
    │   ├─ psutil.virtual_memory().total
    │   └─ On error: log warning, return 0
    │
    ├─ torch.cuda.device_count() if cuda_available else 0 → gpu_count
    │
    ├─ _get_compute_capability() → compute_capability
    │   ├─ If !cuda_available: return None
    │   ├─ If device_count > 0: return get_device_capability(0)
    │   └─ On error: return None
    │
    └─ platform.system() → os_name
```

### Backend Setup Flow

```
Environment.setup_backends()
    │
    ├─ If not cuda_available:
    │   └─ Log "Using CPU", return
    │
    ├─ Log device name
    │
    ├─ torch.cuda.get_device_capability(0) → (major, minor)
    │
    ├─ If (major, minor) >= (8, 0):  # Ampere or newer
    │   ├─ torch.backends.cuda.matmul.allow_tf32 = True
    │   └─ torch.set_float32_matmul_precision("high")
    │
    └─ On error: log warning, continue with defaults
```

## 5. Data Flow

### Environment Data Path

```
Runtime State
    │
    ▼
Environment()
    │
    ├─► cuda_available: bool
    │   └─ Used by: train.py (bf16/fp16 selection), model_setup.py (device placement)
    │
    ├─► bnb_available: bool
    │   └─ Used by: train.py (optimizer fallback), model_setup.py (quantization config)
    │
    ├─► bf16_supported: bool
    │   └─ Used by: train.py (TrainingArguments bf16 flag)
    │
    ├─► compute_dtype: torch.dtype
    │   └─ Used by: model_setup.py (model loading precision)
    │
    └─► device_name: str
        └─ Used by: logging, diagnostics
```

### HardwareProfile Data Path

```
Environment (parent)
    │
    ▼
HardwareProfile()
    │
    ├─► vram_bytes: int
    │   └─ Used by: model_optimizer.py (memory-based recommendations)
    │
    ├─► system_ram_bytes: int
    │   └─ Used by: model_optimizer.py (offloading decisions)
    │
    ├─► gpu_count: int
    │   └─ Used by: model_optimizer.py (multi-GPU config)
    │
    ├─► compute_capability: tuple[int, int] | None
    │   └─ Used by: model_optimizer.py (feature support detection)
    │
    └─► os_name: str
        └─ Used by: model_optimizer.py (platform-specific recommendations)
```

## 6. Integration Points

### External Dependencies

| Dependency       | Usage                                          | Notes                       |
| ---------------- | ---------------------------------------------- | --------------------------- |
| `torch`          | CUDA detection, device properties, dtype       | Core dependency             |
| `torch.cuda`     | Device availability, names, capabilities, VRAM | Requires CUDA toolkit       |
| `psutil`         | System RAM detection                           | Optional, graceful fallback |
| `importlib.util` | bitsandbytes detection                         | Standard library            |
| `platform`       | OS detection, Windows-specific logic           | Standard library            |


### Internal Module Dependencies

| Module                | Functions/Classes Used          | Purpose                                   |
| --------------------- | ------------------------------- | ----------------------------------------- |
| `src.main`            | `Environment`,`HardwareProfile` | Creates instances at startup              |
| `src.train`           | `Environment`                   | bf16/fp16 selection, optimizer fallback   |
| `src.model_setup`     | `Environment`                   | Model loading precision, device placement |
| `src.model_optimizer` | `HardwareProfile`               | Config recommendations based on hardware  |
| `src.utils`           | `is_bitsandbytes_available`     | Utility function for bnb detection        |


### Caller Integration

```
# In src/main.py
from src.utils import Environment
from src.hardware import HardwareProfile

# Basic environment detection
env = Environment()
env.setup_backends()

# Extended hardware profiling for model optimizer
profile = HardwareProfile()
```

<!---->

```
# In src/train.py
from src.utils import Environment

env = Environment()
# Use env.bf16_supported for TrainingArguments.bf16
# Use env.bnb_available for optimizer fallback logic
```

<!---->

```
# In src/model_optimizer.py
from src.hardware import HardwareProfile

profile = HardwareProfile()
# Use profile.vram_bytes, profile.system_ram_bytes for recommendations
```

## 7. Configuration and Conventions

### Compute Dtype Selection Logic

```
Platform Detection
    │
    ├─ Windows:
    │   └─ Always use float16 (bitsandbytes compatibility)
    │
    └─ Linux/Other:
        ├─ If bf16_supported: use bfloat16
        └─ Else: use float16
```

### TF32 Backend Configuration

*   **Condition:** GPU compute capability >= 8.0 (Ampere architecture or newer)

*   **Settings:**

    *   `torch.backends.cuda.matmul.allow_tf32 = True`
    *   `torch.set_float32_matmul_precision("high")`

*   **Benefit:** Improved performance with minimal accuracy loss for float32 operations

### Error Handling Conventions

*   **CUDA device access errors:** Logged as warnings, fall back to "CPU"

*   **VRAM detection errors:** Logged as warnings, return 0

*   **RAM detection errors:** Logged as warnings, return 0

*   **Compute capability errors:** Return None, no fallback needed

*   **Backend configuration errors:** Logged as warnings, continue with defaults

### Constants

| Constant             | Value | Location    | Description                        |
| -------------------- | ----- | ----------- | ---------------------------------- |
| `VRAM_DEFAULT_BYTES` | `0`   | hardware.py | Default VRAM when CUDA unavailable |


## 8. Extension and Testing Guidance

### Adding New Hardware Detection

1.  **New GPU metrics:** Add private helper functions in `hardware.py` following `_get_vram_bytes` pattern

2.  **New environment attributes:** Add to `Environment.__init__()` with safe detection

3.  **Platform-specific logic:** Extend compute\_dtype selection in `Environment.__init__()`

### Testing Patterns

The modules include unit tests in `tests/test_utils.py`:

*   **bitsandbytes detection tests:** Verify importlib.spec behavior

*   **Environment initialization tests:** Mock torch.cuda for various scenarios

*   **Device name fallback tests:** Test error handling for CUDA device access

*   **Compute dtype tests:** Verify Windows vs Linux dtype selection

*   **Backend setup tests:** Test TF32 enablement for different compute capabilities

### Error Handling

All functions include comprehensive error handling:

*   CUDA availability checks before device access
*   Try-except blocks around all torch.cuda operations
*   Graceful fallbacks with logging for all detection failures
*   No exceptions propagated to callers

## 9. Visualizations

### Environment and HardwareProfile Data Model

```
classDiagram
class Environment {
  +cuda_available
  +bnb_available
  +bf16_supported
  +compute_dtype
  +device_name
  +setup_backends()
  +_get_device_name()
}

class HardwareProfile {
  +vram_bytes
  +system_ram_bytes
  +gpu_count
  +compute_capability
  +os_name
}

Environment <|-- HardwareProfile
```

### Environment Detection and Backend Setup

```
flowchart TD
    START["Environment()"] --> CUDA["torch.cuda.is_available()"]
    START --> BNB["find_spec('bitsandbytes')"]
    CUDA --> BF16["cuda_available and torch.cuda.is_bf16_supported()"]
    BF16 --> PLATFORM{"platform.system() == Windows?"}
    PLATFORM -- yes --> FP16["compute_dtype = torch.float16"]
    PLATFORM -- no and bf16_supported --> BF16DT["compute_dtype = torch.bfloat16"]
    PLATFORM -- no and not bf16_supported --> FP16
    CUDA --> NAME{"CUDA available?"}
    NAME -- no --> CPU["device_name = CPU"]
    NAME -- yes --> DEVTRY["device_count() + get_device_name(0)"]
    DEVTRY --> GPU["device_name = GPU name"]
    DEVTRY -. RuntimeError or AttributeError .-> CPU

    ENV["Existing Environment"] --> SETUP["setup_backends()"]
    SETUP --> CUDA2{"cuda_available?"}
    CUDA2 -- no --> NOOP["Log CPU path and return"]
    CUDA2 -- yes --> CAP["torch.cuda.get_device_capability(0)"]
    CAP --> AMPERE{"capability at least 8.0?"}
    AMPERE -- yes --> TF32["Enable TF32<br/>set matmul precision high"]
    AMPERE -- no --> DEFAULT["Leave backend defaults"]
    CAP -. RuntimeError or AttributeError .-> WARN["Log warning and continue"]
```

### Consumer Lineage Across the Codebase

```
flowchart LR
    subgraph HELPERS["hardware.py probe helpers"]
        VRAM["_get_vram_bytes<br/>device 0 only"]
        RAM["_get_system_ram_bytes<br/>psutil optional"]
        CAP["_get_compute_capability<br/>device 0 only"]
    end

    ENV["Environment fields<br/>cuda, bnb, bf16, dtype, device"] --> HW["HardwareProfile"]
    VRAM --> HW
    RAM --> HW
    CAP --> HW

    ENV --> MAIN["src.main"]
    ENV --> TRAIN["src.train"]
    ENV --> SETUP["src.model_setup"]
    HW --> OPT["src.model_optimizer"]
    HW --> OPTCLI["main optimize-config summary"]
```

## 10. Mathematical Framing

### Compute Capability Versioning

GPU compute capability is represented as a tuple (major, minor):

*   **8.0+:** Ampere architecture (RTX 30xx, A100) - TF32 supported

*   **7.0-7.5:** Volta/Turing (V100, RTX 20xx) - No TF32

*   **< 7.0:** Older architectures - Limited mixed precision

### VRAM to GB Conversion

```
VRAM_GB = vram_bytes / (1024^3)
```

### Dtype Memory Requirements

| dtype    | Bytes per parameter | 7B model VRAM | 13B model VRAM |
| -------- | ------------------- | ------------- | -------------- |
| float32  | 4                   | ~28 GB        | ~52 GB         |
| float16  | 2                   | ~14 GB        | ~26 GB         |
| bfloat16 | 2                   | ~14 GB        | ~26 GB         |
| int8     | 1                   | ~7 GB         | ~13 GB         |
| int4     | 0.5                 | ~3.5 GB       | ~6.5 GB        |


### Platform-Specific Dtype Selection

```
compute_dtype = {
    "Windows": float16,
    "Linux": bf16 if bf16_supported else float16,
    "Other": bf16 if bf16_supported else float16
}[platform.system()]
```

### TF32 Performance Impact

For Ampere+ GPUs (compute capability >= 8.0):

*   **Matrix multiplication:** Up to 2x speedup vs float32

*   **Accuracy:** Minimal loss (< 0.1% on most benchmarks)

*   **Mechanism:** Uses 19-bit floating point internally, inputs/outputs remain float32

***

*Last updated: 2026-08-02.*
