# Installation

Install CUVIS.AI and its dependencies.

## Requirements

- **Python**: 3.10+ (tested up to 3.13; **3.11 recommended**)
- **RAM**: 8GB minimum (16GB recommended; **32GB** for large datasets)
- **OS**: Windows / Linux / macOS
- **GPU (optional)**: NVIDIA + **CUDA 12.8** for faster training
- **Storage**: ~2GB for deps (+ space for datasets/outputs)

## Install with uv (recommended)

1. Install **uv**:

   ```bash
   # Linux/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Clone and install (all extras):

   ```bash
   git clone https://github.com/cubert-hyperspectral/cuvis-ai.git
   cd cuvis-ai

   uv sync --all-extras
   ```

## GPU support (optional)

Check CUDA availability:

```python
import torch
print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())
```

## Verify

Run tests:

```bash
uv run pytest tests/ -v
```

Skip GPU tests (CPU-only):

```bash
uv run pytest tests/ -v -m "no gpu"
```

Or quick import:

```python
from cuvis_ai_core.pipeline.graph import Graph
from cuvis_ai_core.anomaly.rx_detector import RXGlobal
print("Installation successful!")
```

## Next steps

* **[Quickstart](quickstart.md)**
* **[Configuration](configuration.md)**
* **[Tutorials](../tutorials/index.md)**
