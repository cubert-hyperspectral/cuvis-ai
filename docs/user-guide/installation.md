# Installation

This guide covers the installation of CUVIS.AI and its dependencies.

## System Requirements

### Minimum Requirements
- **Python**: 3.10 or higher (tested up to 3.13)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB for dependencies, additional space for datasets
- **OS**: Windows, Linux, or macOS

### Recommended Requirements
- **Python**: 3.12
- **RAM**: 32GB for large datasets
- **GPU**: NVIDIA GPU with CUDA 12.8 support (optional but recommended for training)
- **Storage**: 10GB+ for datasets and outputs

## Installation Methods

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer that manages dependencies efficiently.

1. **Install uv** (if not already installed):
   ```bash
   # On Linux/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/cubert-hyperspectral/cuvis.ai.git
   cd cuvis.ai
   ```

3. **Install the package**:
   ```bash
   # Install with all dependencies
   uv sync
   
   # Install with development dependencies
   uv sync --extra dev
   
   # Install with documentation dependencies
   uv sync --extra docs
   ```


## GPU Support

CUVIS.AI uses PyTorch with CUDA 12.8 by default for GPU acceleration.

### Verifying GPU Support

After installation, verify CUDA is available:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## Optional Dependencies

### TensorBoard

TensorBoard is included by default. Launch it with:

```bash
tensorboard --logdir=./outputs/tensorboard
```

## Verification

Verify your installation by running the test suite:

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test module
uv run pytest tests/test_imports.py -v
```

Or try a simple import:

```python
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.anomaly.rx_detector import RXGlobal
print("Installation successful!")
```

## Common Issues


### Memory Errors

**Problem**: Out-of-memory errors during training.

**Solution**: 
- Reduce batch size in configuration
- Use mixed precision training (`precision='16-mixed'`)
- Enable gradient checkpointing
- Use CPU training for small experiments

## Next Steps

Now that CUVIS.AI is installed, continue to:

- **[Quickstart](quickstart.md)**: Get started with a simple example
- **[Configuration](configuration.md)**: Learn about Hydra configuration
- **[Tutorials](../tutorials/phase1_statistical.md)**: Detailed walkthroughs

## Updating

To update to the latest version:

```bash
cd cuvis.ai
git pull
uv sync
```

