# Installation

Install Cuvis.AI and its dependencies.

## Requirements

| Component  | Recommended                                                                                                               |
| ---------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Python** | **3.11** (3.10 minimum, tested up to 3.13)                                                                                |
| **RAM**    | **32 GB** (16 GB minimum; hyperspectral cubes are memory-hungry)                                                          |
| **GPU**    | **NVIDIA + CUDA 12.8** (optional but strongly recommended)                                                                |
| **OS**     | **Windows or Linux** — macOS works for pure-Python use but has no Cuvis SDK build, so `.cu3s` / `.cu3` I/O is unavailable |

Why so much disk?

A single hyperspectral cube at **1000 × 1000 × 61** is **115 MB** in F16 and **230 MB** in F32. At 15 FPS, one minute of video is on the order of **100–200 GB**. Plan dataset and output storage accordingly.

## Install with uv (recommended)

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

### 2. Clone and install (all extras)

```bash
git clone https://github.com/cubert-hyperspectral/cuvis-ai.git
cd cuvis-ai

uv sync --all-extras
```

## Cuvis SDK (required for cu3s/cu3 I/O)

The `cuvis` Python package is just a binding — the **C++ Cuvis SDK** must be installed system-wide, or any `.cu3s` / `.cu3` read fails at runtime.

macOS not supported

Cuvis SDK ships for Windows and Linux only. On macOS, `.cu3s` / `.cu3` reads will fail at runtime; everything else (numpy / video / TIFF input) still works.

Obtain a suitable build of **Cuvis SDK 3.5.1.0** for your OS from the [Cuvis SDK download page](https://cubert-hyperspectral.github.io/cuvis.sdk/installation/), then verify:

```bash
uv run python -c "import cuvis; print(cuvis.__version__)"
```

## FFmpeg (required for video pipelines)

`uv sync` installs the Python video deps but not FFmpeg itself — both the reader ([`torchcodec`](https://github.com/pytorch/torchcodec) shared-lib link) and writer (`ToVideoNode` subprocess) need it at runtime.

```bash
sudo apt install ffmpeg
```

```bash
brew install ffmpeg
```

Use the **shared** build so `torchcodec` can find the DLLs, then put it on PATH:

```powershell
scoop install ffmpeg-shared
$env:Path = "$env:USERPROFILE\scoop\apps\ffmpeg-shared\current\bin;$env:Path"
```

Verify both paths:

```bash
ffmpeg -version                # writer-side binary
python -c "import torchcodec"  # reader-side shared libs
```

## Graphviz (required for pipeline graph rendering)

The Python `graphviz` wrapper shells out to the system `dot` binary, so `pipeline.visualize(format="png" | "svg" | "render_graphviz", ...)` needs it on PATH. Pure DOT/Mermaid output (`format="dot_string"` / `"mermaid"`) doesn't.

```bash
sudo apt install graphviz
```

```bash
brew install graphviz
```

```powershell
scoop install graphviz
```

Verify with `dot -V`.

## GPU support (optional)

Check CUDA availability:

```python
import torch
print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())
```

## Verify

Quick smoke test — imports the package and prints its version:

```bash
uv run python -c "import cuvis_ai; print(f'cuvis_ai {cuvis_ai.__version__} ready')"
```

### Run the test suite (optional)

If you want stronger confidence, run the tests with fast, and CPU-only filter:

```bash
uv run python -m pytest tests/ -v --tb=line -m "not slow and not gpu"
```

## Next steps

- **[Quickstart](https://cubert-hyperspectral.github.io/cuvis-ai/0.8.0/get-started/quickstart/index.md)**
- **[Configuration](https://cubert-hyperspectral.github.io/cuvis-ai/0.8.0/reference/configuration/index.md)**
- **[Use Cases](https://cubert-hyperspectral.github.io/cuvis-ai/0.8.0/tutorials/index.md)**
