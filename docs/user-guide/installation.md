# Installation

Install Cuvis.AI and its dependencies.

## Requirements

| Component | Minimum | Recommended | Notes |
| --- | --- | --- | --- |
| **Python** | 3.10 | **3.11** | Tested up to 3.13 |
| **RAM** | 16 GB | **32 GB** | Hyperspectral cubes are memory-hungry |
| **OS** | Windows, Linux, or macOS | — | All three are first-class |
| **GPU** | — | **NVIDIA + CUDA 12.8** | Strongly recommended even for inference |
| **Storage** | ~10 GB for dependencies | ~50GB for dataset/output budget | See sizing note below |

!!! note "Why so much disk?"
    A single hyperspectral cube at **1000 × 1000 × 61** is **115 MB** in F16 and **230 MB** in F32. At 15 FPS, one minute of video is on the order of **100–200 GB**. Plan dataset and output storage accordingly.

## Install with uv (recommended)

### 1. Install uv

=== "Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "macOS"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    irm https://astral.sh/uv/install.ps1 | iex
    ```

### 2. Clone and install (all extras)

```bash
git clone https://github.com/cubert-hyperspectral/cuvis-ai.git
cd cuvis-ai

uv sync --all-extras
```

## FFmpeg (required for video pipelines)

The reader-side decoder ([torchcodec](https://github.com/pytorch/torchcodec))
is installed automatically by `uv sync` as a regular Python dependency. What
you still need to provide yourself are the FFmpeg **system libraries** that
both reader and writer paths depend on at runtime:

- **Reader-side** (`VideoIterator`, `VideoFrameDataModule`) — `torchcodec`
  links against FFmpeg's **shared libraries** at runtime. Without them,
  `import torchcodec` fails to load its native extension.

- **Writer-side** (`ToVideoNode`) — spawns an `ffmpeg` subprocess directly to
  encode H.264/H.265 at a configurable bitrate, so the `ffmpeg` **binary** must
  be resolvable on `PATH`. Without it, `ToVideoNode.forward(...)` raises
  `RuntimeError: ffmpeg binary not found on PATH`.

A single "full" FFmpeg install satisfies both; no separate packages are needed.

=== "Linux"

    ```bash
    sudo apt install ffmpeg
    ```

=== "macOS"

    ```bash
    brew install ffmpeg
    ```

=== "Windows"

    Use the **shared** build so `torchcodec` can find the FFmpeg DLLs:

    ```powershell
    scoop install ffmpeg-shared
    ```

    Then expose the `bin/` directory on PATH so the OS DLL loader can resolve the shared libs at runtime:

    ```powershell
    $env:Path = "$env:USERPROFILE\scoop\apps\ffmpeg-shared\current\bin;$env:Path"
    ```

Verify both paths with:

```bash
ffmpeg -version     # binary available (writer-side)
python -c "import torchcodec"   # shared libs available (reader-side)
```

## Graphviz (required for pipeline graph rendering)

The Python `graphviz` package is pulled in automatically by `uv sync`, but it
is only a thin wrapper that shells out to the **`dot` system binary** to
rasterise graphs. Any call like
`pipeline.visualize(format="render_graphviz", output_path="...")` (or
`format="png"` / `format="svg"`) requires `dot` on `PATH`; without it,
`graphviz.backend.execute.ExecutableNotFound` is raised.

If you only consume `format="graphviz"` / `format="dot_string"` (raw DOT text)
or `format="mermaid"`, the system install is not needed.

=== "Linux"

    ```bash
    sudo apt install graphviz
    ```

=== "macOS"

    ```bash
    brew install graphviz
    ```

=== "Windows"

    ```powershell
    scoop install graphviz
    ```

Verify:

```bash
dot -V
```

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

* **[Quickstart](quickstart.md)**
* **[Configuration](configuration.md)**
* **[Use Cases](../use_cases/index.md)**
