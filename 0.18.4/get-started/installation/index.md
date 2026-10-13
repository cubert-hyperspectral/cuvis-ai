# Installation

Install Cuvis.AI and its dependencies.

## Requirements

| Component  | Recommended                                                                                                               |
| ---------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Python** | **3.11, 3.12 or 3.13**                                                                                                    |
| **RAM**    | **32 GB** (16 GB minimum; hyperspectral cubes are memory-hungry)                                                          |
| **GPU**    | **NVIDIA + CUDA 12.8** on x86_64, **JetPack 7 / CUDA 13** on Jetson Thor (aarch64); optional but strongly recommended     |
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

### Jetson / aarch64 (JetPack 7)

On aarch64 Linux, `uv sync` installs `torch` and `torchvision` from the cu130 wheel index; the lock carries no cu128 build for this platform. The cu128 index only serves SBSA wheels whose kernels stop at sm_120; on a Jetson Thor (sm_110) they install cleanly and fail at the first CUDA kernel. The cu130 wheels need a CUDA 13 driver (JetPack 7, driver 580 or newer). If no CPython 3.11 to 3.13 is installed, uv downloads a managed one. Verify that the build fits the GPU:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
```

The list must contain `sm_110`. JetPack 6 (Orin) is not covered: its CUDA 12 driver cannot load the cu130 wheels this lock resolves, and its system Python is 3.10. An SBSA host (Grace, GH200) still on a CUDA 12 driver is outside the lock as well: `uv sync` installs the cu130 build there whichever dependency groups are selected. As a workaround, replace torch and torchvision by hand after the sync and skip the re-sync when running, because every `uv sync` and plain `uv run` restores the locked cu130 build:

```bash
uv sync
uv pip install --reinstall "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128
uv run --no-sync python -c "import torch; print(torch.__version__)"
```

## Cuvis SDK (only for cu3s/cu3 I/O)

Reading `.cu3s` / `.cu3` files needs the system-wide **C++ Cuvis SDK** plus the `cuvis` Python binding. Neither ships with cuvis-ai, and pipelines that only use numpy, TIFF, or video input don't need it. The `cuvis` binding is installed by the [`cuvis-ai-dataloader`](https://github.com/cubert-hyperspectral/cuvis-ai-dataloader) plugin's `[cu3s]` extra (which owns the `cuvis` pin); the C++ SDK is a separate system install.

macOS not supported

The Cuvis SDK ships for **Windows and Linux only**. On macOS, `.cu3s` / `.cu3` reads fail at runtime; TIFF, numpy, and video input still work.

Install the binding (`uv pip install "cuvis-ai-dataloader[cu3s,coco]"`), then install the C++ SDK **3.6.0** for your OS (the `cuvis>=3.6.0.0` binding the extra installs fails at import against a 3.5.x runtime with `DLL load failed while importing _cuvis_pyil`) from the [Cuvis SDK installation guide](https://sdk.cuvis.ai/latest/installation/), and verify the binding finds it:

```bash
uv run python -c "import cuvis; print(cuvis.version())"
```

## FFmpeg (required for video output)

`uv sync` installs the Python video deps but not FFmpeg itself. The video writer (`ToVideoNode`) runs the `ffmpeg` binary as a subprocess and needs it on PATH. Video input reads with OpenCV out of the box and needs nothing extra.

```bash
sudo apt install ffmpeg
```

```bash
brew install ffmpeg
```

```powershell
scoop install ffmpeg
$env:Path = "$env:USERPROFILE\scoop\apps\ffmpeg\current\bin;$env:Path"
```

Verify:

```bash
ffmpeg -version
```

### Optional: GPU video decoding with torchcodec

cuvis-ai does not install [`torchcodec`](https://github.com/pytorch/torchcodec). Its shared library is built per torch release, and a torchcodec next to a torch it was not built for fails at import instead of falling back, which is what a plain `pip install` produced (PyPI's newest torch beside torchcodec 0.11). The video reader in `cuvis-ai-core` uses torchcodec when it imports and OpenCV otherwise; the OpenCV path reopens the file per frame, so long MP4 inputs read markedly slower without it.

To decode on the GPU, install the torchcodec that matches the installed torch (torch 2.11 pairs with torchcodec 0.11.x, torch 2.14 with 0.16.x; see the torchcodec README for the table) together with FFmpeg's **shared** libraries:

```bash
sudo apt install ffmpeg
uv pip install "torchcodec==0.11.1"   # the release built for the locked torch 2.11
```

Use the shared build so torchcodec can find the DLLs (cuvis-ai registers every PATH directory holding `avcodec-*.dll` at import), then put it on PATH:

```powershell
scoop install ffmpeg-shared
$env:Path = "$env:USERPROFILE\scoop\apps\ffmpeg-shared\current\bin;$env:Path"
uv pip install "torchcodec==0.11.1"
```

Verify: `python -c "import torchcodec"`. A `uv sync` removes the package again; run with `uv run --no-sync` or add it to your own project.

## Graphviz (required for pipeline graph rendering)

The Python `graphviz` wrapper shells out to the system `dot` binary, so `pipeline.visualize(format="render_graphviz", output_path=...)` (alias `format="render"`) needs it on PATH. Pure DOT/Mermaid string output (`format="dot_string"` / `"mermaid"`) doesn't.

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

- **[Quickstart](https://docs.cuvis.ai/latest/get-started/quickstart/index.md)**
- **[Model Weights](https://docs.cuvis.ai/latest/workflows/model-weights/index.md)** (pretrained plugin weights, no Hugging Face account needed)
- **[Configuration](https://docs.cuvis.ai/latest/reference/configuration/index.md)**
- **[Use Cases](https://docs.cuvis.ai/latest/tutorials/index.md)**
