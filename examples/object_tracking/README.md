# Object Tracking Examples

This folder contains object-tracking workflows for hyperspectral CU3S recordings.

## SAM3 HSI Tracking (Current)

Use `examples/object_tracking/sam3_hsi_tracker.py` for end-to-end SAM3 tracking with a
single pipeline and core `Predictor`.

### What This Script Does

- Loads CU3S frames via `SingleCu3sDataModule` in `predict` mode
- Builds one pipeline:
  - `CU3SDataNode -> CIETristimulusFalseRGBSelector -> SAM3TrackerInference`
  - Tentative sink: `TrackingCocoJsonNode` (COCO JSON)
  - Overlay sink: `TrackingOverlayNode -> ToVideoNode` (MP4)
- Runs inference through `cuvis_ai_core.training.Predictor` (no manual node forwarding)
- Saves pipeline visualizations as PNG and Mermaid markdown

### Prerequisites

- CU3S input file (for example `Auto_002.cu3s`)
- `cuvis-ai-core` with `Predictor` and datamodule predict-stage support installed in the environment
- SAM3 plugin available and discoverable from `examples/object_tracking/plugins.yaml`

Current local plugin manifest:

```yaml
plugins:
  sam3:
    path: "../../../../cuvis-ai-sam3"
    provides:
      - cuvis_ai_sam3.node.SAM3TrackerInference
```

If your plugin checkout is at a different path, update `path` accordingly.

### Run

From repo root:

**Bash / Linux / macOS:**

```bash
uv run python examples/object_tracking/sam3_hsi_tracker.py \
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s"
```

**PowerShell (Windows):**

```powershell
uv run python examples/object_tracking/sam3_hsi_tracker.py `
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s"
```

Quick smoke test (10 frames):

```bash
uv run python examples/object_tracking/sam3_hsi_tracker.py \
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s" \
  --end-frame 10
```

```powershell
uv run python examples/object_tracking/sam3_hsi_tracker.py `
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s" `
  --end-frame 10
```

CPU-only:

```bash
uv run python examples/object_tracking/sam3_hsi_tracker.py \
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s" \
  --device cpu
```

```powershell
uv run python examples/object_tracking/sam3_hsi_tracker.py `
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s" `
  --device cpu
```

With explicit checkpoint/prompt/output:

```bash
uv run python examples/object_tracking/sam3_hsi_tracker.py \
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s" \
  --checkpoint-path "D:\models\sam3_checkpoint.pt" \
  --prompt "person" \
  --output-dir "outputs/sam3_tracking"
```

```powershell
uv run python examples/object_tracking/sam3_hsi_tracker.py `
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s" `
  --checkpoint-path "D:\models\sam3_checkpoint.pt" `
  --prompt "person" `
  --output-dir "outputs/sam3_tracking"
```

Show CLI help:

```bash
uv run python examples/object_tracking/sam3_hsi_tracker.py --help
```

### Main CLI Options

- `--cu3s-path` required input `.cu3s`
- `--processing-mode` `Raw|DarkSubtract|Preview|Reflectance|SpectralRadiance` (default `SpectralRadiance`)
- `--end-frame` limit frames (`-1` means all)
- `--prompt` text prompt for tracker initialization
- `--output-dir` output directory (default `./tracking_output`)
- `--device` `cuda|cpu`
- `--checkpoint-path` optional tracker checkpoint path
- `--mask-alpha` overlay alpha
- `--plugins-yaml` plugin manifest path (default `plugins.yaml` relative to this script)
- `--bf16` enable CUDA bf16 autocast
- `--compile` enable `torch.compile`

### Outputs

For `--output-dir <OUT>`:

- `<OUT>/tracking_results.json` - COCO instance segmentation output (tentative stream)
- `<OUT>/tracking_overlay.mp4` - overlay video (always produced)
- `<OUT>/pipeline/SAM3_HSI_Tracking.png` - graphviz pipeline image
- `<OUT>/pipeline/SAM3_HSI_Tracking.md` - mermaid pipeline markdown

## Channel Selector False RGB Workflow (Training)

`examples/object_tracking/channel_selector_false_rgb.py` remains available for
learnable false-RGB training and inspection.

Inspect:

```bash
uv run python examples/object_tracking/channel_selector_false_rgb.py mode=inspect
```

Train:

```bash
uv run python examples/object_tracking/channel_selector_false_rgb.py mode=train
```

