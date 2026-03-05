# Object Tracking Examples

This folder contains object-tracking workflows for hyperspectral CU3S recordings.

## SAM3 HSI Tracking (Current)

Use `examples/object_tracking/sam3/sam3_hsi_tracker.py` for end-to-end SAM3 tracking with a
single pipeline and core `Predictor`.

### What This Script Does

- Loads CU3S frames via `SingleCu3sDataModule` in `predict` mode
- Builds one pipeline:
  - `CU3SDataNode -> RangeAverageFalseRGBSelector -> SAM3TrackerInference`
  - Confirmed-output sink: `TrackingCocoJsonNode` (COCO JSON with RLE masks, bboxes, scores)
  - Overlay sink: `TrackingOverlayNode -> ToVideoNode` (MP4)
- Runs inference through `cuvis_ai_core.training.Predictor` (no manual node forwarding)
- Enables automatic per-node runtime profiling (CUDA-synchronised, skip first 3 warm-up frames)
- Saves pipeline visualizations as PNG and Mermaid markdown

### Prerequisites

- CU3S input file (for example `Auto_013+01.cu3s`)
- `cuvis-ai-core` with `Predictor` and datamodule predict-stage support installed in the environment
- SAM3 plugin available and discoverable from `configs/plugins/sam3.yaml`

Current local plugin manifest (`configs/plugins/sam3.yaml`):

```yaml
plugins:
  sam3:
    path: "../../../../cuvis-ai-sam3/sam3-init"
    provides:
      - cuvis_ai_sam3.node.SAM3TrackerInference
      - cuvis_ai_sam3.node.SpectralSignatureExtractor
```

If your plugin checkout is at a different path, update `path` accordingly.

### Run

From repo root:

**Bash / Linux / macOS:**

```bash
uv run python examples/object_tracking/sam3/sam3_hsi_tracker.py \
  --cu3s-path "D:/data/your_dataset/Auto_013+01.cu3s" \
  --plugins-yaml "configs/plugins/sam3.yaml"
```

**PowerShell (Windows):**

```powershell
uv run python examples/object_tracking/sam3/sam3_hsi_tracker.py `
  --cu3s-path "D:\data\your_dataset\Auto_013+01.cu3s" `
  --plugins-yaml "configs/plugins/sam3.yaml"
```

Quick smoke test (10 frames):

```bash
uv run python examples/object_tracking/sam3/sam3_hsi_tracker.py \
  --cu3s-path "D:/data/your_dataset/Auto_013+01.cu3s" \
  --plugins-yaml "configs/plugins/sam3.yaml" \
  --end-frame 10
```

With bf16, explicit output, and tuned thresholds (350 frames):

```bash
uv run python examples/object_tracking/sam3/sam3_hsi_tracker.py \
  --cu3s-path "D:/data/your_dataset/Auto_013+01.cu3s" \
  --plugins-yaml "configs/plugins/sam3.yaml" \
  --output-dir "D:/experiments/sam3/v1_sam3_hsi_tracker" \
  --end-frame 350 \
  --bf16
```

Show CLI help:

```bash
uv run python examples/object_tracking/sam3/sam3_hsi_tracker.py --help
```

### Main CLI Options

- `--cu3s-path` required input `.cu3s`
- `--processing-mode` `Raw|DarkSubtract|Preview|Reflectance|SpectralRadiance` (default `SpectralRadiance`)
- `--end-frame` limit frames (`-1` means all)
- `--prompt` text prompt for tracker initialization (default `person`)
- `--output-dir` output directory (default `./tracking_output`)
- `--checkpoint-path` optional tracker checkpoint path
- `--plugins-yaml` plugin manifest path (default `plugins.yaml` relative to script)
- `--bf16` enable CUDA bf16 autocast
- `--compile` enable `torch.compile`
- `--score-threshold-detection` SAM3 detector score threshold (default `0.5`)
- `--new-det-thresh` new-track threshold (default `0.7`)
- `--det-nms-thresh` detector NMS IoU threshold (default `0.1`)
- `--overlap-suppress-thresh` overlap suppression threshold (default `0.7`)
- `--max-tracker-states` max active tracker states (default `5`)
- `--confirmed-output / --tentative-output` use confirmed vs tentative tracker outputs (default confirmed)
- `--progress-log-interval` emit progress log every N frames (default `50`)

### Outputs

For `--output-dir <OUT>`:

- `<OUT>/tracking_results.json` - COCO instance segmentation output (confirmed stream)
- `<OUT>/tracking_overlay.mp4` - overlay video (always produced)
- `<OUT>/pipeline/SAM3_HSI_Tracking.png` - graphviz pipeline image
- `<OUT>/pipeline/SAM3_HSI_Tracking.md` - mermaid pipeline markdown
- `<OUT>/profiling_summary.txt` - per-node runtime profiling breakdown

### Profiling

The script automatically enables per-node pipeline profiling (with CUDA synchronisation and
3 warm-up frames skipped). After inference completes, a table is printed and saved to
`profiling_summary.txt`.

Example run: 350 frames from `Auto_013+01.cu3s` (SpectralRadiance, bf16, CUDA):

| Node | Count | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Total (s) | % |
|---|---|---|---|---|---|---|---|
| sam3_tracker | 347 | 9,552 | 6,619 | 481 | 165,439 | 3,314.6 | 99.2 |
| range_average_false_rgb | 347 | 44 | 27 | 3 | 2,557 | 15.1 | 0.5 |
| overlay | 347 | 11 | 10 | 5 | 77 | 3.9 | 0.1 |
| tracking_coco_json | 347 | 11 | 11 | 6 | 45 | 3.7 | 0.1 |
| to_video | 347 | 8 | 8 | 6 | 11 | 2.6 | 0.1 |
| cu3s_data | 347 | 0.1 | 0.1 | 0.1 | 0.2 | 0.04 | 0.0 |
| **TOTAL** | | | | | | **3,340** | |

Average per-frame: **9,626 ms** (0.1 FPS). SAM3 tracker dominates at 99.2% of pipeline time.
The large std/max reflects tracker state growth — early frames ~480 ms (1 state),
later frames up to 165 s (5 states, 11 tracked objects).

## YOLO + ByteTrack HSI Tracking

Use `examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py` for YOLO detection +
ByteTrack multi-object tracking on CU3S hyperspectral recordings.

### What This Script Does

- Loads CU3S frames via `SingleCu3sDataModule` in `predict` mode
- Builds one pipeline:
  - `CU3SDataNode -> CIETristimulusFalseRGBSelector -> YOLO26Detection -> YOLOPostprocess -> ByteTrack -> BBoxesOverlayNode -> ToVideoNode`
- Runs inference through `cuvis_ai_core.training.Predictor`
- Outputs an overlay video with tracked bounding boxes

### Prerequisites

- CU3S input file (e.g. `Auto_013+01.cu3s`)
- `cuvis-ai-core` with `Predictor` and datamodule predict-stage support installed
- Ultralytics plugin: `configs/plugins/ultralytics.yaml`
- ByteTrack plugin: `configs/plugins/bytetrack.yaml`
- A YOLO model (default `yolo26n.pt`, auto-downloaded on first run)

### Run

From repo root:

**Bash / Linux / macOS:**

```bash
uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py \
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s"
```

**PowerShell (Windows):**

```powershell
uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py `
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s"
```

Quick smoke test (10 frames):

```bash
uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py \
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s" \
  --end-frame 10
```

```powershell
uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py `
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s" `
  --end-frame 10
```

With explicit output and tuning:

```bash
uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py \
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s" \
  --output-dir "outputs/bytetrack_tracking" \
  --confidence-threshold 0.3 \
  --track-thresh 0.4
```

```powershell
uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py `
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s" `
  --output-dir "outputs/bytetrack_tracking" `
  --confidence-threshold 0.3 `
  --track-thresh 0.4
```

Show CLI help:

```bash
uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py --help
```

### Main CLI Options

- `--cu3s-path` required input `.cu3s`
- `--processing-mode` `Raw|DarkSubtract|Preview|Reflectance|SpectralRadiance` (default `SpectralRadiance`)
- `--end-frame` limit frames (`-1` means all)
- `--model-name` YOLO model name/path (default `yolo26n.pt`)
- `--confidence-threshold` YOLO detection threshold (default `0.5`)
- `--track-thresh` ByteTrack high-confidence threshold (default `0.5`)
- `--track-buffer` lost-track buffer in frames (default `30`)
- `--match-thresh` IoU match threshold (default `0.8`)
- `--output-dir` output directory (default `./tracking_output`)
- `--plugins-dir` plugin YAML directory (default `<repo>/configs/plugins/`)
- `--frame-rotation` optional frame rotation (degrees)
- `--bf16` enable CUDA bf16 autocast

### Outputs

For `--output-dir <OUT>`:

- `<OUT>/tracking_overlay.mp4` — CIE false RGB frames with tracked bounding box overlays

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

