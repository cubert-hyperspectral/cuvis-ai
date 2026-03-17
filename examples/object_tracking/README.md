# Object Tracking Examples

This folder contains object-tracking workflows for hyperspectral CU3S recordings.

## Export CU3S to False-RGB Video

Use `examples/object_tracking/export_cu3s_false_rgb_video.py` to convert a CU3S hyperspectral
recording to a false-RGB MP4 without running any tracker.

### Pipeline

```
CU3SDataNode → <FalseRGB node> → ToVideoNode
```

The false-RGB node is selected by `--method`:

| Method | Node | Description |
|---|---|---|
| `range_average` | `RangeAverageFalseRGBSelector` | Per-channel wavelength-range averaging (default) |
| `cie_tristimulus` | `CIETristimulusFalseRGBSelector` | CIE 1931 XYZ → sRGB conversion |
| `camera_emulation` | `CameraEmulationFalseRGBSelector` | Gaussian camera sensitivity curves |
| `baseline` | `FixedWavelengthSelector` | Fixed band selection (650 / 550 / 450 nm) |

### Run

**CIE tristimulus export:**

```powershell
uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
  --cu3s-file-path "D:\data\XMR_notarget_Busstation\20260226\Auto_001+01.cu3s" `
  --output-video-path "D:\data\XMR_notarget_Busstation\20260226\Auto_001+01.mp4" `
  --method cie_tristimulus
```

**Default normalization behavior (used if not specified):**

- `--normalization-mode sampled_fixed`
- `--sample-fraction 0.05`

Equivalent explicit command:

```powershell
uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
  --cu3s-file-path "D:\data\XMR_notarget_Busstation\20260226\Auto_001+01.cu3s" `
  --output-video-path "D:\data\XMR_notarget_Busstation\20260226\Auto_001+01.mp4" `
  --method cie_tristimulus `
  --normalization-mode sampled_fixed `
  --sample-fraction 0.05
```

**With frame ID overlay (renders measurement index in top-left corner):**

```powershell
uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
  --cu3s-file-path "D:\data\XMR_notarget_Busstation\20260226\Auto_001+01.cu3s" `
  --output-video-path "D:\data\XMR_notarget_Busstation\20260226\Auto_001+01.mp4" `
  --method cie_tristimulus `
  --overlay-frame-id
```

**Compare all four methods side-by-side (exports to a directory):**

```powershell
uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
  --cu3s-file-path "D:\data\XMR_notarget_Busstation\20260226\Auto_001+01.cu3s" `
  --compare-all "D:\data\XMR_notarget_Busstation\20260226\compare_methods"
```

Show CLI help:

```powershell
uv run python examples/object_tracking/export_cu3s_false_rgb_video.py --help
```

### Main CLI Options

- `--cu3s-file-path` required input `.cu3s` file
- `--output-video-path` target `.mp4` path (required unless `--compare-all` is used)
- `--method` `range_average|cie_tristimulus|camera_emulation|baseline` (default `range_average`)
- `--compare-all` export all methods into this directory instead of a single file
- `--processing-mode` `Raw|DarkSubtract|Preview|Reflectance|SpectralRadiance` (default `Raw`)
- `--normalization-mode` `sampled_fixed|running|per_frame` (default `sampled_fixed`)
- `--sample-fraction` fraction of frames used for sampled-fixed calibration (default `0.05`, valid `(0,1]`)
- `--freeze-running-bounds-after` freeze running normalization bounds after N frames (default `20`, use `<=0` to disable)
- `--frame-rate` output FPS (default: use session FPS, fallback 10.0)
- `--frame-rotation` rotation in degrees; `+90` = anticlockwise, `-90` = clockwise
- `--max-num-frames` maximum frames to write (`-1` = all frames)
- `--batch-size` dataloader batch size (default `1`)
- `--overlay-frame-id` render measurement index as text in the top-left corner of each frame
- `--red-low` / `--red-high` / `--green-low` / `--green-high` / `--blue-low` / `--blue-high` wavelength ranges for `range_average` (nm)
- `--r-peak` / `--g-peak` / `--b-peak` peak wavelengths for `camera_emulation` (nm)
- `--r-sigma` / `--g-sigma` / `--b-sigma` Gaussian sigmas for `camera_emulation` (nm)

### Outputs

Next to the output `.mp4`:

- `<name>.mp4` — false-RGB video
- `SAM3_FalseRGB_Export.png` — graphviz pipeline diagram
- `SAM3_FalseRGB_Export.yaml` — saved pipeline config (nodes + weights)

---

## Render Tracking Overlay

Use `examples/object_tracking/render_tracking_overlay.py` to apply bbox or mask overlays from a
tracking JSON onto an existing MP4 or directly onto a CU3S file.

### Video mode

Overlay `Auto_013+01-trustimulus.json` on the matching MP4:

```powershell
uv run python examples/object_tracking/render_tracking_overlay.py `
  --video-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01-trustimulus.mp4" `
  --tracking-json "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.json"
```

Output: `D:\data\XMR_notarget_Busstation\20260226\overlay.mp4` (alongside the JSON by default).

### CU3S mode

Supply a `.cu3s` file directly — false-RGB frames are generated on the fly (default method:
`cie_tristimulus`) and overlays are applied without a separate export step:

```powershell
uv run python examples/object_tracking/render_tracking_overlay.py `
  --cu3s-file-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.cu3s" `
  --tracking-json "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.json"
```

Change the false-RGB method:

```powershell
uv run python examples/object_tracking/render_tracking_overlay.py `
  --cu3s-file-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.cu3s" `
  --tracking-json "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.json" `
  --method range_average
```

Show CLI help:

```powershell
uv run python examples/object_tracking/render_tracking_overlay.py --help
```

### Main CLI Options

**Source (mutually exclusive, one required):**
- `--video-path` path to source MP4
- `--cu3s-file-path` path to `.cu3s` file (false-RGB generated on the fly)

**CU3S-mode options:**
- `--method` `range_average|cie_tristimulus|camera_emulation|baseline` (default `cie_tristimulus`)
- `--processing-mode` `Raw|DarkSubtract|Preview|Reflectance|SpectralRadiance` (default `Raw`)

**Common options:**
- `--tracking-json` required path to COCO tracking JSON
- `--output-video-path` output MP4 path (default: `<tracking_json_dir>/overlay.mp4`)
- `--start-frame` / `--end-frame` frame range (default: all)
- `--frame-rate` output FPS (default: source FPS or session FPS, fallback 10.0)
- `--mask-alpha` mask overlay opacity 0–1 (default `0.4`)
- `--line-thickness` bbox line thickness (default `2`)
- `--draw-labels` render track ID text on bboxes
- `--draw-contours` / `--no-draw-contours` mask contours (default on)
- `--draw-ids` / `--no-draw-ids` object ID labels on masks (default on)

### Outputs

- `overlay.mp4` (or `--output-video-path`) — overlay video

---

## TrackEval Metric Nodes

Use `examples/object_tracking/trackeval/evaluate_tracking.py` to evaluate two COCO tracking JSON
files through TrackEval metric nodes inside a `CuvisPipeline`.

The script loads the TrackEval plugin from:

- `configs/plugins/trackeval.yaml`

It instantiates and finalizes:

- `cuvis_ai_trackeval.node.HOTAMetricNode`
- `cuvis_ai_trackeval.node.CLEARMetricNode`
- `cuvis_ai_trackeval.node.IdentityMetricNode`

### Run (your files)

```powershell
uv run python examples/object_tracking/trackeval/evaluate_tracking.py `
  --gt "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.json" `
  --pred "D:\experiments\deepeiou\202060313\video_tracker_reid\Auto_013+01\tracking_results.json" `
  --match-threshold 0.5 `
  --plugins-manifest "configs/plugins/trackeval.yaml"
```

### Perfect-match sanity check (same JSON as GT + pred)

```powershell
uv run python examples/object_tracking/trackeval/evaluate_tracking.py `
  --gt "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.json" `
  --pred "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.json"
```

Expected output (printed):

- `HOTA`, `DetA`, `AssA`, `LocA`
- `MOTA`, `MOTP`, `FP`, `FN`, `IDSW`
- `IDF1`, `IDP`, `IDR`

### Main CLI Options

- `--gt` ground-truth COCO tracking JSON path
- `--pred` prediction COCO tracking JSON path
- `--match-threshold` IoU threshold for CLEAR/Identity (default `0.5`)
- `--plugins-manifest` plugin manifest path (default `configs/plugins/trackeval.yaml`)

---

## SAM3 Tracking

Use `examples/object_tracking/sam3/sam3_tracker.py` for end-to-end SAM3 tracking from exactly one
source:

- `--cu3s-path` for hyperspectral CU3S input (converted to false-RGB on the fly)
- `--video-path` for RGB video input

### What This Script Does

- Loads exactly one source:
  - CU3S via `SingleCu3sDataModule -> CU3SDataNode -> CIETristimulusFalseRGBSelector`
  - video via `VideoFrameDataModule -> VideoFrameNode`
- Builds one shared pipeline:
  - `source RGB -> SAM3TrackerInference`
  - Confirmed-output sink: `TrackingCocoJsonNode` (COCO JSON with RLE masks, bboxes, scores)
  - Overlay sink: `TrackingOverlayNode -> ToVideoNode` (MP4)
- Preserves source frame IDs in outputs:
  - CU3S mode uses `mesu_index`
  - video mode uses the original video `frame_id`, even when `--start-frame` skips frames
- Runs inference through `cuvis_ai_core.training.Predictor` (no manual node forwarding)
- Enables automatic per-node runtime profiling (CUDA-synchronised, skip first 3 warm-up frames)
- Saves pipeline visualizations as PNG and Mermaid markdown

### Prerequisites

- Exactly one source input:
  - a CU3S file (for example `Auto_013+01.cu3s`)
  - or a video file (for example `Auto_013+01-tristimulus.mp4`)
- `cuvis-ai-core` with `Predictor` and datamodule predict-stage support installed in the environment
- SAM3 plugin available and discoverable from `configs/plugins/sam3.yaml`

Current local plugin manifest (`configs/plugins/sam3.yaml`):

```yaml
plugins:
  sam3:
    path: "../../../../cuvis-ai-sam3/sam3-init"
    provides:
      - cuvis_ai_sam3.node.SAM3TrackerInference
      - cuvis_ai_sam3.node.SAM3StreamingPropagation
      - cuvis_ai_sam3.node.SAM3ObjectTracker
      - cuvis_ai_sam3.node.SpectralSignatureExtractor
```

If your plugin checkout is at a different path, update `path` accordingly.

### Run

**CU3S mode:**

```powershell
uv run python examples/object_tracking/sam3/sam3_tracker.py `
  --cu3s-path "D:\data\your_dataset\Auto_013+01.cu3s" `
  --plugins-yaml "configs/plugins/sam3.yaml"
```

**Video mode:**

```powershell
uv run python examples/object_tracking/sam3/sam3_tracker.py `
  --video-path "D:\data\your_dataset\Auto_013+01-tristimulus.mp4" `
  --plugins-yaml "configs/plugins/sam3.yaml"
```

**Quick smoke test with frame range (10 frames starting at frame 25):**

```powershell
uv run python examples/object_tracking/sam3/sam3_tracker.py `
  --video-path "D:\data\your_dataset\Auto_013+01-tristimulus.mp4" `
  --plugins-yaml "configs/plugins/sam3.yaml" `
  --start-frame 25 `
  --end-frame 35
```

**With bf16, explicit output, and tuned thresholds (CU3S mode):**

```powershell
uv run python examples/object_tracking/sam3/sam3_tracker.py `
  --cu3s-path "D:\data\your_dataset\Auto_013+01.cu3s" `
  --plugins-yaml "configs/plugins/sam3.yaml" `
  --output-dir "D:\experiments\sam3" `
  --out-basename "v1_sam3_tracker" `
  --start-frame 0 `
  --end-frame 350 `
  --bf16
```

Show CLI help:

```powershell
uv run python examples/object_tracking/sam3/sam3_tracker.py --help
```

### Main CLI Options

- Source options: exactly one required
- `--cu3s-path` input `.cu3s`
- `--video-path` input video file
- `--processing-mode` `Raw|DarkSubtract|Preview|Reflectance|SpectralRadiance` (default `SpectralRadiance`, CU3S mode only)
- `--start-frame` first frame to process (default `0`)
- `--end-frame` exclusive stop frame (`-1` means all remaining frames)
- `--prompt` text prompt for tracker initialization (default `person`)
- `--output-dir` parent/root output directory (default `./tracking_output`)
- `--out-basename` optional run-folder basename; defaults to
  `Path(cu3s_path).stem` in CU3S mode or `Path(video_path).stem` in video mode
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

For `--output-dir <ROOT>`, the script writes to:
`<RUN> = <ROOT>/<out-basename or input-file-stem>`

- `<RUN>/tracking_results.json` - COCO instance segmentation output (confirmed stream)
- `<RUN>/tracking_overlay.mp4` - overlay video (always produced)
- `<RUN>/pipeline/SAM3_HSI_Tracking.png` or `<RUN>/pipeline/SAM3_Video_Tracking.png` - graphviz pipeline image
- `<RUN>/pipeline/SAM3_HSI_Tracking.md` or `<RUN>/pipeline/SAM3_Video_Tracking.md` - mermaid pipeline markdown
- `<RUN>/profiling_summary.txt` - per-node runtime profiling breakdown

Default basename is only the input stem, so running different SAM3 modes on the
same input reuses the same folder unless you change `--out-basename` or
`--output-dir`.

### Profiling

The script automatically enables per-node pipeline profiling (with CUDA synchronisation and
3 warm-up frames skipped). After inference completes, a table is printed and saved to
`<RUN>/profiling_summary.txt`.

Example CU3S run: 350 frames from `Auto_013+01.cu3s` (SpectralRadiance, bf16, CUDA):

| Node | Count | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Total (s) | % |
|---|---|---|---|---|---|---|---|
| sam3_tracker | 347 | 9,552 | 6,619 | 481 | 165,439 | 3,314.6 | 99.2 |
| cie_false_rgb | 347 | 44 | 27 | 3 | 2,557 | 15.1 | 0.5 |
| overlay | 347 | 11 | 10 | 5 | 77 | 3.9 | 0.1 |
| tracking_coco_json | 347 | 11 | 11 | 6 | 45 | 3.7 | 0.1 |
| to_video | 347 | 8 | 8 | 6 | 11 | 2.6 | 0.1 |
| cu3s_data | 347 | 0.1 | 0.1 | 0.1 | 0.2 | 0.04 | 0.0 |
| **TOTAL** | | | | | | **3,340** | |

Average per-frame: **9,626 ms** (0.1 FPS). SAM3 tracker dominates at 99.2% of pipeline time.
The large std/max reflects tracker state growth — early frames ~480 ms (1 state),
later frames up to 165 s (5 states, 11 tracked objects).

## SAM3 Streaming Propagation

Four scripts demonstrate SAM3's streaming propagation with different prompt types.
All use the same pipeline pattern and produce an overlay video with frame IDs rendered in the top-left corner.

- **Text prompt** → `SAM3StreamingPropagation` (tracks prompt-matching objects)
- **Bbox / Point / Mask prompt** → `SAM3StreamingPropagation` (tracks explicitly prompted objects)

Temporal propagation state is preserved across the streamed sequence.

### Prerequisites

- CU3S input file
- SAM3 plugin: `configs/plugins/sam3.yaml`
- For bbox/point/mask prompts: a COCO-format detection or tracking JSON (e.g. from ByteTrack/DeepEIoU)

### Run Folder Naming

- `--output-dir` is the parent/root directory.
- Final run folder is `<output-dir>/<out-basename or input-file-stem>`.
- Default basename for all four scripts is `Path(cu3s_path).stem`.
- Use `--out-basename` to override the default stem.
- `--out-basename` must be a single folder name (not a path).
- Re-running multiple propagation modes on the same CU3S input reuses the same
  folder unless you change `--out-basename` or `--output-dir`.

### Detection spec: `--detection ID@FRAME`

Bbox, point, and mask scripts use `--detection ID@FRAME` to select which detection to use as a prompt:

- **ID** is matched against `track_id` first (from tracking JSONs); if not found, treated as a 1-based rank by detection score
- **FRAME** is the source `image_id` where the bbox/point/mask is read from
- Default: `1@0` (best detection on frame 0)
- `--start-frame` controls where the video begins (independent of prompt frame), must be `<= FRAME`
- Scripts convert source prompt frame to stream-local frame automatically: `local_prompt_frame = FRAME - start_frame`
- Frames before the prompt frame are rendered without masks so temporal context is visible

Examples: `--detection 1@0` (best detection, frame 0), `--detection 2@76` (track ID 2, frame 76).

For bbox propagation, exactly one `--detection` value is currently supported.

Bbox propagation ID semantics:
- If `--detection ID@FRAME` provides an ID, outputs reuse that ID (`track_id=ID`) for the selected bbox object.
- If no explicit bbox object ID is provided at node level, outputs use the SAM-selected object ID.

### Scripts

#### Text prompt (validated window)

```powershell
uv run python examples/object_tracking/sam3/sam3_text_propagation.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.cu3s" `
  --start-frame 290 `
  --max-frames 100 `
  --prompt "person" `
  --output-dir "D:\experiments\sam3\20260316" `
  --out-basename "text_propagation_f290_100f" `
  --plugins-yaml "configs/plugins/sam3.yaml" `
  --bf16
```

Default prompt is `person`. Change with `--prompt "car"` as needed.

#### Bbox prompt (validated window: `track_id=14@frame=290`)

```powershell
uv run python examples/object_tracking/sam3/sam3_bbox_propagation.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.cu3s" `
  --detection-json "D:\experiments\sam3\20260316\video_tracker_parity_unlimited_states_confirmed\tracking_results.json" `
  --detection 14@290 `
  --start-frame 290 `
  --max-frames 100 `
  --output-dir "D:\experiments\sam3\20260316" `
  --out-basename "bbox_propagation_id14_f290_100f" `
  --plugins-yaml "configs/plugins/sam3.yaml" `
  --bf16
```

This mode tracks a single object for the prompted bbox and writes that object with the requested detection ID in output JSON.

#### Point prompt (validated window: `track_id=14@frame=290`)

```powershell
uv run python examples/object_tracking/sam3/sam3_point_propagation.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.cu3s" `
  --detection-json "D:\experiments\sam3\20260316\video_tracker_parity_unlimited_states_confirmed\tracking_results.json" `
  --detection 14@290 `
  --start-frame 290 `
  --max-frames 100 `
  --output-dir "D:\experiments\sam3\20260316" `
  --out-basename "point_propagation_id14_f290_100f" `
  --plugins-yaml "configs/plugins/sam3.yaml" `
  --bf16
```

Uses the center of the detection bbox as a positive point prompt.

#### Mask prompt (validated window: `track_id=14@frame=290`)

```powershell
uv run python examples/object_tracking/sam3/sam3_mask_propagation.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.cu3s" `
  --detection-json "D:\experiments\sam3\20260316\video_tracker_parity_unlimited_states_confirmed\tracking_results.json" `
  --detection 14@290 `
  --start-frame 290 `
  --max-frames 100 `
  --output-dir "D:\experiments\sam3\20260316" `
  --out-basename "mask_propagation_id14_f290_100f" `
  --plugins-yaml "configs/plugins/sam3.yaml" `
  --bf16
```

Creates a binary mask PNG from the detection bbox and uses it as a mask prompt.

### Shared CLI Options

- `--cu3s-path` required input `.cu3s`
- `--processing-mode` `Raw|DarkSubtract|Preview|Reflectance|SpectralRadiance` (default `SpectralRadiance`)
- `--start-frame` first frame to include in the video (default `0`)
- `--max-frames` maximum frames to process from start-frame (default `-1` = all)
- `--detection` detection spec `ID@FRAME` — which detection to prompt with (bbox/point/mask scripts only; bbox currently supports one value)
- `--output-dir` parent/root output directory
- `--out-basename` optional run-folder basename; defaults to `Path(cu3s_path).stem`
- `--plugins-yaml` plugin manifest path (default `configs/plugins/sam3.yaml`)
- `--checkpoint-path` optional SAM3 checkpoint path
- `--bf16` enable CUDA bf16 autocast
- `--frame-rotation` optional frame rotation (degrees)

### Outputs

For `--output-dir <ROOT>`, each script writes to:
`<RUN> = <ROOT>/<out-basename or Path(cu3s_path).stem>`

- `<RUN>/tracking_results.json` — COCO instance segmentation JSON
- `<RUN>/tracking_overlay.mp4` — overlay video with colored masks and frame IDs
- `<RUN>/profiling_summary.txt` — per-node runtime profiling breakdown
- `<RUN>/<PipelineName>.png` — graphviz pipeline diagram
- `<RUN>/prompt_mask.png` — mask prompt image (mask-prompt mode only)

---

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

```powershell
uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py `
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s"
```

Quick smoke test (10 frames):

```powershell
uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py `
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s" `
  --end-frame 10
```

With explicit output and tuning:

```powershell
uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py `
  --cu3s-path "D:\data\your_dataset\Auto_002.cu3s" `
  --output-dir "outputs" `
  --out-basename "bytetrack_tracking" `
  --confidence-threshold 0.3 `
  --track-thresh 0.4
```

Show CLI help:

```powershell
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
- `--output-dir` parent/root output directory (default `./tracking_output`)
- `--out-basename` optional run-folder basename; defaults to `Path(cu3s_path).stem`
- `--plugins-dir` plugin YAML directory (default `<repo>/configs/plugins/`)
- `--frame-rotation` optional frame rotation (degrees)
- `--bf16` enable CUDA bf16 autocast

### Outputs

For `--output-dir <ROOT>`, the script writes to:
`<RUN> = <ROOT>/<out-basename or Path(cu3s_path).stem>`

- `<RUN>/detection_results.json` — COCO detection JSON (pre-tracking)
- `<RUN>/tracking_results.json` — COCO tracking JSON with `track_id` per annotation
- `<RUN>/tracking_overlay.mp4` — CIE false RGB frames with tracked bounding box overlays
- `<RUN>/experiment_info.txt` — experiment parameters and result metrics
- `<RUN>/profiling_summary.txt` — per-node runtime profiling breakdown
- `<RUN>/YOLO_ByteTrack_HSI.png` — graphviz pipeline diagram

Default basename is only the input stem, so repeated runs on the same CU3S file
reuse the same folder unless you change `--out-basename` or `--output-dir`.

## YOLO + DeepEIoU HSI Tracking

Use `examples/object_tracking/deepeiou/yolo_deepeiou_reid_hsi.py` as the single
DeepEIoU example for either CU3S or RGB video input. Exactly one of
`--cu3s-path` or `--video-path` is required. The script supports EIoU-only
tracking (`--no-reid`, the default) and ReID-enhanced tracking (`--with-reid`).

### What This Script Does

- Loads exactly one source:
  - CU3S via `SingleCu3sDataModule` -> `CU3SDataNode` -> `CIETristimulusFalseRGBSelector`
  - video via `VideoFrameDataModule` -> `VideoFrameNode`
- Builds one pipeline:
  - Shared path: `source RGB → YOLOPreprocess → YOLO26Detection → YOLOPostprocess`
  - Optional ReID branch: `source RGB + bboxes → BBoxRoiCropNode → ChannelNormalizeNode → OSNet/ResNetExtractor`
  - Tracker: `YOLOPostprocess → DeepEIoUTrack`, with embeddings wired only when `--with-reid`
  - Sinks: `DetectionCocoJsonNode`, `ByteTrackCocoJson`, `BBoxesOverlayNode → ToVideoNode`, optional `NumpyFeatureWriterNode`
- Runs inference through `cuvis_ai_core.training.Predictor`
- Preserves absolute frame IDs in outputs: `mesu_index` for CU3S, original `frame_id` for video

### Shared Core Across All Modes

The script uses one common tracking core in all supported modes:

```text
source RGB -> YOLOPreprocess -> YOLO26Detection -> YOLOPostprocess -> DeepEIoUTrack
```

What changes between runs is only:

- the source adapter in front of the shared core
- whether the optional ReID branch is connected
- whether `.npy` embeddings are written

That means the same YOLO detector, the same `DeepEIoUTrack` node, the same JSON
writers, and the same overlay/video sinks are reused in every mode. Switching
between CU3S and video does not create a separate tracking implementation; it
only changes how frames enter the shared core. Switching between `--no-reid`
and `--with-reid` does not swap to a different tracker either; it uses the same
DeepEIoU node with ReID disabled or enabled.

### Behavior By Mode

| Source | ReID mode | Behavior |
|---|---|---|
| video | `--no-reid` | Reads RGB frames directly from `VideoFrameDataModule`, runs YOLO + DeepEIoU in EIoU-only mode, writes detection/tracking JSON and overlay video. |
| video | `--with-reid` | Uses the same RGB video core path, plus bbox crop -> normalize -> extractor embeddings wired into `DeepEIoUTrack`. Optional `.npy` feature export is available with `--write-features`. |
| CU3S | `--no-reid` | Reads hyperspectral frames, converts them to CIE false-RGB once per frame, then runs the same YOLO + DeepEIoU core in EIoU-only mode. |
| CU3S | `--with-reid` | Uses the same CU3S false-RGB front-end, plus the optional ReID branch on top of the shared YOLO + DeepEIoU core. Optional `.npy` feature export is available with `--write-features`. |

### Source-Specific Notes

- CU3S mode generates false-RGB frames on the fly and uses those frames for YOLO, overlay rendering, and ReID crops when ReID is enabled.
- Video mode uses the original RGB frames directly; no false-RGB conversion is involved.
- `--processing-mode` only affects CU3S mode and is ignored when `--video-path` is used.
- `--start-frame` and `--end-frame` work in both modes.
- JSON `image_id` values stay source-native:
  - CU3S mode uses `mesu_index`
  - video mode uses the original video `frame_id`, even when `--start-frame` skips initial frames

### ReID Behavior

- `--no-reid` is the default. In this mode, the tracker uses geometry / motion association only, without appearance embeddings.
- `--with-reid` enables the appearance branch:
  - `BBoxRoiCropNode`
  - `ChannelNormalizeNode`
  - `OSNetExtractor` or `ResNetExtractor`
- `--reid-weights` is required only when `--with-reid` is enabled.
- `--write-features` is optional and only valid together with `--with-reid`.
- `--no-write-features` still uses embeddings for tracking when ReID is enabled; it only disables `.npy` export.

### Output Naming

Pipeline graph names reflect both the source type and whether ReID is enabled:

| Source | ReID mode | Pipeline name |
|---|---|---|
| video | `--no-reid` | `YOLO_DeepEIoU_RGB` |
| video | `--with-reid` | `YOLO_DeepEIoU_ReID_RGB` |
| CU3S | `--no-reid` | `YOLO_DeepEIoU_HSI` |
| CU3S | `--with-reid` | `YOLO_DeepEIoU_ReID_HSI` |

### Prerequisites

- Exactly one source input:
  - a CU3S file (e.g. `Auto_013+01.cu3s`)
  - or a video file (e.g. `Auto_013+01-tristimulus.mp4`)
- `cuvis-ai-core` with `Predictor` and datamodule predict-stage support installed
- Ultralytics plugin: `configs/plugins/ultralytics.yaml`
- DeepEIoU plugin: `configs/plugins/deepeiou.yaml`
- A YOLO model (default `yolo26n.pt`, auto-downloaded on first run)
- ReID backbone weights only when `--with-reid` is used; OSNet paths may be auto-downloaded by the node if missing, while ResNet requires manual weights
- `--processing-mode` is only used in CU3S mode and is ignored in video mode

### Run

**EIoU-only quick smoke test, video mode (60 frames):**

```powershell
uv run python examples/object_tracking/deepeiou/yolo_deepeiou_reid_hsi.py `
  --video-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01-tristimulus.mp4" `
  --no-reid `
  --output-dir "D:\experiments\deepeiou" `
  --out-basename "rgb_eiou_only" `
  --start-frame 25 `
  --end-frame 85
```

**EIoU-only quick smoke test, CU3S mode (60 frames):**

```powershell
uv run python examples/object_tracking/deepeiou/yolo_deepeiou_reid_hsi.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.cu3s" `
  --no-reid `
  --output-dir "D:\experiments\deepeiou" `
  --out-basename "eiou_only" `
  --end-frame 60
```

**ReID quick smoke test, CU3S mode (60 frames, OSNet):**

```powershell
uv run python examples/object_tracking/deepeiou/yolo_deepeiou_reid_hsi.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.cu3s" `
  --with-reid `
  --reid-weights "D:\models\osnet\osnet_x1_0_imagenet.pt" `
  --output-dir "D:\experiments\deepeiou" `
  --out-basename "reid" `
  --end-frame 60
```

**ReID quick smoke test, video mode (60 frames, no feature export):**

```powershell
uv run python examples/object_tracking/deepeiou/yolo_deepeiou_reid_hsi.py `
  --video-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01-tristimulus.mp4" `
  --with-reid `
  --reid-weights "D:\models\osnet_x1_0_imagenet.pth.tar" `
  --no-write-features `
  --output-dir "D:\experiments\deepeiou" `
  --out-basename "rgb_reid" `
  --end-frame 60
```

**ReID run with optional embedding export:**

```powershell
uv run python examples/object_tracking/deepeiou/yolo_deepeiou_reid_hsi.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.cu3s" `
  --with-reid `
  --reid-weights "D:\models\osnet_x1_0_imagenet.pth.tar" `
  --write-features `
  --output-dir "D:\experiments\deepeiou" `
  --out-basename "reid_hsi_run01" `
  --confidence-threshold 0.3 `
  --track-high-thresh 0.5 `
  --appearance-thresh 0.25 `
  --bf16
```

**With ResNet backbone instead of OSNet:**

```powershell
uv run python examples/object_tracking/deepeiou/yolo_deepeiou_reid_hsi.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.cu3s" `
  --with-reid `
  --reid-weights "D:\models\resnet50_market1501.pth.tar" `
  --backbone resnet `
  --end-frame 60
```

Show CLI help:

```powershell
uv run python examples/object_tracking/deepeiou/yolo_deepeiou_reid_hsi.py --help
```

### Main CLI Options

- Source options: exactly one required
- `--cu3s-path` input `.cu3s`
- `--video-path` input video file
- `--with-reid / --no-reid` enable or disable appearance embeddings (default: `--no-reid`)
- `--reid-weights` path to ReID backbone weights, required only with `--with-reid`
- `--backbone` `osnet` or `resnet` (default `osnet`, used only with `--with-reid`)
- `--write-features / --no-write-features` write per-frame `.npy` embeddings when ReID is enabled
- `--processing-mode` `Raw|DarkSubtract|Preview|Reflectance|SpectralRadiance` (default `SpectralRadiance`, CU3S mode only)
- `--start-frame` / `--end-frame` frame range (default: all)
- `--model-name` YOLO model name/path (default `yolo26n.pt`)
- `--confidence-threshold` YOLO detection threshold (default `0.5`)
- `--iou-threshold` NMS IoU threshold (default `0.7`)
- `--agnostic-nms` enable class-agnostic NMS
- `--category-id` class filter: default `0` (person), use `--category-id <N>` for one class, or `--category-id -1` for all classes
- `--track-high-thresh` / `--track-low-thresh` / `--new-track-thresh` DeepEIoU thresholds
- `--track-buffer` lost-track buffer in frames (default `60`)
- `--match-thresh` IoU match threshold (default `0.8`)
- `--proximity-thresh` / `--appearance-thresh` DeepEIoU appearance-gating thresholds
- `--output-dir` parent/root output directory (default `./tracking_output`)
- `--out-basename` optional run-folder basename; defaults to
  `Path(cu3s_path).stem` in CU3S mode or `Path(video_path).stem` in video mode
- `--plugins-dir` plugin YAML directory (default `<repo>/configs/plugins/`)
- `--bf16` enable CUDA bf16 autocast
- `--hide-untracked / --show-untracked` hide bboxes without track ID (default: hide)

### Outputs

For `--output-dir <ROOT>`, the script writes to:
`<RUN> = <ROOT>/<out-basename or input-file-stem>`

- `<RUN>/detection_results.json` — COCO detection JSON (pre-tracking)
- `<RUN>/tracking_results.json` — COCO tracking JSON with `track_id` per annotation
- `<RUN>/tracking_overlay.mp4` — overlay video with tracked bounding boxes
- `<RUN>/experiment_info.txt` — experiment parameters and result metrics
- `<RUN>/profiling_summary.txt` — per-node runtime profiling breakdown
- `<RUN>/YOLO_DeepEIoU_RGB.png` — graphviz pipeline diagram in video EIoU-only mode
- `<RUN>/YOLO_DeepEIoU_ReID_RGB.png` — graphviz pipeline diagram in video ReID mode
- `<RUN>/YOLO_DeepEIoU_HSI.png` — graphviz pipeline diagram when ReID is disabled
- `<RUN>/YOLO_DeepEIoU_ReID_HSI.png` — graphviz pipeline diagram when ReID is enabled
- `<RUN>/features/` — per-frame `.npy` ReID embeddings, only when `--with-reid --write-features`

Default basename is only the input stem, so running different modes on the same
input reuses the same run folder unless you change `--out-basename` or
`--output-dir`.

---

## Channel Selector False RGB Workflow (Training)

`examples/object_tracking/channel_selector_false_rgb.py` remains available for
learnable false-RGB training and inspection.

Inspect:

```powershell
uv run python examples/object_tracking/channel_selector_false_rgb.py mode=inspect
```

Train:

```powershell
uv run python examples/object_tracking/channel_selector_false_rgb.py mode=train
```
