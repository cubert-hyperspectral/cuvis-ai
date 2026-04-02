# Object Tracking Examples

This folder contains object-tracking workflows for hyperspectral CU3S recordings.

## Table of Contents

- [Export CU3S to False-RGB Video](#export-cu3s-to-false-rgb-video)
- [Synthetic Occlusion (Poisson)](#synthetic-occlusion-poisson)
- [Render Tracking Overlay](#render-tracking-overlay)
- [TrackEval Metric Nodes](#trackeval-metric-nodes)
- [SAM3 Text Tracking](#sam3-text-tracking)
- [SAM3 Streaming Propagation](#sam3-streaming-propagation)
- [YOLO + ByteTrack HSI Tracking](#yolo--bytetrack-hsi-tracking)
- [YOLO + DeepEIoU HSI Tracking](#yolo--deepeiou-hsi-tracking)
- [Channel Selector False RGB Workflow (Training)](#channel-selector-false-rgb-workflow-training)

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
| `cie_tristimulus` | `CIETristimulusFalseRGBSelector` | CIE 1931 XYZ → sRGB conversion |
| `cir` | `CIRSelector` | NIR→R, Red→G, Green→B false color |
| `fast_rgb` | `FastRGBSelector` | cuvis-next parity fast range averaging + parity scaling |
| `cuvis-plugin` | `FastRGBSelector` | cuvis user-plugin XML parity (`fast_rgb` ranges/normalization from XML) |

### Run

**CIE tristimulus export:**

```powershell
uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_001+01.cu3s" `
  --output-dir "D:\experiments\sam3\false_rgb_export" `
  --method cie_tristimulus
```

**Default normalization behavior (used if not specified):**

- `--normalization-mode sampled_fixed`
- `--sample-fraction 0.05`

Equivalent explicit command:

```powershell
uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_001+01.cu3s" `
  --output-dir "D:\experiments\video_creation\20260318\video_creation\cir" `
  --out-basename "Auto_001+01_cir" `
  --method cir `
  --normalization-mode sampled_fixed `
  --sample-fraction 0.05
```

**With frame ID overlay (renders measurement index in top-left corner):**

```powershell
uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_001+01.cu3s" `
  --output-dir "D:\experiments\sam3\false_rgb_export" `
  --method cie_tristimulus `
  --overlay-frame-id
```

**cuvis-plugin XML parity export:**

```powershell
uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_001+01.cu3s" `
  --output-dir "D:\experiments\sam3\false_rgb_export" `
  --method cuvis-plugin `
  --plugin-xml-path "C:\Users\nima.ghorbani\CuvisNEXT\invisible_ink.xml"
```

Show CLI help:

```powershell
uv run python examples/object_tracking/export_cu3s_false_rgb_video.py --help
```

### Main CLI Options

- `--cu3s-path` required input `.cu3s` file
- `--output-dir` parent/root output directory
- `--out-basename` optional run-folder basename; defaults to `Path(cu3s_path).stem`
- `--method` `cie_tristimulus|cir|fast_rgb|cuvis-plugin|fastrgb` (default `cie_tristimulus`; `fastrgb` aliases `fast_rgb`)
- `--plugin-xml-path` cuvis user-plugin XML path (required for `--method cuvis-plugin`)
- `--processing-mode` `Raw|DarkSubtract|Preview|Reflectance|SpectralRadiance` (default `Raw`)
- `--normalization-mode` `sampled_fixed|running|per_frame|live_running_fixed` (default `sampled_fixed`)
- `--sample-fraction` fraction of frames used for sampled-fixed calibration (default `0.05`, valid `(0,1]`)
- `--freeze-running-bounds-after` freeze running normalization bounds after N frames (default `20`, use `<=0` to disable)
- `--running-warmup-frames` running-mode warmup frame count (default `10`)
- `--fast-rgb-normalization-strength` optional FastRGB normalization override
- `--frame-rate` output FPS (default: use session FPS, fallback 10.0)
- `--frame-rotation` rotation in degrees; `+90` = anticlockwise, `-90` = clockwise
- `--max-num-frames` maximum frames to write (`-1` = all frames)
- `--batch-size` dataloader batch size (default `1`)
- `--overlay-frame-id` render measurement index as text in the top-left corner of each frame
- `--red-low` / `--red-high` / `--green-low` / `--green-high` / `--blue-low` / `--blue-high` wavelength ranges for `fast_rgb` (nm)

For `--method fast_rgb` and `--method cuvis-plugin`, legacy selector normalization controls are ignored:
`--normalization-mode`, `--sample-fraction`, `--freeze-running-bounds-after`, and
`--running-warmup-frames`.

### Outputs

For `--output-dir <ROOT>`, the script writes to:
`<RUN> = <ROOT>/<out-basename or Path(cu3s_path).stem>`

- `<RUN>/<Path(cu3s_path).stem>.mp4` — false-RGB video
- `<RUN>/SAM3_FalseRGB_Export.png` — graphviz pipeline diagram
- `<RUN>/SAM3_FalseRGB_Export.yaml` — saved pipeline config (nodes + weights)

---

## Synthetic Occlusion (Poisson)

Use `examples/object_tracking/occlusion/occlude_data.py` to generate synthetic occlusions from
tracking masks/bboxes on CU3S data with pure-PyTorch Poisson filling (no OpenCV roundtrip).

### Pipeline

`--occlude-target rgb` (default):

```
CU3SDataNode -> CIETristimulusFalseRGBSelector -> PoissonOcclusionNode(rgb) -> ToVideoNode
```

`--occlude-target cube`:

```
CU3SDataNode -> PoissonOcclusionNode(cube) -> CIETristimulusFalseRGBSelector -> ToVideoNode
```

### Run

Initial Phase-03 style run (occlude tracks 2 and 9 on frames 70..120):

```powershell
uv run python examples/object_tracking/occlusion/occlude_data.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.cu3s" `
  --tracking-json "D:\experiments\sam3\20260316\video_tracker_parity_unlimited_states_confirmed\tracking_results.json" `
  --track-ids "2,9" `
  --occlusion-start-frame 70 --occlusion-end-frame 120 `
  --start-frame 0 --end-frame 301 `
  --occlusion-shape bbox --bbox-mode static --static-bbox-scale 1.2 `
  --output-video-path "D:\experiments\sam3\20260318\ALL_5448\phase03\occluded_videos\static_bbox_poisson_rgb_tid2_9_occ070_120.mp4"
```

Use mask occlusion (tighter to target masks, less collateral occlusion on nearby tracks):

```powershell
uv run python examples/object_tracking/occlusion/occlude_data.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.cu3s" `
  --tracking-json "D:\experiments\sam3\20260316\video_tracker_parity_unlimited_states_confirmed\tracking_results.json" `
  --track-ids "2,9" `
  --occlusion-start-frame 70 --occlusion-end-frame 120 `
  --start-frame 0 --end-frame 301 `
  --occlusion-shape mask `
  --output-video-path "D:\experiments\sam3\20260318\ALL_5448\phase03\occluded_videos\mask_poisson_rgb_tid2_9_occ070_120.mp4"
```

Show CLI help:

```powershell
uv run python examples/object_tracking/occlusion/occlude_data.py --help
```

### Main CLI Options

- `--cu3s-path` required input `.cu3s`
- `--tracking-json` required COCO tracking JSON with `track_id` and `segmentation`
- `--track-ids` comma-separated occlusion target track IDs
- `--occlusion-start-frame` / `--occlusion-end-frame` inclusive occlusion window
- `--start-frame` / `--end-frame` streamed source frame range (`--end-frame` is exclusive)
- `--occlusion-shape` `bbox|mask`
- `--bbox-mode` `static|dynamic` (used when shape is `bbox`)
- `--static-bbox-scale` scale factor for static union bbox (default `1.2`)
- `--static-bbox-padding-px` extra static bbox padding in pixels
- `--static-full-width-x` optionally stretch static bbox to full frame width
- `--occlude-target` `rgb|cube` (default `rgb`)
- `--max-iter` Poisson CG max iterations
- `--tol` Poisson CG convergence tolerance
- `--processing-mode` `Raw|DarkSubtract|Preview|Reflectance|SpectralRadiance` (default `SpectralRadiance`)
- `--sample-fraction` false-RGB calibration frame fraction (default `0.05`)
- `--frame-rate` output FPS override
- `--frame-rotation` output rotation
- `--overlay-frame-id` render frame index text in output
- `--output-video-path` output MP4 path

### Outputs

For `--output-video-path <OUT>.mp4`, the script writes:

- `<OUT>.mp4` - occluded false-RGB video
- `<OUT>.profiling_summary.txt` - per-node runtime profiling summary in the same directory

### Notes

- Static bbox mode can occlude nearby objects if trajectories overlap the static union box.
- Prefer `--occlusion-shape mask` or `--bbox-mode dynamic` when tighter occlusion regions are needed.

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
  --pred "D:\experiments\deepeiou\20260313\video_tracker_reid\Auto_013+01\tracking_results.json" `
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

## SAM3 Text Tracking

Use `examples/object_tracking/sam3/sam3_text_propagation.py` for end-to-end
text-prompt SAM3 tracking from exactly one source:

- `--cu3s-path` for hyperspectral CU3S input (converted to false-RGB on the fly)
- `--video-path` for RGB video input

### What This Script Does

- Loads exactly one source:
  - CU3S via `SingleCu3sDataModule -> CU3SDataNode -> CIETristimulusFalseRGBSelector`
  - video via `VideoFrameDataModule -> VideoFrameNode`
- Builds one shared pipeline:
  - `source RGB -> SAM3TextPropagation`
  - `SAM3TextPropagation -> TrackingCocoJsonNode`
  - `SAM3TextPropagation + source RGB -> TrackingOverlayNode -> ToVideoNode`
- Preserves source frame IDs in outputs:
  - CU3S mode uses `mesu_index`
  - video mode uses the original video `frame_id`, including when `--start-frame` is used
- Runs inference through `cuvis_ai_core.training.Predictor`
- Saves one graphviz pipeline image (no Mermaid output)
- Writes per-node profiling summary

### Prerequisites

- Exactly one source input:
  - a CU3S file (for example `Auto_013+01.cu3s`)
  - or a video file (for example `Auto_013+01-tristimulus.mp4`)
- `cuvis-ai-core` with `Predictor` and datamodule predict-stage support installed in the environment
- SAM3 plugin available and discoverable from `configs/plugins/sam3.yaml`

Required class for this script: `cuvis_ai_sam3.node.SAM3TextPropagation`.

### Run

**CU3S mode:**

```powershell
uv run python examples/object_tracking/sam3/sam3_text_propagation.py `
  --cu3s-path "D:\data\your_dataset\Auto_013+01.cu3s" `
  --plugins-yaml "configs/plugins/sam3.yaml"
```

**Video mode:**

```powershell
uv run python examples/object_tracking/sam3/sam3_text_propagation.py `
  --video-path "D:\data\your_dataset\Auto_013+01-tristimulus.mp4" `
  --plugins-yaml "configs/plugins/sam3.yaml"
```

**Quick smoke test with frame window (10 frames from frame 25):**

```powershell
uv run python examples/object_tracking/sam3/sam3_text_propagation.py `
  --video-path "D:\data\your_dataset\Auto_013+01-tristimulus.mp4" `
  --plugins-yaml "configs/plugins/sam3.yaml" `
  --start-frame 25 `
  --max-frames 10
```

**With bf16, compile, and explicit thresholds (CU3S mode):**

```powershell
n
```

Show CLI help:

```powershell
uv run python examples/object_tracking/sam3/sam3_text_propagation.py --help
```

### Main CLI Options

- Source options: exactly one required
- `--cu3s-path` input `.cu3s`
- `--video-path` input video file
- `--processing-mode` `Raw|DarkSubtract|Preview|Reflectance|SpectralRadiance` (default `SpectralRadiance`, CU3S mode only)
- `--start-frame` first source frame to process (default `0`)
- `--end-frame` exclusive stop frame (`-1` means all remaining frames)
- `--max-frames` deprecated alias for frame window length from `--start-frame`
- `--prompt` text prompt for SAM3 detector (default `person`)
- `--output-dir` parent/root output directory
- `--out-basename` optional run-folder basename; defaults to
  `Path(cu3s_path).stem` in CU3S mode or `Path(video_path).stem` in video mode
- `--checkpoint-path` optional tracker checkpoint path
- `--plugins-yaml` plugin manifest path (default `configs/plugins/sam3.yaml`)
- `--bf16` enable CUDA bf16 autocast
- `--compile` enable `torch.compile`
- `--score-threshold-detection` SAM3 detector score threshold (default `0.5`)
- `--new-det-thresh` new-track threshold (default `0.7`)
- `--det-nms-thresh` detector NMS IoU threshold (default `0.1`)
- `--overlap-suppress-thresh` overlap suppression threshold (default `0.7`)
- `--max-tracker-states` max active tracker states (default `5`)

### Outputs

For `--output-dir <ROOT>`, the script writes to:
`<RUN> = <ROOT>/<out-basename or input-file-stem>`

- `<RUN>/tracking_results.json` - COCO instance segmentation output
- `<RUN>/tracking_overlay.mp4` - overlay video
- `<RUN>/SAM3_Text_Propagation_HSI.png` or `<RUN>/SAM3_Text_Propagation_Video.png` - graphviz pipeline image
- `<RUN>/profiling_summary.txt` - per-node runtime profiling breakdown

## SAM3 Streaming Propagation

Four scripts demonstrate SAM3's streaming propagation with different prompt types.
All use the same pipeline pattern and produce an overlay video with frame IDs rendered in the top-left corner.

- **Text prompt** → `SAM3TextPropagation` (tracks prompt-matching objects)
- **Bbox prompt** → `SAM3BboxPropagation` (tracks a single bbox-prompted object)
- **Point prompt** → `SAM3PointPropagation` (tracks a single point-prompted object)
- **Mask prompt** → `SAM3MaskPropagation` (tracks a single mask-prompted object)

Temporal propagation state is preserved across the streamed sequence.

### Prerequisites

- Exactly one source input:
  - a CU3S file (for example `Auto_013+01.cu3s`)
  - or a video file (for example `Auto_013+01-tristimulus.mp4`)
- SAM3 plugin: `configs/plugins/sam3.yaml`
- For bbox/point prompts: a COCO-format detection or tracking JSON (e.g. from ByteTrack/DeepEIoU)
- For mask prompt: a binary prompt mask PNG (`--prompt-mask-path`) and source prompt frame index (`--prompt-frame-idx`)

### Run Folder Naming

- `--output-dir` is the parent/root directory.
- Final run folder is `<output-dir>/<out-basename or input-file-stem>`.
- Default basename is `Path(cu3s_path).stem` in CU3S mode or `Path(video_path).stem` in video mode.
- Use `--out-basename` to override the default stem.
- `--out-basename` must be a single folder name (not a path).
- Re-running multiple propagation modes on the same input reuses the same
  folder unless you change `--out-basename` or `--output-dir`.

### Detection spec: `--detection ID@FRAME`

Bbox and point scripts use `--detection ID@FRAME` to select which detection to use as a prompt:

- **ID** is matched against `track_id` first (from tracking JSONs); if not found, treated as a 1-based rank by detection score
- **FRAME** is the source `image_id` where the bbox/point is read from
- Default: `1@0` (best detection on frame 0)
- `--start-frame` controls where the video begins (independent of prompt frame), must be `<= FRAME`
- Scripts convert source prompt frame to stream-local frame automatically: `local_prompt_frame = FRAME - start_frame`
- Frames before the prompt frame are rendered without masks so temporal context is visible

Examples: `--detection 1@0` (best detection, frame 0), `--detection 2@76` (track ID 2, frame 76).

For bbox propagation, use exactly one `--detection` value.

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

#### Mask prompt (pre-made mask PNG at frame 290)

```powershell
uv run python examples/object_tracking/sam3/sam3_mask_propagation.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.cu3s" `
  --prompt-mask-path "D:\experiments\sam3\20260318\sam3\prompt_mask.png" `
  --prompt-frame-idx 290 `
  --prompt-obj-id 14 `
  --start-frame 290 `
  --max-frames 100 `
  --output-dir "D:\experiments\sam3\20260316" `
  --out-basename "mask_propagation_obj14_f290_100f" `
  --plugins-yaml "configs/plugins/sam3.yaml" `
  --bf16
```

Uses a provided binary mask PNG (`255=foreground`, `0=background`) as the prompt.

### Shared CLI Options

- Source options: exactly one required
- `--cu3s-path` input `.cu3s`
- `--video-path` input video file
- `--processing-mode` `Raw|DarkSubtract|Preview|Reflectance|SpectralRadiance` (default `SpectralRadiance`, CU3S mode only)
- `--start-frame` first source frame to process (default `0`)
- `--end-frame` exclusive stop frame (`-1` means all remaining frames)
- `--max-frames` deprecated alias for frame window length from `--start-frame`
- `--output-dir` parent/root output directory
- `--out-basename` optional run-folder basename; defaults to input file stem
- `--plugins-yaml` plugin manifest path (default `configs/plugins/sam3.yaml`)
- `--checkpoint-path` optional SAM3 checkpoint path
- `--bf16` enable CUDA bf16 autocast
- `--compile` enable `torch.compile`
- `--score-threshold-detection` SAM3 detector score threshold (default `0.5`)
- `--new-det-thresh` new-track threshold (default `0.7`)
- `--det-nms-thresh` detector NMS IoU threshold (default `0.1`)
- `--overlap-suppress-thresh` overlap suppression threshold (default `0.7`)
- `--max-tracker-states` max active tracker states (default `5`)
- `--frame-rotation` optional frame rotation (degrees)

Prompt-specific options:
- Text: `--prompt`
- Bbox/Point: `--detection-json` and `--detection ID@FRAME`
- Mask: `--prompt-mask-path`, `--prompt-frame-idx`, `--prompt-obj-id`

### Outputs

For `--output-dir <ROOT>`, each script writes to:
`<RUN> = <ROOT>/<out-basename or input-file-stem>`

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
