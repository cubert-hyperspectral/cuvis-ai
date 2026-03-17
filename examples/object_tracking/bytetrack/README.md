# YOLO + ByteTrack Object Tracking

## Scripts

| Script | Input | Description |
|--------|-------|-------------|
| `yolo_bytetrack_rgb.py` | MP4 video | YOLO26 detection + ByteTrack tracking on RGB frames. Outputs COCO JSONs + overlay video. |
| `yolo_bytetrack_hsi.py` | CU3S file | CU3S → false RGB → YOLO → ByteTrack with optional spectral-aware association. |
| `run_bytetrack_sweep.py` | CU3S file | Grid-search over detection, tracking, and spectral thresholds using `yolo_bytetrack_hsi.py`. |
| `regenerate_bytetrack_sweep_html.py` | Sweep output dir | Builds interactive HTML table + CSV from existing sweep results. |
| `convert_overlays_in_place.py` | Sweep output dir | Re-encodes existing overlay videos in place (default codec `VP90`). |

## Pipeline

```
VideoFrameNode → YOLOPreprocess → YOLO26Detection → YOLOPostprocess (NMS) → ByteTrack → overlay + JSON
```

The HSI variant replaces `VideoFrameNode` with `CU3SDataNode → CIETristimulusFalseRGBSelector` and optionally adds `BBoxSpectralExtractor` for spectral-aware tracking.

## Run folder naming

- `--output-dir` is the parent/root directory.
- Final run folder is `<output-dir>/<out-basename or input-file-stem>`.
- Default basename:
  - RGB script: `Path(video_path).stem`
  - HSI script: `Path(cu3s_path).stem`
- Use `--out-basename` to override the default stem.
- `--out-basename` must be a single folder name (not a path).
- Re-running on the same input stem reuses the same run folder unless you change
  `--out-basename` or `--output-dir`.

## Quick start (RGB video)

```powershell
uv run python examples/object_tracking/bytetrack/yolo_bytetrack_rgb.py `
  --video-path "D:\data\video.mp4" `
  --output-dir tracking_output `
  --end-frame 60
```

## Quick start (HSI baseline)

```powershell
uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py `
  --cu3s-path "D:\data\cube.cu3s" `
  --output-dir tracking_output `
  --end-frame 60
```

## Quick start (HSI spectral)

```powershell
uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py `
  --cu3s-path "D:\data\cube.cu3s" `
  --association-mode spectral_cost `
  --spectral-cost-weight 0.3 `
  --output-dir tracking_output `
  --out-basename spectral_run01 `
  --end-frame 60
```

Outputs:
- `tracking_output/<run_name>/detection_results.json` — COCO-format detections (pre-tracking)
- `tracking_output/<run_name>/tracking_results.json` — COCO-format tracks (post-tracking)
- `tracking_output/<run_name>/tracking_overlay.mp4` — overlay video with frame IDs (top-left) and track IDs (bbox labels)
  - `<run_name>` defaults to the input stem, or `--out-basename` if provided

## Threshold reference

| Stage | Parameter | Default | CLI flag | Notes |
|-------|-----------|---------|----------|-------|
| YOLOPostprocess | `confidence_threshold` | 0.1 | `--confidence-threshold` | Primary gate before tracking. |
| YOLOPostprocess | `iou_threshold` | 0.7 | `--iou-threshold` | IoU used by NMS. |
| YOLOPostprocess | `agnostic_nms` | False | `--agnostic-nms` | Class-aware NMS by default. |
| YOLOPostprocess | `classes` | all | `--classes <id> ...` | Filter to specific COCO class ids. |
| ByteTrack | `track_thresh` | 0.5 | `--track-thresh` | Min confidence for track association. |
| ByteTrack | `match_thresh` | 0.8 | `--match-thresh` | IoU+score cost threshold (higher = more permissive). |
| ByteTrack | `track_buffer` | 30 | `--track-buffer` | Frames to keep lost tracks alive. |
| ByteTrack | `second_score_thresh` | 0.1 | `--second-score-thresh` | Low-score floor for second association. |
| ByteTrack | `second_match_thresh` | 0.5 | `--second-match-thresh` | IoU threshold for second association. |
| ByteTrack | `unconfirmed_match_thresh` | 0.7 | `--unconfirmed-match-thresh` | IoU threshold for unconfirmed tracks. |
| ByteTrack | `new_track_thresh_offset` | 0.1 | `--new-track-thresh-offset` | `det_thresh = track_thresh + offset` for new track activation. |

## Spectral association options (HSI only)

| Parameter | Default | CLI flag | Notes |
|-----------|---------|----------|-------|
| `association_mode` | baseline | `--association-mode` | `baseline`, `spectral_cost`, or `spectral_post_gate`. |
| `spectral_cost_weight` | 0.3 | `--spectral-cost-weight` | Weight for spectral cost in all association stages. |
| `prototype_ema_beta` | 0.1 | `--prototype-ema-beta` | EMA weight for track spectral prototypes. |
| `prototype_min_sim` | 0.5 | `--prototype-min-sim` | Min cosine similarity for prototype matching. |
| `prototype_min_det_score` | 0.3 | `--prototype-min-det-score` | Min detection score to update prototype. |
| `spectral_center_crop` | 0.65 | `--spectral-center-crop` | Center-crop scale for spectral extraction. |
| `spectral_sim_floor` | 0.4 | `--spectral-sim-floor` | Similarity floor for post-gate mode. |
| sparklines | off | `--draw-spectral-sparklines` | Draw spectral sparklines on overlay bboxes. |
| `sparkline_height` | 24 | `--sparkline-height` | Pixel height of sparkline bars. |

## Hyperparameter sweep

The sweep uses a two-stage grid strategy defined in `run_bytetrack_sweep.py`:

- **Stage A** (default): baseline detection & tracking thresholds (~375 runs)
- **Stage B**: spectral sweep on reduced threshold subset × spectral grid

### Run the sweep

```powershell
# Stage A — baseline thresholds only
uv run python examples/object_tracking/bytetrack/run_bytetrack_sweep.py `
  --cu3s-path "D:\data\cube.cu3s" `
  --output-root "D:\data\tracker\bytetrack_sweep" `
  --sweep-stage A `
  --num-workers 3 `
  --resume

# Stage B — spectral params on reduced threshold subset
uv run python examples/object_tracking/bytetrack/run_bytetrack_sweep.py `
  --cu3s-path "D:\data\cube.cu3s" `
  --output-root "D:\data\tracker\bytetrack_sweep" `
  --sweep-stage B `
  --num-workers 3 `
  --resume
```

Key flags:
| Flag | Description |
|------|-------------|
| `--sweep-stage A\|B\|AB` | Which sweep stage to run (default A). |
| `--num-workers N` | Parallel jobs (default 1 = sequential). |
| `--max-frames N` | Frames per run (default 60). |
| `--limit-runs N` | Run only first N combos (smoke test). |
| `--grid-filter "<regex>"` | Include only matching run_ids. |
| `--resume` | Skip runs that already have outputs. |
| `--overwrite` | Force rerun even if outputs exist. |
| `--bf16` | Enable bfloat16 autocast. |

Each run produces `<output-root>/<run_id>/` containing:
- `detection_results.json`, `tracking_results.json`
- `tracking_overlay.mp4`
- `run_meta.json` (params, status, command)
- `metrics.json` (per-frame stats)

### Generate results table

After the sweep completes, build the interactive HTML + CSV:

```powershell
uv run python examples/object_tracking/bytetrack/regenerate_bytetrack_sweep_html.py `
  --results-root "D:\data\tracker\bytetrack_sweep"
```

Outputs:
- `results.html` — sortable/filterable interactive table
- `sweep_metrics.csv` — flat CSV for analysis in pandas/Excel

The HTML loads from (in priority order):
1. `sweep_metrics.csv`
2. `sweep_manifest.json`
3. Per-run subdirectories (`run_meta.json` + `metrics.json`)

### Re-encode overlay videos (in place)

To convert existing sweep overlays to a browser-friendly codec without rerunning inference:

```powershell
uv run python examples/object_tracking/bytetrack/convert_overlays_in_place.py `
  --root "D:\data\tracker\bytetrack_sweep" `
  --dry-run
```

Then run the actual conversion:

```powershell
uv run python examples/object_tracking/bytetrack/convert_overlays_in_place.py `
  --root "D:\data\tracker\bytetrack_sweep" `
  --codec VP90 `
  --jobs 8 `
  --keep-backup
```

Notes:
- The script updates files in place (default pattern: `tracking_overlay.mp4`).
- With `--keep-backup`, original files are kept as `*.bak.mp4`.
- Output filenames stay `.mp4`; only the encoded video stream changes.
- Use `--limit N` for a small test batch before converting all files.
