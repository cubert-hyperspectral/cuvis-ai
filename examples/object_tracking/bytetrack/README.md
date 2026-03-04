# YOLO + ByteTrack thresholds (HSI pipeline)

Cheat sheet for every threshold that influences detections before they hit ByteTrack in `yolo_bytetrack_hsi.py`.

## Pipeline (where thresholds apply)
- CU3SData → CIETristimulusFalseRGBSelector → **YOLOPreprocess** → **YOLO26Detection** → **YOLOPostprocess (NMS)** → **ByteTrack** → overlay.

## Threshold knobs and defaults

| Stage | Parameter | Default | How to change (current CLI) | Notes |
| --- | --- | --- | --- | --- |
| YOLO26Detection | — | — | n/a | Emits raw logits; no filtering here. |
| YOLOPostprocess | `confidence_threshold` | 0.1 (CLI default) | `--confidence-threshold` | Primary gate before tracking. |
| YOLOPostprocess | `iou_threshold` | 0.7 | `--iou-threshold` | IoU used by NMS. |
| YOLOPostprocess | `max_detections` | 300 | not exposed | Ultralytics default; usually not limiting. |
| YOLOPostprocess | `agnostic_nms` | False | `--agnostic-nms` | Class-aware NMS by default. |
| YOLOPostprocess | `classes` | None | `--classes <id> [--classes <id>...]` | Keep only selected class ids; otherwise all. |
| ByteTrack | `track_thresh` | 0.5 | `--track-thresh` | Min conf for detections to start/continue tracks. |
| ByteTrack | `match_thresh` | 0.8 | `--match-thresh` | Association cost threshold after IoU+score fusion; higher is more permissive. |
| ByteTrack | `track_buffer` | 30 | `--track-buffer` | Frames to keep lost tracks alive. |
| ByteTrack | `second_score_thresh` | 0.1 | `--second-score-thresh` | Low-score floor for second association candidates. |
| ByteTrack | `second_match_thresh` | 0.5 | `--second-match-thresh` | IoU threshold for second association. |
| ByteTrack | `unconfirmed_match_thresh` | 0.7 | `--unconfirmed-match-thresh` | IoU threshold for unconfirmed-track matching. |
| ByteTrack | `new_track_thresh_offset` | 0.1 | `--new-track-thresh-offset` | Uses `det_thresh = track_thresh + offset` for new track activation. |
| ByteTrack | `mot20` | False | not exposed | Only change if using MOT20 tuning. |
| ByteTrack | `frame_rate` | from dataset fps | derived | Set by datamodule; not user-facing. |

## Quick guidance
- If ByteTrack could recover weaker detections, try lowering NMS confidence (e.g., `--confidence-threshold 0.25`) before relaxing tracker gates.
- Keep IoU at 0.7 unless you see excessive duplicate boxes; lowering increases merges and recall but may raise false positives.
- Adjust `track_thresh` (e.g., 0.25) only if tracks are still being dropped after lowering NMS confidence.

## Suggested validation runs
Run two passes on the same clip and compare detection/track counts per frame (overlay video or logs):
1. Baseline:\
   `uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py --cu3s-path <file.cu3s> --output-dir <out_dir>`
2. Lower NMS confidence:\
   `uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py --cu3s-path <file.cu3s> --output-dir <out_dir_low_conf> --confidence-threshold 0.25`
3. Optional tracker gate relaxation:\
   `uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py --cu3s-path <file.cu3s> --output-dir <out_dir_loose_tracker> --track-thresh 0.25`

Focus on whether targets of interest persist across frames and whether false positives become problematic.

## Sweep runner (grid search)
Use `run_bytetrack_sweep.py` to explore a grid of thresholds and tracker settings and generate an interactive HTML table of results.

Example (60-frame subset, default grid of 336 combos):
```powershell
python examples/object_tracking/bytetrack/run_bytetrack_sweep.py `
  --cu3s-path "D:\data\XMR_notarget_Busstation\20260226\Auto_013+01.cu3s" `
  --output-root "D:\data\XMR_notarget_Busstation\20260226\tracker\hyper_param_search"
```

Key flags:
- `--limit-runs N` run only first N combos (for smoke tests).
- `--grid-filter "<regex>"` include only run_ids matching the regex (e.g., `tb60`).
- `--overwrite` force reruns even if outputs exist; `--resume` skips finished runs.
- `--max-frames` controls `--end-frame` passed to the tracking script (default 60).
- `--verbose` prints per-run command/output tails in addition to progress and per-run metrics.
- Sweep metadata now includes `second_score_thresh`, `second_match_thresh`, `unconfirmed_match_thresh`, and `new_track_thresh_offset`.

Outputs per run are stored under `<output-root>/<run_id>/` (JSON, MP4, metrics), with aggregates at `<output-root>/sweep_metrics.csv` and `<output-root>/results.html`.
