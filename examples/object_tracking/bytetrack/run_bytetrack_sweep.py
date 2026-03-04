"""Hyperparameter sweep runner for YOLO + ByteTrack HSI pipeline.

Runs `yolo_bytetrack_hsi.py` over a predefined grid of detection and tracking
thresholds, records outputs per run, computes simple tracking/detection metrics,
and writes aggregate CSV/Parquet plus an interactive HTML table.
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd

# Optional: plotly is only used for color scales; HTML interactivity relies on DataTables JS.
try:
    import plotly.express as px  # noqa: F401
except ImportError:  # pragma: no cover - plotly is optional
    px = None


# === Sweep space ===
CONFIDENCE_VALUES = [0.01, 0.04, 0.05, 0.10, 0.20, 0.30, 0.50]
IOU_VALUES = [0.10, 0.60]
TRACK_THRESH_VALUES = [0.10, 0.60]
MATCH_THRESH_VALUES = [0.10, 0.60]
TRACK_BUFFER_VALUES = [10, 30, 60]
AGNOSTIC_VALUES = [False, True]
SECOND_SCORE_THRESH_VALUES = [0.10]
SECOND_MATCH_THRESH_VALUES = [0.50]
UNCONFIRMED_MATCH_THRESH_VALUES = [0.70]
NEW_TRACK_THRESH_OFFSET_VALUES = [0.10]


# === Helpers ===
def _fmt_float_token(val: float) -> str:
    """Encode a float as a filesystem-friendly token (e.g., 0.10 -> 0p10)."""
    return f"{val:.2f}".replace(".", "p")


def _build_run_id(
    confidence_threshold: float,
    iou_threshold: float,
    track_thresh: float,
    match_thresh: float,
    track_buffer: int,
    agnostic_nms: bool,
    second_score_thresh: float,
    second_match_thresh: float,
    unconfirmed_match_thresh: float,
    new_track_thresh_offset: float,
) -> str:
    return (
        f"conf{_fmt_float_token(confidence_threshold)}"
        f"_iou{_fmt_float_token(iou_threshold)}"
        f"_tt{_fmt_float_token(track_thresh)}"
        f"_mt{_fmt_float_token(match_thresh)}"
        f"_tb{track_buffer}"
        f"_agn{int(agnostic_nms)}"
        f"_s2{_fmt_float_token(second_score_thresh)}"
        f"_m2{_fmt_float_token(second_match_thresh)}"
        f"_u2{_fmt_float_token(unconfirmed_match_thresh)}"
        f"_nt{_fmt_float_token(new_track_thresh_offset)}"
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _compute_metrics(det_json_path: Path, track_json_path: Path) -> dict[str, Any]:
    """Compute basic per-frame and run-level stats."""
    det_data = _read_json(det_json_path)
    track_data = _read_json(track_json_path)

    frame_ids = [img["id"] for img in det_data.get("images", [])]
    n_frames = len(frame_ids)
    if n_frames == 0:
        raise ValueError("No frames found in detection JSON.")

    det_counts = dict.fromkeys(frame_ids, 0)
    for ann in det_data.get("annotations", []):
        det_counts[ann["image_id"]] = det_counts.get(ann["image_id"], 0) + 1

    track_counts = dict.fromkeys(frame_ids, 0)
    track_ids = set()
    for ann in track_data.get("annotations", []):
        track_counts[ann["image_id"]] = track_counts.get(ann["image_id"], 0) + 1
        tid = ann.get("track_id")
        if tid is not None:
            track_ids.add(tid)

    metrics = {
        "frames": n_frames,
        "avg_dets_per_frame": sum(det_counts.values()) / n_frames,
        "max_dets_per_frame": max(det_counts.values(), default=0),
        "frames_with_zero_dets": sum(1 for v in det_counts.values() if v == 0),
        "avg_tracks_per_frame": sum(track_counts.values()) / n_frames,
        "max_tracks_per_frame": max(track_counts.values(), default=0),
        "frames_with_zero_tracks": sum(1 for v in track_counts.values() if v == 0),
        "unique_track_ids": len(track_ids),
        "total_annotations_det": len(det_data.get("annotations", [])),
        "total_annotations_track": len(track_data.get("annotations", [])),
    }
    return metrics


@dataclass
class RunParams:
    confidence_threshold: float
    iou_threshold: float
    track_thresh: float
    match_thresh: float
    track_buffer: int
    agnostic_nms: bool
    second_score_thresh: float
    second_match_thresh: float
    unconfirmed_match_thresh: float
    new_track_thresh_offset: float


def _generate_grid() -> Iterable[RunParams]:
    for conf, iou, tt, mt, tb, agn, s2, m2, u2, nt in itertools.product(
        CONFIDENCE_VALUES,
        IOU_VALUES,
        TRACK_THRESH_VALUES,
        MATCH_THRESH_VALUES,
        TRACK_BUFFER_VALUES,
        AGNOSTIC_VALUES,
        SECOND_SCORE_THRESH_VALUES,
        SECOND_MATCH_THRESH_VALUES,
        UNCONFIRMED_MATCH_THRESH_VALUES,
        NEW_TRACK_THRESH_OFFSET_VALUES,
    ):
        yield RunParams(conf, iou, tt, mt, tb, agn, s2, m2, u2, nt)


def _run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _maybe_filter(run_id: str, pattern: str | None) -> bool:
    if not pattern:
        return True
    return re.search(pattern, run_id) is not None


def _ensure_web_preview(overlay_path: Path, codec: str = "VP90") -> Path | None:
    """Create a browser-friendly `.webm` preview next to the overlay if possible."""
    if not overlay_path.exists():
        return None

    preview_path = overlay_path.with_suffix(".webm")
    if preview_path.exists() and preview_path.stat().st_mtime >= overlay_path.stat().st_mtime:
        return preview_path

    try:
        import cv2
    except Exception:
        return None

    cap = cv2.VideoCapture(str(overlay_path))
    if not cap.isOpened():
        cap.release()
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return None

    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        cap.release()
        return None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
    finally:
        writer.release()
        cap.release()

    return preview_path if preview_path.exists() else None


def _write_interactive_table(df: pd.DataFrame, output_html: Path) -> None:
    """Render a sortable/filterable HTML table via DataTables (client-side)."""
    preferred_order = [
        "run_id",
        "status",
        "unique_track_ids",
        "max_tracks_per_frame",
        "avg_tracks_per_frame",
        "avg_dets_per_frame",
        "frames_with_zero_tracks",
        "frames_with_zero_dets",
        "confidence_threshold",
        "iou_threshold",
        "track_thresh",
        "match_thresh",
        "track_buffer",
        "agnostic_nms",
        "second_score_thresh",
        "second_match_thresh",
        "unconfirmed_match_thresh",
        "new_track_thresh_offset",
        "wall_time_sec",
        "frames",
        "total_annotations_det",
        "total_annotations_track",
        "overlay_path",
        "detection_json",
        "tracking_json",
    ]
    existing_columns = list(df.columns)
    hidden_columns = {"overlay_web_path"}
    columns = [c for c in preferred_order if c in existing_columns] + [
        c for c in existing_columns if c not in preferred_order and c not in hidden_columns
    ]

    header_meta = {
        "run_id": ("Run", "Sweep run identifier"),
        "status": ("St", "Run status"),
        "unique_track_ids": ("IDs", "Unique tracking IDs in the run"),
        "max_tracks_per_frame": ("MaxTrk", "Maximum tracked boxes in any frame"),
        "avg_tracks_per_frame": ("AvgTrk", "Average tracked boxes per frame"),
        "avg_dets_per_frame": ("AvgDet", "Average detections per frame"),
        "frames_with_zero_tracks": ("ZeroTrk", "Frames without any track"),
        "frames_with_zero_dets": ("ZeroDet", "Frames without any detection"),
        "confidence_threshold": ("Conf", "YOLO confidence threshold"),
        "iou_threshold": ("IoU", "YOLO NMS IoU threshold"),
        "track_thresh": ("TThr", "ByteTrack track threshold"),
        "match_thresh": ("MThr", "ByteTrack match threshold"),
        "track_buffer": ("Buf", "ByteTrack track buffer"),
        "agnostic_nms": ("Agn", "Class-agnostic NMS enabled"),
        "second_score_thresh": ("S2Floor", "Second-association low score floor"),
        "second_match_thresh": ("S2IoU", "Second-association IoU threshold"),
        "unconfirmed_match_thresh": ("UncIoU", "Unconfirmed-track match threshold"),
        "new_track_thresh_offset": ("NewOff", "det_thresh = track_thresh + offset"),
        "wall_time_sec": ("Time", "Run wall time in seconds"),
        "frames": ("Frames", "Processed frame count"),
        "total_annotations_det": ("DetAnn", "Total detection annotations"),
        "total_annotations_track": ("TrkAnn", "Total tracking annotations"),
        "overlay_path": ("Video", "Overlay MP4 path"),
        "detection_json": ("DetJSON", "Detections JSON path"),
        "tracking_json": ("TrkJSON", "Tracking JSON path"),
    }

    def _to_file_uri(value: Any) -> str:
        if not isinstance(value, str) or not value:
            return ""
        try:
            p = Path(value)
            if p.is_absolute():
                return p.as_uri()
        except Exception:  # pragma: no cover
            pass
        return value.replace("\\", "/")

    def _format_value(val: Any) -> str:
        if val is None:
            return ""
        if isinstance(val, bool):
            return "True" if val else "False"
        if isinstance(val, str):
            return val
        try:
            if pd.isna(val):
                return ""
        except Exception:  # pragma: no cover
            pass
        if isinstance(val, Integral):
            return str(int(val))
        if isinstance(val, Real):
            fval = float(val)
            if math.isfinite(fval):
                return f"{fval:.2f}"
            return str(val)
        return str(val)

    numeric_cols = set(df.select_dtypes(include=["number"]).columns.tolist())

    header_cells = []
    for col in columns:
        short, tip = header_meta.get(col, (col, col))
        header_cells.append(
            f'<th data-col-id="{html.escape(col, quote=True)}" data-bs-toggle="tooltip" data-bs-title="{html.escape(tip, quote=True)}">{html.escape(short, quote=False)}</th>'
        )
    filter_cells = [f'<th data-col-id="{html.escape(col, quote=True)}"></th>' for col in columns]

    body_rows = []
    for _, row in df.iterrows():
        overlay_uri = _to_file_uri(row.get("overlay_web_path")) or _to_file_uri(
            row.get("overlay_path")
        )
        overlay_raw_uri = _to_file_uri(row.get("overlay_path"))
        run_id = _format_value(row.get("run_id", ""))
        cells = []
        for col in columns:
            val = row[col]
            if col == "overlay_path":
                cells.append(
                    '<td><button type="button" class="btn btn-sm btn-outline-primary open-video-btn">Play</button></td>'
                )
            elif col in {"detection_json", "tracking_json"} and isinstance(val, str):
                href = _to_file_uri(val)
                cells.append(f'<td><a href="{html.escape(href, quote=True)}">open</a></td>')
            else:
                cell_text = _format_value(val)
                is_numeric = col in numeric_cols and not isinstance(val, bool)
                cell_cls = "numeric-cell" if is_numeric else ""
                cells.append(
                    f'<td class="{cell_cls}" title="Click metric to preview video">{html.escape(cell_text, quote=False)}</td>'
                )
        body_rows.append(
            '<tr data-video="{}" data-video-raw="{}" data-run-id="{}">{}</tr>'.format(
                html.escape(overlay_uri, quote=True),
                html.escape(overlay_raw_uri, quote=True),
                html.escape(run_id, quote=True),
                "".join(cells),
            )
        )

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ByteTrack Sweep Results</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/dataTables.bootstrap5.min.css">
  <style>
    body {{
      background: #f8f9fa;
      font-size: 13px;
    }}
    #results {{
      width: 100%;
    }}
    #results th, #results td {{
      white-space: nowrap;
      padding: 0.30rem 0.45rem;
      vertical-align: middle;
      font-variant-numeric: tabular-nums;
    }}
    #results td.numeric-cell {{
      color: #0d6efd;
      cursor: pointer;
    }}
    #results td.numeric-cell:hover {{
      text-decoration: underline;
    }}
    .dataTables_wrapper .dataTables_filter input {{
      margin-left: 0.5rem;
    }}
    #overlayVideo {{
      width: 100%;
      max-height: 75vh;
      background: #000;
    }}
  </style>
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.8/js/dataTables.bootstrap5.min.js"></script>
</head>
<body>
  <div class="container-fluid py-3">
    <h5 class="mb-3">ByteTrack Sweep Results</h5>
    <div class="card shadow-sm">
      <div class="card-body p-2">
        <table id="results" class="table table-striped table-hover table-sm">
          <thead>
            <tr>{"".join(header_cells)}</tr>
            <tr class="filters">{"".join(filter_cells)}</tr>
          </thead>
          <tbody>
            {"".join(body_rows)}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="modal fade" id="overlayModal" tabindex="-1" aria-hidden="true">
    <div class="modal-dialog modal-xl modal-dialog-centered">
      <div class="modal-content bg-dark text-light">
        <div class="modal-header py-2">
          <h6 class="modal-title" id="overlayModalLabel">Overlay Preview</h6>
          <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body p-0">
          <video id="overlayVideo" controls autoplay></video>
          <div id="overlayError" class="alert alert-warning m-2 d-none" role="alert">
            This video cannot be played in the browser codec pipeline. Use the file path in the table to open it in a local player.
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    $(document).ready(function() {{
        const table = $('#results').DataTable({{
            pageLength: 25,
            order: [[0, 'asc']],
            scrollX: true,
            orderCellsTop: true
        }});

        const tooltipList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipList.forEach((el) => new bootstrap.Tooltip(el));

        const nonFilterableCols = new Set(['overlay_path', 'detection_json', 'tracking_json']);
        table.columns().every(function(colIdx) {{
            const colHeader = $(table.column(colIdx).header());
            const colId = (colHeader.data('col-id') || '').toString();
            const filterCell = $('.filters th').eq(colIdx);
            filterCell.empty();

            if (nonFilterableCols.has(colId)) {{
                return;
            }}

            const rawData = table
                .column(colIdx)
                .data()
                .toArray()
                .map((x) => $('<div>').html(String(x)).text().trim())
                .filter((x) => x.length > 0);
            const uniqueValues = Array.from(new Set(rawData)).sort((a, b) => a.localeCompare(b, undefined, {{ numeric: true }}));

            if (uniqueValues.length > 0 && uniqueValues.length <= 25) {{
                const select = $('<select class="form-select form-select-sm"><option value="">All</option></select>');
                uniqueValues.forEach((v) => {{
                    select.append(`<option value="${{v.replace(/"/g, '&quot;')}}">${{v}}</option>`);
                }});
                select.appendTo(filterCell).on('change', function() {{
                    const val = $(this).val();
                    const escaped = $.fn.dataTable.util.escapeRegex(val);
                    table.column(colIdx).search(val ? '^' + escaped + '$' : '', true, false).draw();
                }});
            }} else {{
                const input = $('<input type="text" class="form-control form-control-sm" placeholder="filter">');
                input.appendTo(filterCell).on('keyup change', function() {{
                    table.column(colIdx).search(this.value).draw();
                }});
            }}
        }});

        const modalEl = document.getElementById('overlayModal');
        const modal = new bootstrap.Modal(modalEl, {{ backdrop: true, keyboard: true }});
        const videoEl = document.getElementById('overlayVideo');
        const titleEl = document.getElementById('overlayModalLabel');
        const errorEl = document.getElementById('overlayError');
        let rawFallbackTried = false;
        let fallbackRawSrc = '';

        function openVideo(rowEl) {{
            const videoSrc = rowEl.getAttribute('data-video');
            fallbackRawSrc = rowEl.getAttribute('data-video-raw') || '';
            rawFallbackTried = false;
            if (!videoSrc) return;
            errorEl.classList.add('d-none');
            videoEl.src = videoSrc;
            titleEl.textContent = rowEl.getAttribute('data-run-id') || 'Overlay Preview';
            modal.show();
            videoEl.play().catch(() => {{}});
        }}

        $('#results tbody').on('click', 'td.numeric-cell', function() {{
            const row = this.closest('tr');
            if (!row) return;
            openVideo(row);
        }});

        $('#results tbody').on('click', 'button.open-video-btn', function(event) {{
            event.preventDefault();
            event.stopPropagation();
            const row = this.closest('tr');
            if (!row) return;
            openVideo(row);
        }});

        videoEl.addEventListener('error', function() {{
            if (!rawFallbackTried && fallbackRawSrc && videoEl.src !== fallbackRawSrc) {{
                rawFallbackTried = true;
                videoEl.src = fallbackRawSrc;
                videoEl.play().catch(() => {{}});
                return;
            }}
            errorEl.classList.remove('d-none');
        }});

        modalEl.addEventListener('hidden.bs.modal', function () {{
            videoEl.pause();
            videoEl.removeAttribute('src');
            videoEl.load();
            errorEl.classList.add('d-none');
            rawFallbackTried = false;
            fallbackRawSrc = '';
        }});
    }});
  </script>
</body>
</html>"""
    output_html.write_text(html_doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a ByteTrack HSI hyperparameter sweep.")
    parser.add_argument("--cu3s-path", type=Path, required=True, help="Input CU3S file.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("hyper_param_search"),
        help="Root directory to store all run outputs.",
    )
    parser.add_argument(
        "--model-name", type=str, default="yolo26n.pt", help="YOLO model name/path."
    )
    parser.add_argument(
        "--plugins-dir", type=Path, default=None, help="Optional plugins directory."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Limit number of frames to process (end_frame). Default: 60.",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=None,
        help="Run only the first N parameter combinations (after filtering).",
    )
    parser.add_argument(
        "--grid-filter",
        type=str,
        default=None,
        help="Regex to filter run_id values (applied before limit).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if outputs already exist in the run directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already have tracking_results.json (default skip unless --overwrite).",
    )
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 autocast for inference.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra per-run details (command and output tails).",
    )
    parser.add_argument(
        "--generate-web-previews",
        action="store_true",
        help="Generate browser-friendly .webm previews from overlay MP4 files.",
    )
    parser.add_argument(
        "--web-preview-codec",
        type=str,
        default="VP90",
        help="FourCC codec for generated web previews (default: VP90).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "examples" / "object_tracking" / "bytetrack" / "yolo_bytetrack_hsi.py"

    _ensure_dir(args.output_root)

    rows: list[dict[str, Any]] = []

    grid_iter: Iterable[RunParams] | list[RunParams] = _generate_grid()
    if args.grid_filter:
        grid_iter = [
            gp for gp in grid_iter if _maybe_filter(_build_run_id(**gp.__dict__), args.grid_filter)
        ]

    if args.limit_runs is not None:
        grid_iter = list(itertools.islice(grid_iter, args.limit_runs))

    grid_items = list(grid_iter)
    total_runs = len(grid_items)
    if total_runs == 0:
        print("No runs executed or found after filtering.")
        return 0

    print(f"Scheduled runs: {total_runs}")
    overall_start = time.time()
    ok_count = 0
    fail_count = 0
    skip_count = 0
    preview_count = 0
    preview_fail_count = 0

    for index, params in enumerate(grid_items, start=1):
        run_id = _build_run_id(
            params.confidence_threshold,
            params.iou_threshold,
            params.track_thresh,
            params.match_thresh,
            params.track_buffer,
            params.agnostic_nms,
            params.second_score_thresh,
            params.second_match_thresh,
            params.unconfirmed_match_thresh,
            params.new_track_thresh_offset,
        )
        print(f"[{index}/{total_runs}] {run_id}")
        run_dir = args.output_root / run_id
        _ensure_dir(run_dir)

        det_path = run_dir / "detection_results.json"
        track_path = run_dir / "tracking_results.json"
        overlay_path = run_dir / "tracking_overlay.mp4"

        should_run = True
        if det_path.exists() and track_path.exists() and not args.overwrite:
            should_run = False
        if not args.overwrite and args.resume and track_path.exists():
            should_run = False

        start_ts = time.time()
        status = "skipped"
        stdout = stderr = ""
        cmd: list[str] | None = None

        if should_run:
            cmd = [
                sys.executable,
                str(script_path),
                "--cu3s-path",
                str(args.cu3s_path),
                "--output-dir",
                str(run_dir),
                "--end-frame",
                str(args.max_frames),
                "--model-name",
                args.model_name,
                "--confidence-threshold",
                str(params.confidence_threshold),
                "--iou-threshold",
                str(params.iou_threshold),
                "--track-thresh",
                str(params.track_thresh),
                "--match-thresh",
                str(params.match_thresh),
                "--track-buffer",
                str(params.track_buffer),
                "--second-score-thresh",
                str(params.second_score_thresh),
                "--second-match-thresh",
                str(params.second_match_thresh),
                "--unconfirmed-match-thresh",
                str(params.unconfirmed_match_thresh),
                "--new-track-thresh-offset",
                str(params.new_track_thresh_offset),
                "--classes",
                "0",
            ]
            if params.agnostic_nms:
                cmd.append("--agnostic-nms")
            if args.plugins_dir:
                cmd.extend(["--plugins-dir", str(args.plugins_dir)])
            if args.bf16:
                cmd.append("--bf16")

            if args.verbose:
                print(f"  cmd: {' '.join(cmd)}")
            proc = _run_command(cmd, cwd=repo_root)
            stdout, stderr = proc.stdout, proc.stderr
            status = "ok" if proc.returncode == 0 else f"fail({proc.returncode})"
            if proc.returncode != 0:
                fail_count += 1
                if args.verbose:
                    if stdout:
                        print(f"  stdout_tail: {stdout[-400:]}")
                    if stderr:
                        print(f"  stderr_tail: {stderr[-400:]}")
                # Record failure and continue to next run without metrics.
                rows.append(
                    {
                        "run_id": run_id,
                        "status": status,
                        "wall_time_sec": time.time() - start_ts,
                        "stderr_tail": stderr[-400:],
                        "stdout_tail": stdout[-400:],
                        **params.__dict__,
                    }
                )
                run_elapsed = time.time() - start_ts
                overall_elapsed = time.time() - overall_start
                print(
                    f"  -> {status} | run={run_elapsed:.1f}s | elapsed={overall_elapsed:.1f}s | ok={ok_count} fail={fail_count} skipped={skip_count}"
                )
                continue

        # Attempt metrics even if skipped (assuming prior successful run).
        metrics = {}
        try:
            metrics = _compute_metrics(det_path, track_path)
            status = status if status != "skipped" else "existing"
        except Exception as exc:  # pylint: disable=broad-except
            status = f"metrics_failed: {exc}"

        overlay_web_path = ""
        if args.generate_web_previews:
            preview_path = _ensure_web_preview(overlay_path, codec=args.web_preview_codec)
            if preview_path is not None:
                overlay_web_path = str(preview_path)
                preview_count += 1
            elif overlay_path.exists():
                preview_fail_count += 1

        rows.append(
            {
                "run_id": run_id,
                "status": status,
                "wall_time_sec": time.time() - start_ts,
                "overlay_path": str(overlay_path),
                "overlay_web_path": overlay_web_path,
                "detection_json": str(det_path),
                "tracking_json": str(track_path),
                **params.__dict__,
                **metrics,
            }
        )

        # Persist per-run metrics for traceability.
        (run_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "params": params.__dict__,
                    "command": cmd if should_run else None,
                    "status": status,
                    "stdout_tail": stdout[-400:],
                    "stderr_tail": stderr[-400:],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if status.startswith("ok"):
            ok_count += 1
        elif status.startswith("fail") or status.startswith("metrics_failed"):
            fail_count += 1
        else:
            skip_count += 1

        if args.verbose:
            if stdout:
                print(f"  stdout_tail: {stdout[-400:]}")
            if stderr:
                print(f"  stderr_tail: {stderr[-400:]}")
        if metrics:
            print(
                "  metrics: unique_ids={} max_tracks={} avg_tracks={:.3f} avg_dets={:.3f} zero_track_frames={}".format(
                    metrics.get("unique_track_ids", 0),
                    metrics.get("max_tracks_per_frame", 0),
                    float(metrics.get("avg_tracks_per_frame", 0.0)),
                    float(metrics.get("avg_dets_per_frame", 0.0)),
                    metrics.get("frames_with_zero_tracks", 0),
                )
            )
        run_elapsed = time.time() - start_ts
        overall_elapsed = time.time() - overall_start
        print(
            f"  -> {status} | run={run_elapsed:.1f}s | elapsed={overall_elapsed:.1f}s | ok={ok_count} fail={fail_count} skipped={skip_count}"
        )
        if args.generate_web_previews:
            print(f"  previews: ready={preview_count} failed={preview_fail_count}")

    df = pd.DataFrame(rows)
    csv_path = args.output_root / "sweep_metrics.csv"
    parquet_path = args.output_root / "sweep_metrics.parquet"
    html_path = args.output_root / "results.html"

    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:  # pragma: no cover - pyarrow/fastparquet may be missing
        parquet_path = None
    _write_interactive_table(df, html_path)

    manifest_path = args.output_root / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows)} rows to {csv_path}")
    if parquet_path:
        print(f"Wrote Parquet: {parquet_path}")
    print(f"Interactive table: {html_path}")
    print(f"Final counts -> ok={ok_count}, fail={fail_count}, skipped={skip_count}")
    if args.generate_web_previews:
        print(f"Web previews -> ready={preview_count}, failed={preview_fail_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
