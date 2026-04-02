"""Regenerate ByteTrack sweep HTML from an existing results folder.

This helper avoids rerunning the full sweep. It loads data in this order:
1) `sweep_metrics.csv`
2) `sweep_manifest.json`
3) per-run `run_meta.json` + `metrics.json` inside result subfolders

Usage:
    uv run python examples/object_tracking/bytetrack/regenerate_bytetrack_sweep_html.py \
        --results-root "D:\\data\\XMR_notarget_Busstation\\20260226\\tracker\\bytetrack_sweep"
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_from_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _load_from_manifest(manifest_path: Path) -> pd.DataFrame:
    raw = _read_json(manifest_path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON list in {manifest_path}")
    return pd.DataFrame(raw)


def _guess_overlay_path(run_dir: Path) -> str:
    webm = run_dir / "tracking_overlay.webm"
    mp4 = run_dir / "tracking_overlay.mp4"
    if webm.exists():
        return str(webm)
    if mp4.exists():
        return str(mp4)
    return str(webm)


def _load_from_run_dirs(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        run_meta_path = child / "run_meta.json"
        metrics_path = child / "metrics.json"
        if not run_meta_path.exists() and not metrics_path.exists():
            continue

        row: dict[str, Any] = {"run_id": child.name}
        if run_meta_path.exists():
            meta = _read_json(run_meta_path)
            if isinstance(meta, dict):
                row.update(
                    {
                        k: v
                        for k, v in meta.items()
                        if k not in {"params", "command", "stdout_tail", "stderr_tail"}
                    }
                )
                params = meta.get("params")
                if isinstance(params, dict):
                    row.update(params)
        if metrics_path.exists():
            metrics = _read_json(metrics_path)
            if isinstance(metrics, dict):
                row.update(metrics)

        row.setdefault("status", "existing")
        row.setdefault("overlay_path", _guess_overlay_path(child))
        row.setdefault("detection_json", str(child / "detection_results.json"))
        row.setdefault("tracking_json", str(child / "tracking_results.json"))
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No run folders with run_meta.json or metrics.json found under: {results_root}"
        )

    return pd.DataFrame(rows)


def _load_dataframe(results_root: Path) -> tuple[pd.DataFrame, str]:
    csv_path = results_root / "sweep_metrics.csv"
    if csv_path.exists():
        return _load_from_csv(csv_path), str(csv_path)

    manifest_path = results_root / "sweep_manifest.json"
    if manifest_path.exists():
        return _load_from_manifest(manifest_path), str(manifest_path)

    return _load_from_run_dirs(results_root), "run directories"


def _overwrite_text_file(path: Path, content: str) -> None:
    """Atomically overwrite a text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _overwrite_csv_file(df: pd.DataFrame, path: Path) -> None:
    """Atomically overwrite a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _write_interactive_table(df: pd.DataFrame, output_html: Path) -> None:
    """Render an interactive HTML table with per-column select filters."""
    if "tracks_whole_video" not in df.columns and "unique_track_ids" in df.columns:
        df = df.copy()
        df["tracks_whole_video"] = df["unique_track_ids"]

    id_col = "__display_id__"
    hyperparam_columns = [
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
        "association_mode",
        "spectral_cost_weight",
        "prototype_ema_beta",
    ]
    metric_columns = [
        "status",
        "tracks_whole_video",
        "max_tracks_per_frame",
        "avg_tracks_per_frame",
        "avg_dets_per_frame",
        "max_dets_per_frame",
        "frames_with_zero_tracks",
        "frames_with_zero_dets",
        "frames",
        "total_annotations_det",
        "total_annotations_track",
        "wall_time_sec",
    ]
    artifact_columns = ["detection_json", "tracking_json"]
    preferred_order = [*metric_columns, *hyperparam_columns, *artifact_columns]

    existing_columns = list(df.columns)
    hidden_columns = {"overlay_web_path", "overlay_path", "run_id", "unique_track_ids"}
    ordered_columns = [c for c in preferred_order if c in existing_columns] + [
        c for c in existing_columns if c not in preferred_order and c not in hidden_columns
    ]
    columns = [id_col] + ordered_columns
    hyperparam_set = set(hyperparam_columns)
    metric_set = set(metric_columns)
    artifact_set = set(artifact_columns)
    first_hyper_col = next((c for c in columns if c in hyperparam_set), None)

    header_meta = {
        id_col: ("ID", "Incrementing run index. Click an ID cell to open the run folder."),
        "status": ("St", "Run status. Use non-ok rows as invalid for model comparison."),
        "tracks_whole_video": (
            "#Tracks",
            "Maximum unique track IDs across the whole video (run-level track count).",
        ),
        "max_tracks_per_frame": (
            "MaxTrk",
            "Peak simultaneous tracks in a frame. Spikes can indicate duplicates/false positives.",
        ),
        "avg_tracks_per_frame": (
            "AvgTrk",
            "Average active tracks per frame. Compare against expected object count in scene.",
        ),
        "avg_dets_per_frame": (
            "AvgDet",
            "Average detections per frame before/with tracking. High values may include extra false positives.",
        ),
        "max_dets_per_frame": (
            "MaxDet",
            "Maximum detections in a frame. Large spikes may indicate noisy detection behavior.",
        ),
        "frames_with_zero_tracks": (
            "ZeroTrk",
            "Frames with no active tracks. Lower is better if targets are expected continuously.",
        ),
        "frames_with_zero_dets": (
            "ZeroDet",
            "Frames with no detections. Lower values mean fewer complete misses.",
        ),
        "frames": (
            "Frames",
            "Processed frame count. Compare rows only when frame counts are equal.",
        ),
        "total_annotations_det": ("DetAnn", "Total detections written to JSON."),
        "total_annotations_track": ("TrkAnn", "Total tracked boxes written to JSON."),
        "wall_time_sec": (
            "Time",
            "Runtime in seconds. Useful for speed/accuracy tradeoff selection.",
        ),
        "confidence_threshold": (
            "Conf",
            "YOLO confidence threshold. Higher cuts weak detections (fewer false positives, more misses).",
        ),
        "iou_threshold": (
            "IoU",
            "YOLO NMS IoU threshold. Higher keeps overlapping boxes; lower suppresses more duplicates.",
        ),
        "track_thresh": (
            "TThr",
            "ByteTrack high-score threshold for primary association. Higher is stricter, often fewer tracks.",
        ),
        "match_thresh": (
            "MThr",
            "IoU needed to match a detection to a track. Higher reduces bad matches but increases fragmentation.",
        ),
        "track_buffer": (
            "Buf",
            "Frames a lost track is kept alive. Higher helps through occlusion but can keep stale tracks longer.",
        ),
        "agnostic_nms": (
            "Agn",
            "Class-agnostic NMS flag. Can change duplicate suppression behavior across classes.",
        ),
        "second_score_thresh": (
            "S2Floor",
            "Low-score floor for ByteTrack second association. Lower recovers weak objects but can add noise.",
        ),
        "second_match_thresh": (
            "S2IoU",
            "IoU threshold for second association. Higher means stricter low-score matching.",
        ),
        "unconfirmed_match_thresh": (
            "UncIoU",
            "IoU threshold for matching unconfirmed tracks. Higher reduces accidental confirmations.",
        ),
        "new_track_thresh_offset": (
            "NewOff",
            "New-track threshold offset where det_thresh = track_thresh + offset. Higher creates fewer new IDs.",
        ),
        "association_mode": (
            "AMode",
            "ByteTrack association mode: baseline (IoU only), spectral_cost, or spectral_post_gate.",
        ),
        "spectral_cost_weight": (
            "SCW",
            "Spectral cost weight blended into association stages. Only active in spectral modes.",
        ),
        "prototype_ema_beta": (
            "EMA",
            "EMA beta for track spectral prototype updates. Only active in spectral modes.",
        ),
        "detection_json": (
            "DetJSON",
            "Detection JSON output path (cell click still opens video preview).",
        ),
        "tracking_json": (
            "TrkJSON",
            "Tracking JSON output path (cell click still opens video preview).",
        ),
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

    def _infer_run_folder_uri(row: pd.Series) -> str:
        for key in ("tracking_json", "detection_json", "overlay_path", "overlay_web_path"):
            raw = row.get(key)
            if not isinstance(raw, str) or not raw:
                continue
            try:
                candidate = Path(raw)
            except Exception:  # pragma: no cover
                continue
            folder = candidate if candidate.is_dir() else candidate.parent
            if not str(folder):
                continue
            folder_uri = _to_file_uri(str(folder))
            if folder_uri:
                return folder_uri
        return ""

    def _col_style_classes(col: str) -> str:
        classes: list[str] = []
        if col in metric_set or col == id_col:
            classes.append("metric-col")
        elif col in hyperparam_set:
            classes.append("hyper-col")
        elif col in artifact_set:
            classes.append("artifact-col")
        if col == first_hyper_col:
            classes.append("section-divider")
        return " ".join(classes)

    def _display_cell_text(col: str, val: Any, display_idx: int | None = None) -> str:
        if col == id_col:
            return "" if display_idx is None else str(display_idx)
        text = _format_value(val)
        if col in {"detection_json", "tracking_json"} and isinstance(val, str):
            try:
                return Path(val).name
            except Exception:  # pragma: no cover
                return _format_value(val)
        return text

    def _natural_sort_key(text: str) -> list[Any]:
        parts = re.split(r"(\d+)", text)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    numeric_cols = set(df.select_dtypes(include=["number"]).columns.tolist())
    filter_values: dict[str, set[str]] = {col: set() for col in columns}

    body_rows = []
    for display_idx, (_, row) in enumerate(df.iterrows(), start=1):
        overlay_uri = _to_file_uri(row.get("overlay_web_path")) or _to_file_uri(
            row.get("overlay_path")
        )
        overlay_raw_uri = _to_file_uri(row.get("overlay_path"))
        run_id = _format_value(row.get("run_id", ""))
        run_folder_uri = _infer_run_folder_uri(row)
        cells = []

        for col in columns:
            if col == id_col:
                hint = f"Click to open folder.\nRun: {run_id}"
                cell_classes = "id-cell"
                col_classes = _col_style_classes(col)
                if col_classes:
                    cell_classes = f"{cell_classes} {col_classes}"
                id_text = _display_cell_text(col, None, display_idx)
                if id_text:
                    filter_values[col].add(id_text)
                cells.append(
                    f'<td class="{cell_classes}" data-run-folder="{html.escape(run_folder_uri, quote=True)}" title="{html.escape(hint, quote=True)}">{display_idx}</td>'
                )
                continue

            val = row.get(col)
            cell_text = _display_cell_text(col, val)
            if cell_text:
                filter_values[col].add(cell_text)

            base_tip = header_meta.get(col, (col, col))[1]
            cell_tip = f"Click to open video.\n{base_tip}"
            classes = ["video-cell"]
            if col in numeric_cols and not isinstance(val, bool):
                classes.append("numeric-cell")
            col_classes = _col_style_classes(col)
            if col_classes:
                classes.extend(col_classes.split())
            cells.append(
                '<td class="{}" title="{}">{}</td>'.format(
                    " ".join(classes),
                    html.escape(cell_tip, quote=True),
                    html.escape(cell_text, quote=False),
                )
            )

        body_rows.append(
            '<tr data-video="{}" data-video-raw="{}" data-run-id="{}" data-run-folder="{}">{}</tr>'.format(
                html.escape(overlay_uri, quote=True),
                html.escape(overlay_raw_uri, quote=True),
                html.escape(run_id, quote=True),
                html.escape(run_folder_uri, quote=True),
                "".join(cells),
            )
        )

    header_cells = []
    for col_idx, col in enumerate(columns):
        short, tip = header_meta.get(col, (col, col))
        col_classes = _col_style_classes(col)
        class_attr = f' class="{col_classes}"' if col_classes else ""
        options = "".join(
            f'<option value="{html.escape(v, quote=True)}">{html.escape(v, quote=False)}</option>'
            for v in sorted(filter_values.get(col, set()), key=_natural_sort_key)
        )
        header_cells.append(
            f'<th{class_attr} data-col-id="{html.escape(col, quote=True)}" data-bs-toggle="tooltip" data-bs-title="{html.escape(tip, quote=True)}"><div class="th-label">{html.escape(short, quote=False)}</div><div class="th-filter" data-col-id="{html.escape(col, quote=True)}"><select class="form-select form-select-sm col-filter-select" data-col-id="{html.escape(col, quote=True)}" data-col-idx="{col_idx}"><option value="">All</option>{options}</select></div></th>'
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
    #results thead tr:first-child th {{
      position: sticky;
      top: 0;
      z-index: 4;
      background: #f8f9fa;
    }}
    #results th .th-label {{
      line-height: 1.05;
      margin-bottom: 0.18rem;
      min-height: 16px;
    }}
    #results th .th-filter {{
      min-width: 92px;
    }}
    #results th.metric-col, #results td.metric-col {{
      background-color: rgba(13, 110, 253, 0.05);
    }}
    #results th.hyper-col, #results td.hyper-col {{
      background-color: rgba(255, 193, 7, 0.12);
    }}
    #results th.section-divider, #results td.section-divider {{
      border-left: 3px solid #6c757d !important;
    }}
    #results td.video-cell {{
      color: #0d6efd;
      cursor: pointer;
    }}
    #results td.video-cell:hover {{
      text-decoration: underline;
    }}
    #results td.id-cell {{
      color: #198754;
      cursor: pointer;
      font-weight: 600;
      text-decoration: underline;
    }}
    #results th .th-filter select {{
      width: 100%;
      min-width: 92px;
      max-width: 180px;
      padding-top: 0.08rem;
      padding-bottom: 0.08rem;
      font-size: 11px;
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
    <h5 class="mb-2">ByteTrack Sweep Results</h5>
    <div class="small text-muted mb-2">Click any non-ID cell to open video. Click ID to open the run folder. Blue columns are metrics, yellow columns are hyperparameters.</div>
    <div class="card shadow-sm">
      <div class="card-body p-2">
        <table id="results" class="table table-striped table-hover table-sm">
          <thead>
            <tr>{"".join(header_cells)}</tr>
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
            This video cannot be played in the browser codec pipeline. Use the run folder path from the ID cell.
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    $(document).ready(function() {{
        const table = $('#results').DataTable({{
            paging: false,
            order: [[0, 'asc']],
            scrollX: true,
            autoWidth: false
        }});

        function adjustTableLayout() {{
            table.columns.adjust();
        }}
        adjustTableLayout();
        setTimeout(adjustTableLayout, 0);
        $(window).on('resize', adjustTableLayout);
        table.on('draw.dt', adjustTableLayout);

        $(document).on('mousedown click', '.col-filter-select', function(event) {{
            event.stopPropagation();
        }});

        $(document).on('change', '.col-filter-select', function(event) {{
            event.stopPropagation();
            const colIdx = Number(this.getAttribute('data-col-idx'));
            if (!Number.isFinite(colIdx)) return;
            const value = this.value || '';
            const escaped = $.fn.dataTable.util.escapeRegex(value);
            table.column(colIdx).search(value ? '^' + escaped + '$' : '', true, false).draw();
            const selector = '.col-filter-select[data-col-idx="' + colIdx + '"]';
            $(selector).not(this).val(value);
        }});

        const tooltipList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipList.forEach((el) => new bootstrap.Tooltip(el));

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

        function openFolder(cellEl) {{
            const rowEl = cellEl.closest('tr');
            if (!rowEl) return;
            const folderUri = cellEl.getAttribute('data-run-folder') || rowEl.getAttribute('data-run-folder');
            if (!folderUri) return;
            const tab = window.open(folderUri, '_blank');
            if (!tab) {{
                window.location.href = folderUri;
            }}
        }}

        $('#results tbody').on('click', 'td.id-cell', function(event) {{
            event.preventDefault();
            event.stopPropagation();
            openFolder(this);
        }});

        $('#results tbody').on('click', 'td.video-cell', function() {{
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
    _overwrite_text_file(output_html, html_doc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate ByteTrack sweep results.html from existing outputs."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Sweep output folder containing sweep_metrics.csv / sweep_manifest.json / run subfolders.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Output HTML path (default: <results-root>/results.html).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Also write a CSV summary (default: <results-root>/sweep_metrics.csv).",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    output_html = (args.output_html or (results_root / "results.html")).resolve()
    output_csv = (args.output_csv or (results_root / "sweep_metrics.csv")).resolve()

    df, source = _load_dataframe(results_root)
    _write_interactive_table(df, output_html)
    _overwrite_csv_file(df, output_csv)

    print(f"Loaded rows: {len(df)} from {source}")
    print(f"Wrote HTML:  {output_html}")
    print(f"Wrote CSV:   {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
