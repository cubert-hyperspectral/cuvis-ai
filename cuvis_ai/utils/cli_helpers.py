"""CLI and experiment bookkeeping helpers shared across tracking examples."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import click


def resolve_run_output_dir(
    *,
    output_root: Path,
    source_path: Path,
    out_basename: str | None,
) -> Path:
    """Resolve the per-run output directory from ``--output-dir`` and ``--out-basename``."""
    resolved_basename = source_path.stem
    if out_basename is not None:
        candidate = out_basename.strip()
        if not candidate:
            raise click.BadParameter(
                "--out-basename must not be empty or whitespace only",
                param_hint="--out-basename",
            )
        if "/" in candidate or "\\" in candidate:
            raise click.BadParameter(
                "--out-basename must be a folder name, not a path",
                param_hint="--out-basename",
            )
        resolved_basename = candidate
    return output_root / resolved_basename


def resolve_end_frame(
    *,
    start_frame: int,
    end_frame: int,
    max_frames: int | None,
) -> int:
    """Reconcile ``--end-frame`` and ``--max-frames`` into an effective end-frame index."""
    if max_frames is None:
        return end_frame
    if max_frames == -1:
        derived_end = -1
    elif max_frames <= 0:
        raise click.BadParameter("--max-frames must be -1 or positive", param_hint="--max-frames")
    else:
        derived_end = start_frame + max_frames
    if end_frame != -1 and derived_end != -1 and end_frame != derived_end:
        raise click.BadParameter(
            "--end-frame and --max-frames conflict; use one or set consistent values.",
            param_hint="--end-frame",
        )
    return derived_end


def write_experiment_info(output_dir: Path, **params: object) -> None:
    """Write an ``experiment_info.txt`` alongside outputs for traceability."""
    lines = [
        f"Experiment: {output_dir.name}",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Parameters:",
    ]
    for k, v in params.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    (output_dir / "experiment_info.txt").write_text("\n".join(lines), encoding="utf-8")


def append_tracking_metrics(info_path: Path, tracking_json_path: Path) -> None:
    """Append diagnostic track-count metrics from a COCO tracking JSON."""
    import collections

    try:
        data = json.loads(tracking_json_path.read_text(encoding="utf-8"))
    except Exception:
        return

    annots = data.get("annotations", [])
    frame_ids = [int(img["id"]) for img in data.get("images", [])]
    n_frames = len(frame_ids)
    frame_tracks: dict[int, set[int]] = collections.defaultdict(set)
    all_ids: set[int] = set()
    for a in annots:
        tid = a.get("track_id", -1)
        if tid == -1:
            continue
        frame_tracks[a["image_id"]].add(tid)
        all_ids.add(tid)

    counts = [len(frame_tracks.get(frame_id, set())) for frame_id in frame_ids]
    avg = sum(counts) / len(counts) if counts else 0.0
    mx = max(counts) if counts else 0
    zeros = sum(1 for c in counts if c == 0)

    lines = [
        "Results:",
        f"  frames: {n_frames}",
        f"  unique_track_ids: {len(all_ids)}",
        f"  avg_tracks_per_frame: {avg:.1f}",
        f"  max_tracks_per_frame: {mx}",
        f"  zero_track_frames: {zeros}",
        "",
    ]
    with info_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
