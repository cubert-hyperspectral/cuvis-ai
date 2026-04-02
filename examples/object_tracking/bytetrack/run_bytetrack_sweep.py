"""Hyperparameter sweep runner for YOLO + ByteTrack HSI pipeline.

Runs `yolo_bytetrack_hsi.py` over a predefined grid of detection, tracking,
and spectral thresholds, records per-run metrics JSON, and prints progress.

Sweep stages:
  A  — baseline threshold sweep (association_mode=baseline, no spectral params)
  B  — spectral sweep on a reduced threshold subset
  AB — run both stages sequentially

Use `regenerate_bytetrack_sweep_html.py` afterwards to collect all run results
into aggregate CSV/Parquet and an interactive HTML table.

Supports parallel execution via --num-workers (default 1 = sequential).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# === Sweep space ===
# Two-phase strategy:
#   Phase 1 — sweep primary detection & tracking thresholds (secondary fixed at defaults)
#   Phase 2 — take best Phase 1 combos, uncomment secondary param grids below

# --- Phase 1: primary params (~375 runs) ---
CONFIDENCE_VALUES = [0.15, 0.20, 0.25]
IOU_VALUES = [0.65, 0.75, 0.85, 0.90, 0.95]
TRACK_THRESH_VALUES = [0.15, 0.25, 0.30, 0.35, 0.45]
MATCH_THRESH_VALUES = [0.4, 0.5, 0.60, 0.70, 0.80]
TRACK_BUFFER_VALUES = [30]
AGNOSTIC_VALUES = [False]

# --- Phase 1 defaults for secondary params (single value = no sweep) ---
SECOND_SCORE_THRESH_VALUES = [0.10]
SECOND_MATCH_THRESH_VALUES = [0.50]
UNCONFIRMED_MATCH_THRESH_VALUES = [0.70]
NEW_TRACK_THRESH_OFFSET_VALUES = [0.10]

# --- Phase 2: uncomment to sweep secondary params ---
# TRACK_BUFFER_VALUES = [30, 40, 60]
# SECOND_SCORE_THRESH_VALUES = [0.05, 0.10, 0.15, 0.20]
# SECOND_MATCH_THRESH_VALUES = [0.35, 0.45, 0.50, 0.55, 0.60]
# UNCONFIRMED_MATCH_THRESH_VALUES = [0.55, 0.60, 0.65, 0.70]
# NEW_TRACK_THRESH_OFFSET_VALUES = [0.05, 0.10]

# --- Stage B: spectral params ---
# Reduced baseline grid for Stage B (best combos from Stage A).
STAGE_B_CONFIDENCE_VALUES = [0.15, 0.20]
STAGE_B_IOU_VALUES = [0.75, 0.85]
STAGE_B_TRACK_THRESH_VALUES = [0.25, 0.35]
STAGE_B_MATCH_THRESH_VALUES = [0.50, 0.70]
# Spectral grid
ASSOCIATION_MODE_VALUES = ["spectral_cost", "spectral_post_gate"]
SPECTRAL_COST_WEIGHT_VALUES = [0.2, 0.3, 0.5]
PROTOTYPE_EMA_BETA_VALUES = [0.05, 0.1, 0.2]


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
    association_mode: str = "baseline",
    spectral_cost_weight: float | None = None,
    prototype_ema_beta: float | None = None,
    spectral_sim_floor: float | None = None,
    prototype_decay_enabled: bool = False,
    prototype_decay_half_life: float | None = None,
    spectral_std_weighting_enabled: bool = False,
    spectral_std_alpha: float | None = None,
) -> str:
    parts = (
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
    if association_mode != "baseline":
        mode_abbr = "sc" if association_mode == "spectral_cost" else "spg"
        parts += f"_am{mode_abbr}"
        if spectral_cost_weight is not None:
            parts += f"_scw{_fmt_float_token(spectral_cost_weight)}"
        if prototype_ema_beta is not None:
            parts += f"_ema{_fmt_float_token(prototype_ema_beta)}"
        if spectral_sim_floor is not None:
            parts += f"_sf{_fmt_float_token(spectral_sim_floor)}"
        if prototype_decay_enabled:
            parts += f"_pd{_fmt_float_token(prototype_decay_half_life or 10.0)}"
        if spectral_std_weighting_enabled:
            parts += f"_sw{_fmt_float_token(spectral_std_alpha or 1.0)}"
    return parts


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
        # Unique track IDs observed over the full clip.
        "tracks_whole_video": len(track_ids),
        # Backward-compatible legacy name.
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
    association_mode: str = "baseline"
    spectral_cost_weight: float | None = None
    prototype_ema_beta: float | None = None
    spectral_sim_floor: float | None = None
    prototype_decay_enabled: bool = False
    prototype_decay_half_life: float | None = None
    spectral_std_weighting_enabled: bool = False
    spectral_std_alpha: float | None = None


def _generate_grid(stage: str = "A") -> Iterable[RunParams]:
    """Generate sweep grid.

    stage "A": baseline threshold sweep (association_mode=baseline).
    stage "B": reduced threshold subset x spectral grid.
    stage "AB": both stages concatenated.
    """
    if stage in ("A", "AB"):
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

    if stage in ("B", "AB"):
        for conf, iou, tt, mt, am, scw, ema in itertools.product(
            STAGE_B_CONFIDENCE_VALUES,
            STAGE_B_IOU_VALUES,
            STAGE_B_TRACK_THRESH_VALUES,
            STAGE_B_MATCH_THRESH_VALUES,
            ASSOCIATION_MODE_VALUES,
            SPECTRAL_COST_WEIGHT_VALUES,
            PROTOTYPE_EMA_BETA_VALUES,
        ):
            yield RunParams(
                confidence_threshold=conf,
                iou_threshold=iou,
                track_thresh=tt,
                match_thresh=mt,
                track_buffer=TRACK_BUFFER_VALUES[0],
                agnostic_nms=False,
                second_score_thresh=SECOND_SCORE_THRESH_VALUES[0],
                second_match_thresh=SECOND_MATCH_THRESH_VALUES[0],
                unconfirmed_match_thresh=UNCONFIRMED_MATCH_THRESH_VALUES[0],
                new_track_thresh_offset=NEW_TRACK_THRESH_OFFSET_VALUES[0],
                association_mode=am,
                spectral_cost_weight=scw,
                prototype_ema_beta=ema,
            )


def _run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _maybe_filter(run_id: str, pattern: str | None) -> bool:
    if not pattern:
        return True
    return re.search(pattern, run_id) is not None


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or not math.isfinite(seconds):
        return "--:--:--"
    whole = int(seconds)
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _estimate_eta(elapsed: float, completed: int, total: int) -> float | None:
    if completed <= 0 or total <= 0 or completed > total:
        return None
    avg = elapsed / completed
    return avg * (total - completed)


def _build_cmd(
    script_path: Path,
    cu3s_path: Path,
    run_dir: Path,
    max_frames: int,
    model_name: str,
    params: RunParams,
    plugins_dir: Path | None,
    bf16: bool,
) -> list[str]:
    """Build the subprocess command for a single run."""
    cmd = [
        sys.executable,
        str(script_path),
        "--cu3s-path",
        str(cu3s_path),
        "--output-dir",
        str(run_dir),
        "--end-frame",
        str(max_frames),
        "--model-name",
        model_name,
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
        "--association-mode",
        params.association_mode,
        "--classes",
        "0",
    ]
    if params.association_mode != "baseline":
        if params.spectral_cost_weight is not None:
            cmd.extend(["--spectral-cost-weight", str(params.spectral_cost_weight)])
        if params.prototype_ema_beta is not None:
            cmd.extend(["--prototype-ema-beta", str(params.prototype_ema_beta)])
        if params.spectral_sim_floor is not None:
            cmd.extend(["--spectral-sim-floor", str(params.spectral_sim_floor)])
        if params.prototype_decay_enabled:
            cmd.append("--prototype-decay")
            if params.prototype_decay_half_life is not None:
                cmd.extend(["--prototype-decay-half-life", str(params.prototype_decay_half_life)])
        if params.spectral_std_weighting_enabled:
            cmd.append("--spectral-std-weighting")
            if params.spectral_std_alpha is not None:
                cmd.extend(["--spectral-std-alpha", str(params.spectral_std_alpha)])
    if params.agnostic_nms:
        cmd.append("--agnostic-nms")
    if plugins_dir:
        cmd.extend(["--plugins-dir", str(plugins_dir)])
    if bf16:
        cmd.append("--bf16")
    return cmd


def _execute_single_run(
    index: int,
    total_runs: int,
    params: RunParams,
    script_path: Path,
    cu3s_path: Path,
    output_root: Path,
    max_frames: int,
    model_name: str,
    plugins_dir: Path | None,
    bf16: bool,
    overwrite: bool,
    resume: bool,
    verbose: bool,
    repo_root: Path,
) -> dict[str, Any]:
    """Execute one sweep run and return its result dict."""
    run_id = _build_run_id(**params.__dict__)
    run_dir = output_root / run_id
    _ensure_dir(run_dir)

    det_path = run_dir / "detection_results.json"
    track_path = run_dir / "tracking_results.json"

    should_run = True
    if det_path.exists() and track_path.exists() and not overwrite:
        should_run = False
    if not overwrite and resume and track_path.exists():
        should_run = False

    start_ts = time.time()
    status = "skipped"
    stdout = stderr = ""
    cmd: list[str] | None = None

    if should_run:
        cmd = _build_cmd(
            script_path,
            cu3s_path,
            run_dir,
            max_frames,
            model_name,
            params,
            plugins_dir,
            bf16,
        )
        proc = _run_command(cmd, cwd=repo_root)
        stdout, stderr = proc.stdout, proc.stderr
        status = "ok" if proc.returncode == 0 else f"fail({proc.returncode})"

    wall_time = time.time() - start_ts

    # Compute metrics if outputs exist (even for skipped/existing runs).
    metrics: dict[str, Any] = {}
    if not status.startswith("fail"):
        try:
            metrics = _compute_metrics(det_path, track_path)
            if status == "skipped":
                status = "existing"
        except Exception as exc:
            status = f"metrics_failed: {exc}"

    # Persist per-run metadata.
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "params": params.__dict__,
                "command": cmd,
                "status": status,
                "stdout_tail": stdout[-400:],
                "stderr_tail": stderr[-400:],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    result = {
        "run_id": run_id,
        "status": status,
        "wall_time_sec": wall_time,
        **params.__dict__,
        **metrics,
    }
    if verbose and stdout:
        result["stdout_tail"] = stdout[-400:]
    if verbose and stderr:
        result["stderr_tail"] = stderr[-400:]
    return result


# -- Thread-safe progress printer --
_print_lock = threading.Lock()


def _print_progress(
    index: int,
    total_runs: int,
    result: dict[str, Any],
    overall_start: float,
    completed: int,
) -> None:
    """Print a single-run completion line (thread-safe)."""
    overall_elapsed = time.time() - overall_start
    eta = _estimate_eta(overall_elapsed, completed, total_runs)
    percent = (completed / total_runs) * 100.0
    status = result["status"]
    wall = result.get("wall_time_sec", 0.0)
    run_id = result["run_id"]
    metrics_line = ""
    if "unique_track_ids" in result:
        metrics_line = " | ids={} maxTrk={} avgTrk={:.1f} avgDet={:.1f}".format(
            result.get("unique_track_ids", 0),
            result.get("max_tracks_per_frame", 0),
            float(result.get("avg_tracks_per_frame", 0.0)),
            float(result.get("avg_dets_per_frame", 0.0)),
        )
    with _print_lock:
        print(
            f"[{completed}/{total_runs} | {percent:5.1f}% | elapsed={_format_duration(overall_elapsed)} | eta={_format_duration(eta)}] {run_id} -> {status} ({wall:.1f}s){metrics_line}"
        )


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
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel jobs. Default: 1 (sequential).",
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
        "--sweep-stage",
        type=str,
        choices=["A", "B", "AB"],
        default="A",
        help="Sweep stage: A=baseline, B=spectral, AB=both. Default: A.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra per-run details (command and output tails).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "examples" / "object_tracking" / "bytetrack" / "yolo_bytetrack_hsi.py"

    _ensure_dir(args.output_root)

    grid_iter: Iterable[RunParams] | list[RunParams] = _generate_grid(stage=args.sweep_stage)
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

    num_workers = max(1, args.num_workers)
    print(f"Scheduled runs: {total_runs} (workers: {num_workers})")
    overall_start = time.time()
    ok_count = 0
    fail_count = 0
    skip_count = 0
    completed = 0

    def _on_result(index: int, result: dict[str, Any]) -> None:
        nonlocal ok_count, fail_count, skip_count, completed
        completed += 1
        status = result["status"]
        if status.startswith("ok"):
            ok_count += 1
        elif status.startswith("fail") or status.startswith("metrics_failed"):
            fail_count += 1
        else:
            skip_count += 1
        _print_progress(index, total_runs, result, overall_start, completed)

    common_kwargs = {
        "script_path": script_path,
        "cu3s_path": args.cu3s_path,
        "output_root": args.output_root,
        "max_frames": args.max_frames,
        "model_name": args.model_name,
        "plugins_dir": args.plugins_dir,
        "bf16": args.bf16,
        "overwrite": args.overwrite,
        "resume": args.resume,
        "verbose": args.verbose,
        "repo_root": repo_root,
    }

    if num_workers == 1:
        for index, params in enumerate(grid_items, start=1):
            result = _execute_single_run(
                index=index, total_runs=total_runs, params=params, **common_kwargs
            )
            _on_result(index, result)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(
                    _execute_single_run,
                    index=index,
                    total_runs=total_runs,
                    params=params,
                    **common_kwargs,
                ): index
                for index, params in enumerate(grid_items, start=1)
            }
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                result = future.result()
                _on_result(index, result)

    elapsed = _format_duration(time.time() - overall_start)
    print(f"Done in {elapsed} -> ok={ok_count}, fail={fail_count}, skipped={skip_count}")
    print(
        f"Run `regenerate_bytetrack_sweep_html.py --results-root {args.output_root}` to build the results table."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
