"""Convert existing ByteTrack overlay videos to browser-friendly codecs in place.

This script only touches already-generated overlay files. It does not run
inference or the sweep.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import time
from pathlib import Path


def _iter_overlay_files(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.rglob(pattern) if p.is_file())


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


def _transcode_in_place(
    src_path: Path,
    codec: str,
    keep_backup: bool,
    dry_run: bool,
) -> tuple[bool, str]:
    if len(codec) != 4:
        return False, "codec must be 4 chars (FourCC)"

    tmp_path = src_path.with_name(f"{src_path.stem}.tmp_reencode{src_path.suffix}")
    bak_path = src_path.with_name(f"{src_path.stem}.bak{src_path.suffix}")

    if dry_run:
        return True, "dry-run"

    try:
        import cv2
    except Exception as exc:
        return False, f"cv2 import failed: {exc}"

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        cap.release()
        return False, "failed to open source video"

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return False, f"invalid frame size ({width}x{height})"

    writer = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        cap.release()
        return False, f"failed to open writer with codec={codec}"

    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frames += 1
    finally:
        writer.release()
        cap.release()

    if frames == 0 or not tmp_path.exists() or tmp_path.stat().st_size <= 0:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False, "no frames written"

    try:
        if keep_backup:
            if bak_path.exists():
                bak_path.unlink()
            os.replace(src_path, bak_path)
        os.replace(tmp_path, src_path)
    except Exception as exc:
        return False, f"replace failed: {exc}"

    return True, f"ok ({frames} frames)"


def _transcode_job(
    idx: int,
    src_path: Path,
    codec: str,
    keep_backup: bool,
    dry_run: bool,
) -> tuple[int, Path, bool, str]:
    ok, msg = _transcode_in_place(
        src_path=src_path,
        codec=codec,
        keep_backup=keep_backup,
        dry_run=dry_run,
    )
    return idx, src_path, ok, msg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcode existing tracking_overlay.mp4 files in place."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing sweep run folders.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="tracking_overlay.mp4",
        help="File pattern to find overlays (default: tracking_overlay.mp4).",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="VP90",
        help="FourCC output codec (default: VP90).",
    )
    parser.add_argument(
        "--keep-backup",
        action="store_true",
        help="Keep original as *.bak.mp4 before replacing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without writing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only first N files after sorting.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel conversion jobs (default: 1).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"Root does not exist: {root}")
        return 1

    if args.jobs < 1:
        print("--jobs must be >= 1")
        return 1

    overlays = _iter_overlay_files(root, args.pattern)
    if args.limit is not None:
        overlays = overlays[: args.limit]

    if not overlays:
        print("No matching overlay files found.")
        return 0

    print(f"Found {len(overlays)} file(s) under {root}", flush=True)
    if args.jobs > 1:
        print(f"Running with {args.jobs} parallel job(s)", flush=True)

    ok_count = 0
    fail_count = 0
    started = time.time()
    total = len(overlays)

    if args.jobs == 1:
        for idx, path in enumerate(overlays, start=1):
            elapsed = time.time() - started
            percent = ((idx - 1) / total) * 100.0
            eta = _estimate_eta(elapsed, idx - 1, total)
            print(
                f"[{idx}/{total} | {percent:5.1f}% | elapsed={_format_duration(elapsed)} | eta={_format_duration(eta)}] {path}",
                flush=True,
            )
            ok, msg = _transcode_in_place(
                src_path=path,
                codec=args.codec,
                keep_backup=args.keep_backup,
                dry_run=args.dry_run,
            )
            if ok:
                ok_count += 1
                print(f"  -> {msg}", flush=True)
            else:
                fail_count += 1
                print(f"  -> FAILED: {msg}", flush=True)

            elapsed = time.time() - started
            percent = (idx / total) * 100.0
            eta = _estimate_eta(elapsed, idx, total)
            print(
                "  progress: {} done | elapsed={} | eta={} | success={} failed={}".format(
                    f"{percent:5.1f}%",
                    _format_duration(elapsed),
                    _format_duration(eta),
                    ok_count,
                    fail_count,
                ),
                flush=True,
            )
    else:
        futures: dict[concurrent.futures.Future[tuple[int, Path, bool, str]], Path] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            for idx, path in enumerate(overlays, start=1):
                future = executor.submit(
                    _transcode_job,
                    idx,
                    path,
                    args.codec,
                    args.keep_backup,
                    args.dry_run,
                )
                futures[future] = path

            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                try:
                    idx, path, ok, msg = future.result()
                except Exception as exc:
                    idx = -1
                    path = futures[future]
                    ok = False
                    msg = f"worker exception: {exc}"

                if ok:
                    ok_count += 1
                else:
                    fail_count += 1

                elapsed = time.time() - started
                percent = (completed / total) * 100.0
                eta = _estimate_eta(elapsed, completed, total)
                run_label = f"{idx}/{total}" if idx > 0 else "?"
                status = msg if ok else f"FAILED: {msg}"
                print(
                    f"[{completed}/{total} | {percent:5.1f}% | elapsed={_format_duration(elapsed)} | eta={_format_duration(eta)}] "
                    f"run={run_label} {path}",
                    flush=True,
                )
                print(f"  -> {status}", flush=True)
                print(
                    "  progress: {} done | elapsed={} | eta={} | success={} failed={}".format(
                        f"{percent:5.1f}%",
                        _format_duration(elapsed),
                        _format_duration(eta),
                        ok_count,
                        fail_count,
                    ),
                    flush=True,
                )

    total_elapsed = time.time() - started
    print(
        f"Done. success={ok_count} failed={fail_count} elapsed={_format_duration(total_elapsed)}",
        flush=True,
    )
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
