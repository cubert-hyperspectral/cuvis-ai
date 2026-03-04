"""Convert existing ByteTrack overlay videos to browser-friendly codecs in place.

This script only touches already-generated overlay files. It does not run
inference or the sweep.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _iter_overlay_files(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.rglob(pattern) if p.is_file())


def _transcode_in_place(
    src_path: Path,
    codec: str,
    keep_backup: bool,
    dry_run: bool,
) -> tuple[bool, str]:
    try:
        import cv2
    except Exception as exc:
        return False, f"cv2 import failed: {exc}"

    if len(codec) != 4:
        return False, "codec must be 4 chars (FourCC)"

    tmp_path = src_path.with_name(f"{src_path.stem}.tmp_reencode{src_path.suffix}")
    bak_path = src_path.with_name(f"{src_path.stem}.bak{src_path.suffix}")

    if dry_run:
        return True, "dry-run"

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
        default="AV01",
        help="FourCC output codec (default: AV01).",
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
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"Root does not exist: {root}")
        return 1

    overlays = _iter_overlay_files(root, args.pattern)
    if args.limit is not None:
        overlays = overlays[: args.limit]

    if not overlays:
        print("No matching overlay files found.")
        return 0

    print(f"Found {len(overlays)} file(s) under {root}")
    ok_count = 0
    fail_count = 0

    for idx, path in enumerate(overlays, start=1):
        print(f"[{idx}/{len(overlays)}] {path}")
        ok, msg = _transcode_in_place(
            src_path=path,
            codec=args.codec,
            keep_backup=args.keep_backup,
            dry_run=args.dry_run,
        )
        if ok:
            ok_count += 1
            print(f"  -> {msg}")
        else:
            fail_count += 1
            print(f"  -> FAILED: {msg}")

    print(f"Done. success={ok_count} failed={fail_count}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
