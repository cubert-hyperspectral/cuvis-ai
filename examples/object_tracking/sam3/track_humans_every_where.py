from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3 text propagation over all videos under an input base folder, "
            "mirroring the same subfolder structure under an output base folder."
        )
    )
    parser.add_argument("--input-base-folder", type=Path, required=True)
    parser.add_argument("--output-base-folder", type=Path, required=True)
    parser.add_argument("--plugins-yaml", type=Path, default=Path("configs/plugins/sam3.yaml"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-tracker-states", type=int, default=50)
    parser.add_argument(
        "--prompt",
        type=str,
        default="person with black hoodie",
        help="Text prompt for SAM3 detector.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile inside sam3_text_propagation.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated commands without executing them.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def collect_videos(input_base: Path) -> list[Path]:
    return sorted(
        [p for p in input_base.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    )


def build_command(
    *,
    video_path: Path,
    plugins_yaml: Path,
    output_dir: Path,
    out_basename: str | None,
    start_frame: int,
    max_tracker_states: int,
    prompt: str,
    compile_model: bool,
) -> list[str]:
    script_path = Path("examples/object_tracking/sam3/sam3_text_propagation.py")
    common_args = [
        str(script_path),
        "--video-path",
        str(video_path),
        "--plugins-yaml",
        str(plugins_yaml),
        "--output-dir",
        str(output_dir),
        "--start-frame",
        str(start_frame),
        "--bf16",
        "--max-tracker-states",
        str(max_tracker_states),
        "--prompt",
        prompt,
    ]
    if out_basename is not None:
        common_args.extend(["--out-basename", out_basename])
    common_args.append("--save-pipeline-yaml")
    common_args.append("--no-save-pipeline-weights")
    if compile_model:
        common_args.append("--compile")
    return ["uv", "run", "python", *common_args]


def main() -> int:
    args = parse_args()

    input_base = args.input_base_folder.resolve()
    output_base = args.output_base_folder.resolve()
    repo_root = Path(__file__).resolve().parents[3]

    if not input_base.is_dir():
        print(f"Input base folder not found: {input_base}", file=sys.stderr)
        return 2
    if not args.dry_run:
        output_base.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(input_base)
    if not videos:
        print(f"No video files found under: {input_base}")
        return 0

    print(f"Input base folder : {input_base}")
    print(f"Output base folder: {output_base}")
    print(f"Videos found      : {len(videos)}")
    print("Out basename      : default per video file stem")
    if args.dry_run:
        print("Mode              : dry-run (print only)")

    for idx, video in enumerate(videos, start=1):
        relative_dir = video.parent.relative_to(input_base)
        target_output_dir = output_base / relative_dir
        out_basename = "." if video.stem == "spam_result" else None
        if not args.dry_run:
            target_output_dir.mkdir(parents=True, exist_ok=True)

        run_dir = target_output_dir if out_basename == "." else (target_output_dir / video.stem)
        if run_dir.exists() and not args.force:
            print(f"[{idx}/{len(videos)}] Output exists, skipping: {run_dir}")
            continue

        print()
        print(f"[{idx}/{len(videos)}] Processing: {video}")
        print(f"             Output dir : {target_output_dir}")
        if out_basename is None:
            print(f"             Out base   : {video.stem} (default)")
        else:
            print("             Out base   : . (write directly to output dir)")

        cmd = build_command(
            video_path=video,
            plugins_yaml=args.plugins_yaml,
            output_dir=target_output_dir,
            out_basename=out_basename,
            start_frame=args.start_frame,
            max_tracker_states=args.max_tracker_states,
            prompt=args.prompt,
            compile_model=args.compile,
        )
        print(f"             Command    : {subprocess.list2cmdline(cmd)}")
        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=repo_root)
        if result.returncode != 0:
            print(f"SAM3 run failed for video: {video}", file=sys.stderr)
            return result.returncode

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
