from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


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
    end_frame: int,
    max_frames: int | None,
    max_tracker_states: int,
    prompts: tuple[str, ...],
    bf16: bool,
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
        "--max-tracker-states",
        str(max_tracker_states),
    ]
    if end_frame != -1:
        common_args.extend(["--end-frame", str(end_frame)])
    if max_frames is not None:
        common_args.extend(["--max-frames", str(max_frames)])
    if bf16:
        common_args.append("--bf16")
    for prompt in prompts:
        common_args.extend(["--prompt", prompt])
    if out_basename is not None:
        common_args.extend(["--out-basename", out_basename])
    if compile_model:
        common_args.append("--compile")
    return [sys.executable, *common_args]


@click.command()
@click.option(
    "--input-base-folder",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Root folder to scan recursively for video files.",
)
@click.option(
    "--output-base-folder",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Root folder where the input subfolder structure will be mirrored.",
)
@click.option(
    "--plugins-yaml",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=Path("configs/plugins/sam3.yaml"),
    show_default=True,
)
@click.option("--start-frame", type=int, default=0, show_default=True)
@click.option("--max-tracker-states", type=int, default=50, show_default=True)
@click.option(
    "--prompt",
    type=str,
    multiple=True,
    default=("person with black hoodie",),
    show_default=True,
    help=(
        "Repeatable text prompt spec: <text>@<frame_id>. Bare <text> means <text>@0. "
        "Prompts are emitted only on their scheduled frames."
    ),
)
@click.option("--end-frame", type=int, default=-1, show_default=True)
@click.option(
    "--max-frames",
    type=int,
    default=None,
    help="Optional frame-window length from --start-frame.",
)
@click.option(
    "--bf16/--no-bf16",
    default=True,
    show_default=True,
    help="Enable bfloat16 autocast in sam3_text_propagation.py.",
)
@click.option(
    "--compile",
    "compile_model",
    is_flag=True,
    default=False,
    help="Enable torch.compile inside sam3_text_propagation.py.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print generated commands without executing them.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-run a video even if its target output directory already exists.",
)
def main(
    input_base_folder: Path,
    output_base_folder: Path,
    plugins_yaml: Path,
    start_frame: int,
    end_frame: int,
    max_frames: int | None,
    max_tracker_states: int,
    prompt: tuple[str, ...],
    bf16: bool,
    compile_model: bool,
    dry_run: bool,
    force: bool,
) -> None:
    """Run SAM3 text propagation over all videos under an input base folder."""
    if max_tracker_states < 1:
        raise click.BadParameter(
            "--max-tracker-states must be >= 1.",
            param_hint="--max-tracker-states",
        )

    input_base = input_base_folder.resolve()
    output_base = output_base_folder.resolve()
    repo_root = Path(__file__).resolve().parents[3]

    if not dry_run:
        output_base.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(input_base)
    if not videos:
        click.echo(f"No video files found under: {input_base}")
        return

    click.echo(f"Input base folder : {input_base}")
    click.echo(f"Output base folder: {output_base}")
    click.echo(f"Videos found      : {len(videos)}")
    click.echo(f"Prompt specs      : {list(prompt)}")
    click.echo("Out basename      : default per video file stem")
    if dry_run:
        click.echo("Mode              : dry-run (print only)")

    for idx, video in enumerate(videos, start=1):
        relative_dir = video.parent.relative_to(input_base)
        target_output_dir = output_base / relative_dir
        out_basename = "." if video.stem == "spam_result" else None
        if not dry_run:
            target_output_dir.mkdir(parents=True, exist_ok=True)

        run_dir = target_output_dir if out_basename == "." else (target_output_dir / video.stem)
        if run_dir.exists() and not force:
            click.echo(f"[{idx}/{len(videos)}] Output exists, skipping: {run_dir}")
            continue

        click.echo()
        click.echo(f"[{idx}/{len(videos)}] Processing: {video}")
        click.echo(f"             Output dir : {target_output_dir}")
        if out_basename is None:
            click.echo(f"             Out base   : {video.stem} (default)")
        else:
            click.echo("             Out base   : . (write directly to output dir)")

        cmd = build_command(
            video_path=video,
            plugins_yaml=plugins_yaml,
            output_dir=target_output_dir,
            out_basename=out_basename,
            start_frame=start_frame,
            end_frame=end_frame,
            max_frames=max_frames,
            max_tracker_states=max_tracker_states,
            prompts=prompt,
            bf16=bf16,
            compile_model=compile_model,
        )
        click.echo(f"             Command    : {subprocess.list2cmdline(cmd)}")
        if dry_run:
            continue

        result = subprocess.run(cmd, cwd=repo_root, check=False)
        if result.returncode != 0:
            raise click.ClickException(f"SAM3 run failed for video: {video}")

    click.echo()
    click.echo("Done.")


if __name__ == "__main__":
    main()
