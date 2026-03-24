from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from cuvis_ai_core.data.datasets import SingleCu3sDataset

PROCESSING_MODE = "SpectralRadiance"
PIPELINE_NAME = "SPAM_Invisible_Ink"
VIDEO_NAME = "spam_result.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SPAM invisible-ink pipeline on all .cu3s files above a minimum size, "
            "mirroring source subfolder structure under an output base folder."
        )
    )
    parser.add_argument("--input-base-folder", type=Path, required=True)
    parser.add_argument(
        "--output-base-folder",
        type=Path,
        default=Path("D:/experiments/20260323/spam_ink_highlight"),
    )
    parser.add_argument("--sam-xml-path", type=Path, required=True)
    parser.add_argument("--rgb-xml-path", type=Path, required=True)
    parser.add_argument("--reference-npy", type=Path, default=None)
    parser.add_argument("--min-size-gb", type=float, default=1.0)
    parser.add_argument("--overlay-alpha", type=float, default=1.0)
    parser.add_argument(
        "--overlay-color",
        type=str,
        default="255,0,0",
        help="Overlay color triplet R,G,B in [0,255] or [0,1].",
    )
    parser.add_argument("--frame-rotation", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=None,
        help="Directory for bootstrap SPAM pipeline YAML/PT; defaults to <output-base-folder>/_restore_template.",
    )
    parser.add_argument(
        "--rebuild-template",
        action="store_true",
        help="Force rebuilding bootstrap pipeline YAML/PT even if already present.",
    )
    return parser.parse_args()


def collect_cu3s(input_base: Path, min_size_bytes: int) -> list[Path]:
    return sorted(
        p
        for p in input_base.rglob("*")
        if p.is_file() and p.suffix.lower() == ".cu3s" and p.stat().st_size > min_size_bytes
    )


def infer_fps(cu3s_path: Path) -> float:
    dataset = SingleCu3sDataset(cu3s_file_path=str(cu3s_path), processing_mode=PROCESSING_MODE)
    fps = float(getattr(dataset, "fps", None) or 10.0)
    return fps if fps > 0.0 else 10.0


def _bootstrap_command(
    *,
    cu3s_path: Path,
    sam_xml_path: Path,
    rgb_xml_path: Path,
    output_dir: Path,
    overlay_alpha: float,
    overlay_color: str,
    frame_rotation: int | None,
    reference_npy: Path | None,
) -> list[str]:
    script_path = Path("examples/spectral_angle_mapper/spam_invisible_ink.py")
    cmd = [
        "uv",
        "run",
        "python",
        str(script_path),
        "--cu3s-path",
        str(cu3s_path),
        "--sam-xml-path",
        str(sam_xml_path),
        "--rgb-xml-path",
        str(rgb_xml_path),
        "--overlay-alpha",
        str(overlay_alpha),
        "--overlay-color",
        overlay_color,
        "--end-frame",
        "1",
        "--output-dir",
        str(output_dir),
    ]
    if frame_rotation is not None:
        cmd.extend(["--frame-rotation", str(frame_rotation)])
    if reference_npy is not None:
        cmd.extend(["--reference-npy", str(reference_npy)])
    return cmd


def _load_to_video_index(pipeline_yaml: Path) -> int:
    config = yaml.safe_load(pipeline_yaml.read_text(encoding="utf-8"))
    nodes = config.get("nodes", [])
    for idx, node_cfg in enumerate(nodes):
        class_name = str(node_cfg.get("class_name", ""))
        name = str(node_cfg.get("name", ""))
        if name == "to_video" or class_name.endswith(".ToVideoNode"):
            return idx
    raise RuntimeError(f"Could not find ToVideoNode in pipeline config: {pipeline_yaml}")


def _restore_command(
    *,
    pipeline_yaml: Path,
    pipeline_pt: Path,
    cu3s_path: Path,
    output_video_path: Path,
    frame_rate: float,
    to_video_index: int,
    device: str,
) -> list[str]:
    override_video = (
        f'nodes.{to_video_index}.hparams.output_video_path="{output_video_path.as_posix()}"'
    )
    override_fps = f"nodes.{to_video_index}.hparams.frame_rate={frame_rate:.6f}"
    return [
        "uv",
        "run",
        "restore-pipeline",
        "--pipeline-path",
        str(pipeline_yaml),
        "--weights-path",
        str(pipeline_pt),
        "--device",
        device,
        "--cu3s-file-path",
        str(cu3s_path),
        "--processing-mode",
        PROCESSING_MODE,
        "--override",
        override_video,
        "--override",
        override_fps,
    ]


def _copy_template_artifacts(template_dir: Path, run_dir: Path) -> None:
    artifact_patterns = [
        f"{PIPELINE_NAME}.yaml",
        f"{PIPELINE_NAME}.pt",
        f"{PIPELINE_NAME}.png",
        f"{PIPELINE_NAME}.md",
        "*.npy",
        "*_config.json",
    ]
    for pattern in artifact_patterns:
        for src in template_dir.glob(pattern):
            dst = run_dir / src.name
            if src.resolve() == dst.resolve():
                continue
            shutil.copy2(src, dst)


def _write_profiling_summary(run_dir: Path, command: list[str], output_text: str) -> None:
    line = ""
    for candidate in output_text.splitlines():
        if "Inference complete:" in candidate:
            line = candidate.strip()
    summary_lines = [
        f"Command: {subprocess.list2cmdline(command)}",
        line if line else "Inference summary line not found in restore-pipeline output.",
    ]
    (run_dir / "profiling_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    input_base = args.input_base_folder.resolve()
    output_base = args.output_base_folder.resolve()
    template_dir = (
        args.template_dir.resolve()
        if args.template_dir is not None
        else (output_base / "_restore_template").resolve()
    )
    repo_root = Path(__file__).resolve().parents[2]

    if not input_base.is_dir():
        print(f"Input base folder not found: {input_base}", file=sys.stderr)
        return 2
    if not args.sam_xml_path.exists():
        print(f"SAM XML not found: {args.sam_xml_path}", file=sys.stderr)
        return 2
    if not args.rgb_xml_path.exists():
        print(f"RGB XML not found: {args.rgb_xml_path}", file=sys.stderr)
        return 2
    if args.reference_npy is not None and not args.reference_npy.exists():
        print(f"Reference NPY not found: {args.reference_npy}", file=sys.stderr)
        return 2

    min_size_bytes = int(args.min_size_gb * (1024**3))
    cu3s_files = collect_cu3s(input_base=input_base, min_size_bytes=min_size_bytes)
    if not cu3s_files:
        print(
            f"No .cu3s files above {args.min_size_gb:.2f} GB found under: {input_base}",
            file=sys.stderr,
        )
        return 0

    pipeline_yaml = template_dir / f"{PIPELINE_NAME}.yaml"
    pipeline_pt = template_dir / f"{PIPELINE_NAME}.pt"

    print(f"Input base folder : {input_base}")
    print(f"Output base folder: {output_base}")
    print(f"Template dir      : {template_dir}")
    print(f"CU3S files found  : {len(cu3s_files)} (> {args.min_size_gb:.2f} GB)")

    need_template = args.rebuild_template or not (pipeline_yaml.exists() and pipeline_pt.exists())
    if need_template:
        template_seed = cu3s_files[0]
        bootstrap_cmd = _bootstrap_command(
            cu3s_path=template_seed,
            sam_xml_path=args.sam_xml_path.resolve(),
            rgb_xml_path=args.rgb_xml_path.resolve(),
            output_dir=template_dir,
            overlay_alpha=float(args.overlay_alpha),
            overlay_color=args.overlay_color,
            frame_rotation=args.frame_rotation,
            reference_npy=args.reference_npy.resolve() if args.reference_npy is not None else None,
        )
        print()
        print("Bootstrapping restore template pipeline...")
        print(f"  Seed CU3S : {template_seed}")
        print(f"  Command   : {subprocess.list2cmdline(bootstrap_cmd)}")
        if not args.dry_run:
            template_dir.mkdir(parents=True, exist_ok=True)
            bootstrap_result = subprocess.run(
                bootstrap_cmd, cwd=repo_root, text=True, capture_output=True
            )
            (template_dir / "bootstrap.log").write_text(
                (bootstrap_result.stdout or "") + "\n" + (bootstrap_result.stderr or ""),
                encoding="utf-8",
            )
            if bootstrap_result.returncode != 0:
                print(
                    "Bootstrap SPAM pipeline failed; see log: "
                    f"{template_dir / 'bootstrap.log'}",
                    file=sys.stderr,
                )
                return bootstrap_result.returncode

    if args.dry_run:
        print("Mode              : dry-run (print only)")
        return 0

    if not pipeline_yaml.exists() or not pipeline_pt.exists():
        print(
            f"Missing template artifacts: {pipeline_yaml} and/or {pipeline_pt}",
            file=sys.stderr,
        )
        return 2

    to_video_index = _load_to_video_index(pipeline_yaml)
    print(f"ToVideo node index: {to_video_index}")

    processed = 0
    skipped = 0
    failures = 0

    for idx, cu3s_path in enumerate(cu3s_files, start=1):
        relative_dir = cu3s_path.parent.relative_to(input_base)
        target_output_dir = output_base / relative_dir
        run_dir = target_output_dir / cu3s_path.stem

        if run_dir.exists() and not args.force:
            skipped += 1
            print(f"[{idx}/{len(cu3s_files)}] Output exists, skipping: {run_dir}")
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = run_dir / VIDEO_NAME

        try:
            fps = infer_fps(cu3s_path)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{idx}/{len(cu3s_files)}] FPS detection failed ({exc}); using fallback 10.0"
            )
            fps = 10.0

        cmd = _restore_command(
            pipeline_yaml=pipeline_yaml,
            pipeline_pt=pipeline_pt,
            cu3s_path=cu3s_path,
            output_video_path=output_video_path,
            frame_rate=fps,
            to_video_index=to_video_index,
            device=args.device,
        )

        print()
        print(f"[{idx}/{len(cu3s_files)}] Processing: {cu3s_path}")
        print(f"             Output dir : {run_dir}")
        print(f"             FPS        : {fps:.3f}")
        print(f"             Command    : {subprocess.list2cmdline(cmd)}")

        result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
        combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
        (run_dir / "restore_pipeline.log").write_text(combined_output, encoding="utf-8")

        _copy_template_artifacts(template_dir=template_dir, run_dir=run_dir)
        _write_profiling_summary(run_dir=run_dir, command=cmd, output_text=combined_output)

        if result.returncode != 0:
            failures += 1
            print(
                f"restore-pipeline failed for {cu3s_path} (see {run_dir / 'restore_pipeline.log'})",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                break
            continue

        processed += 1

    print()
    print(
        "Done. "
        f"processed={processed}, skipped={skipped}, failures={failures}, total={len(cu3s_files)}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
