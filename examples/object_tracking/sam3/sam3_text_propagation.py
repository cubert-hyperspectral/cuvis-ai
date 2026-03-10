"""SAM3 text-prompt propagation: CU3S -> false RGB -> SAM3 streaming -> overlay + JSON.

Uses a text prompt (e.g. ``person``) for SAM3 streaming propagation.
No detection JSON needed — SAM3 finds all matching objects automatically.

Examples::

    # Default: "person" prompt, 100 frames from frame 0
    uv run python examples/object_tracking/sam3/sam3_text_propagation.py `
        --cu3s-path "D:\\data\\Auto_013+01.cu3s" `
        --output-dir "D:\\experiments\\sam3\\text_propagation" `
        --max-frames 100

    # Custom prompt, start at frame 50
    uv run python examples/object_tracking/sam3/sam3_text_propagation.py `
        --cu3s-path "D:\\data\\Auto_013+01.cu3s" `
        --prompt "car" `
        --start-frame 50 `
        --output-dir "D:\\experiments\\sam3\\text_propagation" `
        --max-frames 100
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import click
import torch
from loguru import logger

PROCESSING_MODES = ("Raw", "DarkSubtract", "Preview", "Reflectance", "SpectralRadiance")


@click.command()
@click.option(
    "--cu3s-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option(
    "--processing-mode",
    type=click.Choice(PROCESSING_MODES, case_sensitive=False),
    default="SpectralRadiance",
    show_default=True,
)
@click.option("--frame-rotation", type=int, default=None)
@click.option(
    "--prompt",
    type=str,
    default="person",
    show_default=True,
    help="Text prompt for SAM3 detector (e.g. 'person', 'car').",
)
@click.option(
    "--start-frame",
    type=int,
    default=0,
    show_default=True,
    help="First frame to include in the video.",
)
@click.option(
    "--max-frames",
    type=int,
    default=-1,
    show_default=True,
    help="Maximum frames to process (-1 = all).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("D:/experiments/sam3/text_propagation"),
    show_default=True,
)
@click.option("--checkpoint-path", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option(
    "--plugins-yaml",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=Path("configs/plugins/sam3.yaml"),
    show_default=True,
)
@click.option("--bf16", is_flag=True, default=False)
def main(
    cu3s_path: Path,
    processing_mode: str,
    frame_rotation: int | None,
    prompt: str,
    start_frame: int,
    max_frames: int,
    output_dir: Path,
    checkpoint_path: Path | None,
    plugins_yaml: Path,
    bf16: bool,
) -> None:
    """Run SAM3 text-prompt streaming propagation on a CU3S file."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir.mkdir(parents=True, exist_ok=True)
    if start_frame < 0:
        raise click.ClickException("--start-frame must be >= 0.")

    from cuvis_ai_core.data.datasets import SingleCu3sDataModule
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.training import Predictor
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    from cuvis_ai.node.anomaly_visualization import TrackingOverlayNode
    from cuvis_ai.node.channel_selector import RangeAverageFalseRGBSelector
    from cuvis_ai.node.data import CU3SDataNode
    from cuvis_ai.node.json_writer import TrackingCocoJsonNode
    from cuvis_ai.node.video import ToVideoNode

    predict_ids = list(range(start_frame, start_frame + max_frames)) if max_frames > 0 else None
    datamodule = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        processing_mode=processing_mode,
        batch_size=1,
        predict_ids=predict_ids,
    )
    datamodule.setup(stage="predict")
    target_frames = len(datamodule.predict_ds)
    dataset_fps = float(getattr(datamodule.predict_ds, "fps", None) or 10.0)
    logger.info("Frames: {} (start_frame={}, FPS {:.1f})", target_frames, start_frame, dataset_fps)

    manifest = plugins_yaml
    if not manifest.is_absolute():
        manifest = (Path(__file__).resolve().parents[3] / manifest).resolve()
    registry = NodeRegistry()
    registry.load_plugins(str(manifest))
    sam3_cls = registry.get("cuvis_ai_sam3.node.SAM3StreamingPropagation")

    pipeline = CuvisPipeline("SAM3_Text_Propagation")

    cu3s_data = CU3SDataNode(name="cu3s_data")
    false_rgb = RangeAverageFalseRGBSelector(name="false_rgb")
    sam3_node = sam3_cls(
        num_frames=target_frames,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        prompt_type="text",
        prompt_text=prompt,
        name="sam3_streaming",
    )
    tracking_json = TrackingCocoJsonNode(
        output_json_path=str(output_dir / "tracking_results.json"),
        category_name=prompt,
        name="tracking_coco_json",
    )
    overlay_node = TrackingOverlayNode(alpha=0.2, name="overlay")
    overlay_path = output_dir / "tracking_overlay.mp4"
    to_video = ToVideoNode(
        output_video_path=str(overlay_path),
        frame_rate=dataset_fps,
        frame_rotation=frame_rotation,
        name="to_video",
    )

    pipeline.connect(
        (cu3s_data.outputs.cube, false_rgb.cube),
        (cu3s_data.outputs.wavelengths, false_rgb.wavelengths),
        (false_rgb.rgb_image, sam3_node.rgb_frame),
        (sam3_node.frame_id, tracking_json.frame_id),
        (sam3_node.mask, tracking_json.mask),
        (sam3_node.object_ids, tracking_json.object_ids),
        (sam3_node.detection_scores, tracking_json.detection_scores),
        (false_rgb.rgb_image, overlay_node.rgb_image),
        (cu3s_data.outputs.mesu_index, overlay_node.frame_id),
        (sam3_node.mask, overlay_node.mask),
        (sam3_node.object_ids, overlay_node.object_ids),
        (overlay_node.rgb_with_overlay, to_video.rgb_image),
    )

    pipeline_png = output_dir / f"{pipeline.name}.png"
    pipeline.visualize(
        format="render_graphviz", output_path=str(pipeline_png), show_execution_stage=True
    )

    pipeline.to(device)
    pipeline.set_profiling(enabled=True, synchronize_cuda=(device == "cuda"), skip_first_n=3)
    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16 and device == "cuda"
        else contextlib.nullcontext()
    )

    logger.info(
        "Starting text propagation (prompt='{}', {} frames from {})...",
        prompt,
        target_frames,
        start_frame,
    )
    with amp_ctx:
        predictor.predict(max_batches=target_frames, collect_outputs=False)

    summary = pipeline.format_profiling_summary(total_frames=target_frames)
    logger.info("\n{}", summary)
    (output_dir / "profiling_summary.txt").write_text(summary)

    logger.success("Done")
    logger.info("JSON: {}", output_dir / "tracking_results.json")
    logger.info("Overlay: {}", overlay_path)


if __name__ == "__main__":
    main()
