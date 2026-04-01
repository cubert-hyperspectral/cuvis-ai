"""Occlude CU3S data with pure PyTorch Poisson inpainting.

Pipeline:
    --occlude-target=rgb:
        CU3SDataNode -> CIETristimulusFalseRGBSelector -> PoissonOcclusionNode(rgb) -> ToVideoNode

    --occlude-target=cube:
        CU3SDataNode -> PoissonOcclusionNode(cube) -> CIETristimulusFalseRGBSelector -> ToVideoNode

Both paths are fully PyTorch-based (no cv2 roundtrip).

Example::

    uv run python examples/object_tracking/occlusion/occlude_data.py `
      --cu3s-path "D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01.cu3s" `
      --tracking-json "D:\\experiments\\sam3\\20260316\\...\\tracking_results.json" `
      --track-ids "2,8" `
      --occlusion-start-frame 70 --occlusion-end-frame 90 `
      --start-frame 60 --end-frame 201 `
      --occlusion-shape bbox --bbox-mode static --static-bbox-scale 1.2 `
      --occlude-target cube `
      --output-video-path "D:\\experiments\\sam3\\20260318\\...\\static_bbox_poisson.mp4"
"""

from __future__ import annotations

from pathlib import Path

import click
import torch
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import Predictor
from loguru import logger

from cuvis_ai.node.channel_selector import CIETristimulusFalseRGBSelector, NormMode
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.occlusion import PoissonOcclusionNode
from cuvis_ai.node.video import ToVideoNode
from cuvis_ai.utils.false_rgb_sampling import initialize_false_rgb_sampled_fixed

PROCESSING_MODES = ("Raw", "DarkSubtract", "Preview", "Reflectance", "SpectralRadiance")


def _resolve_processing_mode(mode: str) -> str:
    lookup = {m.lower(): m for m in PROCESSING_MODES}
    resolved = lookup.get(mode.lower())
    if resolved is None:
        raise click.BadParameter(
            f"Invalid processing mode '{mode}'. Supported: {', '.join(PROCESSING_MODES)}"
        )
    return resolved


@click.command()
@click.option(
    "--cu3s-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option(
    "--tracking-json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option(
    "--track-ids", type=str, required=True, help="Comma-separated track IDs (e.g. '2,8')."
)
@click.option("--occlusion-start-frame", type=int, required=True)
@click.option("--occlusion-end-frame", type=int, required=True)
@click.option(
    "--occlusion-shape",
    type=click.Choice(["bbox", "mask"], case_sensitive=False),
    default="bbox",
    show_default=True,
)
@click.option(
    "--bbox-mode",
    type=click.Choice(["static", "dynamic"], case_sensitive=False),
    default="static",
    show_default=True,
)
@click.option("--static-bbox-scale", type=float, default=1.2, show_default=True)
@click.option("--static-bbox-padding-px", type=int, default=0, show_default=True)
@click.option("--static-full-width-x/--no-static-full-width-x", default=False, show_default=True)
@click.option(
    "--max-iter", type=int, default=1000, show_default=True, help="CG solver iteration limit."
)
@click.option(
    "--tol", type=float, default=1e-6, show_default=True, help="CG solver convergence tolerance."
)
@click.option(
    "--occlude-target",
    type=click.Choice(["rgb", "cube"], case_sensitive=False),
    default="rgb",
    show_default=True,
    help="Apply occlusion in false-RGB space (rgb) or raw cube space (cube).",
)
@click.option("--output-video-path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option(
    "--processing-mode",
    type=click.Choice(PROCESSING_MODES, case_sensitive=False),
    default="SpectralRadiance",
    show_default=True,
)
@click.option("--start-frame", type=int, default=0, show_default=True)
@click.option("--end-frame", type=int, default=-1, show_default=True, help="-1 = all frames.")
@click.option("--frame-rate", type=float, default=None)
@click.option("--frame-rotation", type=int, default=None)
@click.option("--overlay-frame-id/--no-overlay-frame-id", default=True, show_default=True)
@click.option(
    "--sample-fraction",
    type=float,
    default=0.05,
    show_default=True,
    help="Fraction of frames for false-RGB calibration.",
)
def main(
    cu3s_path: Path,
    tracking_json: Path,
    track_ids: str,
    occlusion_start_frame: int,
    occlusion_end_frame: int,
    occlusion_shape: str,
    bbox_mode: str,
    static_bbox_scale: float,
    static_bbox_padding_px: int,
    static_full_width_x: bool,
    max_iter: int,
    tol: float,
    occlude_target: str,
    output_video_path: Path,
    processing_mode: str,
    start_frame: int,
    end_frame: int,
    frame_rate: float | None,
    frame_rotation: int | None,
    overlay_frame_id: bool,
    sample_fraction: float,
) -> None:
    parsed_track_ids = [int(t.strip()) for t in track_ids.split(",") if t.strip()]
    if not parsed_track_ids:
        raise click.BadParameter("--track-ids must contain at least one track ID")
    resolved_mode = _resolve_processing_mode(processing_mode)

    # --- Datamodule ---
    predict_ids = None
    if start_frame > 0 or end_frame > 0:
        dm_probe = SingleCu3sDataModule(
            cu3s_file_path=str(cu3s_path),
            processing_mode=resolved_mode,
            batch_size=1,
        )
        dm_probe.setup(stage="predict")
        total = len(dm_probe.predict_ds)
        eff_end = min(end_frame, total) if end_frame > 0 else total
        predict_ids = list(range(start_frame, eff_end))

    datamodule = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        processing_mode=resolved_mode,
        batch_size=1,
        predict_ids=predict_ids,
    )
    datamodule.setup(stage="predict")
    target_frames = len(datamodule.predict_ds)

    dataset_fps = float(
        getattr(datamodule, "fps", None) or getattr(datamodule.predict_ds, "fps", None) or 10.0
    )
    if dataset_fps <= 0:
        dataset_fps = 10.0
    fps = frame_rate if frame_rate is not None else dataset_fps

    # --- Nodes ---
    cu3s_data = CU3SDataNode(name="cu3s_data")

    false_rgb = CIETristimulusFalseRGBSelector(norm_mode=NormMode.STATISTICAL, name="cie_false_rgb")
    sample_positions = initialize_false_rgb_sampled_fixed(
        false_rgb, datamodule.predict_ds, sample_fraction=sample_fraction
    )
    logger.info("False-RGB calibration: {} sample frames", len(sample_positions))

    resolved_input_key = "rgb_image" if occlude_target.lower() == "rgb" else "cube"
    occlusion_node = PoissonOcclusionNode(
        tracking_json_path=str(tracking_json),
        track_ids=parsed_track_ids,
        occlusion_start_frame=occlusion_start_frame,
        occlusion_end_frame=occlusion_end_frame,
        fill_color="poisson",
        input_key=resolved_input_key,
        max_iter=max_iter,
        tol=tol,
        occlusion_shape=occlusion_shape,
        bbox_mode=bbox_mode,
        static_bbox_scale=static_bbox_scale,
        static_bbox_padding_px=static_bbox_padding_px,
        static_full_width_x=static_full_width_x,
        name="occlusion",
    )

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_profiling_summary_path = output_video_path.with_suffix(".profiling_summary.txt")
    resolved_profiling_summary_path.parent.mkdir(parents=True, exist_ok=True)
    to_video = ToVideoNode(
        output_video_path=str(output_video_path),
        frame_rate=fps,
        frame_rotation=frame_rotation,
        name="to_video",
    )

    # --- Pipeline ---
    occlude_target = occlude_target.lower()
    if occlude_target == "rgb":
        pipeline_name = "Occlude_Poisson_RGB"
        connections = [
            (cu3s_data.outputs.cube, false_rgb.cube),
            (cu3s_data.outputs.wavelengths, false_rgb.wavelengths),
            (false_rgb.rgb_image, occlusion_node.inputs.rgb_image),
            (cu3s_data.outputs.mesu_index, occlusion_node.inputs.frame_id),
            (occlusion_node.outputs.rgb_image, to_video.inputs.rgb_image),
        ]
    else:
        pipeline_name = "Occlude_Poisson_Cube"
        connections = [
            (cu3s_data.outputs.cube, occlusion_node.inputs.cube),
            (cu3s_data.outputs.mesu_index, occlusion_node.inputs.frame_id),
            (occlusion_node.outputs.cube, false_rgb.cube),
            (cu3s_data.outputs.wavelengths, false_rgb.wavelengths),
            (false_rgb.rgb_image, to_video.inputs.rgb_image),
        ]

    pipeline = CuvisPipeline(pipeline_name)
    if overlay_frame_id:
        connections.append((cu3s_data.outputs.mesu_index, to_video.inputs.frame_id))
    pipeline.connect(*connections)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    pipeline.set_profiling(enabled=True, synchronize_cuda=(device == "cuda"), skip_first_n=3)

    logger.info(
        "Poisson on {}: {} frames, max_iter={}, tol={}, device={}",
        occlude_target,
        target_frames,
        max_iter,
        tol,
        device,
    )
    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
    predictor.predict(max_batches=target_frames, collect_outputs=False)
    summary = pipeline.format_profiling_summary(total_frames=target_frames)
    logger.info("\n{}", summary)
    resolved_profiling_summary_path.write_text(summary, encoding="utf-8")
    logger.info("Profiling summary -> {}", resolved_profiling_summary_path)
    logger.success("Done -> {}", output_video_path)


if __name__ == "__main__":
    main()
