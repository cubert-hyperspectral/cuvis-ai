"""NDVI blood-perfusion video export for CU3S sessions."""

from __future__ import annotations

from pathlib import Path

import click
import torch
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import Predictor
from loguru import logger

from cuvis_ai.node.channel_selector import NDVISelector
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.video import ToVideoNode
from cuvis_ai.utils.cli_helpers import resolve_run_output_dir

PROCESSING_MODE = "Reflectance"
DEFAULT_CU3S_PATH = Path("data/XMR_Blood_Perfusion/Blood_Perfusion_Refl.cu3s")
DEFAULT_OUTPUT_DIR = Path("tracking_output/ndvi_blood_perfusion")
DEFAULT_WL_1_NM = 750.0
DEFAULT_WL_2_NM = 566.0


def _validate_frame_window(start_frame: int, end_frame: int) -> None:
    if start_frame < 0:
        raise click.BadParameter("--start-frame must be zero or positive.")
    if end_frame == 0 or end_frame < -1:
        raise click.BadParameter("--end-frame must be -1 or a positive integer.")
    if end_frame != -1 and end_frame <= start_frame:
        raise click.BadParameter("--end-frame must be greater than --start-frame.")


def _probe_dataset_window(
    cu3s_file_path: str,
    *,
    start_frame: int,
    end_frame: int,
) -> tuple[list[int], list[int] | None, int, float]:
    probe_dm = SingleCu3sDataModule(
        cu3s_file_path=cu3s_file_path,
        processing_mode=PROCESSING_MODE,
        batch_size=1,
    )
    probe_dm.setup(stage="predict")
    if probe_dm.predict_ds is None:
        raise RuntimeError("Predict dataset was not initialized.")

    total_available = len(probe_dm.predict_ds)
    if total_available <= 0:
        raise click.ClickException("No frames available for prediction.")
    if start_frame >= total_available:
        raise click.BadParameter(
            f"--start-frame {start_frame} is outside the available range [0, {total_available - 1}]"
        )

    effective_end = min(end_frame, total_available) if end_frame > 0 else total_available
    if effective_end <= start_frame:
        raise click.BadParameter("--end-frame must be greater than --start-frame.")

    dataset_fps = float(getattr(probe_dm.predict_ds, "fps", None) or 10.0)
    if dataset_fps <= 0:
        dataset_fps = 10.0
        logger.warning("Could not infer positive FPS from dataset; using fallback 10.0")

    selected_ids = list(range(start_frame, effective_end))
    predict_ids = None if start_frame == 0 and effective_end == total_available else selected_ids
    target_frames = effective_end - start_frame
    return selected_ids, predict_ids, target_frames, dataset_fps


def _save_pipeline_artifacts(
    pipeline: CuvisPipeline,
    run_output_dir: Path,
    *,
    save_pipeline_config: bool,
) -> None:
    pipeline.visualize(
        format="render_graphviz",
        output_path=str(run_output_dir / f"{pipeline.name}.png"),
        show_execution_stage=True,
    )
    pipeline.visualize(
        format="render_mermaid",
        output_path=str(run_output_dir / f"{pipeline.name}.md"),
        direction="LR",
        include_node_class=True,
        wrap_markdown=True,
        show_execution_stage=True,
    )
    if save_pipeline_config:
        pipeline_path = run_output_dir / f"{pipeline.name}.yaml"
        pipeline.save_to_file(str(pipeline_path))
        logger.info("Pipeline config saved: {}", pipeline_path)


def export_ndvi_video(
    *,
    cu3s_file_path: str,
    output_video_path: str,
    nir_nm: float = DEFAULT_WL_1_NM,
    red_nm: float = DEFAULT_WL_2_NM,
    colormap_min: float = -0.7,
    colormap_max: float = 0.5,
    start_frame: int = 0,
    end_frame: int = -1,
    frame_rate: float | None = None,
    frame_rotation: int | None = None,
    save_pipeline_config: bool = False,
) -> Path:
    """Run CU3S -> normalized difference -> Blood_OXY HSV colormap -> MP4 export."""
    _validate_frame_window(start_frame, end_frame)
    selected_ids, predict_ids, target_frames, dataset_fps = _probe_dataset_window(
        cu3s_file_path=cu3s_file_path,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    if frame_rate is not None and frame_rate <= 0:
        raise click.BadParameter("--frame-rate must be > 0.")
    if colormap_max <= colormap_min:
        raise click.BadParameter("--colormap-max must be greater than --colormap-min.")
    resolved_frame_rate = float(frame_rate) if frame_rate is not None else float(dataset_fps)

    datamodule = SingleCu3sDataModule(
        cu3s_file_path=cu3s_file_path,
        processing_mode=PROCESSING_MODE,
        batch_size=1,
        predict_ids=predict_ids,
    )

    pipeline = CuvisPipeline("BloodPerfusion_NDVI_Projection")
    cu3s_data = CU3SDataNode(name="cu3s_data")
    ndvi = NDVISelector(
        nir_nm=nir_nm,
        red_nm=red_nm,
        colormap_min=colormap_min,
        colormap_max=colormap_max,
        name="ndvi",
    )
    to_video = ToVideoNode(
        output_video_path=output_video_path,
        frame_rate=resolved_frame_rate,
        frame_rotation=frame_rotation,
        name="to_video",
    )

    pipeline.connect(
        (cu3s_data.outputs.cube, ndvi.cube),
        (cu3s_data.outputs.wavelengths, ndvi.wavelengths),
        (ndvi.rgb_image, to_video.rgb_image),
        (cu3s_data.outputs.mesu_index, to_video.frame_id),
    )

    run_output_dir = Path(output_video_path).resolve().parent
    _save_pipeline_artifacts(
        pipeline,
        run_output_dir,
        save_pipeline_config=save_pipeline_config,
    )

    device = torch.device("cpu")
    pipeline.to(device)

    logger.info(
        "Starting NDVI export: file={} processing_mode={} frames={} fps={} nir_nm={} red_nm={} "
        "colormap={} colormap_min={} colormap_max={} frame_ids={} device={}",
        cu3s_file_path,
        PROCESSING_MODE,
        target_frames,
        resolved_frame_rate,
        nir_nm,
        red_nm,
        "hsv",
        colormap_min,
        colormap_max,
        selected_ids,
        device,
    )

    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
    predictor.predict(max_batches=target_frames, collect_outputs=False)

    output_path = Path(output_video_path)
    if not output_path.exists():
        raise RuntimeError(f"Expected output video was not created: {output_path}")

    logger.success("NDVI export complete: {}", output_path)
    return output_path


@click.command()
@click.option(
    "--cu3s-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_CU3S_PATH,
    show_default=True,
    help="Path to the blood-perfusion reflectance CU3S session.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help=(
        "Parent output directory. Final run folder is "
        "<output-dir>/<out-basename or input-file-stem>."
    ),
)
@click.option(
    "--out-basename",
    type=str,
    default=None,
    help="Optional leaf run-folder name under --output-dir (must not include '/' or '\\').",
)
@click.option(
    "--nir-nm",
    type=float,
    default=DEFAULT_WL_1_NM,
    show_default=True,
    help="First wavelength operand in nm. Defaults match the Blood_OXY plugin preset.",
)
@click.option(
    "--red-nm",
    type=float,
    default=DEFAULT_WL_2_NM,
    show_default=True,
    help="Second wavelength operand in nm. Defaults match the Blood_OXY plugin preset.",
)
@click.option("--colormap-min", type=float, default=-0.7, show_default=True)
@click.option("--colormap-max", type=float, default=0.5, show_default=True)
@click.option("--start-frame", type=int, default=0, show_default=True)
@click.option(
    "--end-frame",
    type=int,
    default=-1,
    show_default=True,
    help="Stop at this source frame index (exclusive). -1 means all frames.",
)
@click.option(
    "--frame-rate",
    type=float,
    default=None,
    help="Optional FPS override. Defaults to dataset FPS or 10.0 when missing.",
)
@click.option("--frame-rotation", type=int, default=None)
@click.option(
    "--save-pipeline-config/--no-save-pipeline-config",
    default=False,
    show_default=True,
    help="Save pipeline YAML/PT alongside the video output.",
)
def main(
    cu3s_path: Path,
    output_dir: Path,
    out_basename: str | None,
    nir_nm: float,
    red_nm: float,
    colormap_min: float,
    colormap_max: float,
    start_frame: int,
    end_frame: int,
    frame_rate: float | None,
    frame_rotation: int | None,
    save_pipeline_config: bool,
) -> None:
    """Export an NDVI visualization video for the blood-perfusion CU3S session."""
    run_output_dir = resolve_run_output_dir(
        output_root=output_dir,
        source_path=cu3s_path,
        out_basename=out_basename,
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = run_output_dir / "ndvi_projection.mp4"

    export_ndvi_video(
        cu3s_file_path=str(cu3s_path),
        output_video_path=str(output_video_path),
        nir_nm=nir_nm,
        red_nm=red_nm,
        colormap_min=colormap_min,
        colormap_max=colormap_max,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_rate=frame_rate,
        frame_rotation=frame_rotation,
        save_pipeline_config=save_pipeline_config,
    )


if __name__ == "__main__":
    main()
