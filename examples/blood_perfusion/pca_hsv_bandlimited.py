"""Band-limited HSV-colored PCA blood-perfusion video export for CU3S sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import Predictor, StatisticalTrainer
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger

from cuvis_ai.node.colormap import ScalarHSVColormapNode
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.dimensionality_reduction import PCA, TrainablePCA
from cuvis_ai.node.metrics import ExplainedVarianceMetric
from cuvis_ai.node.normalization import MinMaxNormalizer, PerPixelUnitNorm
from cuvis_ai.node.preprocessors import BandpassByWavelength
from cuvis_ai.node.video import ToVideoNode
from cuvis_ai.utils.cli_helpers import resolve_run_output_dir

PROCESSING_MODES = ("Raw", "DarkSubtract", "Preview", "Reflectance", "SpectralRadiance")
PCA_MODES = ("per_frame", "global")
DEFAULT_CU3S_PATH = Path("data/XMR_Blood_Perfusion/Blood_Perfusion_Refl.cu3s")
DEFAULT_OUTPUT_DIR = Path("tracking_output/pca_hsv_bandlimited_blood_perfusion")
DEFAULT_PCA_MODE = "global"
DEFAULT_COLORMAP_MIN = 0.0
DEFAULT_COLORMAP_MAX = 1.0
DEFAULT_BAND_MIN_NM = 540.0
DEFAULT_BAND_MAX_NM = 800.0
PCA_COMPONENTS = 3
DEFAULT_COMPONENT_INDEX = 0
DEFAULT_INVERT_COMPONENT = False


class SelectProjectedComponentNode(Node):
    """Select one component from a PCA-projected BHWC tensor."""

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Projected PCA tensor [B, H, W, C].",
        )
    }
    OUTPUT_SPECS = {
        "selected": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Selected PCA component [B, H, W, 1].",
        )
    }

    def __init__(
        self,
        component_index: int = DEFAULT_COMPONENT_INDEX,
        invert_component: bool = DEFAULT_INVERT_COMPONENT,
        **kwargs: Any,
    ) -> None:
        if component_index < 0:
            raise ValueError("component_index must be >= 0")
        self.component_index = int(component_index)
        self.invert_component = bool(invert_component)
        super().__init__(
            component_index=self.component_index,
            invert_component=self.invert_component,
            **kwargs,
        )

    def forward(self, data: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Extract one component from a BHWC projected tensor."""
        if data.ndim != 4:
            raise ValueError(
                f"Expected projected data with shape [B, H, W, C], got {tuple(data.shape)}"
            )
        if self.component_index >= data.shape[-1]:
            raise ValueError(
                f"component_index {self.component_index} is out of range for last dimension {data.shape[-1]}"
            )
        selected = data[..., self.component_index : self.component_index + 1]
        if self.invert_component:
            selected = -selected
        return {"selected": selected}


def _format_variance_ratios(values: torch.Tensor | np.ndarray | list[float]) -> str:
    ratios = [float(value) for value in values]
    return ", ".join(f"PC{idx}={ratio:.4f}" for idx, ratio in enumerate(ratios))


def _log_pca_component_selection(
    *,
    pca_node: Node,
    resolved_pca_mode: str,
    resolved_component_index: int,
    invert_component: bool,
) -> None:
    logger.info(
        "Selected PCA component: component_index={} invert_component={}",
        resolved_component_index,
        invert_component,
    )
    if resolved_component_index != DEFAULT_COMPONENT_INDEX:
        logger.warning(
            "component_index={} is exploratory for Blood_Perfusion_Refl; component_index={} is the "
            "recommended perfusion comparison view.",
            resolved_component_index,
            DEFAULT_COMPONENT_INDEX,
        )

    if resolved_pca_mode != "global":
        logger.info("Explained variance ratios are frame-dependent in per-frame PCA mode.")
        return

    explained_variance = getattr(pca_node, "_explained_variance", None)
    variance_ratio_fn = getattr(pca_node, "_variance_ratio", None)
    if not isinstance(explained_variance, torch.Tensor) or not callable(variance_ratio_fn):
        logger.info("Explained variance ratios are unavailable for the current PCA node instance.")
        return

    ratios = variance_ratio_fn(explained_variance.detach().cpu()).tolist()
    logger.info("Explained variance ratios: {}", _format_variance_ratios(ratios))


def _resolve_processing_mode(processing_mode: str) -> str:
    lookup = {mode.lower(): mode for mode in PROCESSING_MODES}
    resolved = lookup.get(processing_mode.strip().lower())
    if resolved is None:
        raise click.BadParameter(
            f"Invalid processing mode '{processing_mode}'. Supported: {', '.join(PROCESSING_MODES)}"
        )
    return resolved


def _resolve_pca_mode(pca_mode: str) -> str:
    lookup = {mode.lower(): mode for mode in PCA_MODES}
    resolved = lookup.get(pca_mode.strip().lower())
    if resolved is None:
        raise click.BadParameter(
            f"Invalid pca_mode '{pca_mode}'. Supported: {', '.join(PCA_MODES)}"
        )
    return resolved


def _validate_frame_window(start_frame: int, end_frame: int) -> None:
    if start_frame < 0:
        raise click.BadParameter("--start-frame must be zero or positive.")
    if end_frame == 0 or end_frame < -1:
        raise click.BadParameter("--end-frame must be -1 or a positive integer.")
    if end_frame != -1 and end_frame <= start_frame:
        raise click.BadParameter("--end-frame must be greater than --start-frame.")


def _validate_init_frames(init_frames: int | None) -> int | None:
    if init_frames is None:
        return None
    if init_frames <= 0:
        raise click.BadParameter("--init-frames must be a positive integer.")
    return int(init_frames)


def _validate_component_index(component_index: int) -> int:
    if component_index < 0 or component_index >= PCA_COMPONENTS:
        raise click.BadParameter(
            f"--component-index must be in [0, {PCA_COMPONENTS - 1}] for n_components={PCA_COMPONENTS}."
        )
    return int(component_index)


def _count_bandpass_channels(
    wavelengths: np.ndarray,
    *,
    band_min_nm: float,
    band_max_nm: float,
) -> int:
    keep_mask = (wavelengths >= band_min_nm) & (wavelengths <= band_max_nm)
    return int(np.count_nonzero(keep_mask))


def _probe_dataset_window(
    cu3s_file_path: str,
    processing_mode: str,
    *,
    band_min_nm: float,
    band_max_nm: float,
    start_frame: int,
    end_frame: int,
) -> tuple[list[int], list[int] | None, int, float, int]:
    probe_dm = SingleCu3sDataModule(
        cu3s_file_path=cu3s_file_path,
        processing_mode=processing_mode,
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

    selected_first = probe_dm.predict_ds[start_frame]
    wavelengths = np.asarray(selected_first["wavelengths"], dtype=np.float32).reshape(-1)
    bandpassed_channels = _count_bandpass_channels(
        wavelengths,
        band_min_nm=band_min_nm,
        band_max_nm=band_max_nm,
    )
    if bandpassed_channels == 0:
        raise click.BadParameter(
            f"No wavelengths fall within [{band_min_nm}, {band_max_nm}] nm for this CU3S session."
        )
    if bandpassed_channels < PCA_COMPONENTS:
        raise click.BadParameter(
            f"Bandpass leaves only {bandpassed_channels} channels, but n_components={PCA_COMPONENTS}."
        )

    dataset_fps = float(getattr(probe_dm.predict_ds, "fps", None) or 10.0)
    if dataset_fps <= 0:
        dataset_fps = 10.0
        logger.warning("Could not infer positive FPS from dataset; using fallback 10.0")

    selected_ids = list(range(start_frame, effective_end))
    predict_ids = None if start_frame == 0 and effective_end == total_available else selected_ids
    target_frames = effective_end - start_frame
    return selected_ids, predict_ids, target_frames, dataset_fps, bandpassed_channels


def _resolve_train_ids(
    selected_ids: list[int],
    *,
    pca_mode: str,
    init_frames: int | None,
) -> list[int]:
    if pca_mode != "global" or init_frames is None:
        return selected_ids

    resolved_count = min(init_frames, len(selected_ids))
    if resolved_count < init_frames:
        logger.warning(
            "Requested init_frames={} but only {} frames are available in the selected window; "
            "using all selected frames for initialization.",
            init_frames,
            len(selected_ids),
        )
    return selected_ids[:resolved_count]


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


def export_pca_hsv_bandlimited_video(
    *,
    cu3s_file_path: str,
    output_video_path: str,
    pca_mode: str = DEFAULT_PCA_MODE,
    processing_mode: str = "Reflectance",
    band_min_nm: float = DEFAULT_BAND_MIN_NM,
    band_max_nm: float = DEFAULT_BAND_MAX_NM,
    component_index: int = DEFAULT_COMPONENT_INDEX,
    invert_component: bool = DEFAULT_INVERT_COMPONENT,
    colormap_min: float = DEFAULT_COLORMAP_MIN,
    colormap_max: float = DEFAULT_COLORMAP_MAX,
    start_frame: int = 0,
    end_frame: int = -1,
    init_frames: int | None = None,
    frame_rate: float | None = None,
    frame_rotation: int | None = None,
    save_pipeline_config: bool = False,
) -> Path:
    """Run CU3S -> bandpass -> per-pixel norm -> PCA -> component -> statistical HSV -> MP4 export."""
    output_path = Path(output_video_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".log")
    log_sink_id = logger.add(log_path, level="INFO", mode="w")

    try:
        logger.info("Writing band-limited PCA HSV logs to {}", log_path)

        resolved_mode = _resolve_processing_mode(processing_mode)
        resolved_pca_mode = _resolve_pca_mode(pca_mode)
        resolved_component_index = _validate_component_index(component_index)
        _validate_frame_window(start_frame, end_frame)
        resolved_init_frames = _validate_init_frames(init_frames)
        if band_max_nm <= band_min_nm:
            raise click.BadParameter("--band-max-nm must be greater than --band-min-nm.")
        if colormap_max <= colormap_min:
            raise click.BadParameter("--colormap-max must be greater than --colormap-min.")

        selected_ids, predict_ids, target_frames, dataset_fps, bandpassed_channels = (
            _probe_dataset_window(
                cu3s_file_path=cu3s_file_path,
                processing_mode=resolved_mode,
                band_min_nm=band_min_nm,
                band_max_nm=band_max_nm,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
        train_ids = _resolve_train_ids(
            selected_ids,
            pca_mode=resolved_pca_mode,
            init_frames=resolved_init_frames,
        )

        if frame_rate is not None and frame_rate <= 0:
            raise click.BadParameter("--frame-rate must be > 0.")
        resolved_frame_rate = float(frame_rate) if frame_rate is not None else float(dataset_fps)

        datamodule = SingleCu3sDataModule(
            cu3s_file_path=cu3s_file_path,
            processing_mode=resolved_mode,
            batch_size=1,
            train_ids=train_ids,
            predict_ids=predict_ids,
        )

        pipeline_name = (
            "PCA_PerFrame_HSV_Bandlimited_Projection"
            if resolved_pca_mode == "per_frame"
            else "PCA_Global_HSV_Bandlimited_Projection"
        )
        pipeline = CuvisPipeline(pipeline_name)

        cu3s_data = CU3SDataNode(name="cu3s_data")
        bandpass = BandpassByWavelength(
            min_wavelength_nm=band_min_nm,
            max_wavelength_nm=band_max_nm,
            name="bandpass",
        )
        per_pixel_unit_norm = PerPixelUnitNorm(name="per_pixel_unit_norm")
        if resolved_pca_mode == "per_frame":
            pca_node: Node = PCA(n_components=PCA_COMPONENTS, name="pca_per_frame")
        else:
            pca_node = TrainablePCA(
                num_channels=bandpassed_channels,
                n_components=PCA_COMPONENTS,
                whiten=False,
                init_method="svd",
                name="pca_global",
            )

        explained_variance = ExplainedVarianceMetric(
            execution_stages={ExecutionStage.ALWAYS},
            name="explained_variance_metric",
        )
        component_selector = SelectProjectedComponentNode(
            component_index=resolved_component_index,
            invert_component=invert_component,
            name="component_selector",
        )
        projection_normalizer = MinMaxNormalizer(
            eps=1.0e-6,
            use_running_stats=True,
            name="projection_normalizer",
        )
        hsv_colormap = ScalarHSVColormapNode(
            value_min=colormap_min,
            value_max=colormap_max,
            name="projection_hsv_colormap",
        )
        to_video = ToVideoNode(
            output_video_path=str(output_path),
            frame_rate=resolved_frame_rate,
            frame_rotation=frame_rotation,
            name="to_video",
        )

        pipeline.connect(
            (cu3s_data.outputs.cube, bandpass.data),
            (cu3s_data.outputs.wavelengths, bandpass.wavelengths),
            (bandpass.filtered, per_pixel_unit_norm.data),
            (per_pixel_unit_norm.normalized, pca_node.data),
            (
                pca_node.outputs.explained_variance_ratio,
                explained_variance.explained_variance_ratio,
            ),
            (pca_node.outputs.projected, component_selector.data),
            (component_selector.selected, projection_normalizer.data),
            (projection_normalizer.normalized, hsv_colormap.data),
            (hsv_colormap.rgb_image, to_video.rgb_image),
            (cu3s_data.outputs.mesu_index, to_video.frame_id),
        )

        run_output_dir = output_path.parent
        _save_pipeline_artifacts(
            pipeline,
            run_output_dir,
            save_pipeline_config=save_pipeline_config,
        )

        device = torch.device("cpu")
        pipeline.to(device)

        logger.info(
            "Starting band-limited PCA HSV export: file={} mode={} processing_mode={} frames={} "
            "init_frames={} band_min_nm={} band_max_nm={} band_channels={} component_index={} "
            "invert_component={} fps={} colormap_min={} colormap_max={} device={}",
            cu3s_file_path,
            resolved_pca_mode,
            resolved_mode,
            target_frames,
            len(train_ids),
            band_min_nm,
            band_max_nm,
            bandpassed_channels,
            resolved_component_index,
            invert_component,
            resolved_frame_rate,
            colormap_min,
            colormap_max,
            device,
        )

        stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
        stat_trainer.fit()
        _log_pca_component_selection(
            pca_node=pca_node,
            resolved_pca_mode=resolved_pca_mode,
            resolved_component_index=resolved_component_index,
            invert_component=invert_component,
        )

        predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
        predictor.predict(max_batches=target_frames, collect_outputs=False)

        if not output_path.exists():
            raise RuntimeError(f"Expected output video was not created: {output_path}")

        logger.success("Band-limited PCA HSV export complete: {}", output_path)
        logger.info("Log file: {}", log_path)
        return output_path
    finally:
        logger.remove(log_sink_id)


@click.command()
@click.option(
    "--cu3s-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_CU3S_PATH,
    show_default=True,
    help="Path to the blood-perfusion CU3S session.",
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
    "--pca-mode",
    type=click.Choice(PCA_MODES, case_sensitive=False),
    default=DEFAULT_PCA_MODE,
    show_default=True,
    help="Use independent per-frame PCA or one global PCA fit across the selected frame window.",
)
@click.option(
    "--processing-mode",
    type=click.Choice(PROCESSING_MODES, case_sensitive=False),
    default="Reflectance",
    show_default=True,
)
@click.option("--band-min-nm", type=float, default=DEFAULT_BAND_MIN_NM, show_default=True)
@click.option("--band-max-nm", type=float, default=DEFAULT_BAND_MAX_NM, show_default=True)
@click.option(
    "--component-index",
    type=int,
    default=DEFAULT_COMPONENT_INDEX,
    show_default=True,
    help=(
        f"0-based PCA component to render. Valid range: 0..{PCA_COMPONENTS - 1}. "
        "For Blood_Perfusion_Refl, component 0 is the recommended perfusion comparison view."
    ),
)
@click.option(
    "--invert-component/--no-invert-component",
    default=DEFAULT_INVERT_COMPONENT,
    show_default=True,
    help="Flip the sign of the selected PCA component before normalization and HSV rendering.",
)
@click.option("--colormap-min", type=float, default=DEFAULT_COLORMAP_MIN, show_default=True)
@click.option("--colormap-max", type=float, default=DEFAULT_COLORMAP_MAX, show_default=True)
@click.option("--start-frame", type=int, default=0, show_default=True)
@click.option(
    "--end-frame",
    type=int,
    default=-1,
    show_default=True,
    help="Stop at this source frame index (exclusive). -1 means all frames.",
)
@click.option(
    "--init-frames",
    type=int,
    default=None,
    help=(
        "For global PCA, statistically initialize on the first N frames of the selected "
        "window, then predict over the full selected window. Defaults to the full window."
    ),
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
    pca_mode: str,
    processing_mode: str,
    band_min_nm: float,
    band_max_nm: float,
    component_index: int,
    invert_component: bool,
    colormap_min: float,
    colormap_max: float,
    start_frame: int,
    end_frame: int,
    init_frames: int | None,
    frame_rate: float | None,
    frame_rotation: int | None,
    save_pipeline_config: bool,
) -> None:
    """Export a band-limited HSV-colored PCA visualization video for blood perfusion."""
    run_output_dir = resolve_run_output_dir(
        output_root=output_dir,
        source_path=cu3s_path,
        out_basename=out_basename,
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = run_output_dir / "pca_hsv_bandlimited_projection.mp4"

    export_pca_hsv_bandlimited_video(
        cu3s_file_path=str(cu3s_path),
        output_video_path=str(output_video_path),
        pca_mode=pca_mode,
        processing_mode=processing_mode,
        band_min_nm=band_min_nm,
        band_max_nm=band_max_nm,
        component_index=component_index,
        invert_component=invert_component,
        colormap_min=colormap_min,
        colormap_max=colormap_max,
        start_frame=start_frame,
        end_frame=end_frame,
        init_frames=init_frames,
        frame_rate=frame_rate,
        frame_rotation=frame_rotation,
        save_pipeline_config=save_pipeline_config,
    )


if __name__ == "__main__":
    main()
