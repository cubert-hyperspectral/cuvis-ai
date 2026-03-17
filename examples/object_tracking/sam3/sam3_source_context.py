"""Shared source/context helpers for SAM3 examples.

Supports selecting exactly one source type:
- CU3S (with false-RGB conversion)
- RGB video (direct frames)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from loguru import logger
from torch.utils.data import Subset

PROCESSING_MODES = ("Raw", "DarkSubtract", "Preview", "Reflectance", "SpectralRadiance")


@dataclass
class SourceContext:
    source_type: str
    datamodule: object
    source_rgb_port: Any
    source_frame_id_port: Any
    source_connections: list[tuple[object, object]]
    dataset_fps: float
    target_frames: int


def resolve_processing_mode(processing_mode: str) -> str:
    lookup = {mode.lower(): mode for mode in PROCESSING_MODES}
    resolved = lookup.get(processing_mode.lower())
    if resolved is None:
        raise click.BadParameter(
            f"Invalid processing mode '{processing_mode}'. Supported: {', '.join(PROCESSING_MODES)}"
        )
    return resolved


def resolve_plugin_manifest(plugins_yaml: Path) -> Path:
    plugin_manifest = plugins_yaml
    if not plugin_manifest.is_absolute():
        plugin_manifest = (Path(__file__).resolve().parents[3] / plugin_manifest).resolve()
    plugin_manifest = plugin_manifest.resolve()
    if not plugin_manifest.exists():
        raise click.ClickException(f"Plugins manifest not found: {plugin_manifest}")
    if not plugin_manifest.is_file():
        raise click.ClickException(f"Plugins manifest is not a file: {plugin_manifest}")
    return plugin_manifest


def validate_source_and_window(
    *,
    cu3s_path: Path | None,
    video_path: Path | None,
    start_frame: int,
    end_frame: int,
) -> None:
    if (cu3s_path is None) == (video_path is None):
        raise click.UsageError("Exactly one of --cu3s-path or --video-path must be provided.")
    if start_frame < 0:
        raise click.BadParameter(
            "--start-frame must be zero or positive", param_hint="--start-frame"
        )
    if end_frame == 0 or end_frame < -1:
        raise click.BadParameter("--end-frame must be -1 or positive", param_hint="--end-frame")
    if end_frame != -1 and end_frame <= start_frame:
        raise click.BadParameter(
            "--end-frame must be greater than --start-frame",
            param_hint="--end-frame",
        )


def resolve_end_frame(
    *,
    start_frame: int,
    end_frame: int,
    max_frames: int | None,
) -> int:
    if max_frames is None:
        return end_frame
    if max_frames == -1:
        derived_end = -1
    elif max_frames <= 0:
        raise click.BadParameter("--max-frames must be -1 or positive", param_hint="--max-frames")
    else:
        derived_end = start_frame + max_frames
    if end_frame != -1 and derived_end != -1 and end_frame != derived_end:
        raise click.BadParameter(
            "--end-frame and --max-frames conflict; use one or set consistent values.",
            param_hint="--end-frame",
        )
    return derived_end


def build_source_context(
    *,
    cu3s_path: Path | None,
    video_path: Path | None,
    processing_mode: str,
    start_frame: int,
    end_frame: int,
    single_cu3s_datamodule_cls: type | None = None,
    cu3s_data_node_cls: type | None = None,
    false_rgb_selector_cls: type | None = None,
    false_rgb_norm_mode: object | None = None,
    video_frame_datamodule_cls: type | None = None,
    video_frame_node_cls: type | None = None,
    subset_cls: type = Subset,
    false_rgb_initializer: Any | None = None,
) -> SourceContext:
    source_type = "cu3s" if cu3s_path is not None else "video"

    datamodule: object
    source_connections: list[tuple[object, object]] = []
    source_rgb_port: Any
    source_frame_id_port: Any

    if source_type == "cu3s":
        assert cu3s_path is not None

        if single_cu3s_datamodule_cls is None:
            from cuvis_ai_core.data.datasets import SingleCu3sDataModule

            single_cu3s_datamodule_cls = SingleCu3sDataModule
        if cu3s_data_node_cls is None:
            from cuvis_ai.node.data import CU3SDataNode

            cu3s_data_node_cls = CU3SDataNode
        if false_rgb_selector_cls is None:
            from cuvis_ai.node.channel_selector import CIETristimulusFalseRGBSelector, NormMode

            false_rgb_selector_cls = CIETristimulusFalseRGBSelector
            false_rgb_norm_mode = NormMode.STATISTICAL
        if false_rgb_initializer is None:
            from cuvis_ai.utils.false_rgb_sampling import initialize_false_rgb_sampled_fixed

            false_rgb_initializer = initialize_false_rgb_sampled_fixed

        predict_ids = None
        if start_frame > 0 or end_frame > 0:
            dm_probe = single_cu3s_datamodule_cls(
                cu3s_file_path=str(cu3s_path),
                processing_mode=processing_mode,
                batch_size=1,
            )
            dm_probe.setup(stage="predict")
            if dm_probe.predict_ds is None:
                raise RuntimeError("Predict dataset was not initialized.")
            total_available = len(dm_probe.predict_ds)
            effective_end = min(end_frame, total_available) if end_frame > 0 else total_available
            predict_ids = list(range(start_frame, effective_end))

        datamodule = single_cu3s_datamodule_cls(
            cu3s_file_path=str(cu3s_path),
            processing_mode=processing_mode,
            batch_size=1,
            predict_ids=predict_ids,
        )
        datamodule.setup(stage="predict")
        if datamodule.predict_ds is None:
            raise RuntimeError("Predict dataset was not initialized.")

        cu3s_data = cu3s_data_node_cls(name="cu3s_data")
        false_rgb_kwargs: dict[str, object] = {"name": "cie_false_rgb"}
        if false_rgb_norm_mode is not None:
            false_rgb_kwargs["norm_mode"] = false_rgb_norm_mode
        false_rgb = false_rgb_selector_cls(**false_rgb_kwargs)
        sample_positions = false_rgb_initializer(
            false_rgb,
            datamodule.predict_ds,
            sample_fraction=0.05,
        )
        logger.info(
            "False-RGB sampled-fixed calibration: sample_fraction=0.05, sample_count={}",
            len(sample_positions),
        )
        source_connections.extend(
            [
                (cu3s_data.outputs.cube, false_rgb.cube),
                (cu3s_data.outputs.wavelengths, false_rgb.wavelengths),
            ]
        )
        source_rgb_port = false_rgb.rgb_image
        source_frame_id_port = cu3s_data.outputs.mesu_index
    else:
        assert video_path is not None

        if video_frame_datamodule_cls is None:
            from cuvis_ai.node.video import VideoFrameDataModule

            video_frame_datamodule_cls = VideoFrameDataModule
        if video_frame_node_cls is None:
            from cuvis_ai.node.video import VideoFrameNode

            video_frame_node_cls = VideoFrameNode

        datamodule = video_frame_datamodule_cls(
            video_path=str(video_path),
            end_frame=end_frame,
            batch_size=1,
        )
        datamodule.setup(stage="predict")
        if datamodule.predict_ds is None:
            raise RuntimeError("Predict dataset was not initialized.")

        effective_end = len(datamodule.predict_ds)
        if start_frame > 0:
            datamodule.predict_ds = subset_cls(
                datamodule.predict_ds, range(start_frame, effective_end)
            )

        video_frame = video_frame_node_cls(name="video_frame")
        source_rgb_port = video_frame.outputs.rgb_image
        source_frame_id_port = video_frame.outputs.frame_id

    target_frames = len(datamodule.predict_ds)
    if target_frames <= 0:
        raise click.ClickException("No frames available for prediction.")

    dataset_fps = float(
        getattr(datamodule, "fps", None) or getattr(datamodule.predict_ds, "fps", None) or 10.0
    )
    if dataset_fps <= 0:
        dataset_fps = 10.0
        logger.warning("Invalid FPS from dataset; falling back to 10.0.")

    return SourceContext(
        source_type=source_type,
        datamodule=datamodule,
        source_rgb_port=source_rgb_port,
        source_frame_id_port=source_frame_id_port,
        source_connections=source_connections,
        dataset_fps=dataset_fps,
        target_frames=target_frames,
    )
