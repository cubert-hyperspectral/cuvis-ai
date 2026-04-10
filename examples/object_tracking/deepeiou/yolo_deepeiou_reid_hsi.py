"""YOLO26 + DeepEIoU tracking pipeline for CU3S or RGB video sources.

The script accepts exactly one source:
  - ``--cu3s-path``: hyperspectral CU3S input, converted to false-RGB on the fly
  - ``--video-path``: RGB video input, passed directly into YOLO

ReID embeddings can be enabled via
``BBoxRoiCropNode -> ChannelNormalizeNode -> OSNet/ResNetExtractor``.

Outputs: detection JSON, tracking JSON, overlay video, and optional per-frame
``.npy`` embeddings.

Usage (EIoU-only, video):
    uv run python examples/object_tracking/deepeiou/yolo_deepeiou_reid_hsi.py `
        --video-path source.mp4 --no-reid --end-frame 60
        # default: category 0 (person)
        # use --category-id -1 to track all classes

Usage (ReID, CU3S):
    uv run python examples/object_tracking/deepeiou/yolo_deepeiou_reid_hsi.py `
        --cu3s-path cube.cu3s --with-reid --reid-weights osnet.pth.tar --end-frame 60
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import click
import torch
from loguru import logger
from torch.utils.data import Subset

from cuvis_ai.utils.cli_helpers import (
    append_tracking_metrics,
    resolve_run_output_dir,
    write_experiment_info,
)
from cuvis_ai.utils.false_rgb_sampling import initialize_false_rgb_sampled_fixed


@click.command()
@click.option(
    "--cu3s-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a .cu3s file. False-RGB frames are generated on the fly.",
)
@click.option(
    "--video-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to an MP4 (or other cv2-compatible) video file.",
)
@click.option("--processing-mode", type=str, default="SpectralRadiance", show_default=True)
@click.option("--frame-rotation", type=int, default=None)
@click.option("--start-frame", type=int, default=0, show_default=True)
@click.option(
    "--end-frame",
    type=int,
    default=-1,
    show_default=True,
    help="Stop after this many frames (exclusive). -1 means all frames.",
)
@click.option("--model-name", type=str, default="yolo26n.pt", show_default=True)
@click.option("--confidence-threshold", type=float, default=0.5, show_default=True)
@click.option("--iou-threshold", type=float, default=0.7, show_default=True, help="YOLO NMS IoU")
@click.option(
    "--agnostic-nms",
    is_flag=True,
    default=False,
    help="Enable class-agnostic NMS (off by default).",
)
@click.option(
    "--category-id",
    type=int,
    default=0,
    show_default=True,
    help="Track a single class id. Use -1 to track all classes. Default: 0 (person).",
)
@click.option("--with-reid/--no-reid", default=False, show_default=True)
@click.option(
    "--reid-weights",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to ReID backbone weights. Required when --with-reid is enabled.",
)
@click.option(
    "--backbone",
    type=click.Choice(["osnet", "resnet"]),
    default="osnet",
    show_default=True,
    help="ReID backbone architecture when --with-reid is enabled.",
)
@click.option(
    "--write-features/--no-write-features",
    default=False,
    show_default=True,
    help="Write per-frame .npy embeddings when ReID is enabled.",
)
@click.option("--track-high-thresh", type=float, default=0.6, show_default=True)
@click.option("--track-low-thresh", type=float, default=0.1, show_default=True)
@click.option("--new-track-thresh", type=float, default=0.7, show_default=True)
@click.option("--track-buffer", type=int, default=60, show_default=True)
@click.option("--match-thresh", type=float, default=0.8, show_default=True)
@click.option("--proximity-thresh", type=float, default=0.5, show_default=True)
@click.option("--appearance-thresh", type=float, default=0.25, show_default=True)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("./tracking_output"),
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
    "--plugins-dir",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing ultralytics.yaml and deepeiou.yaml",
)
@click.option("--bf16", is_flag=True, default=False)
@click.option(
    "--hide-untracked/--show-untracked",
    default=True,
    show_default=True,
    help="Hide bboxes without a track ID (track_id=-1) from the overlay.",
)
def main(
    cu3s_path: Path | None,
    video_path: Path | None,
    processing_mode: str,
    frame_rotation: int | None,
    start_frame: int,
    end_frame: int,
    model_name: str,
    confidence_threshold: float,
    iou_threshold: float,
    agnostic_nms: bool,
    category_id: int,
    with_reid: bool,
    reid_weights: Path | None,
    backbone: str,
    write_features: bool,
    track_high_thresh: float,
    track_low_thresh: float,
    new_track_thresh: float,
    track_buffer: int,
    match_thresh: float,
    proximity_thresh: float,
    appearance_thresh: float,
    output_dir: Path,
    out_basename: str | None,
    plugins_dir: Path | None,
    bf16: bool,
    hide_untracked: bool,
) -> None:
    if (cu3s_path is None) == (video_path is None):
        raise click.UsageError("Exactly one of --cu3s-path or --video-path must be provided.")
    if start_frame < 0:
        raise click.BadParameter(
            "--start-frame must be zero or positive", param_hint="--start-frame"
        )
    if end_frame == 0 or end_frame < -1:
        raise click.BadParameter("--end-frame must be -1 or positive", param_hint="--end-frame")
    if category_id < -1:
        raise click.BadParameter(
            "--category-id must be -1 or a non-negative integer",
            param_hint="--category-id",
        )
    if with_reid and reid_weights is None:
        raise click.BadParameter(
            "--reid-weights is required when --with-reid is enabled",
            param_hint="--reid-weights",
        )
    if write_features and not with_reid:
        raise click.BadParameter(
            "--write-features requires --with-reid",
            param_hint="--write-features",
        )

    source_type = "cu3s" if cu3s_path is not None else "video"
    source_path = cu3s_path if source_type == "cu3s" else video_path
    assert source_path is not None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_output_dir = resolve_run_output_dir(
        output_root=output_dir,
        source_path=source_path,
        out_basename=out_basename,
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output run directory: {}", run_output_dir)

    if not with_reid and reid_weights is not None:
        logger.info("Ignoring --reid-weights because --no-reid is set")
    if not with_reid and backbone != "osnet":
        logger.info("Ignoring --backbone because --no-reid is set")
    if source_type == "video":
        logger.info("Ignoring --processing-mode because --video-path is set")

    effective_reid_weights = reid_weights if with_reid else None
    effective_backbone = backbone if with_reid else None
    effective_classes = None if category_id == -1 else [category_id]
    if category_id == -1:
        logger.info("Tracking all classes (--category-id -1)")
    else:
        logger.info("Tracking only class id {}", category_id)
    pipeline_name = (
        f"YOLO_DeepEIoU{'_ReID' if with_reid else ''}_{'HSI' if source_type == 'cu3s' else 'RGB'}"
    )

    write_experiment_info(
        run_output_dir,
        source_type=source_type,
        cu3s_path=cu3s_path,
        video_path=video_path,
        processing_mode=processing_mode if source_type == "cu3s" else None,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        agnostic_nms=agnostic_nms,
        category_id=category_id,
        classes=effective_classes,
        with_reid=with_reid,
        reid_weights=effective_reid_weights,
        backbone=effective_backbone,
        write_features=write_features,
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=new_track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        proximity_thresh=proximity_thresh,
        appearance_thresh=appearance_thresh,
        start_frame=start_frame,
        end_frame=end_frame,
        device=device,
        pipeline_name=pipeline_name,
    )

    from cuvis_ai_core.data.datasets import SingleCu3sDataModule
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.training import Predictor
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    from cuvis_ai.node.anomaly_visualization import BBoxesOverlayNode
    from cuvis_ai.node.channel_selector import CIETristimulusFalseRGBSelector, NormMode
    from cuvis_ai.node.data import CU3SDataNode
    from cuvis_ai.node.json_file import CocoTrackBBoxWriter, DetectionCocoJsonNode
    from cuvis_ai.node.video import ToVideoNode, VideoFrameDataModule, VideoFrameNode

    datamodule: object
    cu3s_data = None
    false_rgb = None
    video_frame = None
    source_rgb_port = None
    source_frame_id_port = None

    if source_type == "cu3s":
        assert cu3s_path is not None

        predict_ids = None
        if start_frame > 0 or end_frame > 0:
            dm_probe = SingleCu3sDataModule(
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

        datamodule = SingleCu3sDataModule(
            cu3s_file_path=str(cu3s_path),
            processing_mode=processing_mode,
            batch_size=1,
            predict_ids=predict_ids,
        )
        datamodule.setup(stage="predict")
        if datamodule.predict_ds is None:
            raise RuntimeError("Predict dataset was not initialized.")

        cu3s_data = CU3SDataNode(name="cu3s_data")
        false_rgb = CIETristimulusFalseRGBSelector(
            norm_mode=NormMode.STATISTICAL,
            name="cie_false_rgb",
        )
        sample_positions = initialize_false_rgb_sampled_fixed(
            false_rgb,
            datamodule.predict_ds,
            sample_fraction=0.05,
        )
        logger.info(
            "False-RGB sampled-fixed calibration: sample_fraction=0.05, sample_count={}",
            len(sample_positions),
        )
        source_rgb_port = false_rgb.rgb_image
        source_frame_id_port = cu3s_data.outputs.mesu_index
    else:
        assert video_path is not None

        datamodule = VideoFrameDataModule(
            video_path=str(video_path),
            end_frame=end_frame,
            batch_size=1,
        )
        datamodule.setup(stage="predict")
        if datamodule.predict_ds is None:
            raise RuntimeError("Predict dataset was not initialized.")

        effective_end = len(datamodule.predict_ds)
        if start_frame > 0:
            datamodule.predict_ds = Subset(datamodule.predict_ds, range(start_frame, effective_end))

        video_frame = VideoFrameNode(name="video_frame")
        source_rgb_port = video_frame.outputs.rgb_image
        source_frame_id_port = video_frame.outputs.frame_id

    target_frames = len(datamodule.predict_ds)
    if target_frames <= 0:
        raise click.ClickException("No frames available for prediction.")
    dataset_fps = float(
        getattr(datamodule, "fps", None) or getattr(datamodule.predict_ds, "fps", None) or 10.0
    )

    _repo_root = Path(__file__).parents[3]
    plugins_dir_resolved = (plugins_dir or (_repo_root / "configs" / "plugins")).resolve()

    registry = NodeRegistry()
    for yaml_name in ("ultralytics.yaml", "deepeiou.yaml"):
        registry.load_plugins(str(plugins_dir_resolved / yaml_name))
        logger.info("Plugin manifest loaded: {}", plugins_dir_resolved / yaml_name)

    yolo_pre_cls = registry.get("cuvis_ai_ultralytics.node.YOLOPreprocess")
    yolo_cls = registry.get("cuvis_ai_ultralytics.node.YOLO26Detection")
    yolo_post_cls = registry.get("cuvis_ai_ultralytics.node.YOLOPostprocess")
    deepeiou_cls = registry.get("cuvis_ai_deepeiou.node.DeepEIoUTrack")

    pipeline = CuvisPipeline(pipeline_name)
    yolo_det = yolo_cls(model_path=model_name, name="yolo26_det")
    yolo_pre = yolo_pre_cls(stride=int(getattr(yolo_det, "stride", 32)), name="yolo_preprocess")
    yolo_post = yolo_post_cls(
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        agnostic_nms=agnostic_nms,
        classes=effective_classes,
        name="yolo_post",
    )
    tracker = deepeiou_cls(
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=new_track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        frame_rate=int(dataset_fps),
        with_reid=with_reid,
        proximity_thresh=proximity_thresh,
        appearance_thresh=appearance_thresh,
        name="deepeiou",
    )
    det_json = DetectionCocoJsonNode(
        output_json_path=str(run_output_dir / "detection_results.json"),
        name="detection_coco_json",
    )
    track_json = CocoTrackBBoxWriter(
        output_json_path=str(run_output_dir / "tracking_results.json"),
        name="tracking_coco_json",
    )
    bbox_overlay = BBoxesOverlayNode(
        name="bboxes_overlay",
        draw_labels=True,
        hide_untracked=hide_untracked,
    )
    overlay_path = run_output_dir / "tracking_overlay.mp4"
    to_video = ToVideoNode(
        output_video_path=str(overlay_path),
        frame_rate=dataset_fps,
        frame_rotation=frame_rotation,
        name="to_video",
    )

    connections: list[tuple[object, object]] = []
    if source_type == "cu3s":
        assert cu3s_data is not None
        assert false_rgb is not None
        connections.extend(
            [
                (cu3s_data.outputs.cube, false_rgb.cube),
                (cu3s_data.outputs.wavelengths, false_rgb.wavelengths),
            ]
        )

    assert source_rgb_port is not None
    assert source_frame_id_port is not None
    connections.extend(
        [
            (source_rgb_port, yolo_pre.rgb_image),
            (yolo_pre.preprocessed, yolo_det.preprocessed),
            (yolo_pre.model_input_hw, yolo_post.model_input_hw),
            (yolo_pre.orig_hw, yolo_post.orig_hw),
            (yolo_det.raw_preds, yolo_post.raw_preds),
            (yolo_post.bboxes, tracker.inputs.bboxes),
            (yolo_post.category_ids, tracker.inputs.category_ids),
            (yolo_post.confidences, tracker.inputs.confidences),
            (source_frame_id_port, det_json.frame_id),
            (yolo_post.bboxes, det_json.bboxes),
            (yolo_post.category_ids, det_json.category_ids),
            (yolo_post.confidences, det_json.confidences),
            (yolo_pre.orig_hw, det_json.orig_hw),
            (source_frame_id_port, track_json.frame_id),
            (tracker.outputs.bboxes, track_json.bboxes),
            (tracker.outputs.category_ids, track_json.category_ids),
            (tracker.outputs.confidences, track_json.confidences),
            (tracker.track_ids, track_json.track_ids),
            (yolo_pre.orig_hw, track_json.orig_hw),
            (source_rgb_port, bbox_overlay.rgb_image),
            (source_frame_id_port, bbox_overlay.frame_id),
            (tracker.outputs.bboxes, bbox_overlay.bboxes),
            (tracker.track_ids, bbox_overlay.category_ids),
            (tracker.outputs.confidences, bbox_overlay.confidences),
            (bbox_overlay.rgb_with_overlay, to_video.rgb_image),
        ]
    )

    if with_reid:
        from cuvis_ai.node.preprocessors import BBoxRoiCropNode, ChannelNormalizeNode

        extractor_cls = registry.get(
            "cuvis_ai_deepeiou.node.OSNetExtractor"
            if backbone == "osnet"
            else "cuvis_ai_deepeiou.node.ResNetExtractor"
        )
        crop = BBoxRoiCropNode(output_size=(256, 128), name="roi_crop")
        normalize = ChannelNormalizeNode(name="normalize")
        extractor = extractor_cls(model_path=str(reid_weights), name="reid_extractor")

        connections.extend(
            [
                (source_rgb_port, crop.images),
                (yolo_post.bboxes, crop.bboxes),
                (crop.crops, normalize.images),
                (normalize.normalized, extractor.crops),
                (extractor.embeddings, tracker.inputs.embeddings),
            ]
        )

        if write_features:
            from cuvis_ai.node.numpy_writer import NumpyFeatureWriterNode

            writer = NumpyFeatureWriterNode(
                output_dir=str(run_output_dir / "features"),
                name="feature_writer",
            )
            connections.extend(
                [
                    (extractor.embeddings, writer.features),
                    (source_frame_id_port, writer.frame_id),
                ]
            )

    pipeline.connect(*connections)

    pipeline_png = run_output_dir / f"{pipeline.name}.png"
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

    if source_type == "video":
        assert video_path is not None
        logger.info("Source: video {}", video_path)
    else:
        assert cu3s_path is not None
        logger.info("Source: CU3S {}", cu3s_path)

    if with_reid:
        logger.info(
            "Starting DeepEIoU+ReID tracking ({} {} frames, {} backbone) on {}",
            target_frames,
            source_type,
            backbone,
            device,
        )
    else:
        logger.info(
            "Starting DeepEIoU tracking ({} {} frames) on {}", target_frames, source_type, device
        )

    with amp_ctx:
        predictor.predict(max_batches=target_frames, collect_outputs=False)

    summary = pipeline.format_profiling_summary(total_frames=target_frames)
    logger.info("\n{}", summary)
    (run_output_dir / "profiling_summary.txt").write_text(summary)

    logger.success("Tracking complete -> {}", run_output_dir)
    logger.info("Overlay: {}", overlay_path)
    logger.info("Detections: {}", run_output_dir / "detection_results.json")
    logger.info("Tracks: {}", run_output_dir / "tracking_results.json")
    if write_features:
        logger.info("Features: {}", run_output_dir / "features")

    append_tracking_metrics(
        run_output_dir / "experiment_info.txt", run_output_dir / "tracking_results.json"
    )


if __name__ == "__main__":
    main()
