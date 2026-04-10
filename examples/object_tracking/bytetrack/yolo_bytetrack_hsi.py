"""YOLO26 + ByteTrack CU3S tracking pipeline with optional spectral association.

Reads a CU3S hyperspectral cube, runs YOLO detection on false-RGB, and tracks
with ByteTrack.  When ``--association-mode`` is set to ``spectral_cost`` or
``spectral_post_gate``, a BBoxSpectralExtractor feeds per-detection spectral
signatures into the tracker for spectral-aware association.

Example (baseline, 60 frames):
    uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py `
        --cu3s-path cube.cu3s --end-frame 60

Example (spectral cost):
    uv run python examples/object_tracking/bytetrack/yolo_bytetrack_hsi.py `
        --cu3s-path cube.cu3s --association-mode spectral_cost `
        --spectral-cost-weight 0.3 --end-frame 60
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import click
import torch
from loguru import logger

from cuvis_ai.utils.cli_helpers import (
    append_tracking_metrics,
    resolve_run_output_dir,
    write_experiment_info,
)
from cuvis_ai.utils.false_rgb_sampling import initialize_false_rgb_sampled_fixed


@click.command()
@click.option(
    "--cu3s-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option("--processing-mode", type=str, default="SpectralRadiance", show_default=True)
@click.option("--frame-rotation", type=int, default=None)
@click.option("--end-frame", type=int, default=-1, show_default=True)
@click.option("--model-name", type=str, default="yolo26n.pt", show_default=True)
@click.option("--confidence-threshold", type=float, default=0.2, show_default=True)
@click.option("--iou-threshold", type=float, default=0.7, show_default=True, help="YOLO NMS IoU")
@click.option(
    "--agnostic-nms",
    is_flag=True,
    default=False,
    help="Enable class-agnostic NMS (off by default).",
)
@click.option(
    "--classes",
    type=int,
    multiple=True,
    help="Limit NMS to these class ids (repeat flag). Default: keep all classes.",
)
@click.option("--track-thresh", type=float, default=0.5, show_default=True)
@click.option("--track-buffer", type=int, default=30, show_default=True)
@click.option("--match-thresh", type=float, default=0.8, show_default=True)
@click.option(
    "--second-score-thresh",
    type=float,
    default=0.1,
    show_default=True,
    help="Low-score floor used for ByteTrack second association.",
)
@click.option(
    "--second-match-thresh",
    type=float,
    default=0.5,
    show_default=True,
    help="IoU match threshold for ByteTrack second association stage.",
)
@click.option(
    "--unconfirmed-match-thresh",
    type=float,
    default=0.7,
    show_default=True,
    help="IoU threshold when matching unconfirmed tracks.",
)
@click.option(
    "--new-track-thresh-offset",
    type=float,
    default=0.1,
    show_default=True,
    help="Offset in det_thresh = track_thresh + offset for new track activation.",
)
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
    show_default=True,
    help="Directory containing ultralytics.yaml and bytetrack.yaml",
)
@click.option("--bf16", is_flag=True, default=False)
# -- Spectral association options ------------------------------------------
@click.option(
    "--association-mode",
    type=click.Choice(["baseline", "spectral_cost", "spectral_post_gate"], case_sensitive=False),
    default="baseline",
    show_default=True,
    help="ByteTrack association strategy.",
)
@click.option(
    "--spectral-cost-weight",
    type=float,
    default=0.3,
    show_default=True,
    help="Weight for spectral cost in all association stages.",
)
@click.option(
    "--prototype-ema-beta",
    type=float,
    default=0.1,
    show_default=True,
    help="EMA weight for track spectral prototypes.",
)
@click.option(
    "--prototype-min-sim",
    type=float,
    default=0.5,
    show_default=True,
    help="Min cosine similarity for prototype matching.",
)
@click.option(
    "--prototype-min-det-score",
    type=float,
    default=0.3,
    show_default=True,
    help="Min detection score to update prototype.",
)
@click.option(
    "--spectral-center-crop",
    type=float,
    default=0.65,
    show_default=True,
    help="Center-crop scale for spectral extraction.",
)
@click.option(
    "--spectral-sim-floor",
    type=float,
    default=0.4,
    show_default=True,
    help="Similarity floor for post-gate mode.",
)
@click.option(
    "--prototype-decay/--no-prototype-decay",
    default=False,
    show_default=True,
    help="Enable prototype staleness decay for lost tracks.",
)
@click.option(
    "--prototype-decay-half-life",
    type=float,
    default=10.0,
    show_default=True,
    help="Half-life (frames) for prototype decay.",
)
@click.option(
    "--spectral-std-weighting/--no-spectral-std-weighting",
    default=False,
    show_default=True,
    help="Use spectral std to modulate spectral cost weight per-detection.",
)
@click.option(
    "--spectral-std-alpha",
    type=float,
    default=1.0,
    show_default=True,
    help="Scaling factor for std confidence: 1/(1+alpha*mean_std).",
)
@click.option(
    "--hide-untracked/--show-untracked",
    default=True,
    show_default=True,
    help="Hide bboxes without a track ID (track_id=-1) from the overlay.",
)
@click.option(
    "--draw-spectral-sparklines/--no-draw-spectral-sparklines",
    default=False,
    show_default=True,
    help="Draw spectral sparklines on overlay bboxes.",
)
@click.option(
    "--sparkline-height",
    type=int,
    default=24,
    show_default=True,
    help="Pixel height of sparkline bars in overlay.",
)
def main(
    cu3s_path: Path,
    processing_mode: str,
    frame_rotation: int | None,
    end_frame: int,
    model_name: str,
    confidence_threshold: float,
    iou_threshold: float,
    agnostic_nms: bool,
    classes: tuple[int, ...],
    track_thresh: float,
    track_buffer: int,
    match_thresh: float,
    second_score_thresh: float,
    second_match_thresh: float,
    unconfirmed_match_thresh: float,
    new_track_thresh_offset: float,
    output_dir: Path,
    out_basename: str | None,
    plugins_dir: Path | None,
    bf16: bool,
    association_mode: str,
    spectral_cost_weight: float,
    prototype_ema_beta: float,
    prototype_min_sim: float,
    prototype_min_det_score: float,
    spectral_center_crop: float,
    spectral_sim_floor: float,
    prototype_decay: bool,
    prototype_decay_half_life: float,
    spectral_std_weighting: bool,
    spectral_std_alpha: float,
    hide_untracked: bool,
    draw_spectral_sparklines: bool,
    sparkline_height: int,
) -> None:
    if end_frame == 0 or end_frame < -1:
        raise click.BadParameter("--end-frame must be -1 or positive")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_output_dir = resolve_run_output_dir(
        output_root=output_dir,
        source_path=cu3s_path,
        out_basename=out_basename,
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output run directory: {}", run_output_dir)

    # Write experiment context file alongside outputs.
    write_experiment_info(
        run_output_dir,
        cu3s_path=cu3s_path,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        association_mode=association_mode,
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        second_score_thresh=second_score_thresh,
        spectral_cost_weight=spectral_cost_weight,
        spectral_sim_floor=spectral_sim_floor,
        prototype_decay=prototype_decay,
        prototype_decay_half_life=prototype_decay_half_life,
        spectral_std_weighting=spectral_std_weighting,
        spectral_std_alpha=spectral_std_alpha,
        end_frame=end_frame,
        device=device,
    )

    from cuvis_ai_core.data.datasets import SingleCu3sDataModule
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.training import Predictor
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    from cuvis_ai.node.anomaly_visualization import BBoxesOverlayNode
    from cuvis_ai.node.channel_selector import CIETristimulusFalseRGBSelector, NormMode
    from cuvis_ai.node.data import CU3SDataNode
    from cuvis_ai.node.json_file import CocoTrackBBoxWriter, DetectionCocoJsonNode
    from cuvis_ai.node.spectral_extractor import BBoxSpectralExtractor
    from cuvis_ai.node.video import ToVideoNode

    use_spectral = association_mode != "baseline"

    predict_ids = list(range(end_frame)) if end_frame > 0 else None
    datamodule = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        processing_mode=processing_mode,
        batch_size=1,
        predict_ids=predict_ids,
    )
    datamodule.setup(stage="predict")
    target_frames = len(datamodule.predict_ds)
    dataset_fps = float(getattr(datamodule.predict_ds, "fps", None) or 10.0)

    _repo_root = Path(__file__).parents[3]
    plugins_dir_resolved = (plugins_dir or (_repo_root / "configs" / "plugins")).resolve()

    registry = NodeRegistry()
    for yaml_name in ("ultralytics.yaml", "bytetrack.yaml"):
        registry.load_plugins(str(plugins_dir_resolved / yaml_name))
        logger.info("Plugin manifest loaded: {}", plugins_dir_resolved / yaml_name)

    yolo_pre_cls = registry.get("cuvis_ai_ultralytics.node.YOLOPreprocess")
    yolo_cls = registry.get("cuvis_ai_ultralytics.node.YOLO26Detection")
    yolo_post_cls = registry.get("cuvis_ai_ultralytics.node.YOLOPostprocess")
    bytetrack_cls = registry.get("cuvis_ai_bytetrack.node.ByteTrack")

    pipeline = CuvisPipeline("YOLO_ByteTrack_HSI")

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
    yolo_det = yolo_cls(model_path=model_name, name="yolo26_det")
    yolo_pre = yolo_pre_cls(stride=int(getattr(yolo_det, "stride", 32)), name="yolo_preprocess")
    yolo_post = yolo_post_cls(
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        agnostic_nms=agnostic_nms,
        classes=list(classes) if classes else None,
        name="yolo_post",
    )
    tracker_kwargs: dict = {
        "track_thresh": track_thresh,
        "track_buffer": track_buffer,
        "match_thresh": match_thresh,
        "second_score_thresh": second_score_thresh,
        "second_match_thresh": second_match_thresh,
        "unconfirmed_match_thresh": unconfirmed_match_thresh,
        "new_track_thresh_offset": new_track_thresh_offset,
        "frame_rate": int(dataset_fps),
        "name": "bytetrack",
    }
    if use_spectral:
        tracker_kwargs.update(
            association_mode=association_mode,
            spectral_cost_weight_first=spectral_cost_weight,
            spectral_cost_weight_second=spectral_cost_weight,
            spectral_cost_weight_unconfirmed=spectral_cost_weight,
            prototype_ema_beta=prototype_ema_beta,
            prototype_min_sim=prototype_min_sim,
            prototype_min_det_score=prototype_min_det_score,
            spectral_sim_floor_post_gate=spectral_sim_floor,
            prototype_decay_enabled=prototype_decay,
            prototype_decay_half_life=prototype_decay_half_life,
            spectral_std_weighting_enabled=spectral_std_weighting,
            spectral_std_alpha=spectral_std_alpha,
        )
    tracker = bytetrack_cls(**tracker_kwargs)
    det_json = DetectionCocoJsonNode(
        output_json_path=str(run_output_dir / "detection_results.json"),
        name="detection_coco_json",
    )
    track_json = CocoTrackBBoxWriter(
        output_json_path=str(run_output_dir / "tracking_results.json"),
        name="tracking_coco_json",
    )
    if use_spectral:
        spectral_extractor = BBoxSpectralExtractor(
            center_crop_scale=spectral_center_crop,
            name="spectral_extractor",
        )

    bbox_overlay = BBoxesOverlayNode(
        name="bboxes_overlay",
        draw_labels=True,
        draw_sparklines=draw_spectral_sparklines and use_spectral,
        sparkline_height=sparkline_height,
        hide_untracked=hide_untracked,
    )
    overlay_path = run_output_dir / "tracking_overlay.mp4"
    to_video = ToVideoNode(
        output_video_path=str(overlay_path),
        frame_rate=dataset_fps,
        frame_rotation=frame_rotation,
        # codec="VP90",
        name="to_video",
    )

    connections = [
        # CU3S → false RGB → YOLO
        (cu3s_data.outputs.cube, false_rgb.cube),
        (cu3s_data.outputs.wavelengths, false_rgb.wavelengths),
        (false_rgb.rgb_image, yolo_pre.rgb_image),
        (yolo_pre.preprocessed, yolo_det.preprocessed),
        (yolo_pre.model_input_hw, yolo_post.model_input_hw),
        (yolo_pre.orig_hw, yolo_post.orig_hw),
        (yolo_det.raw_preds, yolo_post.raw_preds),
        # YOLO → ByteTrack
        (yolo_post.bboxes, tracker.inputs.bboxes),
        (yolo_post.category_ids, tracker.inputs.category_ids),
        (yolo_post.confidences, tracker.inputs.confidences),
        # Detection JSON (pre-tracking)
        (cu3s_data.outputs.mesu_index, det_json.frame_id),
        (yolo_post.bboxes, det_json.bboxes),
        (yolo_post.category_ids, det_json.category_ids),
        (yolo_post.confidences, det_json.confidences),
        (yolo_pre.orig_hw, det_json.orig_hw),
        # Tracking JSON (post-tracking)
        (cu3s_data.outputs.mesu_index, track_json.frame_id),
        (tracker.outputs.bboxes, track_json.bboxes),
        (tracker.outputs.category_ids, track_json.category_ids),
        (tracker.outputs.confidences, track_json.confidences),
        (tracker.track_ids, track_json.track_ids),
        (yolo_pre.orig_hw, track_json.orig_hw),
        # Video overlay with track ID labels
        (false_rgb.rgb_image, bbox_overlay.rgb_image),
        (cu3s_data.outputs.mesu_index, bbox_overlay.frame_id),
        (tracker.outputs.bboxes, bbox_overlay.bboxes),
        (tracker.track_ids, bbox_overlay.category_ids),
        (tracker.outputs.confidences, bbox_overlay.confidences),
        (bbox_overlay.rgb_with_overlay, to_video.rgb_image),
    ]

    if use_spectral:
        connections.extend(
            [
                # Spectral extractor: cube + bboxes → signatures + valid
                (cu3s_data.outputs.cube, spectral_extractor.cube),
                (yolo_post.bboxes, spectral_extractor.bboxes),
                (spectral_extractor.spectral_signatures, tracker.inputs.spectral_signatures),
                (spectral_extractor.spectral_valid, tracker.inputs.spectral_valid),
            ]
        )
        if spectral_std_weighting:
            connections.append(
                (spectral_extractor.spectral_std, tracker.inputs.spectral_std),
            )
        if draw_spectral_sparklines:
            connections.append(
                (spectral_extractor.spectral_signatures, bbox_overlay.spectral_signatures),
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

    logger.info(
        "Starting tracking ({} frames) on {} [mode={}]",
        target_frames,
        device,
        association_mode,
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

    # Append diagnostic metrics to experiment_info.txt.
    append_tracking_metrics(
        run_output_dir / "experiment_info.txt", run_output_dir / "tracking_results.json"
    )


if __name__ == "__main__":
    main()
