"""YOLO26 + DeepEIoU CU3S tracking pipeline.

Reads a CU3S hyperspectral cube, runs YOLO detection on false-RGB, and tracks
with DeepEIoU (EIoU-only mode in Phase 1 — no ReID embeddings).

Example (10 frames):
    uv run python examples/object_tracking/deepeiou/yolo_deepeiou_hsi.py \\
        --cu3s-path cube.cu3s --end-frame 10

Example (full sequence):
    uv run python examples/object_tracking/deepeiou/yolo_deepeiou_hsi.py \\
        --cu3s-path cube.cu3s
"""

from __future__ import annotations

import contextlib
import datetime
from pathlib import Path

import click
import torch
from loguru import logger


def _write_experiment_info(output_dir: Path, **params: object) -> None:
    """Write an ``experiment_info.txt`` alongside outputs for traceability."""
    lines = [
        f"Experiment: {output_dir.name}",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Parameters:",
    ]
    for k, v in params.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    (output_dir / "experiment_info.txt").write_text("\n".join(lines), encoding="utf-8")


def _append_metrics(info_path: Path, tracking_json_path: Path) -> None:
    """Append diagnostic metrics to the experiment info file."""
    import collections
    import json

    try:
        data = json.loads(tracking_json_path.read_text(encoding="utf-8"))
    except Exception:
        return

    annots = data.get("annotations", [])
    n_frames = len(data.get("images", []))
    frame_tracks: dict[int, set[int]] = collections.defaultdict(set)
    all_ids: set[int] = set()
    for a in annots:
        tid = a.get("track_id", -1)
        if tid == -1:
            continue
        frame_tracks[a["image_id"]].add(tid)
        all_ids.add(tid)

    counts = [len(frame_tracks.get(i, set())) for i in range(n_frames)]
    avg = sum(counts) / len(counts) if counts else 0.0
    mx = max(counts) if counts else 0
    zeros = sum(1 for c in counts if c == 0)

    lines = [
        "Results:",
        f"  frames: {n_frames}",
        f"  unique_track_ids: {len(all_ids)}",
        f"  avg_tracks_per_frame: {avg:.1f}",
        f"  max_tracks_per_frame: {mx}",
        f"  zero_track_frames: {zeros}",
        "",
    ]
    with info_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


@click.command()
@click.option(
    "--cu3s-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option("--processing-mode", type=str, default="SpectralRadiance", show_default=True)
@click.option("--frame-rotation", type=int, default=None)
@click.option("--end-frame", type=int, default=-1, show_default=True)
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
    "--classes",
    type=int,
    multiple=True,
    help="Limit NMS to these class ids (repeat flag). Default: keep all classes.",
)
@click.option("--track-high-thresh", type=float, default=0.6, show_default=True)
@click.option("--track-low-thresh", type=float, default=0.1, show_default=True)
@click.option("--new-track-thresh", type=float, default=0.7, show_default=True)
@click.option("--track-buffer", type=int, default=60, show_default=True)
@click.option("--match-thresh", type=float, default=0.8, show_default=True)
@click.option("--with-reid/--no-reid", default=False, show_default=True)
@click.option("--proximity-thresh", type=float, default=0.5, show_default=True)
@click.option("--appearance-thresh", type=float, default=0.25, show_default=True)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("./tracking_output"),
    show_default=True,
)
@click.option(
    "--plugins-dir",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    default=None,
    show_default=True,
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
    cu3s_path: Path,
    processing_mode: str,
    frame_rotation: int | None,
    end_frame: int,
    model_name: str,
    confidence_threshold: float,
    iou_threshold: float,
    agnostic_nms: bool,
    classes: tuple[int, ...],
    track_high_thresh: float,
    track_low_thresh: float,
    new_track_thresh: float,
    track_buffer: int,
    match_thresh: float,
    with_reid: bool,
    proximity_thresh: float,
    appearance_thresh: float,
    output_dir: Path,
    plugins_dir: Path | None,
    bf16: bool,
    hide_untracked: bool,
) -> None:
    if end_frame == 0 or end_frame < -1:
        raise click.BadParameter("--end-frame must be -1 or positive")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_experiment_info(
        output_dir,
        cu3s_path=cu3s_path,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=new_track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        with_reid=with_reid,
        proximity_thresh=proximity_thresh,
        appearance_thresh=appearance_thresh,
        end_frame=end_frame,
        device=device,
    )

    from cuvis_ai_core.data.datasets import SingleCu3sDataModule
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.training import Predictor
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    from cuvis_ai.node.anomaly_visualization import BBoxesOverlayNode
    from cuvis_ai.node.channel_selector import CIETristimulusFalseRGBSelector
    from cuvis_ai.node.data import CU3SDataNode
    from cuvis_ai.node.json_writer import ByteTrackCocoJson, DetectionCocoJsonNode
    from cuvis_ai.node.video import ToVideoNode

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
    for yaml_name in ("ultralytics.yaml", "deepeiou.yaml"):
        registry.load_plugins(str(plugins_dir_resolved / yaml_name))
        logger.info("Plugin manifest loaded: {}", plugins_dir_resolved / yaml_name)

    yolo_pre_cls = registry.get("cuvis_ai_ultralytics.node.YOLOPreprocess")
    yolo_cls = registry.get("cuvis_ai_ultralytics.node.YOLO26Detection")
    yolo_post_cls = registry.get("cuvis_ai_ultralytics.node.YOLOPostprocess")
    deepeiou_cls = registry.get("cuvis_ai_deepeiou.node.DeepEIoUTrack")

    pipeline = CuvisPipeline("YOLO_DeepEIoU_HSI")

    cu3s_data = CU3SDataNode(name="cu3s_data")
    false_rgb = CIETristimulusFalseRGBSelector(name="cie_false_rgb")
    yolo_det = yolo_cls(model_path=model_name, name="yolo26_det")
    yolo_pre = yolo_pre_cls(stride=int(getattr(yolo_det, "stride", 32)), name="yolo_preprocess")
    yolo_post = yolo_post_cls(
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        agnostic_nms=agnostic_nms,
        classes=list(classes) if classes else None,
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
        output_json_path=str(output_dir / "detection_results.json"),
        name="detection_coco_json",
    )
    track_json = ByteTrackCocoJson(
        output_json_path=str(output_dir / "tracking_results.json"),
        name="tracking_coco_json",
    )

    bbox_overlay = BBoxesOverlayNode(
        name="bboxes_overlay",
        draw_labels=True,
        hide_untracked=hide_untracked,
    )
    overlay_path = output_dir / "tracking_overlay.mp4"
    to_video = ToVideoNode(
        output_video_path=str(overlay_path),
        frame_rate=dataset_fps,
        frame_rotation=frame_rotation,
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
        # YOLO → DeepEIoU
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

    pipeline.connect(*connections)

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
        "Starting DeepEIoU tracking ({} frames) on {} [with_reid={}]",
        target_frames,
        device,
        with_reid,
    )
    with amp_ctx:
        predictor.predict(max_batches=target_frames, collect_outputs=False)

    summary = pipeline.format_profiling_summary(total_frames=target_frames)
    logger.info("\n{}", summary)
    (output_dir / "profiling_summary.txt").write_text(summary)

    logger.success("Tracking complete")
    logger.info("Overlay: {}", overlay_path)
    logger.info("Detections: {}", output_dir / "detection_results.json")
    logger.info("Tracks: {}", output_dir / "tracking_results.json")

    _append_metrics(output_dir / "experiment_info.txt", output_dir / "tracking_results.json")


if __name__ == "__main__":
    main()
