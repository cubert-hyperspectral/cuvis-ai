"""SAM3 bbox-prompt propagation: CU3S -> false RGB -> SAM3 streaming -> overlay + JSON.

Uses bounding boxes from a COCO-format JSON as prompts.  Specify which
detections to track with ``--detection ID@FRAME`` (repeatable).  The ID is
either a ``track_id`` (from a tracking JSON) or a 1-based detection rank by
score (from a detection JSON).  FRAME is the image_id where the bbox is read.

``--start-frame`` controls where the video begins (default 0) — frames before
the prompt frame are rendered without masks so context is visible.

Examples::

    # Best detection on frame 0 (default), video from frame 0
    uv run python examples/object_tracking/sam3/sam3_bbox_propagation.py `
        --cu3s-path "D:\\data\\Auto_013+01.cu3s" `
        --detection-json "D:\\data\\detection_results.json" `
        --output-dir "D:\\experiments\\sam3\\bbox_propagation" `
        --max-frames 100

    # Track ID 2 from frame 76, video starts at frame 0
    uv run python examples/object_tracking/sam3/sam3_bbox_propagation.py `
        --cu3s-path "D:\\data\\Auto_013+01.cu3s" `
        --detection-json "D:\\experiments\\deepeiou\\reid\\tracking_results.json" `
        --detection 2@76 `
        --output-dir "D:\\experiments\\sam3\\bbox_propagation" `
        --max-frames 200

    # Multiple detections, video starts at frame 50
    uv run python examples/object_tracking/sam3/sam3_bbox_propagation.py `
        --cu3s-path "D:\\data\\Auto_013+01.cu3s" `
        --detection-json "D:\\experiments\\deepeiou\\reid\\tracking_results.json" `
        --detection 2@76 --detection 5@76 `
        --start-frame 50 `
        --output-dir "D:\\experiments\\sam3\\bbox_propagation" `
        --max-frames 200
"""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path

import click
import torch
from loguru import logger

PROCESSING_MODES = ("Raw", "DarkSubtract", "Preview", "Reflectance", "SpectralRadiance")


def _parse_detection_spec(spec: str) -> tuple[int, int]:
    """Parse ``ID@FRAME`` into ``(track_or_rank_id, frame_idx)``."""
    m = re.fullmatch(r"(\d+)@(\d+)", spec.strip())
    if not m:
        raise click.BadParameter(
            f"Invalid detection spec '{spec}'. Expected format: ID@FRAME (e.g. 2@76)."
        )
    return int(m.group(1)), int(m.group(2))


def _extract_bbox_prompts(
    detection_json: Path,
    detections: list[tuple[int, int]],
) -> tuple[list[dict], int]:
    """Extract bbox prompts from a COCO-format JSON.

    Parameters
    ----------
    detections
        List of ``(id, frame_idx)`` tuples.  ``id`` is matched against
        ``track_id`` first; if not present, it's treated as a 1-based
        detection rank by score.

    Returns
    -------
    prompts
        List of ``{"obj_id": int, "bbox_xywh": [x, y, w, h]}`` dicts
        with normalised coordinates.
    prompt_frame_idx
        The frame index where prompts are applied (all must share the same frame).
    """
    data = json.loads(detection_json.read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data["images"]}

    # All detections must reference the same frame
    frame_indices = sorted({frame for _, frame in detections})
    if len(frame_indices) != 1:
        raise click.ClickException(
            f"All --detection specs must reference the same frame. Got frames: {frame_indices}"
        )
    frame_idx = frame_indices[0]

    if frame_idx not in images:
        raise click.ClickException(f"Frame {frame_idx} not found in {detection_json}.")
    img = images[frame_idx]
    w_img, h_img = img["width"], img["height"]

    frame_annots = [a for a in data["annotations"] if a["image_id"] == frame_idx]
    has_track_ids = any("track_id" in a for a in frame_annots)

    if has_track_ids:
        by_track = {a["track_id"]: a for a in frame_annots if "track_id" in a}
    else:
        by_track = {}

    # Sort by score for rank-based fallback
    frame_annots_by_score = sorted(frame_annots, key=lambda a: a.get("score", 0.0), reverse=True)

    prompts = []
    for det_id, _ in detections:
        if det_id in by_track:
            a = by_track[det_id]
            obj_id = det_id
        else:
            # Treat as 1-based rank
            rank = det_id - 1
            if rank < 0 or rank >= len(frame_annots_by_score):
                raise click.ClickException(
                    f"Detection rank {det_id} out of range on frame {frame_idx} "
                    f"(have {len(frame_annots_by_score)} detections)."
                )
            a = frame_annots_by_score[rank]
            obj_id = det_id

        x, y, w, h = a["bbox"]
        bbox_norm = [x / w_img, y / h_img, w / w_img, h / h_img]
        prompts.append({"obj_id": obj_id, "bbox_xywh": bbox_norm})
        logger.info(
            "  obj_id={} bbox: [{:.3f}, {:.3f}, {:.3f}, {:.3f}] score={:.3f}",
            obj_id,
            *bbox_norm,
            a.get("score", 0),
        )

    return prompts, frame_idx


@click.command()
@click.option(
    "--cu3s-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option(
    "--detection-json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option(
    "--processing-mode",
    type=click.Choice(PROCESSING_MODES, case_sensitive=False),
    default="SpectralRadiance",
    show_default=True,
)
@click.option("--frame-rotation", type=int, default=None)
@click.option(
    "--detection",
    "detection_specs",
    type=str,
    multiple=True,
    help="Detection to track: ID@FRAME (e.g. 2@76). Repeatable. "
    "Default: best detection on frame 0 (1@0).",
)
@click.option(
    "--start-frame",
    type=int,
    default=0,
    show_default=True,
    help="First frame to include in the video (default 0).",
)
@click.option(
    "--max-frames",
    type=int,
    default=-1,
    show_default=True,
    help="Maximum number of frames to process from start-frame.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("D:/experiments/sam3/bbox_propagation"),
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
    detection_json: Path,
    processing_mode: str,
    frame_rotation: int | None,
    detection_specs: tuple[str, ...],
    start_frame: int,
    max_frames: int,
    output_dir: Path,
    checkpoint_path: Path | None,
    plugins_yaml: Path,
    bf16: bool,
) -> None:
    """Run SAM3 bbox-prompt propagation on a CU3S file."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir.mkdir(parents=True, exist_ok=True)
    if start_frame < 0:
        raise click.ClickException("--start-frame must be >= 0.")

    # Parse detection specs (default: best detection on frame 0)
    if not detection_specs:
        detection_specs = ("1@0",)
    detections = [_parse_detection_spec(s) for s in detection_specs]

    logger.info("Extracting bbox prompts from {}", detection_json)
    prompts, prompt_frame_idx = _extract_bbox_prompts(detection_json, detections)

    if start_frame > prompt_frame_idx:
        raise click.ClickException(
            f"--start-frame ({start_frame}) must be <= prompt frame ({prompt_frame_idx})."
        )

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

    manifest = plugins_yaml
    if not manifest.is_absolute():
        manifest = (Path(__file__).resolve().parents[3] / manifest).resolve()
    registry = NodeRegistry()
    registry.load_plugins(str(manifest))
    sam3_cls = registry.get("cuvis_ai_sam3.node.SAM3ObjectTracker")

    pipeline = CuvisPipeline("SAM3_Bbox_Propagation")

    cu3s_data = CU3SDataNode(name="cu3s_data")
    false_rgb = RangeAverageFalseRGBSelector(name="false_rgb")
    sam3_node = sam3_cls(
        num_frames=target_frames,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        prompt_frame_idx=prompt_frame_idx,
        prompt_bboxes=prompts,
        name="sam3_object_tracker",
    )
    tracking_json = TrackingCocoJsonNode(
        output_json_path=str(output_dir / "tracking_results.json"),
        category_name="person",
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
        (cu3s_data.outputs.mesu_index, sam3_node.frame_id),
        (false_rgb.rgb_image, sam3_node.rgb_frame),
        (cu3s_data.outputs.mesu_index, tracking_json.frame_id),
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
        "Starting bbox propagation ({} prompts on frame {}, {} frames from {})...",
        len(prompts),
        prompt_frame_idx,
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
