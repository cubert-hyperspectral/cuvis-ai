"""Statistical-style training for class-wise Stateful SAM pipelines.

This mirrors the staged style used by statistical examples:
1) setup data module
2) build pipeline graph
3) fit class signatures from COCO bbox supervision
4) save deployable YAML/PT artifacts
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import torch
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.pipeline import PipelineMetadata
from loguru import logger

from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.spectral_angle_mapper import StatefulSpectralAngleMapper


@dataclass(frozen=True)
class ClassSample:
    class_id: int
    class_name: str
    frame_index: int
    bbox_xywh: tuple[float, float, float, float]


def _to_xyxy(bbox_xywh: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox_xywh
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    return x1, y1, x2, y2


def _extract_signature_from_bbox(
    cube_hwc: np.ndarray,
    bbox_xywh: tuple[float, float, float, float],
    center_crop_scale: float = 0.65,
    trim_fraction: float = 0.10,
) -> np.ndarray:
    height, width, channels = cube_hwc.shape
    x1, y1, x2, y2 = _to_xyxy(bbox_xywh)
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after clamping: {(x1, y1, x2, y2)}")

    bw = x2 - x1
    bh = y2 - y1
    mx = int(np.floor(bw * (1.0 - center_crop_scale) * 0.5))
    my = int(np.floor(bh * (1.0 - center_crop_scale) * 0.5))
    cx1 = min(max(x1 + mx, 0), width)
    cy1 = min(max(y1 + my, 0), height)
    cx2 = min(max(x2 - mx, 0), width)
    cy2 = min(max(y2 - my, 0), height)
    if cx2 <= cx1 or cy2 <= cy1:
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2

    crop = cube_hwc[cy1:cy2, cx1:cx2, :]
    pixels = crop.reshape(-1, channels)
    finite = np.isfinite(pixels).all(axis=1)
    pixels = pixels[finite]
    if pixels.shape[0] == 0:
        raise ValueError("No valid pixels inside bbox.")

    # Robust per-band trimming to reduce edge/background contamination.
    sorted_vals = np.sort(pixels.astype(np.float32), axis=0)
    n = sorted_vals.shape[0]
    trim_k = int(np.floor(n * trim_fraction))
    if trim_k > 0 and (n - 2 * trim_k) > 0:
        sorted_vals = sorted_vals[trim_k : n - trim_k]
    signature = sorted_vals.mean(axis=0).astype(np.float32)
    mean_val = float(signature.mean())
    if abs(mean_val) < 1e-12:
        raise ValueError("Signature has near-zero mean, cannot normalize.")
    return signature / mean_val


def _parse_coco_samples(coco_path: Path) -> list[ClassSample]:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    frame_indices = None
    if "videos" in data and data["videos"]:
        frame_indices = data["videos"][0].get("frame_indices")
    name_by_id = {int(c["id"]): str(c["name"]) for c in data["categories"]}

    samples: list[ClassSample] = []
    for ann in data["annotations"]:
        class_id = int(ann["category_id"])
        # Support both legacy custom format (bboxes + videos.frame_indices)
        # and standard COCO format (bbox + image_id).
        if "bboxes" in ann:
            if frame_indices is None:
                raise ValueError("COCO has 'bboxes' annotations but no videos.frame_indices.")
            boxes = ann["bboxes"]
            for local_idx, box in enumerate(boxes):
                if not box:
                    continue
                if local_idx >= len(frame_indices):
                    raise ValueError(
                        f"Annotation local frame {local_idx} has no matching videos.frame_indices entry."
                    )
                frame_index = int(frame_indices[local_idx])
                bbox_xywh = tuple(float(v) for v in box)
                samples.append(
                    ClassSample(
                        class_id=class_id,
                        class_name=name_by_id.get(class_id, str(class_id)),
                        frame_index=frame_index,
                        bbox_xywh=bbox_xywh,  # type: ignore[arg-type]
                    )
                )
        else:
            if "bbox" not in ann or "image_id" not in ann:
                continue
            bbox_xywh = tuple(float(v) for v in ann["bbox"])
            frame_index = int(ann["image_id"])
            samples.append(
                ClassSample(
                    class_id=class_id,
                    class_name=name_by_id.get(class_id, str(class_id)),
                    frame_index=frame_index,
                    bbox_xywh=bbox_xywh,  # type: ignore[arg-type]
                )
            )

    if not samples:
        raise ValueError("No non-empty bbox annotations found in COCO file.")
    return samples


def _load_target_frames(
    cu3s_path: Path, frame_indices: list[int], processing_mode: str
) -> dict[int, np.ndarray]:
    dm = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        processing_mode=processing_mode,
        batch_size=1,
        predict_ids=frame_indices,
    )
    dm.setup(stage="predict")
    if dm.predict_ds is None:
        raise RuntimeError("Predict dataset was not initialized.")
    # We only need cubes here; disable label usage even if sibling COCO JSON exists.
    if hasattr(dm.predict_ds, "has_labels"):
        dm.predict_ds.has_labels = False
    if hasattr(dm.predict_ds, "_coco"):
        dm.predict_ds._coco = None

    cubes_by_frame: dict[int, np.ndarray] = {}
    for i in range(len(dm.predict_ds)):
        sample = dm.predict_ds[i]
        cube = sample["cube"]
        mesu_index = int(sample["mesu_index"])
        cube_np = cube.detach().cpu().numpy() if isinstance(cube, torch.Tensor) else np.asarray(cube)
        if cube_np.ndim == 4:
            cube_np = cube_np[0]
        cubes_by_frame[mesu_index] = cube_np.astype(np.float32, copy=False)
    return cubes_by_frame


def _build_pipeline(
    *,
    class_sample: ClassSample,
    num_channels: int,
) -> tuple[CuvisPipeline, StatefulSpectralAngleMapper]:
    """Build a minimal inference graph: CU3S data node -> Stateful SAM."""
    pipeline_name = f"stateful_sam_class_{class_sample.class_id}_{class_sample.class_name}"
    pipeline = CuvisPipeline(name=pipeline_name)
    data = CU3SDataNode(name="cu3s_data")
    sam = StatefulSpectralAngleMapper(num_channels=num_channels, name="stateful_sam")
    pipeline.connect((data.outputs.cube, sam.cube))
    return pipeline, sam


def _save_pipeline(
    *,
    pipeline: CuvisPipeline,
    output_dir: Path,
    class_sample: ClassSample,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / f"{pipeline.name}.yaml"
    pipeline.save_to_file(
        str(yaml_path),
        metadata=PipelineMetadata(
            name=pipeline.name,
            description=(
                "Stateful Spectral Angle Mapper pipeline trained from one COCO bbox sample "
                f"for class {class_sample.class_id} ({class_sample.class_name})"
            ),
            tags=["sam", "stateful", "statistical", "material_detection"],
            author="cuvis.ai",
        ),
    )
    return yaml_path, yaml_path.with_suffix(".pt")


def _fit_and_save_single_class(
    *,
    class_sample: ClassSample,
    cube: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    signature = _extract_signature_from_bbox(cube, class_sample.bbox_xywh)
    num_channels = int(signature.shape[0])
    pipeline, sam = _build_pipeline(class_sample=class_sample, num_channels=num_channels)

    # Statistical-style fit step: estimate one class signature from the annotated bbox.
    sam.fit_signature(signature)
    return _save_pipeline(pipeline=pipeline, output_dir=output_dir, class_sample=class_sample)


@click.command()
@click.option(
    "--cu3s-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input CU3S file.",
)
@click.option(
    "--coco-json-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="COCO JSON containing one bbox sample per class.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory where class-wise YAML/PT artifacts are written.",
)
@click.option(
    "--processing-mode",
    type=click.Choice(["Raw", "Reflectance", "SpectralRadiance"]),
    default="Reflectance",
    show_default=True,
)
@click.option("--center-crop-scale", type=float, default=0.65, show_default=True)
@click.option("--trim-fraction", type=float, default=0.10, show_default=True)
def main(
    cu3s_path: Path,
    coco_json_path: Path,
    output_dir: Path,
    processing_mode: str,
    center_crop_scale: float,
    trim_fraction: float,
) -> None:
    logger.info("=== Stateful SAM Statistical Training ===")

    # Stage 1: Prepare samples from COCO
    samples = _parse_coco_samples(coco_json_path)
    by_class: dict[int, ClassSample] = {}
    for sample in samples:
        if sample.class_id in by_class:
            raise ValueError(
                f"Multiple bbox samples detected for class_id={sample.class_id}. "
                "Expected one sample per class for this workflow."
            )
        by_class[sample.class_id] = sample

    # Stage 2: Prepare data module (statistical example style) and load only target frames.
    frame_indices = sorted({s.frame_index for s in by_class.values()})
    cubes_by_frame = _load_target_frames(cu3s_path, frame_indices, processing_mode)

    # Stage 3: Fit one class pipeline each and export YAML/PT artifacts.
    for class_id, sample in sorted(by_class.items()):
        if sample.frame_index not in cubes_by_frame:
            raise ValueError(f"Frame {sample.frame_index} is missing from loaded CU3S samples.")
        cube = cubes_by_frame[sample.frame_index]
        signature = _extract_signature_from_bbox(
            cube,
            sample.bbox_xywh,
            center_crop_scale=center_crop_scale,
            trim_fraction=trim_fraction,
        )
        num_channels = int(signature.shape[0])
        pipeline, sam = _build_pipeline(class_sample=sample, num_channels=num_channels)
        sam.fit_signature(signature)
        yaml_path, pt_path = _save_pipeline(
            pipeline=pipeline,
            output_dir=output_dir,
            class_sample=sample,
        )
        logger.info(
            "Saved class {} ({}) pipeline: {} and {}",
            class_id,
            sample.class_name,
            yaml_path,
            pt_path,
        )

    logger.success(
        "Generated {} class-specific Stateful SAM pipelines (mode={}, center_crop_scale={}, trim_fraction={}).",
        len(by_class),
        processing_mode,
        center_crop_scale,
        trim_fraction,
    )


if __name__ == "__main__":
    main()
