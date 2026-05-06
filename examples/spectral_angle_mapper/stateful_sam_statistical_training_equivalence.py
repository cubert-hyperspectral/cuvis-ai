"""Stateful SAM statistical-style training with equivalence check.

This example keeps the existing COCO+bbox signature extraction workflow, but
uses ``StatefulSpectralAngleMapper.statistical_initialization(...)`` for
learning the persisted signature buffer. It also verifies equivalence against
the legacy ``fit_signature(...)`` path for each class.
"""

from __future__ import annotations

from pathlib import Path

import click
import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.pipeline import PipelineMetadata
from loguru import logger

from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.spectral_angle_mapper import StatefulSpectralAngleMapper

from train_stateful_sam_from_coco import (
    _extract_signature_from_bbox,
    _load_target_frames,
    _parse_coco_samples,
)


def _build_pipeline(class_id: int, class_name: str, num_channels: int) -> tuple[CuvisPipeline, StatefulSpectralAngleMapper]:
    name = f"stateful_sam_class_{class_id}_{class_name}"
    pipeline = CuvisPipeline(name=name)
    data_node = CU3SDataNode(name="cu3s_data")
    sam_node = StatefulSpectralAngleMapper(num_channels=num_channels, name="stateful_sam")
    pipeline.connect((data_node.outputs.cube, sam_node.cube))
    return pipeline, sam_node


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
    help="COCO JSON with one bbox sample per class.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory for trained YAML/PT artifacts.",
)
@click.option(
    "--processing-mode",
    type=click.Choice(["Raw", "Reflectance", "SpectralRadiance"]),
    default="Reflectance",
    show_default=True,
)
@click.option("--center-crop-scale", type=float, default=0.65, show_default=True)
@click.option("--trim-fraction", type=float, default=0.10, show_default=True)
@click.option("--atol", type=float, default=1e-6, show_default=True)
def main(
    cu3s_path: Path,
    coco_json_path: Path,
    output_dir: Path,
    processing_mode: str,
    center_crop_scale: float,
    trim_fraction: float,
    atol: float,
) -> None:
    logger.info("=== Stateful SAM Statistical Equivalence Training ===")

    samples = _parse_coco_samples(coco_json_path)
    by_class = {}
    for sample in samples:
        if sample.class_id in by_class:
            raise ValueError(
                f"Multiple bbox samples for class_id={sample.class_id}; expected one for this workflow."
            )
        by_class[sample.class_id] = sample

    frame_indices = sorted({sample.frame_index for sample in by_class.values()})
    cubes_by_frame = _load_target_frames(cu3s_path, frame_indices, processing_mode)

    output_dir.mkdir(parents=True, exist_ok=True)

    for class_id, sample in sorted(by_class.items()):
        cube = cubes_by_frame.get(sample.frame_index)
        if cube is None:
            raise ValueError(f"Frame {sample.frame_index} missing from loaded dataset.")

        signature_np = _extract_signature_from_bbox(
            cube,
            sample.bbox_xywh,
            center_crop_scale=center_crop_scale,
            trim_fraction=trim_fraction,
        )
        signature_t = torch.from_numpy(signature_np)
        num_channels = int(signature_t.shape[0])

        # New statistical path
        pipeline, sam_stat = _build_pipeline(class_id, sample.class_name, num_channels)
        sam_stat.statistical_initialization([{"spectral_signature": signature_t}])

        # Legacy path for explicit equivalence check
        sam_legacy = StatefulSpectralAngleMapper(num_channels=num_channels)
        sam_legacy.fit_signature(signature_t)
        if not torch.allclose(sam_stat.learned_signature, sam_legacy.learned_signature, atol=atol):
            raise RuntimeError(
                f"Signature mismatch for class_id={class_id} between statistical_initialization and fit_signature."
            )

        yaml_path = output_dir / f"{pipeline.name}.yaml"
        pipeline.save_to_file(
            str(yaml_path),
            metadata=PipelineMetadata(
                name=pipeline.name,
                description=(
                    "Stateful Spectral Angle Mapper pipeline trained with "
                    "statistical_initialization and COCO bbox supervision."
                ),
                tags=["sam", "stateful", "statistical", "material_detection"],
                author="cuvis.ai",
            ),
        )
        logger.info(
            "Saved class {} ({}) artifacts: {} and {}",
            class_id,
            sample.class_name,
            yaml_path,
            yaml_path.with_suffix(".pt"),
        )

    logger.success(
        "Completed training with equivalence check for {} classes (mode={}, center_crop_scale={}, trim_fraction={}).",
        len(by_class),
        processing_mode,
        center_crop_scale,
        trim_fraction,
    )


if __name__ == "__main__":
    main()
