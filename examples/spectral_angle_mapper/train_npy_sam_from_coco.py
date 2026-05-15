"""Train a per-class Spectral-Angle-Mapper pipeline from a CU3S file + COCO annotations.

This script replaces the earlier ``train_stateful_sam_from_coco.py`` approach
(which required a bespoke ``StatefulSpectralAngleMapper`` node) by using only
nodes that already exist in the cuvis-ai catalog:

    CU3SDataNode  +  COCO mask  →  SpectralSignatureExtractor
                                         ↓
                                    NpyReader(file_path=None)   ← stores per-class sigs
                                         ↓
                                    SpectralAngleMapper

**Buffer-mode NpyReader** (``file_path=None``) is the key ingredient: once
``load_from_array()`` is called with the extracted class signatures, the
buffer is frozen into the companion ``.pt`` file by ``pipeline.save_to_file()``.
At inference time ``restore_pipeline(yaml, pt)`` re-populates the buffer from
the ``.pt`` without needing any external ``.npy`` file.

Usage::

    uv run python examples/spectral_angle_mapper/train_npy_sam_from_coco.py \\
        --cu3s-path  D:/swir_cu3s/medical/Medical_Ac_EtOH97_H2O_left_to_right_000.cu3s \\
        --output-dir outputs/swir_liquids/medical

The sibling ``.json`` file (same stem as the ``.cu3s``) is used automatically
as the annotation source, or you can pass ``--coco-json-path`` explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
from cuvis_ai_schemas.pipeline import PipelineMetadata
from loguru import logger

from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.numpy_file import NpyReader
from cuvis_ai.node.spectral_angle_mapper import SpectralAngleMapper
from cuvis_ai.node.spectral_extractor import (
    BBoxSpectralExtractor,  # noqa: F401 (also available: SpectralSignatureExtractor for mask-based workflows)
)
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

# ---------------------------------------------------------------------------
# Signature extraction (bbox-based, no polygon rasterisation required)
# ---------------------------------------------------------------------------


def _xyxy_from_xywh(bbox_xywh: list[float]) -> list[float]:
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]


def _extract_class_signatures(
    cu3s_path: Path,
    coco_json_path: Path,
    processing_mode: str,
    trim_fraction: float,
    center_crop_scale: float,
) -> tuple[np.ndarray, dict[int, str]]:
    """Return ``[N, C]`` signatures (one per class) and a ``{id: name}`` map.

    Loads each annotated frame from the CU3S file and feeds the bbox through
    :class:`~cuvis_ai.node.spectral_extractor.BBoxSpectralExtractor` to
    obtain a trimmed-mean spectral signature.  This avoids the polygon
    rasterisation path in the data module (which requires a ``file_name``
    field that the SWIR liquids COCO json does not have).

    Parameters
    ----------
    cu3s_path, coco_json_path:
        Paths to the CU3S measurement file and its COCO annotation.
    processing_mode:
        ``"Raw"`` | ``"Reflectance"`` | ``"SpectralRadiance"``
    trim_fraction, center_crop_scale:
        Forwarded to :class:`BBoxSpectralExtractor`.

    Returns
    -------
    signatures : np.ndarray  shape ``[N, C]``
        One row per class, ordered by ascending ``category_id``.
    id_to_name : dict[int, str]
    """
    coco_data = json.loads(coco_json_path.read_text(encoding="utf-8"))
    name_by_id: dict[int, str] = {int(c["id"]): str(c["name"]) for c in coco_data["categories"]}

    # Map category_id → first annotation (image_id + bbox).
    ann_by_class: dict[int, dict[str, Any]] = {}
    for ann in coco_data["annotations"]:
        class_id = int(ann["category_id"])
        if class_id not in ann_by_class and "bbox" in ann:
            ann_by_class[class_id] = ann

    if not ann_by_class:
        raise ValueError(f"No bbox annotations found in {coco_json_path}.")

    # Unique frame indices needed.
    frame_indices = sorted({int(a["image_id"]) for a in ann_by_class.values()})
    logger.info("Loading {} annotated frames from {} …", len(frame_indices), cu3s_path.name)

    # Load CU3S cubes WITHOUT annotation.
    # The data module auto-detects sibling JSON files, but the SWIR liquids
    # COCO json lacks the `file_name` field required by coco_labels.Image.
    # We disable label loading after setup to avoid that code path.
    dm = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        processing_mode=processing_mode,
        batch_size=1,
        predict_ids=frame_indices,
    )
    dm.setup(stage="predict")
    assert dm.predict_ds is not None
    # Disable COCO label loading retroactively (same pattern as original script).
    if hasattr(dm.predict_ds, "has_labels"):
        dm.predict_ds.has_labels = False
    if hasattr(dm.predict_ds, "_coco"):
        dm.predict_ds._coco = None

    cubes_by_frame: dict[int, torch.Tensor] = {}
    for i in range(len(dm.predict_ds)):
        sample = dm.predict_ds[i]
        mesu_index = int(sample["mesu_index"])
        cube = sample["cube"]
        if not isinstance(cube, torch.Tensor):
            cube = torch.from_numpy(np.asarray(cube, dtype=np.float32))
        cube = cube.to(torch.float32)
        if cube.ndim == 3:
            cube = cube.unsqueeze(0)  # [H,W,C] → [1,H,W,C]
        cubes_by_frame[mesu_index] = cube

    extractor = BBoxSpectralExtractor(
        center_crop_scale=center_crop_scale,
        trim_fraction=trim_fraction,
        l2_normalize=False,  # keep raw reflectance values; SAM normalises internally
        aggregation="mean",
    )

    class_sigs: dict[int, np.ndarray] = {}
    for class_id in sorted(ann_by_class):
        ann = ann_by_class[class_id]
        frame_id = int(ann["image_id"])
        bbox_xyxy = _xyxy_from_xywh(ann["bbox"])

        cube = cubes_by_frame.get(frame_id)
        if cube is None:
            raise RuntimeError(f"Frame {frame_id} was not loaded from {cu3s_path}.")

        # BBoxSpectralExtractor expects bboxes [B, N, 4] in xyxy format.
        bboxes = torch.tensor([[bbox_xyxy]], dtype=torch.float32)  # [1, 1, 4]
        result = extractor.forward(cube=cube, bboxes=bboxes)

        valid = int(result["spectral_valid"][0, 0].item())
        sig = result["spectral_signatures"][0, 0].cpu().numpy()  # [C]

        if not valid:
            raise RuntimeError(
                f"Class {class_id} ({name_by_id.get(class_id, '?')}) bbox "
                f"{bbox_xyxy} produced an invalid (near-zero) signature on frame "
                f"{frame_id}.  Check that the bbox is inside the image."
            )

        logger.info(
            "  Class {:2d} {:>12s}  bbox {:s}  →  sig [{:.4f} … {:.4f}]",
            class_id,
            name_by_id.get(class_id, "?"),
            str([round(v, 1) for v in bbox_xyxy]),
            float(sig.min()),
            float(sig.max()),
        )
        class_sigs[class_id] = sig

    sigs_array = np.stack([class_sigs[cid] for cid in sorted(class_sigs)], axis=0)  # [N, C]
    return sigs_array, {cid: name_by_id.get(cid, str(cid)) for cid in sorted(class_sigs)}


# ---------------------------------------------------------------------------
# Pipeline construction + save
# ---------------------------------------------------------------------------


def _build_and_save_pipeline(
    *,
    signatures: np.ndarray,
    id_to_name: dict[int, str],
    cu3s_path: Path,
    output_dir: Path,
    pipeline_name: str,
) -> tuple[Path, Path]:
    """Build the inference pipeline, populate the NpyReader buffer, and save."""
    num_channels = int(signatures.shape[1])

    pipeline = CuvisPipeline(name=pipeline_name)

    cu3s_node = CU3SDataNode(name="cu3s_data")
    ref_node = NpyReader(file_path=None, name="reference_signatures")
    sam_node = SpectralAngleMapper(num_channels=num_channels, name="sam")

    # Populate the NpyReader buffer with the extracted signatures [N, C].
    # _pad_to_bhwc4 will convert [N, C] → [N, 1, 1, C] internally.
    ref_node.load_from_array(signatures)

    # Wire: cube → SAM, reference → SAM
    pipeline.connect((cu3s_node.outputs.cube, sam_node.cube))
    pipeline.connect((ref_node.outputs.data, sam_node.spectral_signature))

    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / f"{pipeline_name}.yaml"

    class_list = ", ".join(f"{cid}={n}" for cid, n in id_to_name.items())
    pipeline.save_to_file(
        str(yaml_path),
        metadata=PipelineMetadata(
            name=pipeline_name,
            description=(
                f"SAM inference pipeline for {cu3s_path.name}. "
                f"Classes: {class_list}. "
                "Spectral signatures stored in NpyReader buffer (file_path=null); "
                "restored from companion .pt at inference time."
            ),
            tags=["sam", "statistical", "material_classification", "swir_liquids", "npy_reader"],
            author="cuvis.ai",
        ),
    )
    pt_path = yaml_path.with_suffix(".pt")
    logger.success(
        "Saved pipeline:\n  YAML → {}\n  PT   → {}",
        yaml_path,
        pt_path,
    )
    return yaml_path, pt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
    default=None,
    help="COCO JSON annotation file. Defaults to sibling .json of the CU3S file.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to write YAML + PT artifacts.",
)
@click.option(
    "--pipeline-name",
    default=None,
    help="Pipeline name. Defaults to the CU3S file stem.",
)
@click.option(
    "--processing-mode",
    type=click.Choice(["Raw", "Reflectance", "SpectralRadiance"]),
    default="Reflectance",
    show_default=True,
)
@click.option("--trim-fraction", type=float, default=0.10, show_default=True)
@click.option("--center-crop-scale", type=float, default=0.65, show_default=True)
def main(
    cu3s_path: Path,
    coco_json_path: Path | None,
    output_dir: Path,
    pipeline_name: str | None,
    processing_mode: str,
    trim_fraction: float,
    center_crop_scale: float,
) -> None:
    """Extract per-class spectral signatures and save a portable SAM pipeline."""
    if coco_json_path is None:
        coco_json_path = cu3s_path.with_suffix(".json")
        if not coco_json_path.exists():
            raise click.UsageError(
                f"No COCO json found at {coco_json_path}. Pass --coco-json-path explicitly."
            )

    if pipeline_name is None:
        pipeline_name = cu3s_path.stem

    logger.info("=== NpyReader-based SAM Training ===")
    logger.info("CU3S       : {}", cu3s_path)
    logger.info("COCO JSON  : {}", coco_json_path)
    logger.info("Output dir : {}", output_dir)
    logger.info("Mode       : {}", processing_mode)

    signatures, id_to_name = _extract_class_signatures(
        cu3s_path=cu3s_path,
        coco_json_path=coco_json_path,
        processing_mode=processing_mode,
        trim_fraction=trim_fraction,
        center_crop_scale=center_crop_scale,
    )
    logger.info(
        "Extracted {} class signatures, shape {}  (will be stored as [N,1,1,C])",
        signatures.shape[0],
        signatures.shape,
    )

    _build_and_save_pipeline(
        signatures=signatures,
        id_to_name=id_to_name,
        cu3s_path=cu3s_path,
        output_dir=output_dir,
        pipeline_name=pipeline_name,
    )

    logger.success(
        "Done. Run inference with:\n  uv run restore-pipeline {out}/{name}.yaml --cu3s <path>",
        out=output_dir,
        name=pipeline_name,
    )


if __name__ == "__main__":
    main()
