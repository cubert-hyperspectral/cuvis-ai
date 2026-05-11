"""Create 3 single-class NpyReader+SAM pipelines from industrial CU3S dataset.

Like the medical version, but for D:/swir_cu3s/industrial/Industrial_H2O_Ac_IPA_000.cu3s

All 3 classes are in the same frame (image_id=0), so we extract all 3 signatures
in one pass, then create 3 pipelines (one per class).

Output directory: configs/pipeline/sam/industrial_npy_sam_single_class/

Base pipelines (3):
    npy_sam_class_{1,2,3}.yaml + .pt

Thresholded pipelines (3 classes × 5 thresholds = 15):
    npy_sam_class_{1,2,3}_thr_0p{10,15,20,25,30}.yaml + .pt

Usage::

    uv run python examples/spectral_angle_mapper/create_npy_sam_single_class_industrial.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.pipeline import PipelineMetadata
from loguru import logger

from cuvis_ai.deciders.binary_decider import BinaryDecider
from cuvis_ai.node.conversion import ScoreToLogit
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.numpy_reader import NpyReader
from cuvis_ai.node.spectral_angle_mapper import SpectralAngleMapper
from cuvis_ai.node.spectral_extractor import BBoxSpectralExtractor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CU3S_PATH = Path("D:/swir_cu3s/industrial/Industrial_H2O_Ac_IPA_000.cu3s")
COCO_JSON = CU3S_PATH.with_suffix(".json")
OUTPUT_DIR = Path("configs/pipeline/sam/industrial_npy_sam_single_class")

NUM_CHANNELS = 39
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _xyxy_from_xywh(bbox: list[float]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return int(x), int(y), int(x + w), int(y + h)


def _extract_class_signatures() -> dict[int, torch.Tensor]:
    """Extract one signature per class from industrial COCO annotations.

    All 3 classes are in the same frame (image_id=0).
    """
    from cuvis_ai_core.data.datasets import SingleCu3sDataModule
    import numpy as np

    coco_data = json.loads(COCO_JSON.read_text(encoding="utf-8"))
    anns_by_class = {}
    for ann in coco_data["annotations"]:
        cid = int(ann["category_id"])
        if cid not in anns_by_class:
            anns_by_class[cid] = ann

    logger.info("Found {} classes", len(anns_by_class))

    # Load the frame (all classes are in image_id=0)
    frame_id = int(anns_by_class[1]["image_id"])
    dm = SingleCu3sDataModule(
        cu3s_file_path=str(CU3S_PATH),
        processing_mode="Reflectance",
        batch_size=1,
        predict_ids=[frame_id],
    )
    dm.setup(stage="predict")
    ds = dm.predict_ds
    if hasattr(ds, "has_labels"):
        ds.has_labels = False
    if hasattr(ds, "_coco"):
        ds._coco = None

    sample = ds[0]
    cube = sample["cube"]
    if not isinstance(cube, torch.Tensor):
        cube = torch.from_numpy(np.asarray(cube, dtype=np.float32))
    else:
        cube = cube.float()
    if cube.ndim == 3:
        cube = cube.unsqueeze(0)

    # Extract signature per class
    extractor = BBoxSpectralExtractor(
        center_crop_scale=0.65, trim_fraction=0.10, l2_normalize=False, aggregation="mean"
    )
    sigs = {}
    for cid in sorted(anns_by_class):
        ann = anns_by_class[cid]
        bbox_xywh = ann["bbox"]
        x1, y1, x2, y2 = _xyxy_from_xywh(bbox_xywh)
        bboxes = torch.tensor([[[x1, y1, x2, y2]]], dtype=torch.float32)  # [1,1,4]

        result = extractor.forward(cube=cube, bboxes=bboxes)
        valid = int(result["spectral_valid"][0, 0].item())
        sig = result["spectral_signatures"][0, 0].cpu().numpy()  # [C]

        if not valid:
            raise RuntimeError(f"Class {cid} bbox produced invalid signature")

        sigs[cid] = torch.from_numpy(sig.astype(np.float32))
        logger.info("  Class {}: bbox={} -> sig shape={}", cid, (x1, y1, x2, y2), sig.shape)

    return sigs


def _build_base_pipeline(class_idx: int, sig: torch.Tensor) -> tuple[Path, Path]:
    """Create base NpyReader+SAM pipeline (no threshold) and save."""
    name = f"npy_sam_class_{class_idx}"
    pipeline = CuvisPipeline(name=name)

    cu3s_node = CU3SDataNode(name="cu3s_data")
    ref_node = NpyReader(file_path=None, name="reference_signature")
    sam_node = SpectralAngleMapper(num_channels=NUM_CHANNELS, name="sam")

    # Populate buffer — sig is [39], NpyReader pads to [1,1,1,39]
    ref_node.load_from_array(sig)

    pipeline.connect((cu3s_node.outputs.cube, sam_node.cube))
    pipeline.connect((ref_node.outputs.data, sam_node.spectral_signature))

    yaml_path = OUTPUT_DIR / f"{name}.yaml"
    pipeline.save_to_file(
        str(yaml_path),
        metadata=PipelineMetadata(
            name=name,
            description=(
                f"Single-class SAM pipeline for class {class_idx} (industrial dataset). "
                "Reference signature stored in NpyReader buffer (file_path=null). "
                "Lower best_scores value = better match."
            ),
            tags=["sam", "statistical", "material_detection", "npy_reader", "single_class"],
            author="cuvis.ai",
        ),
    )
    pt_path = yaml_path.with_suffix(".pt")
    logger.success("  Saved base pipeline: {}", yaml_path.name)
    return yaml_path, pt_path


def _build_thresholded_pipeline(
    class_idx: int, sig: torch.Tensor, threshold: float
) -> tuple[Path, Path]:
    """Create NpyReader+SAM+ScoreToLogit+BinaryDecider pipeline and save."""
    thr_str = f"0p{int(threshold * 100):02d}"
    name = f"npy_sam_class_{class_idx}_thr_{thr_str}"
    pipeline = CuvisPipeline(name=name)

    cu3s_node = CU3SDataNode(name="cu3s_data")
    ref_node = NpyReader(file_path=None, name="reference_signature")
    sam_node = SpectralAngleMapper(num_channels=NUM_CHANNELS, name="sam")
    s2l_node = ScoreToLogit(init_scale=-1.0, init_bias=threshold, name="score_to_logit")
    dec_node = BinaryDecider(threshold=0.5, name="decider")

    ref_node.load_from_array(sig)

    pipeline.connect((cu3s_node.outputs.cube, sam_node.cube))
    pipeline.connect((ref_node.outputs.data, sam_node.spectral_signature))
    pipeline.connect((sam_node.outputs.best_scores, s2l_node.scores))
    pipeline.connect((s2l_node.outputs.logits, dec_node.logits))

    yaml_path = OUTPUT_DIR / f"{name}.yaml"
    pipeline.save_to_file(
        str(yaml_path),
        metadata=PipelineMetadata(
            name=name,
            description=(
                f"Single-class SAM pipeline for class {class_idx} (industrial) "
                f"with absolute angle threshold {threshold:.2f} rad "
                f"({threshold * 57.296:.1f} deg). "
                "Pixels with angle < threshold → True (match)."
            ),
            tags=[
                "sam",
                "statistical",
                "material_detection",
                "npy_reader",
                "single_class",
                "thresholded",
            ],
            author="cuvis.ai",
        ),
    )
    pt_path = yaml_path.with_suffix(".pt")
    logger.success("  Saved threshold={:.2f} pipeline: {}", threshold, yaml_path.name)
    return yaml_path, pt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: {}", OUTPUT_DIR.resolve())

    logger.info("Extracting class signatures from industrial CU3S...")
    sigs = _extract_class_signatures()

    for class_idx in sorted(sigs.keys()):
        sig = sigs[class_idx]
        logger.info("=== Class {} ===", class_idx)

        # Base pipeline
        _build_base_pipeline(class_idx, sig)

        # Thresholded variants
        for thr in THRESHOLDS:
            _build_thresholded_pipeline(class_idx, sig, thr)

    logger.success(
        "Done. Created {} files in {}",
        len(list(OUTPUT_DIR.glob("*.yaml"))),
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
