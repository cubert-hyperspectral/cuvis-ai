"""Verify 3 single-class NpyReader+SAM pipelines for industrial dataset.

Like medical version but for D:/swir_cu3s/industrial/Industrial_H2O_Ac_IPA_000.cu3s
All 3 classes are in the same frame (image_id=0).

Usage::

    uv run python examples/spectral_angle_mapper/verify_npy_sam_single_class_industrial.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from loguru import logger

CU3S_PATH = Path("D:/swir_cu3s/industrial/Industrial_H2O_Ac_IPA_000.cu3s")
COCO_JSON = CU3S_PATH.with_suffix(".json")
PIPELINE_DIR = Path("configs/pipeline/sam/industrial_npy_sam_single_class")
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _xyxy_from_xywh(bbox: list[float]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return int(x), int(y), int(x + w), int(y + h)


def _load_coco_info() -> dict[int, dict]:
    """Returns {category_id: {name, image_id, bbox_xyxy}}."""
    data = json.loads(COCO_JSON.read_text(encoding="utf-8"))
    name_by_id = {c["id"]: c["name"] for c in data["categories"]}
    info = {}
    for ann in data["annotations"]:
        cid = int(ann["category_id"])
        if cid not in info:
            info[cid] = {
                "name": name_by_id.get(cid, str(cid)),
                "image_id": int(ann["image_id"]),
                "bbox_xyxy": _xyxy_from_xywh(ann["bbox"]),
            }
    return info


def _load_frame(image_id: int) -> dict:
    """Load a single CU3S frame → batch dict with cube [1,H,W,C] uint16 and wavelengths [1,C]."""
    dm = SingleCu3sDataModule(
        cu3s_file_path=str(CU3S_PATH),
        processing_mode="Reflectance",
        batch_size=1,
        predict_ids=[image_id],
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
        cube = torch.from_numpy(np.asarray(cube))
    if cube.ndim == 3:
        cube = cube.unsqueeze(0)

    wavelengths = sample["wavelengths"]
    if not isinstance(wavelengths, torch.Tensor):
        wavelengths = torch.from_numpy(np.asarray(wavelengths))
    if wavelengths.ndim == 1:
        wavelengths = wavelengths.unsqueeze(0)

    return {"cube": cube, "wavelengths": wavelengths}


def _bbox_mask(height: int, width: int, bbox_xyxy: tuple[int, int, int, int]) -> torch.Tensor:
    """Binary mask [H,W] — True inside bbox."""
    x1, y1, x2, y2 = bbox_xyxy
    mask = torch.zeros(height, width, dtype=torch.bool)
    mask[y1:y2, x1:x2] = True
    return mask


def _get_output(out: dict, node_name: str, port: str):
    """Retrieve output from pipeline forward result (keys are (node_name, port) tuples)."""
    return out.get((node_name, port))


def _run_base_pipeline(class_idx: int, batch: dict, coco_info: dict) -> None:
    """Load and run the base (no-threshold) pipeline; report inside vs. outside score."""
    info = coco_info[class_idx]
    yaml_path = PIPELINE_DIR / f"npy_sam_class_{class_idx}.yaml"
    pt_path = yaml_path.with_suffix(".pt")

    pipeline = CuvisPipeline.load_pipeline(str(yaml_path), str(pt_path))
    with torch.no_grad():
        out = pipeline.forward(batch=batch)

    scores = _get_output(out, "sam", "best_scores")
    if scores is None:
        logger.error("  No 'best_scores'/'scores' in output. Keys: {}", list(out.keys()))
        return

    scores_f = scores.float()
    B, H, W, C = scores_f.shape
    score_map = scores_f[0, :, :, 0]

    bbox_mask = _bbox_mask(H, W, info["bbox_xyxy"])
    inside_mean = float(score_map[bbox_mask].mean())
    outside_mean = float(score_map[~bbox_mask].mean())
    ratio = inside_mean / (outside_mean + 1e-9)

    status = "PASS" if inside_mean < outside_mean else "FAIL"
    logger.info(
        "  Class {} ({}) | bbox {} | inside={:.4f} outside={:.4f} ratio={:.3f} [{}]",
        class_idx,
        info["name"],
        info["bbox_xyxy"],
        inside_mean,
        outside_mean,
        ratio,
        status,
    )


def _run_thresholded_pipelines(class_idx: int, batch: dict, coco_info: dict) -> None:
    """Run all threshold variants and report True-pixel coverage inside the bbox."""
    info = coco_info[class_idx]
    cube = batch["cube"]
    B, H, W, Ch = cube.shape

    logger.info("  Threshold variants for class {}:", class_idx)
    for thr in THRESHOLDS:
        thr_str = f"0p{int(thr * 100):02d}"
        yaml_path = PIPELINE_DIR / f"npy_sam_class_{class_idx}_thr_{thr_str}.yaml"
        pt_path = yaml_path.with_suffix(".pt")

        pipeline = CuvisPipeline.load_pipeline(str(yaml_path), str(pt_path))
        with torch.no_grad():
            out = pipeline.forward(batch=batch)

        decisions = _get_output(out, "decider", "decisions")
        if decisions is None:
            logger.warning("    thr={:.2f}: no 'decisions' key. Keys: {}", thr, list(out.keys()))
            continue

        dec_map = decisions[0, :, :, 0]
        bbox_mask = _bbox_mask(H, W, info["bbox_xyxy"])

        total_true = int(dec_map.sum())
        inside_true = int((dec_map & bbox_mask).sum())
        inside_pixels = int(bbox_mask.sum())
        recall = inside_true / (inside_pixels + 1e-9)
        precision = inside_true / (total_true + 1e-9)

        logger.info(
            "    thr={:.2f} ({:.1f} deg): total_true={} | inside_true={}/{} "
            "| recall={:.2f} precision={:.2f}",
            thr,
            thr * 57.296,
            total_true,
            inside_true,
            inside_pixels,
            recall,
            precision,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("Loading COCO annotations from {}", COCO_JSON)
    coco_info = _load_coco_info()

    # Industrial: all 3 classes in same frame (image_id=0)
    frame_id = coco_info[1]["image_id"]
    batch = _load_frame(frame_id)
    cube = batch["cube"]
    logger.info("  Cube shape: {}", list(cube.shape))

    for class_idx in sorted(coco_info.keys()):
        info = coco_info[class_idx]
        logger.info(
            "=== Class {} ({}) | frame={} | bbox={} ===",
            class_idx,
            info["name"],
            info["image_id"],
            info["bbox_xyxy"],
        )

        _run_base_pipeline(class_idx, batch, coco_info)
        _run_thresholded_pipelines(class_idx, batch, coco_info)
        print()


if __name__ == "__main__":
    main()
