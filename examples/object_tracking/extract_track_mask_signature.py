"""Extract mask + spectral signature for a specific track from tracking results.

Given a COCO-format tracking JSON (with compressed RLE masks) and a CU3S file,
this script:

1. Extracts the binary mask for a given ``track_id`` at a seed frame and saves
   it as a PNG (255=foreground, 0=background).
2. Emits a COCO-format subset JSON containing only the requested track's
   annotations on the specified frames. For compatibility with
   ``SingleCu3sDataset(annotation_json_path=...)``, each annotation includes an
   uncompressed list-RLE in the ``mask`` field.
3. Computes the per-channel spectral mean/std over the masked region at the
   seed frame and saves it as a JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import cv2
import numpy as np
from loguru import logger


def _load_tracking_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _decode_rle_mask(segmentation: dict) -> np.ndarray:
    """Decode a COCO compressed RLE segmentation dict to a binary uint8 mask."""
    from cuvis_ai_core.data.rle import coco_rle_decode

    return coco_rle_decode(segmentation)


def _mask_to_rle_list(mask: np.ndarray) -> list[int]:
    """Encode a binary mask to uncompressed COCO-style RLE counts list.

    Counts alternate [bg, fg, bg, ...] in Fortran order and always start with
    background.
    """
    flat = np.asarray(mask > 0, dtype=np.uint8).reshape(-1, order="F")
    counts: list[int] = []
    current = 0
    run_len = 0
    for value in flat:
        value_int = int(value)
        if value_int == current:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
            current = value_int
    counts.append(run_len)
    return counts


def _filter_annotations(
    data: dict,
    track_id: int,
    frame_ids: list[int],
) -> tuple[list[dict], list[dict]]:
    """Filter images and annotations for a given track_id and frame list."""
    images_by_id = {img["id"]: img for img in data["images"]}

    filtered_images = []
    for fid in frame_ids:
        if fid in images_by_id:
            filtered_images.append(images_by_id[fid])
        else:
            logger.warning("Frame {} not found in tracking JSON images", fid)

    frame_set = {img["id"] for img in filtered_images}
    filtered_annots = [
        a
        for a in data["annotations"]
        if a.get("track_id") == track_id and a["image_id"] in frame_set
    ]

    return filtered_images, filtered_annots


@click.command()
@click.option(
    "--cu3s-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the CU3S file for spectral signature extraction.",
)
@click.option(
    "--tracking-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the COCO tracking results JSON (with RLE masks).",
)
@click.option("--track-id", type=int, required=True, help="Target track ID to extract.")
@click.option(
    "--frame-id", type=int, required=True, help="Primary seed frame index for mask + signature."
)
@click.option(
    "--frame-list",
    type=str,
    default=None,
    help="Comma-separated frame indices for multi-frame COCO subset (default: just --frame-id).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output directory for mask PNG, subset JSON, and signature JSON.",
)
@click.option(
    "--processing-mode",
    type=str,
    default="SpectralRadiance",
    show_default=True,
    help="CU3S processing mode for spectral data loading.",
)
def main(
    cu3s_path: Path,
    tracking_json: Path,
    track_id: int,
    frame_id: int,
    frame_list: str | None,
    output_dir: Path,
    processing_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_basename = cu3s_path.stem

    # Parse frame list
    if frame_list is not None:
        frame_ids = sorted({int(f.strip()) for f in frame_list.split(",")})
    else:
        frame_ids = [frame_id]
    if frame_id not in frame_ids:
        frame_ids = sorted(set(frame_ids) | {frame_id})

    logger.info("Track ID: {}, seed frame: {}, frame list: {}", track_id, frame_id, frame_ids)

    # Load tracking JSON
    data = _load_tracking_json(tracking_json)
    filtered_images, filtered_annots = _filter_annotations(data, track_id, frame_ids)

    if not filtered_annots:
        raise click.ClickException(
            f"No annotations found for track_id={track_id} on frames {frame_ids}"
        )
    logger.info(
        "Found {} annotations for track_id={} across {} frames",
        len(filtered_annots),
        track_id,
        len(filtered_images),
    )

    # --- 1. Extract and save seed mask as PNG ---
    seed_annots = [a for a in filtered_annots if a["image_id"] == frame_id]
    if not seed_annots:
        raise click.ClickException(
            f"No annotation for track_id={track_id} on seed frame {frame_id}"
        )
    seed_annot = seed_annots[0]

    if "segmentation" not in seed_annot or not isinstance(seed_annot["segmentation"], dict):
        raise click.ClickException(
            f"Seed annotation lacks compressed RLE segmentation (got {type(seed_annot.get('segmentation'))})"
        )

    mask = _decode_rle_mask(seed_annot["segmentation"])
    mask_png_name = f"{video_basename}_track{track_id}_frame{frame_id}_mask.png"
    mask_png_path = output_dir / mask_png_name
    # Save as 255/0 PNG
    cv2.imwrite(str(mask_png_path), (mask * 255).astype(np.uint8))
    mask_area = int(mask.sum())
    logger.info(
        "Mask saved: {} ({}x{}, area={} px)", mask_png_path, mask.shape[1], mask.shape[0], mask_area
    )

    # --- 2. Emit COCO subset JSON ---
    frame_min = min(frame_ids)
    frame_max = max(frame_ids)
    subset_json_name = f"track{track_id}_frames_{frame_min}_{frame_max}.json"
    subset_json_path = output_dir / subset_json_name

    # Re-number annotation IDs sequentially and add dataset-compatible ``mask`` RLE.
    subset_annots = []
    for idx, a in enumerate(filtered_annots, start=1):
        ann_copy = dict(a)
        ann_copy["id"] = idx
        seg = ann_copy.get("segmentation")
        if isinstance(seg, dict):
            decoded = _decode_rle_mask(seg)
            ann_copy["mask"] = {
                "size": [int(decoded.shape[0]), int(decoded.shape[1])],
                "counts": _mask_to_rle_list(decoded),
            }
        subset_annots.append(ann_copy)

    subset_data = {
        "images": filtered_images,
        "annotations": subset_annots,
        "categories": data.get("categories", []),
    }
    subset_json_path.write_text(json.dumps(subset_data, indent=2), encoding="utf-8")
    logger.info(
        "COCO subset JSON saved: {} ({} images, {} annotations)",
        subset_json_path,
        len(filtered_images),
        len(subset_annots),
    )

    # --- 3. Spectral signature extraction ---
    from cuvis_ai_core.data.datasets import SingleCu3sDataModule

    dm = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        processing_mode=processing_mode,
        batch_size=1,
        predict_ids=[frame_id],
    )
    dm.setup(stage="predict")
    if dm.predict_ds is None:
        raise RuntimeError("Predict dataset was not initialized.")

    sample = dm.predict_ds[0]
    cube = sample["cube"]  # (C, H, W) or (H, W, C) tensor
    if hasattr(cube, "numpy"):
        cube = cube.numpy()
    # Ensure (H, W, C)
    if cube.ndim == 3 and cube.shape[0] < cube.shape[-1]:
        # Likely (C, H, W) — transpose
        cube = np.transpose(cube, (1, 2, 0))

    wavelengths = sample.get("wavelengths")
    if wavelengths is not None and hasattr(wavelengths, "numpy"):
        wavelengths = wavelengths.numpy()

    # Resize mask to match cube spatial dims if needed
    h_cube, w_cube = cube.shape[:2]
    if mask.shape != (h_cube, w_cube):
        logger.warning(
            "Mask shape {} differs from cube spatial shape ({}, {}); resizing mask",
            mask.shape,
            h_cube,
            w_cube,
        )
        mask_resized = cv2.resize(
            mask.astype(np.uint8), (w_cube, h_cube), interpolation=cv2.INTER_NEAREST
        )
        mask_bool = mask_resized.astype(bool)
    else:
        mask_bool = mask.astype(bool)

    masked_pixels = cube[mask_bool]  # (N, C)
    if masked_pixels.size == 0:
        raise click.ClickException("No pixels in mask — signature extraction failed.")

    per_channel_mean = masked_pixels.mean(axis=0).tolist()
    per_channel_std = masked_pixels.std(axis=0).tolist()

    signature = {
        "video_basename": video_basename,
        "track_id": track_id,
        "frame_id": frame_id,
        "processing_mode": processing_mode,
        "num_channels": len(per_channel_mean),
        "masked_pixel_count": int(masked_pixels.shape[0]),
        "per_channel_mean": per_channel_mean,
        "per_channel_std": per_channel_std,
    }
    if wavelengths is not None:
        signature["wavelengths"] = (
            wavelengths.tolist() if hasattr(wavelengths, "tolist") else list(wavelengths)
        )

    sig_json_name = f"{video_basename}_track{track_id}_frame{frame_id}_signature.json"
    sig_json_path = output_dir / sig_json_name
    sig_json_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    logger.info(
        "Spectral signature saved: {} ({} channels, {} masked pixels)",
        sig_json_path,
        signature["num_channels"],
        signature["masked_pixel_count"],
    )

    logger.success("Done -> {}", output_dir)


if __name__ == "__main__":
    main()
