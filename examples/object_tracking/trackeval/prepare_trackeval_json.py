"""Prepare a filtered COCO tracking JSON for TrackEval evaluation.

Filters a source COCO tracking/GT JSON by frame range and optionally by
track IDs.  Useful for extracting a frame-window subset of ground-truth
annotations to compare against prediction JSONs.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from loguru import logger


@click.command()
@click.option(
    "--input-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Source COCO tracking/GT JSON.",
)
@click.option(
    "--output-json",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output path for filtered JSON.",
)
@click.option("--start-frame", type=int, required=True, help="Start of frame window (inclusive).")
@click.option("--end-frame", type=int, required=True, help="End of frame window (exclusive).")
@click.option(
    "--track-ids",
    type=str,
    default=None,
    help="Comma-separated track IDs to keep (default: all tracks).",
)
@click.option(
    "--fill-missing-score",
    type=float,
    default=None,
    help="Replace null/missing annotation scores with this value (e.g. 1.0 for GT).",
)
def main(
    input_json: Path,
    output_json: Path,
    start_frame: int,
    end_frame: int,
    track_ids: str | None,
    fill_missing_score: float | None,
) -> None:
    if end_frame <= start_frame:
        raise click.BadParameter(
            "--end-frame must be greater than --start-frame", param_hint="--end-frame"
        )

    data = json.loads(input_json.read_text(encoding="utf-8"))
    logger.info(
        "Loaded {} images, {} annotations from {}",
        len(data.get("images", [])),
        len(data.get("annotations", [])),
        input_json,
    )

    # Parse track IDs filter
    keep_track_ids: set[int] | None = None
    if track_ids is not None:
        keep_track_ids = {int(t.strip()) for t in track_ids.split(",")}
        logger.info("Filtering to track IDs: {}", sorted(keep_track_ids))

    # Filter images by frame window
    filtered_images = [
        img for img in data.get("images", []) if start_frame <= img["id"] < end_frame
    ]
    valid_image_ids = {img["id"] for img in filtered_images}
    logger.info(
        "Images after frame filter [{}, {}): {} / {}",
        start_frame,
        end_frame,
        len(filtered_images),
        len(data.get("images", [])),
    )

    # Filter annotations
    filtered_annots = []
    for a in data.get("annotations", []):
        if a["image_id"] not in valid_image_ids:
            continue
        if keep_track_ids is not None:
            tid = a.get("track_id")
            if tid is None or tid not in keep_track_ids:
                continue
        filtered_annots.append(a)

    logger.info(
        "Annotations after filtering: {} / {}",
        len(filtered_annots),
        len(data.get("annotations", [])),
    )

    # Fill missing scores
    if fill_missing_score is not None:
        filled = 0
        for a in filtered_annots:
            if a.get("score") is None:
                a["score"] = fill_missing_score
                filled += 1
        if filled:
            logger.info("Filled {} null scores with {}", filled, fill_missing_score)

    # Re-number annotation IDs sequentially
    for idx, a in enumerate(filtered_annots, start=1):
        a["id"] = idx

    output_data = {
        "images": filtered_images,
        "annotations": filtered_annots,
        "categories": data.get("categories", []),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    logger.success(
        "Wrote {} images, {} annotations -> {}",
        len(filtered_images),
        len(filtered_annots),
        output_json,
    )


if __name__ == "__main__":
    main()
