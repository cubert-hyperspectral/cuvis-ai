"""Render the dataset videos for the lentils HuggingFace card.

Produces eight mp4 files used by the dataset README:

    measurements/cu3s/2026_04_15_13_32_55/
        Auto_003+01_rgb_input.mp4
        Auto_003+01_rgb_overlay.mp4
        Auto_003+01_cir_input.mp4
        Auto_003+01_cir_overlay.mp4
        Auto_003+01_concrete_input.mp4
        Auto_003+01_concrete_overlay.mp4
    assets/
        lentils_3method_teaser.mp4   # 3-method overlay side-by-side
        lentils_3method_input.mp4    # 3-method input-only (no annotations)

Inputs are fetched on demand by ``resolve_default_config()`` from
``cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils``. Output
mp4s land in the local HuggingFace mirror at ``D:/huggingface_data/data/...``
ready for an ``HfApi.upload_folder(...)`` push.

Run with::

    uv run python tools/render_lentils_dataset_videos.py

Re-run is idempotent — files are overwritten in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
USE_CASES = REPO_ROOT / "notebooks" / "use_cases"
sys.path.insert(0, str(USE_CASES))

from utils import (  # noqa: E402  - sys.path tweak above
    METHOD_TITLES,
    load_pipeline_for_inference,
    make_predict_loader,
    resolve_default_config,
    run_method_on_frame_subset,
    to_uint8_rgb,
)
from utils import _overlay_frame_from_row  # noqa: E402  - intentional internal reuse

HF_DATA_ROOT = Path(r"D:/huggingface_data/data")
DATA_MEASUREMENTS_DIR = HF_DATA_ROOT / "measurements" / "cu3s" / "2026_04_15_13_32_55"
DATA_ASSETS_DIR = HF_DATA_ROOT / "assets"
VIDEO_FPS = 4.0
TEASER_PANEL_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _write_mp4(frames: list[np.ndarray], output_path: Path, fps: float) -> Path:
    """Write a list of HxWx3 uint8 RGB frames to an mp4 via OpenCV mp4v."""
    if not frames:
        raise ValueError(f"No frames to write for {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = int(frames[0].shape[0]), int(frames[0].shape[1])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open mp4 writer for {output_path}")
    try:
        for frame in frames:
            if frame.shape[0] != h or frame.shape[1] != w:
                raise ValueError(
                    f"Frame size mismatch in {output_path}: "
                    f"got {frame.shape[:2]}, expected {(h, w)}"
                )
            writer.write(frame[..., ::-1])  # RGB -> BGR
    finally:
        writer.release()
    if not output_path.is_file():
        raise RuntimeError(f"mp4 writer reported success but file is missing: {output_path}")
    return output_path


def _input_frames(rows: list[dict]) -> list[np.ndarray]:
    rows_sorted = sorted(rows, key=lambda r: int(r["frame_idx"]))
    return [to_uint8_rgb(r["rgb_image"]) for r in rows_sorted]


def _overlay_frames(rows: list[dict]) -> list[np.ndarray]:
    rows_sorted = sorted(rows, key=lambda r: int(r["frame_idx"]))
    return [_overlay_frame_from_row(r) for r in rows_sorted]


def _label_panel(frame: np.ndarray, label: str) -> np.ndarray:
    """Draw a left-aligned label across the top of an RGB frame (uint8)."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    pad = max(8, h // 80)
    text_size, _ = cv2.getTextSize(label, TEASER_PANEL_LABEL_FONT, 0.9, 2)
    cv2.rectangle(canvas, (0, 0), (w, text_size[1] + 2 * pad), (0, 0, 0), thickness=-1)
    cv2.putText(
        canvas,
        label,
        (pad, text_size[1] + pad),
        TEASER_PANEL_LABEL_FONT,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _build_teaser_frames(
    overlays_by_method: dict[str, list[np.ndarray]],
    titles_by_method: dict[str, str],
) -> list[np.ndarray]:
    if len(overlays_by_method) != 3:
        raise ValueError(
            f"Teaser expects exactly 3 methods, got {sorted(overlays_by_method.keys())}"
        )
    method_order = ["dinomaly_rgb", "dinomaly_cir", "dinomaly_concrete"]
    n_frames = min(len(overlays_by_method[m]) for m in method_order)
    teaser_frames: list[np.ndarray] = []
    for i in range(n_frames):
        panels = []
        for m in method_order:
            panel = overlays_by_method[m][i]
            panel = _label_panel(panel, titles_by_method[m])
            panels.append(panel)
        teaser_frames.append(np.hstack(panels))
    return teaser_frames


def _delete_stale(*paths: Path) -> None:
    for path in paths:
        if path.is_file():
            print(f"deleting stale {path}")
            path.unlink()


def main() -> None:
    cfg = resolve_default_config()
    method_names_present = [m.name for m in cfg["methods"]]
    print(f"methods on HF: {method_names_present}")
    expected = {"dinomaly_rgb", "dinomaly_cir", "dinomaly_concrete"}
    missing = expected - set(method_names_present)
    if missing:
        raise SystemExit(
            f"Cannot render teaser: methods missing from HF: {sorted(missing)}. "
            f"Upload them and re-run."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dm, loader = make_predict_loader(
        cfg["cu3s_path"], cfg["annotation_json_path"], processing_mode="Reflectance", batch_size=1
    )
    n = len(dm.predict_ds)
    print(f"frames: {n}")
    frame_indices = set(range(n))

    # method.name -> rows
    rows_by_method: dict[str, list[dict]] = {}

    for method in cfg["methods"]:
        print(f"\n--- {method.title} ---")
        pipe = load_pipeline_for_inference(
            yaml_path=method.yaml_path,
            pt_path=method.pt_path,
            plugins_manifest=cfg["plugins_manifest"],
            device=str(device),
        )
        rows = run_method_on_frame_subset(
            pipeline=pipe,
            loader=loader,
            frame_indices=frame_indices,
            image_threshold=method.image_threshold,
            quantile=method.quantile,
            top_k_fraction=method.top_k_fraction,
            device=device,
            normal_class_ids={0},
        )
        if len(rows) != n:
            print(f"  WARNING: produced {len(rows)} rows for {method.name}, expected {n}")
        rows_by_method[method.name] = rows

        short = method.name.replace("dinomaly_", "")
        input_path = DATA_MEASUREMENTS_DIR / f"Auto_003+01_{short}_input.mp4"
        overlay_path = DATA_MEASUREMENTS_DIR / f"Auto_003+01_{short}_overlay.mp4"
        _write_mp4(_input_frames(rows), input_path, fps=VIDEO_FPS)
        print(f"  wrote {input_path}")
        _write_mp4(_overlay_frames(rows), overlay_path, fps=VIDEO_FPS)
        print(f"  wrote {overlay_path}")

    print("\n--- overlay teaser side-by-side ---")
    overlays_by_method = {m: _overlay_frames(rows_by_method[m]) for m in rows_by_method}
    titles_by_method = {m: METHOD_TITLES[m] for m in rows_by_method}
    teaser_frames = _build_teaser_frames(overlays_by_method, titles_by_method)
    teaser_path = DATA_ASSETS_DIR / "lentils_3method_teaser.mp4"
    _write_mp4(teaser_frames, teaser_path, fps=VIDEO_FPS)
    print(f"  wrote {teaser_path}")

    print("\n--- input teaser side-by-side (no annotations) ---")
    inputs_by_method = {m: _input_frames(rows_by_method[m]) for m in rows_by_method}
    input_teaser_frames = _build_teaser_frames(inputs_by_method, titles_by_method)
    input_teaser_path = DATA_ASSETS_DIR / "lentils_3method_input.mp4"
    _write_mp4(input_teaser_frames, input_teaser_path, fps=VIDEO_FPS)
    print(f"  wrote {input_teaser_path}")

    print("\n--- cleanup stale mp4s ---")
    _delete_stale(
        DATA_MEASUREMENTS_DIR / "Auto_003+01_rgb_4fps.mp4",
        DATA_MEASUREMENTS_DIR / "Auto_003+01_cir_4fps.mp4",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
