"""Spectral re-identification validation script.

Loads a CU3S file + COCO detection JSON, extracts per-bbox spectral signatures,
then measures how accurately cosine distance can re-identify detections under
several perturbation modes.  All variants run in a single pass and results are
written to a Markdown report.

Usage::

    python test_spectral_reid.py \
        --cu3s-path       D:/data/.../Auto_013+01.cu3s \
        --detection-json  D:/data/.../detection_results.json \
        --output-md       D:/data/.../spectral_reid_report.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_detections(json_path: Path) -> dict[int, np.ndarray]:
    """Parse COCO detection JSON into ``{image_id: array[N, 4]}`` (xyxy)."""
    with open(json_path) as f:
        data = json.load(f)

    frame_bboxes: dict[int, list[list[float]]] = {}
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]  # COCO [x, y, w, h]
        frame_bboxes.setdefault(ann["image_id"], []).append(
            [x, y, x + w, y + h],
        )

    return {img_id: np.array(boxes, dtype=np.float32) for img_id, boxes in frame_bboxes.items()}


def spectral_distance(sigs_a: np.ndarray, sigs_b: np.ndarray) -> np.ndarray:
    """Vectorized cosine distance.  Returns cost_matrix [N, M] in [0, 1]."""
    norms_a = np.linalg.norm(sigs_a, axis=1, keepdims=True).clip(min=1e-8)
    norms_b = np.linalg.norm(sigs_b, axis=1, keepdims=True).clip(min=1e-8)
    sim = np.clip((sigs_a / norms_a) @ (sigs_b / norms_b).T, 0.0, 1.0)
    return 1.0 - sim


def perturb_bboxes(
    bboxes: np.ndarray,
    jitter_frac: float,
    rng: np.random.Generator,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Randomly shift and scale bboxes.  Returns new array clamped to image."""
    out = bboxes.copy()
    widths = out[:, 2] - out[:, 0]
    heights = out[:, 3] - out[:, 1]

    # random shift
    dx = rng.uniform(-jitter_frac, jitter_frac, size=len(out)) * widths
    dy = rng.uniform(-jitter_frac, jitter_frac, size=len(out)) * heights
    out[:, [0, 2]] += dx[:, None]
    out[:, [1, 3]] += dy[:, None]

    # random scale (around center)
    cx = (out[:, 0] + out[:, 2]) / 2
    cy = (out[:, 1] + out[:, 3]) / 2
    sw = rng.uniform(1 - jitter_frac, 1 + jitter_frac, size=len(out))
    sh = rng.uniform(1 - jitter_frac, 1 + jitter_frac, size=len(out))
    half_w = widths * sw / 2
    half_h = heights * sh / 2
    out[:, 0] = cx - half_w
    out[:, 1] = cy - half_h
    out[:, 2] = cx + half_w
    out[:, 3] = cy + half_h

    # clamp
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, img_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, img_h)
    return out


def hungarian_accuracy(
    cost: np.ndarray,
    gt_col_for_row: np.ndarray,
) -> tuple[int, int]:
    """Run Hungarian and return (n_correct, n_total)."""
    row_ind, col_ind = linear_sum_assignment(cost)
    correct = int(np.sum(col_ind == gt_col_for_row[row_ind]))
    return correct, len(row_ind)


def extract_signatures(
    extractor: object,
    cube: torch.Tensor,
    bboxes_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run BBoxSpectralExtractor, return (sigs[N,C], valid[N]) numpy arrays."""
    bboxes_t = torch.tensor(bboxes_np, dtype=torch.float32).unsqueeze(0)  # [1, N, 4]
    out = extractor.forward(cube=cube, bboxes=bboxes_t)
    sigs = out["spectral_signatures"][0].cpu().numpy()  # [N, C]
    valid = out["spectral_valid"][0].cpu().numpy()  # [N]
    return sigs, valid


# ---------------------------------------------------------------------------
# Per-variant scorers
# ---------------------------------------------------------------------------


@dataclass
class FrameResult:
    frame_id: int
    n_dets: int
    n_valid: int
    n_correct: int


@dataclass
class VariantResults:
    name: str
    frames: list[FrameResult] = field(default_factory=list)

    @property
    def total_dets(self) -> int:
        return sum(f.n_valid for f in self.frames)

    @property
    def total_correct(self) -> int:
        return sum(f.n_correct for f in self.frames)

    @property
    def accuracy(self) -> float:
        return self.total_correct / self.total_dets if self.total_dets else 0.0


def run_stock(
    sigs: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator,
    frame_id: int,
    **_: object,
) -> FrameResult:
    """Shuffle and re-match within frame."""
    idx_valid = np.where(valid)[0]
    if len(idx_valid) < 2:
        return FrameResult(frame_id, len(valid), len(idx_valid), len(idx_valid))

    v_sigs = sigs[idx_valid]
    perm = rng.permutation(len(v_sigs))
    shuffled = v_sigs[perm]

    cost = spectral_distance(v_sigs, shuffled)
    # ground truth: row i should match column inv_perm[i]
    inv_perm = np.argsort(perm)
    correct, total = hungarian_accuracy(cost, inv_perm)
    return FrameResult(frame_id, len(valid), len(idx_valid), correct)


def run_bbox_jitter(
    sigs: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator,
    frame_id: int,
    *,
    extractor: object,
    cube: torch.Tensor,
    bboxes_np: np.ndarray,
    jitter_frac: float,
    img_h: int,
    img_w: int,
    **_: object,
) -> FrameResult:
    """Perturb bboxes, re-extract, match against originals."""
    idx_valid = np.where(valid)[0]
    if len(idx_valid) < 2:
        return FrameResult(frame_id, len(valid), len(idx_valid), len(idx_valid))

    ref_sigs = sigs[idx_valid]
    valid_bboxes = bboxes_np[idx_valid]

    jittered = perturb_bboxes(valid_bboxes, jitter_frac, rng, img_h, img_w)
    query_sigs, query_valid = extract_signatures(extractor, cube, jittered)

    # only score where both ref and query are valid
    both_valid = query_valid.astype(bool)
    if both_valid.sum() < 2:
        return FrameResult(frame_id, len(valid), len(idx_valid), int(both_valid.sum()))

    ref_v = ref_sigs[both_valid]
    qry_v = query_sigs[both_valid]

    cost = spectral_distance(ref_v, qry_v)
    # ground truth is identity (same order)
    gt = np.arange(len(ref_v))
    correct, total = hungarian_accuracy(cost, gt)
    return FrameResult(frame_id, len(valid), len(idx_valid), correct)


def run_sig_noise(
    sigs: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator,
    frame_id: int,
    *,
    noise_std: float,
    **_: object,
) -> FrameResult:
    """Add Gaussian noise to signatures, re-normalize, match."""
    idx_valid = np.where(valid)[0]
    if len(idx_valid) < 2:
        return FrameResult(frame_id, len(valid), len(idx_valid), len(idx_valid))

    ref = sigs[idx_valid]
    noisy = ref + rng.normal(0, noise_std, size=ref.shape).astype(ref.dtype)
    norms = np.linalg.norm(noisy, axis=1, keepdims=True).clip(min=1e-8)
    noisy = noisy / norms

    perm = rng.permutation(len(ref))
    shuffled_noisy = noisy[perm]

    cost = spectral_distance(ref, shuffled_noisy)
    inv_perm = np.argsort(perm)
    correct, total = hungarian_accuracy(cost, inv_perm)
    return FrameResult(frame_id, len(valid), len(idx_valid), correct)


def run_drop(
    sigs: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator,
    frame_id: int,
    *,
    drop_frac: float,
    **_: object,
) -> FrameResult:
    """Drop a fraction of detections, match remaining."""
    idx_valid = np.where(valid)[0]
    if len(idx_valid) < 2:
        return FrameResult(frame_id, len(valid), len(idx_valid), len(idx_valid))

    ref = sigs[idx_valid]
    n = len(ref)
    n_keep = max(2, int(n * (1 - drop_frac)))
    keep_idx = np.sort(rng.choice(n, size=n_keep, replace=False))

    query = ref[keep_idx]
    # shuffle query
    perm = rng.permutation(len(query))
    shuffled_query = query[perm]

    cost = spectral_distance(ref, shuffled_query)
    # Hungarian on [n, n_keep] matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind[k] matched to col_ind[k]; correct if row_ind[k] == keep_idx[perm^-1[col_ind[k]]]
    inv_perm = np.argsort(perm)
    correct = 0
    for r, c in zip(row_ind, col_ind, strict=False):
        original_query_idx = inv_perm[c]
        if r == keep_idx[original_query_idx]:
            correct += 1
    return FrameResult(frame_id, len(valid), n_keep, correct)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_report(
    all_results: dict[str, VariantResults],
    params: dict[str, object],
    output_path: Path,
) -> None:
    """Write Markdown report."""
    lines: list[str] = []
    lines.append("# Spectral Re-ID Validation Report\n")

    lines.append("## Parameters\n")
    for k, v in params.items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Variant | Frames | Total Dets | Correct | Accuracy | F1 |")
    lines.append("|---------|--------|------------|---------|----------|----|")
    for name, res in all_results.items():
        acc = res.accuracy
        lines.append(
            f"| {name} | {len(res.frames)} | {res.total_dets} "
            f"| {res.total_correct} | {acc:.3f} | {acc:.3f} |"
        )
    lines.append("")

    # Per-frame details
    for name, res in all_results.items():
        lines.append(f"<details><summary>{name} — per-frame</summary>\n")
        lines.append("| Frame | Dets | Valid | Correct | Accuracy |")
        lines.append("|-------|------|-------|---------|----------|")
        for fr in res.frames:
            acc = fr.n_correct / fr.n_valid if fr.n_valid else 0.0
            lines.append(
                f"| {fr.frame_id:>5d} | {fr.n_dets:>4d} | {fr.n_valid:>5d} "
                f"| {fr.n_correct:>7d} | {acc:>8.3f} |"
            )
        lines.append("\n</details>\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to {}", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--cu3s-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--detection-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-md",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("spectral_reid_report.md"),
    show_default=True,
    help="Path for the Markdown report.",
)
@click.option("--processing-mode", type=str, default="SpectralRadiance", show_default=True)
@click.option("--center-crop-scale", type=float, default=0.65, show_default=True)
@click.option("--end-frame", type=int, default=-1, show_default=True)
@click.option("--jitter-frac", type=float, default=0.1, show_default=True)
@click.option("--noise-std", type=float, default=0.05, show_default=True)
@click.option("--drop-frac", type=float, default=0.3, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def main(
    cu3s_path: Path,
    detection_json: Path,
    output_md: Path,
    processing_mode: str,
    center_crop_scale: float,
    end_frame: int,
    jitter_frac: float,
    noise_std: float,
    drop_frac: float,
    seed: int,
) -> None:
    from cuvis_ai_core.data.datasets import SingleCu3sDataModule

    from cuvis_ai.node.data import CU3SDataNode
    from cuvis_ai.node.spectral_extractor import BBoxSpectralExtractor

    rng = np.random.default_rng(seed)
    frame_dets = load_detections(detection_json)

    # --- data loading ---
    predict_ids = list(range(end_frame)) if end_frame > 0 else None
    datamodule = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        processing_mode=processing_mode,
        batch_size=1,
        predict_ids=predict_ids,
    )
    datamodule.setup(stage="predict")

    cu3s_node = CU3SDataNode(name="cu3s_norm")
    extractor = BBoxSpectralExtractor(
        center_crop_scale=center_crop_scale,
        l2_normalize=True,
        name="spec_extract",
    )

    # --- variant accumulators ---
    variant_names = ["stock", "bbox-jitter", "sig-noise", "drop"]
    all_results: dict[str, VariantResults] = {
        name: VariantResults(name=name) for name in variant_names
    }

    from torch.utils.data import DataLoader

    loader = DataLoader(datamodule.predict_ds, batch_size=1, shuffle=False)
    for batch in loader:
        norm = cu3s_node.forward(**dict(batch.items()))
        cube = norm["cube"]  # [1, H, W, C]
        frame_idx = int(batch["mesu_index"].item())
        img_h, img_w = cube.shape[1], cube.shape[2]

        bboxes_np = frame_dets.get(frame_idx)
        if bboxes_np is None or len(bboxes_np) < 2:
            continue

        # extract signatures once for stock / sig-noise / drop
        sigs, valid = extract_signatures(extractor, cube, bboxes_np)

        common = {
            "sigs": sigs,
            "valid": valid,
            "frame_id": frame_idx,
            "extractor": extractor,
            "cube": cube,
            "bboxes_np": bboxes_np,
            "jitter_frac": jitter_frac,
            "noise_std": noise_std,
            "drop_frac": drop_frac,
            "img_h": img_h,
            "img_w": img_w,
        }

        # each variant gets its own rng fork for reproducibility
        all_results["stock"].frames.append(
            run_stock(rng=np.random.default_rng(rng.integers(2**63)), **common)
        )
        all_results["bbox-jitter"].frames.append(
            run_bbox_jitter(rng=np.random.default_rng(rng.integers(2**63)), **common)
        )
        all_results["sig-noise"].frames.append(
            run_sig_noise(rng=np.random.default_rng(rng.integers(2**63)), **common)
        )
        all_results["drop"].frames.append(
            run_drop(rng=np.random.default_rng(rng.integers(2**63)), **common)
        )

        logger.info(
            "Frame {:>4d} | dets={:>3d} | stock={}/{} jitter={}/{} noise={}/{} drop={}/{}",
            frame_idx,
            len(bboxes_np),
            all_results["stock"].frames[-1].n_correct,
            all_results["stock"].frames[-1].n_valid,
            all_results["bbox-jitter"].frames[-1].n_correct,
            all_results["bbox-jitter"].frames[-1].n_valid,
            all_results["sig-noise"].frames[-1].n_correct,
            all_results["sig-noise"].frames[-1].n_valid,
            all_results["drop"].frames[-1].n_correct,
            all_results["drop"].frames[-1].n_valid,
        )

    # --- report ---
    params = {
        "cu3s_path": str(cu3s_path),
        "detection_json": str(detection_json),
        "processing_mode": processing_mode,
        "center_crop_scale": center_crop_scale,
        "end_frame": end_frame,
        "jitter_frac": jitter_frac,
        "noise_std": noise_std,
        "drop_frac": drop_frac,
        "seed": seed,
    }

    output_md.parent.mkdir(parents=True, exist_ok=True)
    write_report(all_results, params, output_md)

    # stdout summary
    print("\n" + "=" * 60)
    for name, res in all_results.items():
        print(f"  {name:<14s}  acc={res.accuracy:.3f}  ({res.total_correct}/{res.total_dets})")
    print("=" * 60)


if __name__ == "__main__":
    main()
