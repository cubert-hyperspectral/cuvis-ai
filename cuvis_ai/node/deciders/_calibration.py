"""Threshold-calibration sweep math shared by the deciders and the calibrate CLI.

A trained anomaly pipeline ships its decider with fixed threshold hparams that go stale
per checkpoint: training moves the score distribution, so values tuned for one set of
weights misfit the next. These functions sweep a decider's actual decision rule against
ground truth (F1-max) so the operating point can be re-fitted on a labelled split. The
sweep is dispatched on the decider class:

- two-stage: a 2-D grid over the stage-1 image gate (mean of top-k pixel scores, raw
  space) and an absolute stage-2 ``pixel_threshold`` (raw space). The stages couple - a
  frame killed by the gate contributes all-zero pixels - so the grid is joint.

- binary: a single sweep over an absolute elementwise cutoff. ``BinaryDecider`` thresholds
  ``sigmoid(logits)``, so the sweep runs in raw space and the caller maps the optimum
  through sigmoid.

- quantile: a single sweep over the per-frame adaptive ``quantile``; the decider recomputes
  its cutoff from each frame's own scores, so the sweep fixes the flagged-pixel fraction
  rather than an absolute threshold.

All functions are numpy-based and free of any pipeline / node state, so both the in-training
calibration phase and the offline ``calibrate-thresholds`` CLI call the same code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch


def reduce_scores_targets(
    scores: torch.Tensor, targets: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce stacked val scores/targets to the numpy arrays the sweeps consume.

    ``scores`` is the decider-input tensor stacked over the split, ``[N, H, W, C]`` (or
    ``[N, H, W]``); ``targets`` the ground-truth mask, bool-ish ``[N, H, W, C]`` or
    ``[N, H, W]``. Returns ``(full_scores[N,H,W,C], pixel_scores[N,H,W],
    gt_masks[N,H,W] bool, frame_labels[N] bool)``. ``pixel_scores`` is the per-pixel max
    over channels - the same reduction the deciders apply in ``forward`` - and
    ``full_scores`` is kept for the per-frame quantile probes.
    """
    full = scores.detach().to("cpu", torch.float32).numpy()
    if full.ndim == 3:
        full = full[..., None]
    pixel = full.max(axis=-1)
    gt = targets.detach().to("cpu").numpy()
    gt = (gt.any(axis=-1) if gt.ndim == 4 else gt).astype(bool)
    frame_labels = gt.any(axis=(1, 2))
    return full, pixel, gt, frame_labels


def prf(tp: float, fp: float, fn: float) -> dict[str, float]:
    """Precision, recall, F1 and IoU from raw counts (0 when undefined)."""
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def image_metrics_at(image_scores: np.ndarray, labels: np.ndarray, thr: float) -> dict[str, float]:
    """Frame-level precision/recall/F1/IoU at one image threshold (``>=``)."""
    flagged = image_scores >= thr
    tp = float((flagged & labels).sum())
    return prf(tp, float((flagged & ~labels).sum()), float((~flagged & labels).sum()))


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Exact rank-based AUROC (Mann-Whitney U) for a small sample count."""
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Midranks for ties, so equal scores contribute 0.5 each.
    for value in np.unique(scores):
        tied = scores == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    u_statistic = ranks[labels].sum() - positives * (positives + 1) / 2.0
    return float(u_statistic / (positives * negatives))


def sigmoid(values: np.ndarray | float) -> Any:
    """Numerically plain sigmoid; raw anomaly scores are small in practice."""
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


def topk_mean_scores(pixel_scores: np.ndarray, top_k_fraction: float) -> np.ndarray:
    """Stage-1 image score per frame: mean of the top-k fraction of pixel scores.

    ``k`` replicates the decider bit-for-bit: the product is cast to float32 before ceiling
    (``TwoStageBinaryDecider.forward``), so calibration and runtime agree even when the
    float64 product lands epsilon above an integer.
    """
    flat = pixel_scores.reshape(pixel_scores.shape[0], -1)
    k = max(1, int(np.ceil(np.float32(flat.shape[1] * top_k_fraction))))
    top = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:]
    return top.mean(axis=1)


def pixel_candidates(pixel_scores: np.ndarray, num_candidates: int) -> np.ndarray:
    """Absolute-threshold candidates from pooled score quantiles.

    The tail reaches ``1/pixel_scores.size`` (a single pooled pixel) so the achievable
    precision frontier for sparse anomalies lies inside the grid.
    """
    tail_lo = max(1.0 / pixel_scores.size, 1e-12)
    upper_tail = 1.0 - np.logspace(np.log10(tail_lo), np.log10(0.5), num_candidates)
    return np.unique(np.quantile(pixel_scores, np.sort(upper_tail)))


def quantile_grid_builder(
    preset_quantile: float, num_candidates: int
) -> Callable[[int], np.ndarray]:
    """Quantile-sweep grid deferred on the per-frame pixel count.

    The tail reaches one pixel per frame so the sweep can trade all the way down to a single
    flagged pixel; the shipped preset value is always included for before/after.
    """

    def build(numel: int) -> np.ndarray:
        """Build the quantile grid for frames of ``numel`` score values."""
        tail_lo = max(1.0 / numel, 1e-12)
        grid = 1.0 - np.logspace(np.log10(tail_lo), np.log10(0.5), num_candidates)
        return np.unique(np.append(grid, preset_quantile))

    return build


def preset_probe_builder(preset_quantile: float) -> Callable[[int], np.ndarray]:
    """Single-probe builder: just the shipped preset quantile (before/after reference)."""

    def build(_: int) -> np.ndarray:
        """Return the preset quantile regardless of frame size."""
        return np.asarray([preset_quantile])

    return build


def sweep_two_stage(
    pixel_scores: np.ndarray,
    gt_masks: np.ndarray,
    image_scores: np.ndarray,
    frame_labels: np.ndarray,
    num_candidates: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """2-D sweep (image gate x absolute pixel threshold) for ``TwoStageBinaryDecider``.

    Returns ``(best_image, joint_best, conditional_best)``; gated-out frames contribute their
    full ground truth as misses, which is what couples the two stages.
    """
    image_candidates = np.unique(image_scores)
    image_grid = [
        {"threshold": float(thr), **image_metrics_at(image_scores, frame_labels, thr)}
        for thr in image_candidates
    ]
    best_image = max(image_grid, key=lambda row: (row["f1"], row["threshold"]))

    candidates = pixel_candidates(pixel_scores, num_candidates)
    flat_scores = pixel_scores.reshape(len(frame_labels), -1)
    flat_gt = gt_masks.reshape(len(frame_labels), -1)
    not_gt = ~flat_gt
    gt_per_frame = flat_gt.sum(axis=1).astype(np.float64)
    # Per-frame TP/FP for every pixel candidate: [n_frames, n_candidates]
    tp = np.empty((len(frame_labels), len(candidates)), dtype=np.float64)
    fp = np.empty_like(tp)
    for j, thr in enumerate(candidates):
        flagged = flat_scores >= thr
        tp[:, j] = (flagged & flat_gt).sum(axis=1)
        fp[:, j] = (flagged & not_gt).sum(axis=1)

    total_gt = float(gt_per_frame.sum())
    joint_best: dict[str, Any] | None = None
    conditional_best: dict[str, Any] | None = None
    for image_thr in image_candidates:
        gate = image_scores >= image_thr
        gated_tp = tp[gate].sum(axis=0)
        gated_fp = fp[gate].sum(axis=0)
        fn = total_gt - gated_tp  # gated-out frames contribute their full GT as misses
        for j, pixel_thr in enumerate(candidates):
            row = {
                "image_threshold": float(image_thr),
                "pixel_threshold": float(pixel_thr),
                **prf(float(gated_tp[j]), float(gated_fp[j]), float(fn[j])),
            }
            if joint_best is None or row["f1"] > joint_best["f1"]:
                joint_best = row
            if image_thr == best_image["threshold"] and (
                conditional_best is None or row["f1"] > conditional_best["f1"]
            ):
                conditional_best = row
    assert joint_best is not None and conditional_best is not None
    return best_image, joint_best, conditional_best


def sweep_absolute(
    pixel_scores: np.ndarray, gt_masks: np.ndarray, num_candidates: int
) -> dict[str, Any]:
    """Single ungated sweep over an absolute elementwise cutoff (``BinaryDecider``)."""
    total_gt = float(gt_masks.sum())
    not_gt = ~gt_masks
    best: dict[str, Any] | None = None
    for thr in pixel_candidates(pixel_scores, num_candidates):
        flagged = pixel_scores >= thr
        tp = float((flagged & gt_masks).sum())
        row = {
            "raw_threshold": float(thr),
            **prf(tp, float((flagged & not_gt).sum()), total_gt - tp),
        }
        if best is None or row["f1"] > best["f1"]:
            best = row
    assert best is not None
    return best


def sweep_quantile(
    pixel_scores: np.ndarray,
    gt_masks: np.ndarray,
    frame_quantiles: dict[float, np.ndarray],
) -> dict[str, Any]:
    """Sweep the per-frame adaptive ``quantile`` (``QuantileBinaryDecider``)."""
    total_gt = float(gt_masks.sum())
    not_gt = ~gt_masks
    best: dict[str, Any] | None = None
    for q in sorted(frame_quantiles):
        thresholds = frame_quantiles[q][:, None, None]
        flagged = pixel_scores >= thresholds
        tp = float((flagged & gt_masks).sum())
        row = {
            "quantile": float(q),
            **prf(tp, float((flagged & not_gt).sum()), total_gt - tp),
        }
        if best is None or row["f1"] > best["f1"]:
            best = row
    assert best is not None
    return best
