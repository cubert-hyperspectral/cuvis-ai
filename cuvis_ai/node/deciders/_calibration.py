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
  ``sigmoid(logits)`` in float32, so the caller hands in those probabilities and the sweep
  runs in the exact space ``forward`` compares in.

- quantile: a single sweep over the per-frame adaptive ``quantile``; the decider recomputes
  its cutoff from each frame's own scores, so the sweep fixes the flagged-pixel fraction
  rather than an absolute threshold.

Numerics, so that what the sweep scores is what ``forward`` decides:

- Pixel-level thresholds are compared in float32 at runtime (``TwoStageBinaryDecider`` casts
  ``pixel_threshold`` to the score dtype, ``BinaryDecider`` compares float32 probabilities
  against a Python float that torch casts to float32). Candidates are therefore float32 values
  and the per-frame counts use ``searchsorted`` on float32 arrays: exact, and one pass per
  frame instead of one pass per candidate.

- The image gate compares two Python floats (``image_score`` comes from ``.item()``), so image
  thresholds stay float64. ``frame_image_score`` is the very statistic ``forward`` computes.

- F1 ties resolve to the highest candidate (the most restrictive threshold), then the
  applied value steps down to the midpoint of the F1 plateau (``margin_below``): same
  validation F1, a real margin against near-duplicate scores at deployment.

All functions are free of pipeline / node state, so both the in-training calibration phase
and the offline ``calibrate-thresholds`` CLI call the same code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch


class CalibrationError(ValueError):
    """Raised when a decider cannot be calibrated on the given scores and targets.

    Covers unusable data (shape mismatch, non-finite scores, a single-class split) and a
    decider configuration whose decision rule the sweep cannot reproduce (a ``reduce_dims``
    that keeps a real channel axis out of the quantile). A caller that runs calibration as an
    optional phase catches this, logs it, and leaves the shipped thresholds untouched.
    """


def topk_count(numel: int, top_k_fraction: float) -> int:
    """``k`` for the stage-1 image score: ``ceil(top_k_fraction * numel)``, at least 1.

    The product is rounded to float32 before the ceiling, exactly as
    ``TwoStageBinaryDecider.forward`` does, so a float64 product that lands epsilon above an
    integer does not add a pixel.
    """
    return max(
        1,
        int(torch.ceil(torch.tensor(numel * top_k_fraction, dtype=torch.float32)).item()),
    )


def frame_image_score(pixel_scores: torch.Tensor, top_k_fraction: float) -> float:
    """Stage-1 image score of one frame: mean of the top-k per-pixel scores.

    This is the statistic ``TwoStageBinaryDecider.forward`` gates on, shared so calibration
    and runtime agree bit for bit. ``pixel_scores`` is the frame's ``[H, W]`` per-pixel score
    map (already reduced over channels).
    """
    flat = pixel_scores.reshape(-1)
    k = topk_count(flat.numel(), top_k_fraction)
    return torch.topk(flat, k).values.mean().item()


def topk_mean_scores(pixel_scores: np.ndarray, top_k_fraction: float) -> np.ndarray:
    """Stage-1 image score per frame for stacked ``[N, H, W]`` pixel scores (float64).

    Delegates to :func:`frame_image_score` frame by frame, so the values are the ones
    ``forward`` will compute, widened to float64 the way ``.item()`` widens them.
    """
    frames = torch.from_numpy(np.ascontiguousarray(pixel_scores, dtype=np.float32))
    return np.asarray(
        [frame_image_score(frame, top_k_fraction) for frame in frames], dtype=np.float64
    )


def sigmoid_float32(values: np.ndarray) -> np.ndarray:
    """``torch.sigmoid`` in float32: the probabilities ``BinaryDecider.forward`` thresholds.

    Computed with torch rather than numpy so the CLI and the decider see identical values,
    including the saturation to exactly ``1.0`` above logits of roughly 17.
    """
    return torch.sigmoid(torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))).numpy()


def reduce_scores_targets(
    scores: torch.Tensor, targets: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce stacked val scores/targets to the numpy arrays the sweeps consume.

    ``scores`` is the decider-input tensor stacked over the split, ``[N, H, W, C]`` (or
    ``[N, H, W]``); ``targets`` the ground-truth mask, bool-ish ``[N, H, W, C]`` or
    ``[N, H, W]``. Returns ``(full_scores[N,H,W,C] float32, pixel_scores[N,H,W] float32,
    gt_masks[N,H,W] bool, frame_labels[N] bool)``. ``pixel_scores`` is the per-pixel max
    over channels - the same reduction the deciders apply in ``forward`` - and
    ``full_scores`` is kept for the per-frame quantile probes.

    Raises:
        CalibrationError: when the shapes disagree, a score is nan or inf (a threshold fitted
            on it would be refused when the yaml is loaded), or the targets are single-class
            (no anomalous pixel, or nothing but anomalous pixels), which no threshold can fit.
    """
    if scores.dim() not in (3, 4):
        raise CalibrationError(
            f"scores must be [N, H, W] or [N, H, W, C], got shape {tuple(scores.shape)}"
        )
    if targets.dim() not in (3, 4):
        raise CalibrationError(
            f"targets must be [N, H, W] or [N, H, W, C], got shape {tuple(targets.shape)}"
        )
    if tuple(scores.shape[:3]) != tuple(targets.shape[:3]):
        raise CalibrationError(
            f"scores {tuple(scores.shape)} and targets {tuple(targets.shape)} disagree on "
            "[N, H, W]; every frame needs its own ground-truth mask"
        )
    full = scores.detach().to("cpu", torch.float32).numpy()
    if full.ndim == 3:
        full = full[..., None]
    if not np.isfinite(full).all():
        raise CalibrationError(
            "scores contain nan or inf; a threshold fitted on them could not be loaded back"
        )
    pixel = full.max(axis=-1)
    gt = targets.detach().to("cpu").numpy()
    gt = (gt.any(axis=-1) if gt.ndim == 4 else gt).astype(bool)
    if not gt.any():
        raise CalibrationError(
            "targets mark no anomalous pixel; a single-class split cannot calibrate a threshold"
        )
    if gt.all():
        raise CalibrationError(
            "targets mark every pixel anomalous; a single-class split cannot calibrate a threshold"
        )
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


def pixel_candidates(pixel_scores: np.ndarray, num_candidates: int) -> np.ndarray:
    """Absolute-threshold candidates from pooled score quantiles, exact in float32.

    The tail reaches ``1/pixel_scores.size`` (a single pooled pixel) so the achievable
    precision frontier for sparse anomalies lies inside the grid. Candidates are cast to
    float32 - the dtype the deciders compare pixel scores in - and the pooled maximum is
    always included, so every candidate is a threshold ``forward`` applies exactly.
    """
    flat = pixel_scores.reshape(-1)
    tail_lo = max(1.0 / flat.size, 1e-12)
    upper_tail = 1.0 - np.logspace(np.log10(tail_lo), np.log10(0.5), num_candidates)
    quantiles = np.quantile(flat, np.sort(upper_tail)).astype(np.float32)
    return np.unique(np.append(quantiles, np.float32(flat.max())))


def margin_below(threshold: float, samples: np.ndarray, *, float32: bool) -> float:
    """Midpoint between ``threshold`` and the largest sample strictly below it.

    An F1-max sweep returns a threshold that sits on a validation sample (or on a quantile of
    the samples). Every threshold in ``(lower_sample, threshold]`` flags the same set, so the
    plateau midpoint keeps the validation F1 and adds a margin against near-duplicate scores
    at deployment. With ``float32`` the midpoint is rounded to float32 (the dtype the deciders
    compare pixel scores in) and kept only if it still separates the two samples; otherwise,
    and when no sample lies below, ``threshold`` itself is returned.
    """
    threshold = float(threshold)
    below = samples[samples < threshold]
    if below.size == 0:
        return threshold
    lower = float(below.max())
    midpoint = (threshold + lower) / 2.0
    if float32:
        midpoint = float(np.float32(midpoint))
        if not lower < midpoint <= threshold:
            return threshold
    return midpoint


def _count_at_or_above(sorted_values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """How many of the ascending ``sorted_values`` are ``>=`` each candidate."""
    return sorted_values.size - np.searchsorted(sorted_values, candidates, side="left")


def frame_confusions(
    pixel_scores: np.ndarray, gt_masks: np.ndarray, candidates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame true/false positive counts for every candidate, ``[N, len(candidates)]`` each.

    One sort per frame and class plus a ``searchsorted`` against the candidate edges gives the
    counts for all candidates at once; the cost no longer scales with the candidate count.
    ``pixel_scores`` and ``candidates`` must share a dtype (float32) so the comparison is exact.
    """
    n_frames = pixel_scores.shape[0]
    flat_scores = pixel_scores.reshape(n_frames, -1)
    flat_gt = gt_masks.reshape(n_frames, -1)
    tp = np.empty((n_frames, candidates.size), dtype=np.float64)
    fp = np.empty_like(tp)
    for i in range(n_frames):
        positives = np.sort(flat_scores[i][flat_gt[i]])
        negatives = np.sort(flat_scores[i][~flat_gt[i]])
        tp[i] = _count_at_or_above(positives, candidates)
        fp[i] = _count_at_or_above(negatives, candidates)
    return tp, fp


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
    full ground truth as misses, which is what couples the two stages. ``best_image`` carries
    the on-point ``threshold`` and the plateau-midpoint ``margin_threshold``; the two pixel
    rows carry ``pixel_threshold`` / ``margin_pixel_threshold`` and ``image_threshold`` /
    ``margin_image_threshold``. Ties resolve to the highest threshold.
    """
    pixel_scores = np.ascontiguousarray(pixel_scores, dtype=np.float32)
    image_candidates = np.unique(image_scores)
    image_grid = [
        {"threshold": float(thr), **image_metrics_at(image_scores, frame_labels, thr)}
        for thr in image_candidates
    ]
    best_image = max(image_grid, key=lambda row: (row["f1"], row["threshold"]))
    best_image["margin_threshold"] = margin_below(
        best_image["threshold"], image_scores, float32=False
    )

    candidates = pixel_candidates(pixel_scores, num_candidates)
    tp, fp = frame_confusions(pixel_scores, gt_masks, candidates)
    total_gt = float(gt_masks.sum())
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
            # ``>=``: candidates ascend, so a tie goes to the more restrictive threshold.
            if joint_best is None or row["f1"] >= joint_best["f1"]:
                joint_best = row
            if image_thr == best_image["threshold"] and (
                conditional_best is None or row["f1"] >= conditional_best["f1"]
            ):
                conditional_best = row
    assert joint_best is not None and conditional_best is not None
    for row in (joint_best, conditional_best):
        row["margin_image_threshold"] = margin_below(
            row["image_threshold"], image_scores, float32=False
        )
        row["margin_pixel_threshold"] = margin_below(
            row["pixel_threshold"], pixel_scores, float32=True
        )
    return best_image, joint_best, conditional_best


def sweep_absolute(
    pixel_scores: np.ndarray, gt_masks: np.ndarray, num_candidates: int
) -> dict[str, Any]:
    """Single ungated sweep over an absolute elementwise cutoff (``BinaryDecider``).

    ``pixel_scores`` are the values ``forward`` compares (for ``BinaryDecider`` the float32
    sigmoid probabilities). Returns the F1-max row with the on-point ``threshold`` and the
    plateau-midpoint ``margin_threshold``; ties resolve to the highest threshold.
    """
    pixel_scores = np.ascontiguousarray(pixel_scores, dtype=np.float32)
    candidates = pixel_candidates(pixel_scores, num_candidates)
    tp, fp = frame_confusions(pixel_scores, gt_masks, candidates)
    tp_total = tp.sum(axis=0)
    fp_total = fp.sum(axis=0)
    total_gt = float(gt_masks.sum())
    best: dict[str, Any] | None = None
    for j, thr in enumerate(candidates):
        row = {
            "threshold": float(thr),
            **prf(float(tp_total[j]), float(fp_total[j]), total_gt - float(tp_total[j])),
        }
        if best is None or row["f1"] >= best["f1"]:
            best = row
    assert best is not None
    best["margin_threshold"] = margin_below(best["threshold"], pixel_scores, float32=True)
    return best


def sweep_quantile(
    pixel_scores: np.ndarray,
    gt_masks: np.ndarray,
    frame_quantiles: dict[float, np.ndarray],
) -> dict[str, Any]:
    """Sweep the per-frame adaptive ``quantile`` (``QuantileBinaryDecider``).

    Ties resolve to the highest quantile (the most restrictive cutoff).
    """
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
        if best is None or row["f1"] >= best["f1"]:
            best = row
    assert best is not None
    return best
