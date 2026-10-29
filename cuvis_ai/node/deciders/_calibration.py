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

import math
import numbers
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

CALIBRATION_OBJECTIVES = ("f1", "precision", "recall")
CALIBRATION_LEVELS = ("pixels", "frames")
# A split with at most this many distinct pixel scores is swept over every one of them, so a
# target lands on the exact point; a larger split starts from the sampled grid and refines.
EXACT_GRID_LIMIT = 4096
# The refinement sweeps at most this many distinct scores per round inside the bracket of the
# best cell; a wider bracket is thinned evenly and the next round zooms into the new best.
REFINE_LIMIT = 4096
REFINE_ROUNDS = 4
_EXACT_PROBE = 65536


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


# --- calibration to a precision or recall target --------------------------------------------


@dataclass(frozen=True)
class CalibrationObjective:
    """What ``calibrate`` optimises: the F1 optimum, or a precision or recall target.

    ``objective`` is ``"f1"``, ``"precision"`` or ``"recall"``; ``target`` the value in [0, 1] a
    target objective must reach; ``level`` the level it is measured at, ``"pixels"`` (pooled
    over the split) or ``"frames"`` (a frame counts as flagged when any of its pixels is, the
    evaluation's definition). Built by :func:`calibration_objective_of` from the deciders'
    ``calibration_objective``, ``calibration_target`` and ``calibration_level`` hparams.
    """

    objective: str = "f1"
    target: float | None = None
    level: str = "pixels"

    @property
    def targeted(self) -> bool:
        """Whether a target replaces the F1 optimum."""
        return self.objective != "f1"


def calibration_objective_of(
    objective: Any,
    target: Any,
    level: Any,
    *,
    levels: Sequence[str] = CALIBRATION_LEVELS,
) -> CalibrationObjective:
    """Validate the three objective hparams of a decider, refusing each by name.

    ``levels`` lists the levels the decider can serve; ``QuantileBinaryDecider`` flags pixels
    in every frame and serves ``pixels`` only.
    """
    if objective not in CALIBRATION_OBJECTIVES:
        raise ValueError(
            f"calibration_objective must be one of {CALIBRATION_OBJECTIVES}, got {objective!r}"
        )
    if level not in levels:
        raise ValueError(f"calibration_level must be one of {tuple(levels)}, got {level!r}")
    if target is None:
        if objective != "f1":
            raise ValueError(
                f"calibration_target is required for calibration_objective {objective!r}"
            )
        return CalibrationObjective(objective, None, level)
    if isinstance(target, bool) or not isinstance(target, numbers.Real):
        raise ValueError(f"calibration_target must be a number in [0, 1], got {target!r}")
    value = float(target)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"calibration_target must be a number in [0, 1], got {target!r}")
    return CalibrationObjective(objective, value, level)


def _row_thresholds(row: dict[str, Any]) -> tuple[float, float]:
    """The tie-breaking thresholds of a row, the image gate first."""
    cutoff = row.get("pixel_threshold", row.get("threshold", row.get("quantile", 0.0)))
    return float(row.get("image_threshold", 0.0)), float(cutoff)


def select_calibration_row(
    rows: Sequence[dict[str, Any]], objective: CalibrationObjective
) -> tuple[dict[str, Any], bool]:
    """The row a target objective picks, and whether the target was reached.

    Rows carry ``precision``, ``recall`` and ``f1`` at the pixel level and ``frame_precision``,
    ``frame_recall`` and ``frame_f1`` at the frame level, plus their thresholds. A recall
    objective takes, among the rows whose recall at the objective's level is at least the
    target, the one with the highest precision; a precision objective the one with the highest
    recall among the rows whose precision is at least the target. Ties go to the higher F1 at
    the other level, then to the higher thresholds (image gate first). When no row reaches the
    target, the row with the highest value of the targeted metric is returned with ``False``
    (ties broken the same way, the other metric first).
    """
    if not objective.targeted or objective.target is None:
        raise ValueError("select_calibration_row needs a precision or recall objective")
    prefix = "" if objective.level == "pixels" else "frame_"
    other_f1 = "frame_f1" if objective.level == "pixels" else "f1"
    primary, secondary = (
        ("recall", "precision") if objective.objective == "recall" else ("precision", "recall")
    )
    reaching = [row for row in rows if row[prefix + primary] >= objective.target]
    if reaching:
        best = max(
            reaching,
            key=lambda row: (row[prefix + secondary], row[other_f1], _row_thresholds(row)),
        )
        return best, True
    best = max(
        rows,
        key=lambda row: (
            row[prefix + primary],
            row[prefix + secondary],
            row[other_f1],
            _row_thresholds(row),
        ),
    )
    return best, False


def target_pixel_candidates(
    pixel_scores: np.ndarray, num_candidates: int
) -> tuple[np.ndarray, bool]:
    """Ascending float32 cutoffs for a target sweep, and whether they are every distinct score.

    A split with at most :data:`EXACT_GRID_LIMIT` distinct scores (probed on a sample first, so
    a large split is not sorted twice) sweeps all of them. A larger split starts from the F1
    sweep's log-spaced quantiles above the pooled median, their mirror below it down to the
    pooled minimum (so recall 1.0 is on the grid) and the extremes; the refinement then zooms
    into the bracket of the best cell.
    """
    flat = pixel_scores.reshape(-1)
    probe = flat[:: max(1, flat.size // _EXACT_PROBE)]
    if np.unique(probe).size <= EXACT_GRID_LIMIT:
        distinct = np.unique(flat)
        if distinct.size <= EXACT_GRID_LIMIT:
            return distinct, True
    tail_lo = max(1.0 / flat.size, 1e-12)
    tail = np.logspace(np.log10(tail_lo), np.log10(0.5), num_candidates)
    pooled = np.quantile(flat, np.concatenate((tail, 1.0 - tail)))
    values = np.concatenate((pooled, [flat.min(), flat.max()])).astype(np.float32)
    return np.unique(values), False


def gated_rows(
    pixel_scores: np.ndarray,
    gt_masks: np.ndarray,
    image_scores: np.ndarray,
    frame_labels: np.ndarray,
    gates: np.ndarray,
    cutoffs: np.ndarray,
) -> list[dict[str, Any]]:
    """One row per (gate, cutoff) with the metrics at both levels.

    A gated-out frame contributes its ground truth as misses at the pixel level and counts as
    not flagged at the frame level; an admitted frame is flagged when any of its pixels reaches
    the cutoff. Every row carries ``image_threshold`` and ``pixel_threshold`` on-point.
    """
    tp, fp = frame_confusions(pixel_scores, gt_masks, cutoffs)
    any_pixel = (tp + fp) > 0
    total_gt = float(gt_masks.sum())
    positive_frames = float(frame_labels.sum())
    rows: list[dict[str, Any]] = []
    for gate in gates:
        admitted = image_scores >= gate
        gated_tp = tp[admitted].sum(axis=0)
        gated_fp = fp[admitted].sum(axis=0)
        flagged = any_pixel & admitted[:, None]
        frame_tp = (flagged & frame_labels[:, None]).sum(axis=0)
        frame_fp = (flagged & ~frame_labels[:, None]).sum(axis=0)
        for j, cutoff in enumerate(cutoffs):
            frame = prf(float(frame_tp[j]), float(frame_fp[j]), positive_frames - frame_tp[j])
            rows.append(
                {
                    "image_threshold": float(gate),
                    "pixel_threshold": float(cutoff),
                    **prf(float(gated_tp[j]), float(gated_fp[j]), total_gt - float(gated_tp[j])),
                    "frame_precision": frame["precision"],
                    "frame_recall": frame["recall"],
                    "frame_f1": frame["f1"],
                    "frame_iou": frame["iou"],
                }
            )
    return rows


def _refine_cutoffs(
    pixel_scores: np.ndarray,
    gt_masks: np.ndarray,
    image_scores: np.ndarray,
    frame_labels: np.ndarray,
    objective: CalibrationObjective,
    best: dict[str, Any],
    cutoffs: np.ndarray,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Zoom into the bracket of the best cell until every distinct score in it was swept.

    Each round sweeps, at the best cell's gate, the distinct pixel scores between the cell's
    neighbouring cutoffs (thinned evenly to :data:`REFINE_LIMIT` when there are more), selects
    again over every row seen and stops when the bracket was swept in full or nothing moved.
    """
    distinct = np.unique(pixel_scores.reshape(-1))
    reached = False
    for _ in range(REFINE_ROUNDS):
        j = int(np.searchsorted(cutoffs, np.float32(best["pixel_threshold"])))
        low = cutoffs[j - 1] if j > 0 else -np.inf
        high = cutoffs[j + 1] if j + 1 < cutoffs.size else np.inf
        inside = distinct[(distinct > low) & (distinct < high)]
        if inside.size == 0:
            break
        thinned = inside.size > REFINE_LIMIT
        if thinned:
            inside = inside[np.linspace(0, inside.size - 1, REFINE_LIMIT).astype(np.int64)]
        gate = np.asarray([best["image_threshold"]], dtype=np.float64)
        rows = rows + gated_rows(pixel_scores, gt_masks, image_scores, frame_labels, gate, inside)
        cutoffs = np.unique(np.concatenate((cutoffs, inside)))
        previous = best
        best, reached = select_calibration_row(rows, objective)
        if not thinned or best is previous:
            break
    return best, reached


def sweep_two_stage_target(
    pixel_scores: np.ndarray,
    gt_masks: np.ndarray,
    image_scores: np.ndarray,
    frame_labels: np.ndarray,
    objective: CalibrationObjective,
    num_candidates: int,
) -> tuple[dict[str, Any], bool]:
    """The (gate, cutoff) a target objective picks for ``TwoStageBinaryDecider``.

    Gates are every distinct image score. At the frame level the cutoffs hold every frame's
    maximum pixel score beside the sampled grid, so the grid is exact in the frame flags; at
    the pixel level a sampled grid is refined around the best cell. The row carries the
    on-point thresholds and their plateau midpoints ``margin_image_threshold`` and
    ``margin_pixel_threshold``.
    """
    pixel_scores = np.ascontiguousarray(pixel_scores, dtype=np.float32)
    gates = np.unique(image_scores)
    cutoffs, exact = target_pixel_candidates(pixel_scores, num_candidates)
    if objective.level == "frames":
        frame_max = pixel_scores.reshape(pixel_scores.shape[0], -1).max(axis=1)
        cutoffs = np.unique(np.concatenate((cutoffs, frame_max)))
    rows = gated_rows(pixel_scores, gt_masks, image_scores, frame_labels, gates, cutoffs)
    best, reached = select_calibration_row(rows, objective)
    if not exact and objective.level == "pixels":
        best, reached = _refine_cutoffs(
            pixel_scores, gt_masks, image_scores, frame_labels, objective, best, cutoffs, rows
        )
    best = dict(best)
    best["margin_image_threshold"] = margin_below(
        best["image_threshold"], image_scores, float32=False
    )
    best["margin_pixel_threshold"] = margin_below(
        best["pixel_threshold"], pixel_scores, float32=True
    )
    return best, reached


def sweep_absolute_target(
    pixel_scores: np.ndarray,
    gt_masks: np.ndarray,
    frame_labels: np.ndarray,
    objective: CalibrationObjective,
    num_candidates: int,
) -> tuple[dict[str, Any], bool]:
    """The cutoff a target objective picks for ``BinaryDecider`` (every frame admitted).

    The row carries the on-point ``threshold`` and its plateau midpoint ``margin_threshold``.
    """
    pixel_scores = np.ascontiguousarray(pixel_scores, dtype=np.float32)
    admit_all = np.zeros(pixel_scores.shape[0], dtype=np.float64)
    best, reached = sweep_two_stage_target(
        pixel_scores, gt_masks, admit_all, frame_labels, objective, num_candidates
    )
    row = {
        key: value
        for key, value in best.items()
        if key not in ("image_threshold", "margin_image_threshold")
    }
    row["threshold"] = row.pop("pixel_threshold")
    row["margin_threshold"] = row.pop("margin_pixel_threshold")
    return row, reached


def quantile_target_grid(numel: int, preset_quantile: float, num_candidates: int) -> np.ndarray:
    """Ascending quantiles in [0, 1] for a target sweep: the F1 grid, its mirror and the ends."""
    upper = quantile_grid_builder(preset_quantile, num_candidates)(numel)
    tail_lo = max(1.0 / numel, 1e-12)
    lower = np.logspace(np.log10(tail_lo), np.log10(0.5), num_candidates)
    values = np.concatenate((upper, lower, [0.0, 1.0]))
    return np.unique(np.clip(values, 0.0, 1.0))


def quantile_rows(
    pixel_scores: np.ndarray,
    gt_masks: np.ndarray,
    frame_labels: np.ndarray,
    frame_quantiles: dict[float, np.ndarray],
) -> list[dict[str, Any]]:
    """One row per quantile for ``QuantileBinaryDecider``, metrics at both levels.

    Every frame flags at least its maximum, so the frame numbers are the same in every row.
    """
    total_gt = float(gt_masks.sum())
    not_gt = ~gt_masks
    positive_frames = float(frame_labels.sum())
    frame = prf(positive_frames, float(frame_labels.size - positive_frames), 0.0)
    rows: list[dict[str, Any]] = []
    for q in sorted(frame_quantiles):
        thresholds = frame_quantiles[q][:, None, None]
        flagged = pixel_scores >= thresholds
        tp = float((flagged & gt_masks).sum())
        rows.append(
            {
                "quantile": float(q),
                **prf(tp, float((flagged & not_gt).sum()), total_gt - tp),
                "frame_precision": frame["precision"],
                "frame_recall": frame["recall"],
                "frame_f1": frame["f1"],
                "frame_iou": frame["iou"],
            }
        )
    return rows


def counts_at_point(
    pixel_scores: np.ndarray,
    gt_masks: np.ndarray,
    frame_labels: np.ndarray,
    admitted: np.ndarray,
    cutoffs: np.ndarray,
) -> dict[str, float]:
    """The metrics at one operating point, counted the way ``forward`` decides.

    ``admitted`` marks the frames the gate lets through, ``cutoffs`` holds each frame's pixel
    cutoff (one value per frame, float32). Returns ``precision``, ``recall``, ``f1`` and ``iou``
    at the pixel level with the ``tp``, ``fp`` and ``fn`` counts behind them, and the same
    seven with a ``frame_`` prefix.
    """
    hits = (pixel_scores >= np.asarray(cutoffs, dtype=np.float32)[:, None, None]) & admitted[
        :, None, None
    ]
    tp = float((hits & gt_masks).sum())
    fp = float((hits & ~gt_masks).sum())
    fn = float(gt_masks.sum()) - tp
    flagged = hits.reshape(hits.shape[0], -1).any(axis=1)
    frame_tp = float((flagged & frame_labels).sum())
    frame_fp = float((flagged & ~frame_labels).sum())
    frame_fn = float(frame_labels.sum()) - frame_tp
    pixel = {**prf(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}
    frame = {**prf(frame_tp, frame_fp, frame_fn), "tp": frame_tp, "fp": frame_fp, "fn": frame_fn}
    return {**pixel, **{f"frame_{key}": value for key, value in frame.items()}}
