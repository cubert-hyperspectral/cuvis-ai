"""
Two-Stage Binary Decision Module.

This module provides a two-stage binary decision node: an optional image-level
gate on the mean of the top-k per-pixel scores (off until ``image_threshold`` is
set or calibrated), then a pixel-level cut with the calibrated absolute
``pixel_threshold`` or, without one, the per-frame ``quantile``.

The gate reduces false positives by blanking frames whose overall anomaly
score stays low before any pixel-level decision is made.

See Also
--------
cuvis_ai.node.deciders.binary_decider : Simple threshold-based binary decisions
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger
from torch import Tensor

from cuvis_ai_core.deciders.base_decider import BinaryDecider as BaseDecider

from . import _calibration
from ._port_inputs import call_threshold, optional_finite

_WITHHOLDING_STAGES = frozenset({ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST})


class TwoStageBinaryDecider(BaseDecider):
    """Two-stage binary decider: optional image-level gate + pixel mask.

    Stage 1 is the image gate: the mean of the top ``top_k_fraction`` per-pixel scores must
    reach ``image_threshold`` or the frame gets a blank mask. ``image_threshold=None`` (the
    default) turns the gate off, so every frame reaches stage 2; a training preset ships it
    as ``null`` and :meth:`calibrate` (the in-training calibration phase, or the
    ``calibrate-thresholds`` CLI) fills in the value fitted on the labelled validation split.

    Stage 2 uses the calibrated absolute ``pixel_threshold`` (raw score space) when one is
    set and otherwise falls back to the per-frame ``quantile`` cutoff, which flags a fixed
    fraction of every gated frame.

    At the TRAIN, VAL and TEST stages the quantile fallback only runs once the decider is
    calibrated (:attr:`is_calibrated`: both ``image_threshold`` and ``pixel_threshold``
    set); an uncalibrated decider emits no ``decisions`` there at all, so stale fallback
    numbers never reach the thresholded metrics before a calibration has run. At INFERENCE,
    or with no execution context, the quantile fallback always applies: with both
    thresholds unset this node is ``QuantileBinaryDecider`` for ``[B, H, W, 1]`` input,
    decision for decision.

    Both thresholds are also optional inputs of shape ``[1]``: a value on the port replaces
    the hparam for that call only (a value on a ``None`` ``image_threshold`` turns the gate
    on for the call), so a GUI slider needs no pipeline reload and the yaml keeps the
    default. The withholding above looks at the hparams alone.

    Beside ``decisions`` the node reports per-frame statistics named after the port they
    describe, the convention a GUI can rely on for any tunable port ``<p>``: ``<p>_range``
    ``[B, 2]`` (the slider span), ``<p>_value`` ``[B]`` (the value in effect, NaN while the
    stage is off) and ``<p>_score`` ``[B]`` (the frame statistic compared against it).
    Here: ``pixel_threshold_range`` (the finite per-pixel score span after the channel
    max), ``pixel_threshold_value`` (the stage-2 cutoff: input, hparam or the frame's
    quantile, reported for gated-out frames too), ``image_threshold_range`` (0 and the
    frame's maximum score, ordered), ``image_threshold_value`` (the gate, NaN while off)
    and ``image_threshold_score`` (the top-k mean, computed for every frame). They are
    absent exactly when ``decisions`` is.

    :meth:`calibrate` fits the F1 optimum unless the ``calibration_objective`` hparam names a
    target: ``"precision"`` or ``"recall"`` with ``calibration_target`` in [0, 1], measured at
    ``calibration_level`` ``"pixels"`` (pooled over the split) or ``"frames"`` (a frame counts
    as flagged when any of its pixels is, the evaluation's definition). The three persist in
    the pipeline yaml, so a retrain calibrates to the same target.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {NodeTag.CLASSIFICATION, NodeTag.POSTPROCESSING, NodeTag.NUMPY, NodeTag.CALIBRATION}
    )

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Anomaly scores (BHWC).",
        ),
        "image_threshold": PortSpec(
            dtype=torch.float32,
            shape=(1,),
            description="Image gate for this call [1]; replaces the `image_threshold` hparam "
            "and turns the gate on when that is null.",
            optional=True,
        ),
        "pixel_threshold": PortSpec(
            dtype=torch.float32,
            shape=(1,),
            description="Pixel cutoff for this call [1]; replaces the `pixel_threshold` hparam "
            "and the quantile fallback.",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly mask (BHWC). Absent at TRAIN, VAL and TEST while the "
            "decider holds no thresholds.",
            optional=True,
        ),
        "pixel_threshold_range": PortSpec(
            dtype=torch.float32,
            shape=(-1, 2),
            description="Per frame the minimum and maximum finite per-pixel score after the "
            "channel max [B, 2], [0, 0] without a finite score. Absent whenever `decisions` is.",
            optional=True,
        ),
        "pixel_threshold_value": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Per frame the stage-2 cutoff in effect [B]: the `pixel_threshold` "
            "input, else the hparam, else the frame's quantile; reported for gated-out frames "
            "too.",
            optional=True,
        ),
        "image_threshold_range": PortSpec(
            dtype=torch.float32,
            shape=(-1, 2),
            description="Per frame 0 and the maximum finite per-pixel score, ordered [B, 2].",
            optional=True,
        ),
        "image_threshold_value": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Per frame the image gate in effect [B]: the `image_threshold` input, "
            "else the hparam; NaN while the gate is off.",
            optional=True,
        ),
        "image_threshold_score": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Per frame the stage-1 image score [B]: the mean of the top-k per-pixel "
            "scores, gate on or off.",
            optional=True,
        ),
    }

    def __init__(
        self,
        image_threshold: float | None = None,
        top_k_fraction: float = 0.001,
        quantile: float = 0.995,
        pixel_threshold: float | None = None,
        reduce_dims: Sequence[int] | None = None,
        calibration_objective: str = "f1",
        calibration_target: float | None = None,
        calibration_level: str = "pixels",
        **kwargs,
    ) -> None:
        # Both thresholds are compared in raw score space (see forward), and raw anomaly
        # scores are unbounded - a [0, 1] cap would make calibrated values unrepresentable.
        # None switches the respective stage off; non-numbers are refused by name.
        self.image_threshold = optional_finite("image_threshold", image_threshold)
        if not 0.0 < top_k_fraction <= 1.0:
            raise ValueError("top_k_fraction must be in (0, 1]")
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be within [0, 1]")
        self.pixel_threshold = optional_finite("pixel_threshold", pixel_threshold)

        self.top_k_fraction = float(top_k_fraction)
        self.quantile = float(quantile)
        self.reduce_dims = (
            tuple(int(dim) for dim in reduce_dims) if reduce_dims is not None else None
        )
        self._calibration_objective = _calibration.calibration_objective_of(
            calibration_objective, calibration_target, calibration_level
        )
        super().__init__(
            image_threshold=self.image_threshold,
            top_k_fraction=self.top_k_fraction,
            quantile=self.quantile,
            pixel_threshold=self.pixel_threshold,
            reduce_dims=self.reduce_dims,
            calibration_objective=self._calibration_objective.objective,
            calibration_target=self._calibration_objective.target,
            calibration_level=self._calibration_objective.level,
            **kwargs,
        )
        self._withheld_stages_logged: set[ExecutionStage] = set()

    @property
    def is_calibrated(self) -> bool:
        """Whether both ``image_threshold`` and ``pixel_threshold`` are set.

        A decider still missing either value has never been calibrated (or the pipeline
        ships it uncalibrated on purpose); :meth:`forward` then withholds ``decisions`` at
        the TRAIN, VAL and TEST stages instead of falling back to the per-frame quantile.
        """
        return self.image_threshold is not None and self.pixel_threshold is not None

    def forward(
        self,
        logits: Tensor,
        context: Context | None = None,
        *,
        image_threshold: Tensor | None = None,
        pixel_threshold: Tensor | None = None,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Apply the optional image-level gate, then the pixel-level cutoff.

        Stage 1 (only while the gate is set): the image score is the mean of the top-k
        per-pixel scores (``k = ceil(top_k_fraction * pixels)``, at least 1). A frame below
        the gate gets a blank mask. With the gate off every frame goes straight to stage 2;
        the image score is still reported.

        Stage 2: the absolute pixel cutoff when set, otherwise the per-frame ``quantile``
        cutoff over the frame's own scores.

        Each threshold is the ``[1]`` input when one is given for this call, else the hparam.

        Before either stage, an uncalibrated decider (:attr:`is_calibrated` false) at the
        TRAIN, VAL or TEST stage returns no outputs at all and logs once per stage, at info
        level, that thresholds are unset; the quantile fallback survives only at INFERENCE,
        or when no context is given.

        Parameters
        ----------
        logits : Tensor
            Anomaly scores [B, H, W, C] or [B, H, W, 1].
        image_threshold : Tensor | None
            Image gate for this call, shape [1]; ``None`` uses the hparam.
        pixel_threshold : Tensor | None
            Pixel cutoff for this call, shape [1]; ``None`` uses the hparam or the quantile.
        context : Context | None
            Execution context carrying the stage; ``None`` behaves like INFERENCE.
        **_ : Any
            Additional unused keyword arguments.

        Returns
        -------
        dict[str, Tensor]
            ``decisions`` [B, H, W, 1] and the five per-frame statistics of the class
            docstring, or an empty dictionary while uncalibrated outside INFERENCE.

        Raises
        ------
        ValueError
            An input of another shape than [1] or a non-finite value, named by its port.

        Notes
        -----
        Multi-channel inputs reduce to the per-pixel max across channels before either
        comparison. The ranges are taken over finite scores only (``[0, 0]`` without one).
        Debug logging is lazy: nothing is formatted, and no tensor is synchronized for
        logging, unless the debug level is enabled.
        """
        if not self.is_calibrated and context is not None and context.stage in _WITHHOLDING_STAGES:
            if context.stage not in self._withheld_stages_logged:
                self._withheld_stages_logged.add(context.stage)
                logger.info("thresholds unset, no decisions until the calibration")
            return {}

        gate = call_threshold("image_threshold", image_threshold, self.image_threshold)
        pixel_cut = call_threshold("pixel_threshold", pixel_threshold, self.pixel_threshold)

        tensor = logits
        decisions = []
        pixel_ranges, cutoffs, image_ranges, image_scores = [], [], [], []
        for b in range(tensor.shape[0]):
            scores = tensor[b]  # [H, W, C] or [H, W]
            pixel_scores = scores.max(dim=-1)[0] if scores.dim() == 3 else scores

            # Per-frame statistics for a GUI control: finite scores only, no device sync.
            finite = torch.isfinite(pixel_scores)
            zero = torch.zeros((), dtype=pixel_scores.dtype, device=pixel_scores.device)
            low = pixel_scores.masked_fill(~finite, math.inf).amin()
            high = pixel_scores.masked_fill(~finite, -math.inf).amax()
            low = torch.where(torch.isinf(low), zero, low)
            high = torch.where(torch.isinf(high), zero, high)
            pixel_ranges.append(torch.stack((low, high)))
            image_ranges.append(torch.stack((torch.minimum(zero, high), torch.maximum(zero, high))))

            # Stage 1 statistic, computed for every frame: the gate compares it when on, a
            # GUI marks it on the slider when off. Shared with calibrate(), so a calibrated
            # gate is exact here.
            image_score = _calibration.frame_image_score(pixel_scores, self.top_k_fraction)
            image_scores.append(image_score)

            # Stage 2 cutoff, resolved before the gate decision so a gated-out frame reports
            # the cutoff it would use. An absolute threshold takes precedence: it is compared
            # in raw score space (the same space this node receives - no sigmoid), so the
            # flagged region follows the anomaly's size. The per-frame quantile fallback
            # flags a fixed fraction of every frame regardless of how much of it is
            # anomalous; over the full [H, W, C] slice it matches QuantileBinaryDecider's
            # reduction over (H, W, C).
            if pixel_cut is not None:
                threshold = torch.tensor(pixel_cut, dtype=scores.dtype, device=scores.device)
            else:
                threshold = torch.quantile(scores, self.quantile)
            cutoffs.append(threshold.reshape(()))

            # Stage 1: image-level gate, skipped when no threshold is set.
            if gate is not None:
                k = _calibration.topk_count(pixel_scores.numel(), self.top_k_fraction)
                # Lazy args (bound as defaults so each frame logs its own values): nothing
                # is formatted unless the debug level is enabled.
                if image_score < gate:
                    logger.opt(lazy=True).debug(
                        "TwoStageDecider[batch={}]: image_score={} < image_threshold={}, "
                        "returning blank mask",
                        lambda b=b: b,
                        lambda s=image_score: f"{s:.6f}",
                        lambda g=gate: f"{g:.6f}",
                    )
                    decisions.append(
                        torch.zeros(
                            (*pixel_scores.shape, 1), dtype=torch.bool, device=tensor.device
                        )
                    )
                    continue
                logger.opt(lazy=True).debug(
                    "TwoStageDecider[batch={}]: k={}, image_score={} >= image_threshold={}",
                    lambda b=b: b,
                    lambda k=k: k,
                    lambda s=image_score: f"{s:.6f}",
                    lambda g=gate: f"{g:.6f}",
                )

            decisions.append((pixel_scores >= threshold).unsqueeze(-1).to(torch.bool))

        return {
            "decisions": torch.stack(decisions, dim=0),
            "pixel_threshold_range": torch.stack(pixel_ranges).to(torch.float32),
            "pixel_threshold_value": torch.stack(cutoffs).to(torch.float32),
            "image_threshold_range": torch.stack(image_ranges).to(torch.float32),
            "image_threshold_value": torch.full(
                (tensor.shape[0],),
                math.nan if gate is None else gate,
                dtype=torch.float32,
                device=tensor.device,
            ),
            "image_threshold_score": torch.tensor(
                image_scores, dtype=torch.float32, device=tensor.device
            ),
        }

    def calibrate(
        self,
        scores: Tensor,
        targets: Tensor,
        *,
        num_candidates: int = 256,
    ) -> dict[str, Any]:
        """Refit ``image_threshold`` + ``pixel_threshold`` on labelled validation scores.

        With the default ``f1`` objective: runs the joint 2-D sweep (image gate x absolute
        pixel cutoff, raw score space, with gated-out frames contributing their full ground
        truth as misses). Sets the F1-max image gate and, at that gate, the F1-max absolute
        pixel threshold - moving stage 2 off the fixed-fraction quantile onto a cutoff that
        tracks anomaly size. The report also carries the on-point optima and the joint
        optimum, which may beat the conditional one.

        With a precision or recall objective (the ``calibration_objective``,
        ``calibration_target`` and ``calibration_level`` hparams): keeps the (gate, cutoff)
        that reaches the target with the best other metric
        (:func:`_calibration.select_calibration_row`). Gates are every distinct image score;
        at the frame level the cutoffs hold every frame's maximum pixel score, so the grid is
        exact in the frame flags; at the pixel level the sampled grid is refined around the
        best cell. A target no cell reaches lands on the closest one and the report says so.

        Either way both values are the midpoints of their plateaus
        (``_calibration.margin_below``); the pixel cutoff is exact in float32, the dtype
        ``forward`` compares in, and the gate uses the very ``frame_image_score`` statistic
        ``forward`` computes. The values go to the live attributes and to ``hparams``;
        ``pipeline.save_to_file`` then carries them in the pipeline yaml. The ``.pt`` weights
        are unchanged, so load the saved yaml rather than the preset plus weights. ``scores``
        is the decider-input tensor stacked over the split (``[N, H, W, C]``); ``targets`` the
        ground-truth mask.

        The report's numbers at the applied point, counted as ``forward`` decides: precision,
        recall, F1 and IoU at the pixel level (``pixel_*``) and at the frame level
        (``frame_*``), the ``frames`` and ``anomalous_frames`` of the split and, for a target
        objective, ``target`` and ``reached`` (1 or 0).

        Raises:
            CalibrationError: shape mismatch, non-finite scores, or a single-class split.
        """
        _, pixel, gt, frame_labels = _calibration.reduce_scores_targets(scores, targets)
        image_scores = _calibration.topk_mean_scores(pixel, self.top_k_fraction)
        objective = self._calibration_objective
        old_image, old_pixel = self.image_threshold, self.pixel_threshold
        report: dict[str, Any] = {"class": type(self).__name__}
        if objective.targeted:
            best, reached = _calibration.sweep_two_stage_target(
                pixel, gt, image_scores, frame_labels, objective, num_candidates
            )
            new_image = float(best["margin_image_threshold"])
            new_pixel = float(best["margin_pixel_threshold"])
            on_point = {
                "image_threshold": best["image_threshold"],
                "pixel_threshold": best["pixel_threshold"],
            }
            report["target"] = objective.target
            report["reached"] = 1.0 if reached else 0.0
        else:
            best_image, joint, conditional = _calibration.sweep_two_stage(
                pixel, gt, image_scores, frame_labels, num_candidates
            )
            new_image = float(best_image["margin_threshold"])
            new_pixel = float(conditional["margin_pixel_threshold"])
            on_point = {
                "image_threshold": best_image["threshold"],
                "pixel_threshold": conditional["pixel_threshold"],
            }
            report["image_f1"] = best_image["f1"]
            report["joint"] = {
                "image_threshold": joint["margin_image_threshold"],
                "pixel_threshold": joint["margin_pixel_threshold"],
                "f1": joint["f1"],
            }
        counts = _calibration.counts_at_point(
            pixel,
            gt,
            frame_labels,
            image_scores >= new_image,
            np.full(pixel.shape[0], np.float32(new_pixel)),
        )
        self.image_threshold = new_image
        self.pixel_threshold = new_pixel
        self.hparams["image_threshold"] = new_image
        self.hparams["pixel_threshold"] = new_pixel
        report.update(
            {
                "image_threshold": {"old": old_image, "new": new_image},
                "pixel_threshold": {"old": old_pixel, "new": new_pixel},
                "on_point": on_point,
                "objective": objective.objective,
                "level": objective.level,
                "pixel_f1": counts["f1"],
                "pixel_precision": counts["precision"],
                "pixel_recall": counts["recall"],
                "pixel_iou": counts["iou"],
                "frame_f1": counts["frame_f1"],
                "frame_precision": counts["frame_precision"],
                "frame_recall": counts["frame_recall"],
                "frame_iou": counts["frame_iou"],
                "frames": int(frame_labels.size),
                "anomalous_frames": int(frame_labels.sum()),
            }
        )
        return report
