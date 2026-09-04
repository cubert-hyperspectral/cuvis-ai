"""
Two-Stage Binary Decision Module.

This module provides a two-stage binary decision node that first applies
an image-level anomaly gate based on top-k statistics, then applies
pixel-level quantile thresholding only for images that pass the gate.

This approach reduces false positives by filtering out images with low
overall anomaly scores before applying pixel-level decisions.

See Also
--------
cuvis_ai.node.deciders.binary_decider : Simple threshold-based binary decisions
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger
from torch import Tensor

from cuvis_ai_core.deciders.base_decider import BinaryDecider as BaseDecider

from . import _calibration


def _optional_finite(name: str, value: Any) -> float | None:
    """Return ``value`` as a float, pass ``None`` through, refuse anything else by name.

    A pipeline yaml or the CuvisNEXT picker can hand a decider a string or a bool; a
    silent coercion would gate at a number nobody chose, so the error names the hparam.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number or None, got {value!r}")
    return float(value)


class TwoStageBinaryDecider(BaseDecider):
    """Two-stage binary decider: optional image-level gate + pixel mask.

    Stage 1 is the image gate: the mean of the top ``top_k_fraction`` per-pixel scores must
    reach ``image_threshold`` or the frame gets a blank mask. ``image_threshold=None`` (the
    default) turns the gate off, so every frame reaches stage 2; a training preset ships it
    as ``null`` and :meth:`calibrate` (the in-training calibration phase, or the
    ``calibrate-thresholds`` CLI) fills in the value fitted on the labelled validation split.

    Stage 2 uses the calibrated absolute ``pixel_threshold`` (raw score space) when one is
    set and otherwise falls back to the per-frame ``quantile`` cutoff, which flags a fixed
    fraction of every gated frame. With both ``image_threshold`` and ``pixel_threshold``
    unset this node is ``QuantileBinaryDecider`` for ``[B, H, W, 1]`` input, decision for
    decision.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.CLASSIFICATION, NodeTag.POSTPROCESSING, NodeTag.NUMPY})

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Anomaly scores (BHWC).",
        )
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly mask (BHWC).",
        )
    }

    def __init__(
        self,
        image_threshold: float | None = None,
        top_k_fraction: float = 0.001,
        quantile: float = 0.995,
        pixel_threshold: float | None = None,
        reduce_dims: Sequence[int] | None = None,
        **kwargs,
    ) -> None:
        # Both thresholds are compared in raw score space (see forward), and raw anomaly
        # scores are unbounded - a [0, 1] cap would make calibrated values unrepresentable.
        # None switches the respective stage off; non-numbers are refused by name.
        self.image_threshold = _optional_finite("image_threshold", image_threshold)
        if not 0.0 < top_k_fraction <= 1.0:
            raise ValueError("top_k_fraction must be in (0, 1]")
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be within [0, 1]")
        self.pixel_threshold = _optional_finite("pixel_threshold", pixel_threshold)

        self.top_k_fraction = float(top_k_fraction)
        self.quantile = float(quantile)
        self.reduce_dims = (
            tuple(int(dim) for dim in reduce_dims) if reduce_dims is not None else None
        )
        super().__init__(
            image_threshold=self.image_threshold,
            top_k_fraction=self.top_k_fraction,
            quantile=self.quantile,
            pixel_threshold=self.pixel_threshold,
            reduce_dims=self.reduce_dims,
            **kwargs,
        )

    def forward(self, logits: Tensor, **_: Any) -> dict[str, Tensor]:
        """Apply the optional image-level gate, then the pixel-level cutoff.

        Stage 1 (only when ``image_threshold`` is set): the image score is the mean of
        the top-k per-pixel scores (``k = ceil(top_k_fraction * pixels)``, at least 1).
        A frame below the threshold gets a blank mask. With ``image_threshold=None``
        every frame goes straight to stage 2 and no top-k statistics are computed.

        Stage 2: the calibrated absolute ``pixel_threshold`` when set, otherwise the
        per-frame ``quantile`` cutoff over the frame's own scores.

        Parameters
        ----------
        logits : Tensor
            Anomaly scores [B, H, W, C] or [B, H, W, 1].
        **_ : Any
            Additional unused keyword arguments.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "decisions" key containing binary masks [B, H, W, 1].

        Notes
        -----
        Multi-channel inputs reduce to the per-pixel max across channels before either
        comparison. Debug logging is lazy: nothing is formatted, and no tensor is
        synchronized for logging, unless the debug level is enabled.
        """
        tensor = logits
        decisions = []
        for b in range(tensor.shape[0]):
            scores = tensor[b]  # [H, W, C] or [H, W]
            pixel_scores = scores.max(dim=-1)[0] if scores.dim() == 3 else scores

            # Stage 1: image-level gate, skipped entirely when no threshold is set. The
            # statistic is shared with calibrate(), so a calibrated gate is exact here.
            if self.image_threshold is not None:
                k = _calibration.topk_count(pixel_scores.numel(), self.top_k_fraction)
                image_score = _calibration.frame_image_score(pixel_scores, self.top_k_fraction)
                # Lazy args (bound as defaults so each frame logs its own values): nothing
                # is formatted unless the debug level is enabled.
                if image_score < self.image_threshold:
                    logger.opt(lazy=True).debug(
                        "TwoStageDecider[batch={}]: image_score={} < image_threshold={}, "
                        "returning blank mask",
                        lambda b=b: b,
                        lambda s=image_score: f"{s:.6f}",
                        lambda: f"{self.image_threshold:.6f}",
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
                    lambda: f"{self.image_threshold:.6f}",
                )

            # Stage 2: pixel-level cutoff. A calibrated absolute threshold takes precedence:
            # it is compared in raw score space (the same space this node receives - no
            # sigmoid), so the flagged region follows the anomaly's size. The per-frame
            # quantile fallback flags a fixed fraction of every frame regardless of how much
            # of it is anomalous; over the full [H, W, C] slice it matches
            # QuantileBinaryDecider's reduction over (H, W, C).
            if self.pixel_threshold is not None:
                threshold = torch.tensor(
                    self.pixel_threshold, dtype=scores.dtype, device=scores.device
                )
            else:
                threshold = torch.quantile(scores, self.quantile)

            decisions.append((pixel_scores >= threshold).unsqueeze(-1).to(torch.bool))

        return {"decisions": torch.stack(decisions, dim=0)}

    def calibrate(
        self, scores: Tensor, targets: Tensor, *, num_candidates: int = 256
    ) -> dict[str, Any]:
        """Refit ``image_threshold`` + ``pixel_threshold`` to F1-max on labelled val scores.

        Runs the joint 2-D sweep (image gate x absolute pixel cutoff, raw score space, with
        gated-out frames contributing their full ground truth as misses). Sets the F1-max
        image gate and, at that gate, the F1-max absolute pixel threshold - moving stage 2
        off the fixed-fraction quantile onto a cutoff that tracks anomaly size. Both values
        are the midpoints of their F1 plateaus (``_calibration.margin_below``); the pixel
        cutoff is exact in float32, the dtype ``forward`` compares in, and the gate uses the
        very ``frame_image_score`` statistic ``forward`` computes. The values go to the live
        attributes and to ``hparams``; ``pipeline.save_to_file`` then carries them in the
        pipeline yaml. The ``.pt`` weights are unchanged, so load the saved yaml rather than
        the preset plus weights. ``scores`` is the decider-input tensor stacked over the split
        (``[N, H, W, C]``); ``targets`` the ground-truth mask. The report also carries the
        on-point optima and the joint optimum, which may beat the conditional one.

        Raises:
            CalibrationError: shape mismatch, non-finite scores, or a single-class split.
        """
        _, pixel, gt, frame_labels = _calibration.reduce_scores_targets(scores, targets)
        image_scores = _calibration.topk_mean_scores(pixel, self.top_k_fraction)
        best_image, joint, conditional = _calibration.sweep_two_stage(
            pixel, gt, image_scores, frame_labels, num_candidates
        )
        old_image, old_pixel = self.image_threshold, self.pixel_threshold
        new_image = float(best_image["margin_threshold"])
        new_pixel = float(conditional["margin_pixel_threshold"])
        self.image_threshold = new_image
        self.pixel_threshold = new_pixel
        self.hparams["image_threshold"] = new_image
        self.hparams["pixel_threshold"] = new_pixel
        return {
            "class": type(self).__name__,
            "image_threshold": {"old": old_image, "new": new_image},
            "pixel_threshold": {"old": old_pixel, "new": new_pixel},
            "on_point": {
                "image_threshold": best_image["threshold"],
                "pixel_threshold": conditional["pixel_threshold"],
            },
            "image_f1": best_image["f1"],
            "pixel_f1": conditional["f1"],
            "pixel_precision": conditional["precision"],
            "pixel_recall": conditional["recall"],
            "pixel_iou": conditional["iou"],
            "joint": {
                "image_threshold": joint["margin_image_threshold"],
                "pixel_threshold": joint["margin_pixel_threshold"],
                "f1": joint["f1"],
            },
        }
