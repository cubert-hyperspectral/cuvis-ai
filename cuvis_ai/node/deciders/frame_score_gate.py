"""Frame-score gate: pass an anomaly map only when a frame-level score clears a threshold.

A deploy pipeline emits a continuous heatmap; a viewer shows it on every frame, including clean
ones where the map is just low-amplitude texture. This node gates the map by the top-``topk_frac``
mean of a per-frame alarm score: a frame whose score is at or below ``threshold`` is blanked (all
zeros), and only above-threshold frames show their heatmap (or a binary mask). The boolean
``decisions`` output is the object mask of a passing frame: its pixels above ``mask_threshold``.
Calibrated on clean frames (e.g. their highest pixel), an absolute ``mask_threshold`` gives a mask
whose area follows the object, where a per-frame quantile would mark the same area on every frame.
The ``core`` output marks the confident part of that mask, the pixels above ``core_ratio`` x
``mask_threshold``: a post-processing step that cuts the mask (e.g. to the objects of a frame) can
put it back, so it never removes a clear detection; it follows ``mask_threshold`` when that is
recalibrated.

By default the alarm score is read from the map that is displayed. The optional ``alarm_scores``
input alarms on a *different* map than the one shown, e.g. alarm on one detector's map while
displaying a sharper fusion map. The node is a stateless post-processor: ``threshold`` is a
hyper-parameter tuned per session (the operating point drifts with illumination), not a fitted
buffer.

Live calibration helpers: ``log_scores`` logs the per-frame score, threshold, gate decision and the
display map's highest pixel (``pmax``), so an operator can read a session's clean band and set
``threshold`` and ``mask_threshold`` above it; ``smooth_k`` gates on a
rolling median of the last k frame scores, so a single-frame perturbation does not flick the gate
on and off on clean frames.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger
from torch import Tensor

from cuvis_ai_core.node.metric_utils import check_topk_frac, topk_mean
from cuvis_ai_core.node.node import Node

_MODES = ("heatmap", "mask")


class FrameScoreGate(Node):
    """Zero a display map unless its per-frame top-k score exceeds a threshold (heatmap or mask)."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.ANOMALY, NodeTag.POSTPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Display map [B, H, W, C] (a heatmap has C = 1): the map that is passed "
            "through or blanked, e.g. a fusion or normalized bank map.",
        ),
        "alarm_scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            optional=True,
            description="Optional separate map [B, H, W, C] whose top-k frame score drives the "
            "gate. When omitted the gate scores `scores` itself. Use it to alarm on one bank (e.g. "
            "a feature bank) while displaying another map (e.g. a two-bank fusion).",
        ),
    }
    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Same shape as `scores`: the display map when the frame's alarm score > "
            "threshold, else all zeros. With mode='mask' the passing frame is binarised (at "
            "`mask_threshold`, defaulting to `threshold`) instead of passed through.",
        ),
        "frame_score": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Per-frame alarm score [B] = mean of the top `topk_frac` pixels of the "
            "alarm map (`alarm_scores` when given, else `scores`). Raw, before smoothing.",
        ),
        "passed": PortSpec(
            dtype=torch.int32,
            shape=(-1,),
            description="Per-frame gate [B]: 1 when the (optionally smoothed) score > threshold, "
            "else 0.",
        ),
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Boolean mask, same shape as `scores`: the pixels of the display map "
            "above `mask_threshold` (default `threshold`) on passing frames, all False on the "
            "others. Viewers show a `decisions` port as a mask overlay.",
        ),
        "core": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Boolean mask, same shape as `scores`: the pixels of the display map above "
            "`core_ratio` x `mask_threshold` on passing frames (equal to `decisions` with the "
            "default ratio 1): the confident core of the mask.",
        ),
    }

    def __init__(
        self,
        threshold: float,
        topk_frac: float = 0.001,
        mode: str = "heatmap",
        mask_threshold: float | None = None,
        log_scores: bool = False,
        smooth_k: int = 1,
        core_ratio: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Create the gate.

        Parameters
        ----------
        threshold : alarm-score cutoff; a frame passes when the (smoothed) mean of its top
            ``topk_frac`` alarm-map pixels exceeds it. Set it on the running session's own clean
            frames (the operating point drifts).
        topk_frac : fraction of map pixels averaged into the per-frame score, ``0.001`` = top
            0.1 %; ``k = max(1, ceil(topk_frac * n))``, the rule of
            ``cuvis_ai_core.node.metric_utils.topk_count`` that ``TwoStageBinaryDecider`` uses too.
        mode : ``"heatmap"`` passes the display map through on a passing frame; ``"mask"`` emits
            the binary ``scores > mask_threshold`` map instead. Both blank a non-passing frame.
        mask_threshold : per-pixel cutoff of ``decisions`` and, in ``mode="mask"``, of the binary
            ``scores``; defaults to ``threshold``. Set it on the display map's own scale, e.g. to
            the highest pixel of the session's clean frames.
        log_scores : log each frame's raw score, threshold, gate decision and the display map's
            highest pixel (``pmax``) at INFO (read the server log during a session to find the clean
            band of both thresholds). Off in production.
        smooth_k : gate on a rolling median of the last ``smooth_k`` frame scores (default 1 = no
            smoothing). Runtime-only state (not serialized); needs one frame per forward (a batch
            of several frames raises); ``reset()`` forgets the history (e.g. a new recording).
        core_ratio : the ``core`` output marks the pixels above ``core_ratio`` x ``mask_threshold``
            on passing frames (default 1.0: the pixels of ``decisions``; must be >= 1, for a
            positive ``mask_threshold``). E.g. 1.3 keeps the confident part of a mark through a
            cut of the mask to the objects.
        """
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"threshold must be a number, got {threshold!r}")
        topk_frac = check_topk_frac(topk_frac)
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if mask_threshold is not None and (
            isinstance(mask_threshold, bool) or not isinstance(mask_threshold, (int, float))
        ):
            raise ValueError(f"mask_threshold must be a number or None, got {mask_threshold!r}")
        if isinstance(smooth_k, bool) or not isinstance(smooth_k, int) or smooth_k < 1:
            raise ValueError(f"smooth_k must be an integer >= 1, got {smooth_k!r}")
        if (
            isinstance(core_ratio, bool)
            or not isinstance(core_ratio, (int, float))
            or not core_ratio >= 1.0
        ):
            raise ValueError(f"core_ratio must be a number >= 1, got {core_ratio!r}")
        self.threshold = float(threshold)
        self.topk_frac = float(topk_frac)
        self.mode = str(mode)
        self.mask_threshold = None if mask_threshold is None else float(mask_threshold)
        self.log_scores = bool(log_scores)
        self.smooth_k = int(smooth_k)
        self.core_ratio = float(core_ratio)
        self._recent: list[float] = []  # rolling frame scores for smoothing (runtime only)
        super().__init__(
            threshold=self.threshold,
            topk_frac=self.topk_frac,
            mode=self.mode,
            mask_threshold=self.mask_threshold,
            log_scores=self.log_scores,
            smooth_k=self.smooth_k,
            core_ratio=self.core_ratio,
            **kwargs,
        )

    def reset(self) -> None:
        """Forget the rolling frame scores of ``smooth_k`` (e.g. a new recording)."""
        self._recent.clear()

    def forward(
        self, scores: Tensor, alarm_scores: Tensor | None = None, **_: Any
    ) -> dict[str, Tensor]:
        """Gate the display map by the per-frame top-k score of the alarm map."""
        if alarm_scores is not None and alarm_scores.shape[0] != scores.shape[0]:
            raise RuntimeError(
                f"alarm_scores batch {alarm_scores.shape[0]} != scores batch {scores.shape[0]}"
            )
        src = scores if alarm_scores is None else alarm_scores
        frame = topk_mean(src, self.topk_frac)  # [B] raw per-frame score

        if self.smooth_k > 1:
            if scores.shape[0] != 1:
                raise RuntimeError(
                    f"smooth_k={self.smooth_k} needs one frame per forward, got a batch of "
                    f"{scores.shape[0]}"
                )
            smoothed = []
            for f in frame.tolist():
                self._recent.append(float(f))
                del self._recent[: -self.smooth_k]  # keep only the last smooth_k
                ordered = sorted(self._recent)
                smoothed.append(ordered[len(ordered) // 2])  # median
            gate_score = torch.tensor(smoothed, dtype=frame.dtype, device=frame.device)
        else:
            gate_score = frame

        passed = gate_score > self.threshold  # [B] bool
        gate = passed.reshape(scores.shape[0], *([1] * (scores.ndim - 1)))
        thr = self.threshold if self.mask_threshold is None else self.mask_threshold
        hot = scores > thr  # [B, H, W, C] bool
        base = hot.to(scores.dtype) if self.mode == "mask" else scores

        if self.log_scores:
            name = getattr(self, "name", None) or type(self).__name__
            # the display map's highest pixel: what mask_threshold cuts
            peak = scores.detach().reshape(scores.shape[0], -1).amax(dim=1)
            for i in range(frame.shape[0]):
                extra = f" smoothed={float(gate_score[i]):.4f}" if self.smooth_k > 1 else ""
                logger.info(
                    f"[{name}] frame_score={float(frame[i]):.4f}{extra} "
                    f"threshold={self.threshold:.4f} passed={int(passed[i])} "
                    f"pmax={float(peak[i]):.4f}"
                )

        decisions = hot & gate
        core = decisions if self.core_ratio == 1.0 else (scores > self.core_ratio * thr) & gate
        return {
            "scores": base * gate.to(scores.dtype),
            "frame_score": frame.to(torch.float32),
            "passed": passed.to(torch.int32),
            "decisions": decisions,
            "core": core,
        }


__all__ = ["FrameScoreGate"]
