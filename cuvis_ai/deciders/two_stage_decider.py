"""
Two-Stage Binary Decision Module.

This module provides a two-stage binary decision node that first applies
an image-level anomaly gate based on top-k statistics, then applies
pixel-level quantile thresholding only for images that pass the gate.

This approach reduces false positives by filtering out images with low
overall anomaly scores before applying pixel-level decisions.

See Also
--------
cuvis_ai.deciders.binary_decider : Simple threshold-based binary decisions
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from cuvis_ai_core.deciders.base_decider import BinaryDecider as BaseDecider
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger
from torch import Tensor


class TwoStageBinaryDecider(BaseDecider):
    """Two-stage binary decider: image-level gate + pixel quantile mask."""

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
        image_threshold: float = 0.5,
        top_k_fraction: float = 0.001,
        quantile: float = 0.995,
        reduce_dims: Sequence[int] | None = None,
        **kwargs,
    ) -> None:
        if not 0.0 <= image_threshold <= 1.0:
            raise ValueError("image_threshold must be within [0, 1]")
        if not 0.0 < top_k_fraction <= 1.0:
            raise ValueError("top_k_fraction must be in (0, 1]")
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be within [0, 1]")

        self.image_threshold = float(image_threshold)
        self.top_k_fraction = float(top_k_fraction)
        self.quantile = float(quantile)
        self.reduce_dims = (
            tuple(int(dim) for dim in reduce_dims) if reduce_dims is not None else None
        )
        super().__init__(
            image_threshold=self.image_threshold,
            top_k_fraction=self.top_k_fraction,
            quantile=self.quantile,
            reduce_dims=self.reduce_dims,
            **kwargs,
        )

    def forward(self, logits: Tensor, **_: Any) -> dict[str, Tensor]:
        """Apply two-stage binary decision: image-level gate + pixel quantile.

        Stage 1: Compute image-level anomaly score from top-k pixel scores.
        If below threshold, return blank mask (no anomalies).

        Stage 2: For images passing the gate, apply pixel-level quantile
        thresholding to create binary anomaly mask.

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
        The image-level score is computed as the mean of the top-k% highest
        pixel scores. For multi-channel inputs, the max across channels is
        used for each pixel.
        """
        tensor = logits
        bsz = tensor.shape[0]

        # DEBUG: Log input tensor stats
        logger.debug(
            f"TwoStageDecider input: shape={tensor.shape}, device={tensor.device}, "
            f"dtype={tensor.dtype}, min={tensor.min().item():.6f}, "
            f"max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}"
        )

        decisions = []
        for b in range(bsz):
            scores = tensor[b]  # [H, W, C]
            # Reduce to per-pixel max for image score
            if scores.dim() == 3:
                pixel_scores = scores.max(dim=-1)[0]
            else:
                pixel_scores = scores
            flat = pixel_scores.reshape(-1)
            k = max(
                1,
                int(
                    torch.ceil(
                        torch.tensor(flat.numel() * self.top_k_fraction, dtype=torch.float32)
                    ).item()
                ),
            )
            topk_vals, _ = torch.topk(flat, k)
            image_score = topk_vals.mean().item()  # Convert to Python float for comparison

            # DEBUG: Log intermediate computation values
            logger.debug(
                f"TwoStageDecider[batch={b}]: k={k}, topk_min={topk_vals.min().item():.6f}, "
                f"topk_max={topk_vals.max().item():.6f}, image_score={image_score:.6f}"
            )

            # Stage 1: Image-level gate
            if image_score < self.image_threshold:
                # Gate failed: return blank mask
                logger.debug(
                    f"TwoStageDecider: image_score={image_score:.6f} < threshold={self.image_threshold:.6f}, "
                    f"returning blank mask"
                )
                decisions.append(
                    torch.zeros((*pixel_scores.shape, 1), dtype=torch.bool, device=tensor.device)
                )
                continue

            # Stage 2: Gate passed, apply pixel-level quantile thresholding
            logger.debug(
                f"TwoStageDecider: image_score={image_score:.6f} >= threshold={self.image_threshold:.6f}, "
                f"applying quantile thresholding (q={self.quantile})"
            )
            # Compute quantile threshold: reduce over all dimensions to get scalar per batch item
            # This matches QuantileBinaryDecider behavior: for [B, H, W, C] it reduces over (H, W, C)
            # For single batch item [H, W, C], we reduce over all dims (0, 1, 2)
            threshold = torch.quantile(scores, self.quantile)

            # Apply threshold: for multi-channel scores, take max across channels first
            if scores.dim() == 3:  # [H, W, C]
                # Take max across channels to get per-pixel score, then threshold
                pixel_scores = scores.max(dim=-1, keepdim=False)[0]  # [H, W]
                binary_map = (pixel_scores >= threshold).unsqueeze(-1).to(torch.bool)  # [H, W, 1]
            else:  # [H, W] - single channel
                binary_map = (scores >= threshold).unsqueeze(-1).to(torch.bool)  # [H, W, 1]

            decisions.append(binary_map)

        return {"decisions": torch.stack(decisions, dim=0)}
