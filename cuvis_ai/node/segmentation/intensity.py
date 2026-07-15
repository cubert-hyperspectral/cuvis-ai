"""Intensity-threshold segmentation node.

This module provides a stateless node that reduces a hyperspectral cube to a
per-pixel intensity and thresholds it into a binary foreground/background mask.
"""

from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_core.node import Node


class IntensityThresholdSegmenter(Node):
    """Segment foreground by thresholding a per-pixel reduced intensity.

    The cube is collapsed over its channel axis to a single per-pixel intensity
    using the chosen ``reduction``, then pixels whose intensity lies inside the
    closed interval ``[low, high]`` are marked as foreground (``1``); all other
    pixels are background (``0``).

    Parameters
    ----------
    low : float
        Inclusive lower bound of the foreground intensity interval. Default: 0.0.
    high : float
        Inclusive upper bound of the foreground intensity interval. Default: 1.0.
    reduction : str
        How to collapse the channel axis to a scalar intensity. One of
        ``"mean"`` (channel mean), ``"max"`` (channel max), or ``"band"``
        (single band at ``band_index``). Default: ``"mean"``.
    band_index : int
        Channel index used when ``reduction == "band"``. Default: 0.

    Examples
    --------
    >>> seg = IntensityThresholdSegmenter(low=0.2, high=0.8, reduction="mean")
    >>> cube = torch.rand(2, 8, 8, 16)
    >>> seg.forward(cube=cube)["mask"].shape
    torch.Size([2, 8, 8])
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.MASK, NodeTag.POSTPROCESSING, NodeTag.TORCH})

    _VALID_REDUCTIONS = ("mean", "max", "band")

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        ),
    }

    OUTPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Foreground mask [B, H, W]; 1 = intensity in [low, high], else 0",
        ),
    }

    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        reduction: str = "mean",
        band_index: int = 0,
        **kwargs: Any,
    ) -> None:
        if reduction not in self._VALID_REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {self._VALID_REDUCTIONS}, got {reduction!r}"
            )
        if float(low) > float(high):
            raise ValueError(f"low must be <= high, got low={low}, high={high}")
        self.low = float(low)
        self.high = float(high)
        self.reduction = reduction
        self.band_index = int(band_index)
        super().__init__(
            low=self.low,
            high=self.high,
            reduction=self.reduction,
            band_index=self.band_index,
            **kwargs,
        )

    @torch.no_grad()
    def forward(self, cube: Tensor, **_: Any) -> dict[str, Tensor]:
        """Reduce the cube over channels and threshold it into a foreground mask.

        Parameters
        ----------
        cube : Tensor
            Input hyperspectral cube ``[B, H, W, C]``.

        Returns
        -------
        dict[str, Tensor]
            ``{"mask": Tensor [B, H, W]}`` of int32 foreground labels.
        """
        if self.reduction == "mean":
            intensity = cube.mean(dim=-1)
        elif self.reduction == "max":
            intensity = cube.amax(dim=-1)
        else:  # "band"
            intensity = cube[..., self.band_index]

        mask = ((intensity >= self.low) & (intensity <= self.high)).to(torch.int32)
        return {"mask": mask}
