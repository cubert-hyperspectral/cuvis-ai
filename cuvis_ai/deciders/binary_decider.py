from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from cuvis_ai.deciders.base_decider import BaseDecider
from cuvis_ai.pipeline.ports import PortSpec


class BinaryDecider(BaseDecider):
    """Simple decider node using a static threshold to classify data.

    Accepts logits as input, applies sigmoid transformation to convert to
    probabilities [0, 1], then applies threshold to produce binary decisions.

    Parameters
    ----------
    threshold : float
        The threshold to use for classification after sigmoid:
        result = (sigmoid(input) >= threshold)
    """

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input logits to threshold (BHWC format)",
        )
    }

    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary decision mask (BHWC format)",
        )
    }

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        self.threshold = threshold
        # Forward threshold to BaseDecider so Serializable captures it in hparams
        super().__init__(threshold=threshold, **kwargs)

    def forward(
        self,
        logits: Tensor,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Apply sigmoid and threshold-based decisioning on channels-last data.

        Args:
            logits: Tensor shaped (B, H, W, C) containing logits.

        Returns:
            Dictionary with "decisions" key containing (B, H, W, 1) decision mask.
        """

        # Apply sigmoid if needed to convert logits to probabilities
        tensor = torch.sigmoid(logits)

        # Apply threshold to get binary decisions
        decisions = tensor >= self.threshold
        return {"decisions": decisions}


class QuantileBinaryDecider(BaseDecider):
    """Quantile-based thresholding node operating on BHWC logits or scores.

    This decider computes a tensor-valued threshold per batch item using the
    requested quantile over one or more non-batch dimensions, then produces a
    binary mask where values greater than or equal to that threshold are marked
    as anomalies.

    Parameters
    ----------
    quantile : float, optional
        Quantile in the closed interval [0, 1] used for the threshold
        computation (default: 0.995).
    reduce_dims : Sequence[int] | None, optional
        Axes (relative to the input tensor) over which to compute the quantile.
        When ``None`` (default), all non-batch dimensions are reduced.
    """

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input logits or anomaly scores (BHWC format)",
        )
    }

    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary decision mask (BHWC format)",
        )
    }

    def __init__(
        self,
        quantile: float = 0.995,
        reduce_dims: Sequence[int] | None = None,
        **kwargs,
    ) -> None:
        self._validate_quantile(quantile)
        self.quantile = float(quantile)
        self.reduce_dims = (
            tuple(int(dim) for dim in reduce_dims) if reduce_dims is not None else None
        )
        # Forward init params so Serializable records them for config serialization
        super().__init__(quantile=self.quantile, reduce_dims=self.reduce_dims, **kwargs)

    def forward(self, logits: Tensor, **_: Any) -> dict[str, Tensor]:
        tensor = logits
        reduce_dims = self._resolve_reduce_dims(tensor.dim())

        if len(reduce_dims) == 1:
            threshold = torch.quantile(
                tensor,
                self.quantile,
                dim=reduce_dims[0],
                keepdim=True,
            )
        else:
            tensor_ndim = tensor.dim()
            dims_to_keep = tuple(i for i in range(tensor_ndim) if i not in reduce_dims)
            new_order = (*dims_to_keep, *reduce_dims)
            permuted = tensor.permute(new_order)
            sizes_keep = [permuted.size(i) for i in range(len(dims_to_keep))]
            flattened = permuted.reshape(*sizes_keep, -1)
            threshold_flat = torch.quantile(
                flattened,
                self.quantile,
                dim=len(dims_to_keep),
                keepdim=True,
            )
            threshold_permuted = threshold_flat.reshape(
                *sizes_keep,
                *([1] * len(reduce_dims)),
            )
            inverse_order = [0] * tensor_ndim
            for original_idx, permuted_idx in enumerate(new_order):
                inverse_order[permuted_idx] = original_idx
            threshold = threshold_permuted.permute(*inverse_order)

        decisions = (tensor >= threshold).to(torch.bool)
        return {"decisions": decisions}

    def _resolve_reduce_dims(self, tensor_ndim: int) -> tuple[int, ...]:
        """Resolve reduce_dims, handling negative indices and defaulting to non-batch dims."""
        if self.reduce_dims is None:
            # Default: reduce over all non-batch dimensions (H, W, C for BHWC)
            return tuple(range(1, tensor_ndim))

        resolved_dims: list[int] = []
        for dim in self.reduce_dims:
            adjusted = dim if dim >= 0 else tensor_ndim + dim
            resolved_dims.append(adjusted)

        return tuple(sorted(set(resolved_dims)))

    @staticmethod
    def _validate_quantile(quantile: float) -> None:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Quantile must be within [0.0, 1.0]; received quantile={quantile}")
