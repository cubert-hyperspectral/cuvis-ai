from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor

from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.ports import PortSpec


class BinaryAnomalyLabelMapper(Node):
    """Convert multi-class segmentation masks to binary anomaly targets.

    Masks are remapped to torch.long tensors with 0 representing normal pixels and
    1 indicating anomalies.

    Parameters
    ----------
    normal_class_ids : Iterable[int], optional
        Class IDs that should be considered normal (default: (0, 2)).
    anomaly_class_ids : Iterable[int] | None, optional
        Explicit anomaly IDs. When ``None`` all IDs not in ``normal_class_ids`` are
        treated as anomalies. When provided, only these IDs are treated as anomalies
        and all others (including those not in normal_class_ids) are treated as normal.

    """

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Features/scores to pass through [B, H, W, C]",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1, 1),
            description="Multi-class segmentation masks [B, H, W, 1]",
        ),
    }

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Pass-through features/scores [B, H, W, C]",
        ),
        "mask": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly labels (0=normal, 1=anomaly) [B, H, W, 1]",
        ),
    }

    def __init__(
        self,
        normal_class_ids: Iterable[int],
        anomaly_class_ids: Iterable[int] | None = None,
        **kwargs,
    ) -> None:
        self.normal_class_ids = tuple(int(c) for c in normal_class_ids)
        self.anomaly_class_ids = (
            tuple(int(c) for c in anomaly_class_ids) if anomaly_class_ids is not None else None
        )

        # Validate that there are no overlaps between normal and anomaly class IDs
        if self.anomaly_class_ids is not None:
            overlap = set(self.normal_class_ids) & set(self.anomaly_class_ids)
            if overlap:
                raise ValueError(
                    f"Overlap detected between normal_class_ids and anomaly_class_ids: {overlap}. "
                    "Class IDs cannot be both normal and anomaly."
                )

            # Check for gaps in coverage and issue warning
            all_specified_ids = set(self.normal_class_ids) | set(self.anomaly_class_ids)
            max_id = max(all_specified_ids) if all_specified_ids else 0

            # Find gaps (missing class IDs)
            expected_ids = set(range(max_id + 1))
            gaps = expected_ids - all_specified_ids

            if gaps:
                warnings.warn(
                    f"Gap detected in class ID coverage. The following class IDs are not specified "
                    f"in either normal_class_ids or anomaly_class_ids: {gaps}. "
                    f"These will be treated as normal classes. To specify all classes explicitly, "
                    f"include them in normal_class_ids or anomaly_class_ids.",
                    UserWarning,
                    stacklevel=2,
                )
                # Add gaps to normal_class_ids as requested
                self.normal_class_ids = tuple(sorted(set(self.normal_class_ids) | gaps))

        self._target_dtype = torch.long

        super().__init__(
            normal_class_ids=self.normal_class_ids,
            anomaly_class_ids=self.anomaly_class_ids,
            **kwargs,
        )

    @staticmethod
    def _membership_mask(values: Tensor, class_ids: Sequence[int]) -> Tensor:
        """Return mask where ``values`` belong to ``class_ids``."""
        if len(class_ids) == 0:
            return torch.zeros_like(values, dtype=torch.bool)

        class_tensor = torch.as_tensor(class_ids, dtype=values.dtype, device=values.device)
        return (values.unsqueeze(-1) == class_tensor).any(dim=-1)

    def forward(self, cube: Tensor, mask: Tensor, **_: Any) -> dict[str, Tensor]:
        """Map multi-class labels to binary anomaly labels.

        Parameters
        ----------
        cube : Tensor
            Features/scores to pass through [B, H, W, C]
        mask : Tensor
            Multi-class segmentation masks [B, H, W, 1]

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "cube" (pass-through) and "mask" (binary bool) keys
        """
        if self.anomaly_class_ids is not None:
            # Explicit anomaly class IDs: only these are anomalies, rest are normal
            mask_anomaly = self._membership_mask(mask, self.anomaly_class_ids)
        else:
            # Original behavior: normal_class_ids are normal, everything else is anomaly
            mask_normal = self._membership_mask(mask, self.normal_class_ids)
            mask_anomaly = ~mask_normal

        mapped = torch.zeros_like(mask, dtype=self._target_dtype, device=mask.device)
        mapped = torch.where(mask_anomaly, torch.ones_like(mapped), mapped)

        # Convert to bool for smaller tensor size
        mapped = mapped.bool()

        return {"cube": cube, "mask": mapped}


__all__ = ["BinaryAnomalyLabelMapper"]
