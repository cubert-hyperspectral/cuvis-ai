from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
from torch import Tensor

from .node import LabelLike, MetaLike, Node, NodeOutput


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
        treated as anomalies.
    add_channel_axis : bool, optional
        If ``True`` ensure the binary mask is returned as [..., 1] to align with
        BHWC scores (default: True).
    """

    def __init__(
        self,
        normal_class_ids: Iterable[int] = (0, 2),
        anomaly_class_ids: Iterable[int] | None = None,
        add_channel_axis: bool = True,
    ) -> None:
        self.normal_class_ids = tuple(int(c) for c in normal_class_ids)
        self.anomaly_class_ids = (
            tuple(int(c) for c in anomaly_class_ids) if anomaly_class_ids is not None else None
        )
        self.add_channel_axis = add_channel_axis
        self._target_dtype = torch.long

        super().__init__(
            normal_class_ids=self.normal_class_ids,
            anomaly_class_ids=self.anomaly_class_ids,
            add_channel_axis=add_channel_axis,
        )

    @staticmethod
    def _membership_mask(values: Tensor, class_ids: Sequence[int]) -> Tensor:
        """Return mask where ``values`` belong to ``class_ids``."""
        if len(class_ids) == 0:
            return torch.zeros_like(values, dtype=torch.bool)

        class_tensor = torch.as_tensor(class_ids, dtype=values.dtype, device=values.device)
        return (values.unsqueeze(-1) == class_tensor).any(dim=-1)

    def forward(
        self,
        x: Tensor,
        y: LabelLike = None,
        m: MetaLike = None,
        **_: Any,
    ) -> NodeOutput:
        if y is None:
            return x, y, m

        y_tensor = torch.as_tensor(y)
        y_tensor = y_tensor.to(device=x.device, dtype=torch.long)

        if y_tensor.dim() == 2:
            y_tensor = y_tensor.unsqueeze(0)
        if y_tensor.dim() == 4 and y_tensor.shape[-1] == 1:
            y_tensor = y_tensor.squeeze(-1)
        if y_tensor.dim() != 3:
            raise ValueError(
                "BinaryAnomalyLabelMapper expects labels shaped as [B, H, W] or [B, H, W, 1]. "
                f"Received shape {tuple(y_tensor.shape)}."
            )

        mask_normal = self._membership_mask(y_tensor, self.normal_class_ids)

        if self.anomaly_class_ids is None:
            mask_anomaly = ~mask_normal
        else:
            mask_anomaly = self._membership_mask(y_tensor, self.anomaly_class_ids)

        mapped = torch.zeros_like(y_tensor, dtype=self._target_dtype, device=y_tensor.device)
        mapped = torch.where(mask_anomaly, torch.ones_like(mapped), mapped)
        mapped = torch.where(mask_normal, torch.zeros_like(mapped), mapped)

        if self.add_channel_axis and mapped.dim() == 3:
            mapped = mapped.unsqueeze(-1)

        return x, mapped, m

    @property
    def input_dim(self) -> tuple[int, int, int, int]:
        return (-1, -1, -1, -1)

    @property
    def output_dim(self) -> tuple[int, int, int, int]:
        return (-1, -1, -1, -1)

    def load(self, params: dict, serial_dir: str) -> None:
        config = params.get("config", {})

        self.normal_class_ids = tuple(int(c) for c in config.get("normal_class_ids", self.normal_class_ids))
        anomaly_ids = config.get(
            "anomaly_class_ids",
            self.anomaly_class_ids if self.anomaly_class_ids is not None else None,
        )
        self.anomaly_class_ids = (
            tuple(int(c) for c in anomaly_ids) if anomaly_ids is not None else None
        )
        self.add_channel_axis = config.get("add_channel_axis", self.add_channel_axis)
        self._target_dtype = torch.long


__all__ = ["BinaryAnomalyLabelMapper"]
