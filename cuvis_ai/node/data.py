"""Data loading nodes for hyperspectral anomaly detection pipelines.

This module provides specialized data nodes that convert multi-class segmentation
datasets into binary anomaly detection tasks. Data nodes handle type conversions,
label mapping, and format transformations required for pipeline processing.
"""

from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node.labels import BinaryAnomalyLabelMapper


class LentilsAnomalyDataNode(Node):
    """Data node for Lentils anomaly detection dataset with binary label mapping.

    Converts multi-class Lentils segmentation data to binary anomaly detection format.
    Maps specified class IDs to normal (0) or anomaly (1) labels, and handles type
    conversions from uint16 to float32 for hyperspectral cubes.

    Parameters
    ----------
    normal_class_ids : list[int]
        List of class IDs to treat as normal background (e.g., [0, 1] for unlabeled
        and black lentils)
    anomaly_class_ids : list[int] | None, optional
        List of class IDs to treat as anomalies. If None, all classes not in
        normal_class_ids are treated as anomalies (default: None)
    **kwargs : dict
        Additional arguments passed to Node base class

    Attributes
    ----------
    _binary_mapper : BinaryAnomalyLabelMapper
        Internal label mapper for converting multi-class to binary masks

    Examples
    --------
    >>> from cuvis_ai.node.data import LentilsAnomalyDataNode
    >>> from cuvis_ai_core.data.datasets import SingleCu3sDataModule
    >>>
    >>> # Create datamodule for Lentils dataset
    >>> datamodule = SingleCu3sDataModule(
    ...     data_dir="data/lentils",
    ...     batch_size=4,
    ... )
    >>>
    >>> # Create data node with normal class specification
    >>> data_node = LentilsAnomalyDataNode(
    ...     normal_class_ids=[0, 1],  # Unlabeled and black lentils are normal
    ... )
    >>>
    >>> # Use in pipeline
    >>> pipeline.add_node(data_node)
    >>> pipeline.connect(
    ...     (data_node.cube, normalizer.data),
    ...     (data_node.mask, metrics.targets),
    ... )

    See Also
    --------
    BinaryAnomalyLabelMapper : Label mapping utility used internally
    SingleCu3sDataModule : DataModule for loading CU3S hyperspectral data
    docs/tutorials/rx-statistical.md : Complete example with LentilsAnomalyDataNode

    Notes
    -----
    The node performs the following transformations:
    - Converts hyperspectral cube from uint16 to float32
    - Maps multi-class mask [B, H, W] to binary mask [B, H, W, 1]
    - Extracts wavelengths from first batch element (assumes consistent wavelengths)
    """

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.uint16,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Multi-class segmentation mask [B, H, W]",
            optional=True,
        ),
        "wavelengths": PortSpec(
            dtype=torch.int32, shape=(-1, -1), description="Wavelengths for each channel"
        ),
    }
    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        ),
        "mask": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly mask (0=normal, 1=anomaly) [B, H, W, 1]",
            optional=True,
        ),
        # wavelength must be a required input port and not optional
        "wavelengths": PortSpec(
            dtype=np.int32, shape=(-1,), description="Wavelengths for each channel", optional=True
        ),
    }

    def __init__(
        self, normal_class_ids: list[int], anomaly_class_ids: list[int] | None = None, **kwargs
    ) -> None:
        super().__init__(
            normal_class_ids=normal_class_ids, anomaly_class_ids=anomaly_class_ids, **kwargs
        )

        self._binary_mapper = BinaryAnomalyLabelMapper(  # could have be used as a node as well
            normal_class_ids=normal_class_ids,
            anomaly_class_ids=anomaly_class_ids,
        )

    def forward(
        self,
        cube: torch.Tensor,
        mask: torch.Tensor | None = None,
        wavelengths: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """Process hyperspectral cube and convert labels to binary anomaly format.

        Parameters
        ----------
        cube : torch.Tensor
            Input hyperspectral cube, shape (B, H, W, C), dtype uint16
        mask : torch.Tensor | None, optional
            Multi-class segmentation mask, shape (B, H, W), dtype int32.
            If None, only cube is returned (default: None)
        wavelengths : torch.Tensor | None, optional
            Wavelengths for each channel, shape (B, C), dtype int32.
            If None, wavelengths are not included in output (default: None)

        Returns
        -------
        dict[str, torch.Tensor | np.ndarray]
            Dictionary containing:
            - "cube" : torch.Tensor
                Converted hyperspectral cube, shape (B, H, W, C), dtype float32
            - "mask" : torch.Tensor (optional)
                Binary anomaly mask, shape (B, H, W, 1), dtype bool.
                Only included if input mask is provided.
            - "wavelengths" : np.ndarray (optional)
                Wavelength array, shape (C,), dtype int32.
                Only included if input wavelengths are provided.
        """
        result: dict[str, torch.Tensor | np.ndarray] = {"cube": cube.to(torch.float32)}

        # wavelengths passthrough, could check that in all batch elements the same wavelengths are used
        # input B x C -> output C
        if wavelengths is not None:
            result["wavelengths"] = wavelengths[0].cpu().numpy()

        if mask is not None:
            # Add channel dimension for mapper: BHW -> BHWC
            mask_4d = mask.unsqueeze(-1)

            # Always apply binary mapper
            mapped = self._binary_mapper.forward(
                cube=cube,
                mask=mask_4d,
                **_,  # Pass through additional kwargs
            )
            result["mask"] = mapped["mask"]  # Already BHWC bool

        return result
