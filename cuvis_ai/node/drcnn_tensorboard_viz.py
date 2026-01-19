"""TensorBoard visualization node for DRCNN-AdaClip training.

This node creates image artifacts for TensorBoard logging to visualize:
- Input HSI cube (false-color RGB visualization)
- Mixer output (what AdaClip sees as input)
- Ground truth anomaly masks
- AdaClip anomaly scores (as heatmap)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.ports import PortSpec
from cuvis_ai_core.utils.types import Artifact, ArtifactType, Context, ExecutionStage


class DRCNNTensorBoardViz(Node):
    """TensorBoard visualization node for DRCNN-AdaClip pipeline.

    Creates image artifacts for logging to TensorBoard:
    - Input HSI cube visualization (false-color RGB from selected channels)
    - Mixer output (3-channel RGB-like image that AdaClip sees)
    - Ground truth anomaly mask
    - AdaClip anomaly scores (as heatmap)

    Parameters
    ----------
    hsi_channels : list[int], optional
        Channel indices to use for false-color RGB visualization of HSI input
        (default: [0, 20, 40] for a simple false-color representation)
    max_samples : int, optional
        Maximum number of samples to log per batch (default: 4)
    log_every_n_batches : int, optional
        Log images every N batches to reduce TensorBoard size (default: 1, log every batch)
    """

    INPUT_SPECS = {
        "hsi_cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input HSI cube [B, H, W, C]",
        ),
        "mixer_output": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Mixer output (RGB-like) [B, H, W, 3]",
        ),
        "ground_truth_mask": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Ground truth anomaly mask [B, H, W, 1]",
        ),
        "adaclip_scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="AdaClip anomaly scores [B, H, W, 1]",
        ),
    }

    OUTPUT_SPECS = {
        "artifacts": PortSpec(
            dtype=list,
            shape=(),
            description="List of Artifact objects for TensorBoard logging",
        )
    }

    def __init__(
        self,
        hsi_channels: list[int] | None = None,
        max_samples: int = 4,
        log_every_n_batches: int = 1,
        **kwargs,
    ) -> None:
        if hsi_channels is None:
            hsi_channels = [0, 20, 40]  # Default: use channels 0, 20, 40 for false-color RGB
        self.hsi_channels = hsi_channels
        self.max_samples = max_samples
        self.log_every_n_batches = log_every_n_batches
        self._batch_counter = 0

        super().__init__(
            execution_stages={ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST},
            hsi_channels=hsi_channels,
            max_samples=max_samples,
            log_every_n_batches=log_every_n_batches,
            **kwargs,
        )

    def forward(
        self,
        hsi_cube: Tensor,
        mixer_output: Tensor,
        ground_truth_mask: Tensor,
        adaclip_scores: Tensor,
        context: Context | None = None,
        **_: Any,
    ) -> dict[str, list[Artifact]]:
        """Create image artifacts for TensorBoard logging.

        Parameters
        ----------
        hsi_cube : Tensor
            Input HSI cube [B, H, W, C]
        mixer_output : Tensor
            Mixer output (RGB-like) [B, H, W, 3]
        ground_truth_mask : Tensor
            Ground truth anomaly mask [B, H, W, 1]
        adaclip_scores : Tensor
            AdaClip anomaly scores [B, H, W, 1]
        context : Context, optional
            Execution context with stage, epoch, batch_idx info

        Returns
        -------
        dict[str, list[Artifact]]
            Dictionary with "artifacts" key containing list of Artifact objects
        """
        if context is None:
            context = Context()

        # Skip logging if not the right batch interval
        self._batch_counter += 1
        if (self._batch_counter - 1) % self.log_every_n_batches != 0:
            return {"artifacts": []}

        artifacts = []
        B = hsi_cube.shape[0]
        num_samples = min(B, self.max_samples)

        # Convert tensors to numpy for visualization
        hsi_np = hsi_cube.detach().cpu().numpy()
        mixer_np = mixer_output.detach().cpu().numpy()
        mask_np = ground_truth_mask.detach().cpu().numpy()
        scores_np = adaclip_scores.detach().cpu().numpy()

        for b in range(num_samples):
            # 1. HSI Input Visualization (false-color RGB)
            hsi_img = self._create_hsi_visualization(hsi_np[b])
            artifact = Artifact(
                name=f"hsi_input_sample_{b}",
                value=hsi_img,
                el_id=b,
                desc=f"HSI input (false-color RGB) for sample {b}",
                type=ArtifactType.IMAGE,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
            artifacts.append(artifact)

            # 2. Mixer Output (what AdaClip sees as input)
            mixer_img = self._normalize_image(mixer_np[b])  # Already [H, W, 3]
            artifact = Artifact(
                name=f"mixer_output_adaclip_input_sample_{b}",
                value=mixer_img,
                el_id=b,
                desc=f"Mixer output (AdaClip input) for sample {b}",
                type=ArtifactType.IMAGE,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
            artifacts.append(artifact)

            # 3. Ground Truth Mask
            mask_img = self._create_mask_visualization(mask_np[b])  # [H, W, 1] -> [H, W, 3]
            artifact = Artifact(
                name=f"ground_truth_mask_sample_{b}",
                value=mask_img,
                el_id=b,
                desc=f"Ground truth anomaly mask for sample {b}",
                type=ArtifactType.IMAGE,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
            artifacts.append(artifact)

            # 4. AdaClip Scores (as heatmap)
            scores_img = self._create_scores_heatmap(scores_np[b])  # [H, W, 1] -> [H, W, 3]
            artifact = Artifact(
                name=f"adaclip_scores_heatmap_sample_{b}",
                value=scores_img,
                el_id=b,
                desc=f"AdaClip anomaly scores (heatmap) for sample {b}",
                type=ArtifactType.IMAGE,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
            artifacts.append(artifact)

        return {"artifacts": artifacts}

    def _create_hsi_visualization(self, hsi: np.ndarray) -> np.ndarray:
        """Create false-color RGB visualization from HSI cube.

        Parameters
        ----------
        hsi : np.ndarray
            HSI cube [H, W, C]

        Returns
        -------
        np.ndarray
            RGB image [H, W, 3] in range [0, 1]
        """
        H, W, C = hsi.shape

        # Select channels for false-color RGB
        # Clamp indices to valid range
        channels = [min(ch_idx, C - 1) if ch_idx < C else C - 1 for ch_idx in self.hsi_channels[:3]]

        # Extract selected channels
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        for i, ch_idx in enumerate(channels):
            rgb[:, :, i] = hsi[:, :, ch_idx]

        # Normalize to [0, 1]
        rgb = self._normalize_image(rgb)
        return rgb

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range.

        Parameters
        ----------
        img : np.ndarray
            Image array [H, W, C]

        Returns
        -------
        np.ndarray
            Normalized image [H, W, C] in range [0, 1]
        """
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def _create_mask_visualization(self, mask: np.ndarray) -> np.ndarray:
        """Create RGB visualization of binary mask.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask [H, W, 1]

        Returns
        -------
        np.ndarray
            RGB image [H, W, 3] where anomalies are red
        """
        H, W = mask.shape[:2]
        mask_2d = mask.squeeze() if mask.ndim == 3 else mask

        # Create RGB visualization: anomalies in red, normal in black
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[:, :, 0] = mask_2d.astype(np.float32)  # Red channel for anomalies
        return rgb

    def _create_scores_heatmap(self, scores: np.ndarray) -> np.ndarray:
        """Create heatmap visualization of anomaly scores.

        Parameters
        ----------
        scores : np.ndarray
            Anomaly scores [H, W, 1]

        Returns
        -------
        np.ndarray
            RGB heatmap [H, W, 3] using colormap (blue=low, red=high)
        """
        H, W = scores.shape[:2]
        scores_2d = scores.squeeze() if scores.ndim == 3 else scores

        # Normalize scores to [0, 1]
        scores_norm = self._normalize_image(scores_2d[..., np.newaxis]).squeeze()

        # Create heatmap: blue (low) -> green -> yellow -> red (high)
        rgb = np.zeros((H, W, 3), dtype=np.float32)

        # Blue to red colormap
        # Low values (0): blue
        # High values (1): red
        rgb[:, :, 0] = scores_norm  # Red channel
        rgb[:, :, 2] = 1.0 - scores_norm  # Blue channel (inverse)

        return rgb


__all__ = ["DRCNNTensorBoardViz"]
