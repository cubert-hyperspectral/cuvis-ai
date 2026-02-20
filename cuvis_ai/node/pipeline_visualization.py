"""Pipeline and data visualization sink nodes for monitoring training progress."""

from __future__ import annotations

import matplotlib

# Use non-interactive backend to avoid GUI/threading issues in tests
matplotlib.use("Agg")
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import ArtifactType, ExecutionStage
from cuvis_ai_schemas.execution import Artifact, Context
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger
from torch import Tensor

from cuvis_ai.utils.vis_helpers import fig_to_array


class CubeRGBVisualizer(Node):
    """Creates false-color RGB images from hyperspectral cube using channel weights.

    Selects 3 channels with highest weights for R, G, B channels and creates
    a false-color visualization with wavelength annotations.
    """

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C]",
        ),
        "weights": PortSpec(
            dtype=torch.float32, shape=(-1,), description="Channel selection weights [C]"
        ),
        "wavelengths": PortSpec(
            dtype=np.int32, shape=(-1,), description="Wavelengths for each channel [C]"
        ),
    }

    OUTPUT_SPECS = {
        "artifacts": PortSpec(
            dtype=list, shape=(), description="List of Artifact objects with RGB visualizations"
        )
    }

    def __init__(self, name: str | None = None, up_to: int = 5) -> None:
        super().__init__(name=name, execution_stages={ExecutionStage.INFERENCE, ExecutionStage.VAL})
        self.up_to = up_to

    def forward(self, cube, weights, wavelengths, context) -> dict[str, list[Artifact]]:
        """Generate false-color RGB visualizations from hyperspectral cube.

        Selects the 3 channels with highest weights and creates RGB images
        with wavelength annotations. Also generates a bar chart showing
        channel weights with the selected channels highlighted.

        Parameters
        ----------
        cube : Tensor
            Hyperspectral cube [B, H, W, C].
        weights : Tensor
            Channel selection weights [C] indicating importance of each channel.
        wavelengths : Tensor
            Wavelengths for each channel [C] in nanometers.
        context : Context
            Execution context with stage, epoch, batch_idx information.

        Returns
        -------
        dict[str, list[Artifact]]
            Dictionary with "artifacts" key containing list of visualization artifacts.
        """
        top3_indices = torch.topk(weights, k=3).indices.cpu().numpy()
        top3_wavelengths = wavelengths[top3_indices]

        batch_size = min(cube.shape[0], self.up_to)
        artifacts = []

        for b in range(batch_size):
            rgb_channels = cube[b, :, :, top3_indices].cpu().numpy()

            rgb_img = (rgb_channels - rgb_channels.min()) / (
                rgb_channels.max() - rgb_channels.min() + 1e-8
            )

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.imshow(rgb_img)
            ax1.set_title(
                f"False RGB: R={top3_wavelengths[0]:.1f}nm, "
                f"G={top3_wavelengths[1]:.1f}nm, B={top3_wavelengths[2]:.1f}nm"
            )
            ax1.axis("off")

            ax2.bar(range(len(wavelengths)), weights.detach().cpu().numpy())
            ax2.scatter(
                top3_indices,
                weights[top3_indices].detach().cpu().numpy(),
                c="red",
                s=100,
                zorder=3,
            )
            ax2.set_xlabel("Channel Index")
            ax2.set_ylabel("Weight")
            ax2.set_title("Channel Selection Weights")
            ax2.grid(True, alpha=0.3)

            for idx in top3_indices:
                ax2.annotate(
                    f"{wavelengths[idx]:.0f}nm",
                    xy=(idx, weights[idx].item()),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

            plt.tight_layout()

            img_array = fig_to_array(fig, dpi=150)

            artifact = Artifact(
                name=f"viz_rgb_sample_{b}",
                value=img_array,
                el_id=b,
                desc=f"False RGB visualization for sample {b}",
                type=ArtifactType.IMAGE,
            )
            artifacts.append(artifact)
            plt.close(fig)

        return {"artifacts": artifacts}


class PCAVisualization(Node):
    """Visualize PCA-projected data with scatter and image plots.

    Creates visualizations for each batch element showing:
    1. Scatter plot of H*W points in 2D PC space (using first 2 PCs)
    2. Image representation of the 2D projection reshaped to [H, W, 2]

    Points in scatter plot are colored by spatial position. Returns artifacts
    for monitoring systems.

    Executes only during validation stage.

    Parameters
    ----------
    up_to : int, optional
        Maximum number of batch elements to visualize. If None, visualizes all (default: None)

    Examples
    --------
    >>> pca_viz = PCAVisualization(up_to=10)
    >>> tensorboard_node = TensorBoardMonitorNode(output_dir="./runs")
    >>> graph.connect(
    ...     (pca.projected, pca_viz.data),
    ...     (pca_viz.artifacts, tensorboard_node.artifacts),
    ... )
    """

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="PCA-projected data [B, H, W, C] (uses first 2 components)",
        ),
    }

    OUTPUT_SPECS = {
        "artifacts": PortSpec(
            dtype=list,
            shape=(),
            description="List of Artifact objects with PCA visualizations",
        )
    }

    def __init__(self, up_to: int | None = None, **kwargs) -> None:
        self.up_to = up_to

        super().__init__(execution_stages={ExecutionStage.VAL}, up_to=up_to, **kwargs)

    def forward(self, data: torch.Tensor, context: Context) -> dict:
        """Create PCA projection visualizations as Artifact objects.

        Parameters
        ----------
        data : torch.Tensor
            PCA-projected data tensor [B, H, W, C] (uses first 2 components)
        context : Context
            Execution context with stage, epoch, batch_idx

        Returns
        -------
        dict
            Dictionary with "artifacts" key containing list of Artifact objects
        """
        # Convert to numpy
        data_np = data.detach().cpu().numpy()

        # Handle input shape: [B, H, W, C]
        if data_np.ndim != 4:
            raise ValueError(f"Expected 4D input [B, H, W, C], got shape: {data_np.shape}")

        B, H, W, C = data_np.shape

        if C < 2:
            raise ValueError(f"Expected at least 2 components, got {C}")

        # Extract context information
        stage = context.stage.value
        epoch = context.epoch
        batch_idx = context.batch_idx

        # Determine how many images to visualize from this batch
        up_to_batch = B if self.up_to is None else min(B, self.up_to)

        # List to collect artifacts
        artifacts = []

        # Loop through each batch element
        for i in range(up_to_batch):
            # Get projection for this batch element: [H, W, C]
            projection = data_np[i]

            # Use only first 2 components
            projection_2d = projection[:, :, :2]  # [H, W, 2]

            # Flatten spatial dimensions for scatter plot
            projection_flat = projection_2d.reshape(-1, 2)  # [H*W, 2]

            # Create spatial position colors using 2D HSV encoding
            # x-coordinate maps to Hue (0-1)
            # y-coordinate maps to Saturation (0-1)
            # Value is constant at 1.0 for brightness
            y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

            # Normalize coordinates to [0, 1]
            x_norm = x_coords / (W - 1) if W > 1 else np.zeros_like(x_coords)
            y_norm = y_coords / (H - 1) if H > 1 else np.zeros_like(y_coords)

            # Create HSV colors: H from x, S from y, V constant
            hsv_colors = np.stack(
                [
                    x_norm.flatten(),  # Hue from x-coordinate
                    y_norm.flatten(),  # Saturation from y-coordinate
                    np.ones(H * W),  # Value constant at 1.0
                ],
                axis=-1,
            )

            # Convert HSV to RGB for matplotlib
            from matplotlib.colors import hsv_to_rgb

            rgb_colors = hsv_to_rgb(hsv_colors)

            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))

            # Subplot 1: Scatter plot colored by 2D spatial position
            axes[0].scatter(
                projection_flat[:, 0],
                projection_flat[:, 1],
                c=rgb_colors,
                alpha=0.6,
                s=20,
            )
            axes[0].set_xlabel("PC1 (1st component)")
            axes[0].set_ylabel("PC2 (2nd component)")
            axes[0].set_title(f"PCA Scatter - {stage} E{epoch} B{batch_idx} Img{i}")
            axes[0].grid(True, alpha=0.3)

            # Subplot 2: Spatial reference image
            # Create reference image showing the spatial color coding
            spatial_reference = hsv_to_rgb(
                np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=-1)
            )
            axes[1].imshow(spatial_reference, aspect="auto")
            axes[1].set_xlabel("Width (→ Hue)")
            axes[1].set_ylabel("Height (→ Saturation)")
            axes[1].set_title("Spatial Color Reference")

            # Subplot 3: Image representation
            # Normalize each channel to [0, 1] for visualization
            pc1_norm = (projection_2d[:, :, 0] - projection_2d[:, :, 0].min()) / (
                projection_2d[:, :, 0].max() - projection_2d[:, :, 0].min() + 1e-8
            )
            pc2_norm = (projection_2d[:, :, 1] - projection_2d[:, :, 1].min()) / (
                projection_2d[:, :, 1].max() - projection_2d[:, :, 1].min() + 1e-8
            )

            # Create RGB image: PC1 in red channel, PC2 in green channel, zeros in blue
            img_rgb = np.stack([pc1_norm, pc2_norm, np.zeros_like(pc1_norm)], axis=-1)

            axes[2].imshow(img_rgb, aspect="auto")
            axes[2].set_xlabel("Width")
            axes[2].set_ylabel("Height")
            axes[2].set_title("PCA Image (R=PC1, G=PC2)")

            # Add statistics text
            pc1_min = projection_2d[:, :, 0].min()
            pc1_max = projection_2d[:, :, 0].max()
            pc2_min = projection_2d[:, :, 1].min()
            pc2_max = projection_2d[:, :, 1].max()
            stats_text = (
                f"Shape: [{H}, {W}]\n"
                f"Points: {H * W}\n"
                f"PC1 range: [{pc1_min:.3f}, {pc1_max:.3f}]\n"
                f"PC2 range: [{pc2_min:.3f}, {pc2_max:.3f}]"
            )
            fig.text(
                0.98,
                0.5,
                stats_text,
                ha="left",
                va="center",
                bbox={
                    "boxstyle": "round",
                    "facecolor": "wheat",
                    "alpha": 0.5,
                },
            )

            plt.tight_layout()

            # Convert figure to numpy array (RGB format)
            img_array = fig_to_array(fig, dpi=150)

            # Create Artifact object
            artifact = Artifact(
                name=f"pca_projection_img{i:02d}",
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
                value=img_array,
                el_id=i,
                desc=f"PCA projection for {stage} epoch {epoch}, batch {batch_idx}, image {i}",
                type=ArtifactType.IMAGE,
            )
            artifacts.append(artifact)

            progress_total = self.up_to if self.up_to else B
            description = (
                f"Created PCA projection artifact ({i + 1}/{progress_total}): {artifact.name}"
            )
            logger.info(description)

            plt.close(fig)

        # Return artifacts
        return {"artifacts": artifacts}


class PipelineComparisonVisualizer(Node):
    """TensorBoard visualization node for comparing pipeline stages.

    Creates image artifacts for logging to TensorBoard:
    - Input HSI cube visualization (false-color RGB from selected channels)
    - Mixer output (3-channel RGB-like image that downstream model sees)
    - Ground truth anomaly mask
    - Anomaly scores (as heatmap)

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
            description="Anomaly scores [B, H, W, 1]",
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
            Anomaly scores [B, H, W, 1]
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

            # 2. Mixer Output (what downstream model sees as input)
            mixer_img = self._normalize_image(mixer_np[b])  # Already [H, W, 3]
            artifact = Artifact(
                name=f"mixer_output_adaclip_input_sample_{b}",
                value=mixer_img,
                el_id=b,
                desc=f"Mixer output (model input) for sample {b}",
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

            # 4. Anomaly Scores (as heatmap)
            scores_img = self._create_scores_heatmap(scores_np[b])  # [H, W, 1] -> [H, W, 3]
            artifact = Artifact(
                name=f"adaclip_scores_heatmap_sample_{b}",
                value=scores_img,
                el_id=b,
                desc=f"Anomaly scores (heatmap) for sample {b}",
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

        # Create heatmap: blue (low) -> red (high)
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[:, :, 0] = scores_norm  # Red channel
        rgb[:, :, 2] = 1.0 - scores_norm  # Blue channel (inverse)

        return rgb


__all__ = [
    "CubeRGBVisualizer",
    "PCAVisualization",
    "PipelineComparisonVisualizer",
]
