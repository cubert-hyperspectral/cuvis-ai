"""Visualization sink nodes for monitoring training progress (port-based architecture)."""

from __future__ import annotations

import matplotlib

# Use non-interactive backend to avoid GUI/threading issues in tests
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torchmetrics.functional.classification import binary_average_precision

from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.utils.types import Artifact, ArtifactType, Context, ExecutionStage
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


class AnomalyMask(Node):
    """Visualize anomaly detection with GT and predicted masks.

    Creates side-by-side visualizations showing ground truth masks, predicted masks,
    and overlay comparisons on hyperspectral cube images. The overlay shows:
    - Green: True Positives (correct anomaly detection)
    - Red: False Positives (false alarms)
    - Yellow: False Negatives (missed anomalies)

    Also displays IoU and other metrics. Returns a list of Artifact objects for
    logging to monitoring systems.

    Executes during validation and inference stages.

    Parameters
    ----------
    channel : int
        Channel index to use for cube visualization (required)
    up_to : int, optional
        Maximum number of images to visualize. If None, visualizes all (default: None)

    Examples
    --------
    >>> decider = BinaryDecider(threshold=0.2)
    >>> viz_mask = AnomalyMask(channel=30, up_to=5)
    >>> tensorboard_node = TensorBoardMonitorNode(output_dir="./runs")
    >>> graph.connect(
    ...     (logit_head.logits, decider.data),
    ...     (decider.decisions, viz_mask.decisions),
    ...     (data_node.mask, viz_mask.mask),
    ...     (data_node.cube, viz_mask.cube),
    ...     (viz_mask.artifacts, tensorboard_node.artifacts),
    ... )
    """

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly decisions [B, H, W, 1]",
        ),
        "mask": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Ground truth anomaly mask [B, H, W, 1] (optional)",
            optional=True,
        ),
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Original cube [B, H, W, C] for visualization",
        ),
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Optional anomaly logits/scores [B, H, W, 1]",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "artifacts": PortSpec(
            dtype=list,
            shape=(),
            description="List of Artifact objects with heatmap visualizations",
        )
    }

    def __init__(self, channel: int, up_to: int | None = None, **kwargs) -> None:
        self.channel = channel
        self.up_to = up_to

        super().__init__(
            execution_stages={ExecutionStage.VAL, ExecutionStage.TEST, ExecutionStage.INFERENCE},
            channel=channel,
            up_to=up_to,
            **kwargs,
        )

    def forward(
        self,
        decisions: torch.Tensor,
        cube: torch.Tensor,
        context: Context,
        mask: torch.Tensor | None = None,
        scores: torch.Tensor | None = None,
    ) -> dict:
        """Create anomaly mask visualizations with GT/pred comparison.

        Parameters
        ----------
        decisions : torch.Tensor
            Binary anomaly decisions [B, H, W, 1]
        mask : torch.Tensor | None
            Ground truth anomaly mask [B, H, W, 1] (optional)
        cube : torch.Tensor
            Original cube [B, H, W, C] for visualization
        context : Context
            Execution context with stage, epoch, batch_idx

        Returns
        -------
        dict
            Dictionary with "artifacts" key containing list of Artifact objects
        """
        # Extract context information
        stage = context.stage.value
        epoch = context.epoch
        batch_idx = context.batch_idx

        # Use decisions directly (already binary)
        pred_mask = decisions.float()

        # Convert to numpy and squeeze channel dimension
        pred_mask_np = pred_mask.detach().cpu().numpy().squeeze(-1)  # [B, H, W]
        cube_np = cube.detach().cpu().numpy()  # [B, H, W, C]

        # Determine if we should use ground truth
        # Skip GT comparison if: mask not provided, inference stage, or mask is all zeros
        use_gt = (
            mask is not None and context.stage != ExecutionStage.INFERENCE and mask.any().item()
        )

        # Process ground truth mask if available
        gt_mask_np = None
        batch_iou = None
        if use_gt:
            gt_mask_np = mask.detach().cpu().numpy().squeeze(-1)  # [B, H, W]

            # Add binary mask assertion
            unique_values = np.unique(gt_mask_np)
            if not np.all(np.isin(unique_values, [0, 1, True, False])):
                raise ValueError(
                    f"AnomalyMask expects binary masks with only values {{0, 1}}. "
                    f"Found unique values: {unique_values}. "
                    f"Ensure LentilsAnomolyDataNode is configured with anomaly_class_ids "
                    f"to convert multi-class masks to binary."
                )

            # Compute batch-level IoU (matches AnomalyDetectionMetrics computation)
            batch_gt = gt_mask_np > 0.5  # [B, H, W] bool
            batch_pred = pred_mask_np > 0.5  # [B, H, W] bool
            batch_tp = np.logical_and(batch_pred, batch_gt).sum()
            batch_fp = np.logical_and(batch_pred, ~batch_gt).sum()
            batch_fn = np.logical_and(~batch_pred, batch_gt).sum()
            batch_iou = batch_tp / (batch_tp + batch_fp + batch_fn + 1e-8)

        # Determine how many images to visualize from this batch
        batch_size = pred_mask_np.shape[0]
        up_to_batch = batch_size if self.up_to is None else min(batch_size, self.up_to)

        # List to collect artifacts
        artifacts = []

        # Loop through each image in the batch up to the limit
        for i in range(up_to_batch):
            # Get predicted mask for this image
            pred = pred_mask_np[i] > 0.5  # [H, W] bool

            # Get cube channel for visualization
            cube_img = cube_np[i]  # [H, W, C]
            cube_channel = cube_img[:, :, self.channel]

            # Normalize cube channel to [0, 1] for display
            cube_norm = (cube_channel - cube_channel.min()) / (
                cube_channel.max() - cube_channel.min() + 1e-8
            )

            if use_gt:
                # Mode A: Full comparison with ground truth
                assert gt_mask_np is not None, "gt_mask_np should not be None when use_gt is True"
                gt = gt_mask_np[i] > 0.5  # [H, W] bool

                # Compute confusion matrix
                tp = np.logical_and(pred, gt)  # True Positives
                fp = np.logical_and(pred, ~gt)  # False Positives
                fn = np.logical_and(~pred, gt)  # False Negatives
                # Compute metrics
                tp_count = tp.sum()
                fp_count = fp.sum()
                fn_count = fn.sum()

                precision = tp_count / (tp_count + fp_count + 1e-8)
                recall = tp_count / (tp_count + fn_count + 1e-8)
                iou = tp_count / (tp_count + fp_count + fn_count + 1e-8)

                # Create figure with 3 subplots
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # Subplot 1: Ground truth mask
                axes[0].imshow(gt, cmap="gray", aspect="auto")
                axes[0].set_title("Ground Truth Mask")
                axes[0].set_xlabel("Width")
                axes[0].set_ylabel("Height")

                # Subplot 2: Cube with TP/FP/FN overlay
                per_image_ap = None
                if scores is not None:
                    raw_scores = scores[i, ..., 0]
                    probs = torch.sigmoid(raw_scores).flatten()
                    target_tensor = mask[i, ..., 0].flatten().to(dtype=torch.long)
                    if probs.numel() == target_tensor.numel():
                        per_image_ap = binary_average_precision(probs, target_tensor).item()

                axes[1].imshow(cube_norm, cmap="gray", aspect="auto")

                # Create color overlay
                overlay = np.zeros((*gt.shape, 4))
                overlay[tp] = [0, 1, 0, 0.6]  # Green: True Positives
                overlay[fp] = [1, 0, 0, 0.6]  # Red: False Positives
                overlay[fn] = [1, 1, 0, 0.6]  # Yellow: False Negatives
                # TN pixels remain transparent (no overlay)

                overlay_title = f"Overlay (Channel {self.channel}) - IoU: {iou:.3f}"
                if per_image_ap is not None:
                    overlay_title += f" | AP: {per_image_ap:.3f}"
                overlay_title += "\nGreen=TP, Red=FP, Yellow=FN"

                axes[1].imshow(overlay, aspect="auto")
                axes[1].set_title(overlay_title)
                axes[1].set_xlabel("Width")
                axes[1].set_ylabel("Height")

                # Subplot 3: Predicted mask with metrics in title
                axes[2].imshow(pred, cmap="gray", aspect="auto")

                # Add metrics as title (smaller font)
                metrics_title = (
                    f"Predicted Mask\nIoU: {iou:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}"
                )
                if per_image_ap is not None:
                    metrics_title += f" | AP: {per_image_ap:.4f}"
                metrics_title += f"\nBatch IoU: {batch_iou:.4f} (all {batch_size} imgs) | Ch: {self.channel}/{cube_img.shape[2]}"
                axes[2].set_title(metrics_title, fontsize=9)
                axes[2].set_xlabel("Width")
                axes[2].set_ylabel("Height")

                log_msg = f"Created anomaly mask artifact ({i + 1}/{up_to_batch}): IoU: {iou:.3f}"
            else:
                # Mode B: Prediction-only visualization (no ground truth)
                # Create figure with 2 subplots
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                # Subplot 1: Cube with predicted overlay
                axes[0].imshow(cube_norm, cmap="gray", aspect="auto")

                # Create prediction overlay (cyan for predicted anomalies)
                overlay = np.zeros((*pred.shape, 4))
                overlay[pred] = [0, 1, 1, 0.6]  # Cyan: Predicted anomalies

                axes[0].imshow(overlay, aspect="auto")
                axes[0].set_title(
                    f"Prediction Overlay (Channel {self.channel})\nCyan=Predicted Anomalies"
                )
                axes[0].set_xlabel("Width")
                axes[0].set_ylabel("Height")

                # Subplot 2: Predicted mask
                axes[1].imshow(pred, cmap="gray", aspect="auto")
                axes[1].set_title("Predicted Mask")
                axes[1].set_xlabel("Width")
                axes[1].set_ylabel("Height")

                # Add statistics as text
                pred_pixels = pred.sum()
                total_pixels = pred.size
                pred_ratio = pred_pixels / total_pixels

                stats_text = (
                    f"Prediction Stats:\n"
                    f"Anomaly pixels: {pred_pixels}\n"
                    f"Total pixels: {total_pixels}\n"
                    f"Anomaly ratio: {pred_ratio:.4f}\n"
                    f"\n"
                    f"Channel: {self.channel}/{cube_img.shape[2]}\n"
                    f"\n"
                    f"Mode: Inference/No GT"
                )

                fig.text(
                    0.98,
                    0.5,
                    stats_text,
                    ha="left",
                    va="center",
                    bbox={
                        "boxstyle": "round",
                        "facecolor": "lightblue",
                        "alpha": 0.5,
                    },
                    fontfamily="monospace",
                )

                log_msg = (
                    f"Created anomaly mask artifact ({i + 1}/{up_to_batch}): prediction-only mode"
                )

            # Add main title with epoch/batch info
            fig.suptitle(
                f"Anomaly Mask Visualization - {stage} E{epoch} B{batch_idx} Img{i}",
                fontsize=14,
                fontweight="bold",
            )

            plt.tight_layout()

            # Convert figure to numpy array (RGB format)
            img_array = fig_to_array(fig, dpi=150)

            # Create Artifact object
            artifact = Artifact(
                name=f"anomaly_mask_img{i:02d}",
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
                value=img_array,
                el_id=i,
                desc=f"Anomaly mask for {stage} epoch {epoch}, batch {batch_idx}, image {i}",
                type=ArtifactType.IMAGE,
            )
            artifacts.append(artifact)

            logger.info(log_msg)

            plt.close(fig)

        # Return artifacts
        return {"artifacts": artifacts}


class ScoreHeatmapVisualizer(Node):
    """Log LAD/RX score heatmaps as TensorBoard artifacts."""

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Anomaly scores [B, H, W, 1]",
        )
    }

    OUTPUT_SPECS = {
        "artifacts": PortSpec(
            dtype=list,
            shape=(),
            description="List of Artifact objects with score heatmaps",
        )
    }

    def __init__(
        self,
        normalize_scores: bool = True,
        cmap: str = "inferno",
        up_to: int | None = 5,
        **kwargs,
    ) -> None:
        self.normalize_scores = normalize_scores
        self.cmap = cmap
        self.up_to = up_to
        super().__init__(
            execution_stages={ExecutionStage.VAL, ExecutionStage.TEST, ExecutionStage.INFERENCE},
            normalize_scores=normalize_scores,
            cmap=cmap,
            up_to=up_to,
            **kwargs,
        )

    def forward(self, scores: torch.Tensor, context: Context) -> dict[str, list[Artifact]]:
        artifacts: list[Artifact] = []
        batch_limit = scores.shape[0] if self.up_to is None else min(scores.shape[0], self.up_to)

        for idx in range(batch_limit):
            score_map = scores[idx, ..., 0].detach().cpu().numpy()

            if self.normalize_scores:
                min_v = float(score_map.min())
                max_v = float(score_map.max())
                if max_v - min_v > 1e-9:
                    score_map = (score_map - min_v) / (max_v - min_v)
                else:
                    score_map = np.zeros_like(score_map)

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            im = ax.imshow(score_map, cmap=self.cmap)
            ax.set_title(f"Score Heatmap #{idx}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            img_array = fig_to_array(fig, dpi=150)
            plt.close(fig)

            artifact = Artifact(
                name=f"score_heatmap_img{idx:02d}",
                value=img_array,
                el_id=idx,
                desc="Anomaly score heatmap",
                type=ArtifactType.IMAGE,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
            artifacts.append(artifact)

        return {"artifacts": artifacts}


__all__ = [
    "PCAVisualization",
    "AnomalyMask",
    "ScoreHeatmapVisualizer",
]
