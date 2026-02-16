"""Anomaly detection visualization sink nodes for monitoring training progress."""

from __future__ import annotations

import matplotlib

# Use non-interactive backend to avoid GUI/threading issues in tests
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import ArtifactType, ExecutionStage
from cuvis_ai_schemas.execution import Artifact, Context
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger
from torchmetrics.functional.classification import binary_average_precision

from cuvis_ai.utils.vis_helpers import fig_to_array, tensor_to_numpy


class ImageArtifactVizBase(Node):
    """Base class for visualization nodes that produce image artifacts."""

    OUTPUT_SPECS = {
        "artifacts": PortSpec(
            dtype=list,
            shape=(),
            description="List of Artifact objects for TensorBoard",
        )
    }

    def __init__(
        self,
        max_samples: int = 4,
        log_every_n_batches: int = 1,
        execution_stages: set[ExecutionStage] | None = None,
        **kwargs,
    ) -> None:
        self.max_samples = max_samples
        self.log_every_n_batches = log_every_n_batches
        self._batch_counter = 0
        if execution_stages is None:
            execution_stages = {ExecutionStage.TRAIN, ExecutionStage.VAL, ExecutionStage.TEST}
        super().__init__(
            execution_stages=execution_stages,
            max_samples=max_samples,
            log_every_n_batches=log_every_n_batches,
            **kwargs,
        )

    def _should_log(self) -> bool:
        """Increment batch counter and return True if this batch should be logged."""
        self._batch_counter += 1
        return (self._batch_counter - 1) % self.log_every_n_batches == 0

    @staticmethod
    def _normalize_image(img: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        return np.clip(img, 0.0, 1.0).astype(np.float32)


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
        """Generate heatmap visualizations of anomaly scores.

        Creates color-mapped heatmaps of anomaly scores for visualization
        in TensorBoard. Optionally normalizes scores to [0, 1] range for
        consistent visualization across batches.

        Parameters
        ----------
        scores : Tensor
            Anomaly scores [B, H, W, 1] from detection nodes (e.g., RX, LAD).
        context : Context
            Execution context with stage, epoch, batch_idx information.

        Returns
        -------
        dict[str, list[Artifact]]
            Dictionary with "artifacts" key containing list of heatmap artifacts.
        """
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


class RGBAnomalyMask(Node):
    """Visualize anomaly detection with GT and predicted masks on RGB images.

    Similar to AnomalyMask but designed for RGB images (e.g., from band selectors).
    Creates side-by-side visualizations showing ground truth masks, predicted masks,
    and overlay comparisons on RGB images. The overlay shows:
    - Green: True Positives (correct anomaly detection)
    - Red: False Positives (false alarms)
    - Yellow: False Negatives (missed anomalies)

    Also displays IoU and other metrics. Returns a list of Artifact objects for
    logging to monitoring systems.

    Executes during validation and inference stages.

    Parameters
    ----------
    up_to : int, optional
        Maximum number of images to visualize. If None, visualizes all (default: None)

    Examples
    --------
    >>> decider = BinaryDecider(threshold=0.2)
    >>> viz_mask = RGBAnomalyMask(up_to=5)
    >>> tensorboard_node = TensorBoardMonitorNode(output_dir="./runs")
    >>> graph.connect(
    ...     (decider.decisions, viz_mask.decisions),
    ...     (data_node.mask, viz_mask.mask),
    ...     (band_selector.rgb_image, viz_mask.rgb_image),
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
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB image [B, H, W, 3] for visualization",
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
            description="List of Artifact objects with RGB visualization artifacts",
        )
    }

    def __init__(self, up_to: int | None = None, **kwargs) -> None:
        """Initialize RGBAnomalyMask visualizer.

        Parameters
        ----------
        up_to : int | None, optional
            Maximum number of images to visualize. If None, visualizes all (default: None)
        """
        self.up_to = up_to
        super().__init__(
            execution_stages={ExecutionStage.VAL, ExecutionStage.TEST, ExecutionStage.INFERENCE},
            up_to=up_to,
            **kwargs,
        )

    def _compute_metrics(self, pred: np.ndarray, gt: np.ndarray) -> dict:
        """Compute IoU, precision, recall from boolean masks."""
        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, ~gt).sum()
        fn = np.logical_and(~pred, gt).sum()
        denom = tp + fp + fn + 1e-8
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "iou": tp / denom,
            "precision": tp / (tp + fp + 1e-8),
            "recall": tp / (tp + fn + 1e-8),
        }

    def _create_overlay(self, pred: np.ndarray, gt: np.ndarray | None) -> np.ndarray:
        """Create RGBA overlay: Green=TP, Red=FP, Yellow=FN."""
        overlay = np.zeros((*pred.shape, 4))
        if gt is not None:
            tp, fp, fn = (
                np.logical_and(pred, gt),
                np.logical_and(pred, ~gt),
                np.logical_and(~pred, gt),
            )
            overlay[tp] = [0, 1, 0, 0.6]  # Green
            overlay[fp] = [1, 0, 0, 0.6]  # Red
            overlay[fn] = [1, 1, 0, 0.6]  # Yellow
        else:
            overlay[pred] = [0, 1, 1, 0.6]  # Cyan for prediction-only
        return overlay

    def _plot_with_gt(
        self,
        axes,
        rgb: np.ndarray,
        pred: np.ndarray,
        gt: np.ndarray,
        metrics: dict,
        batch_iou: float,
        batch_size: int,
        per_image_ap: float | None,
    ) -> None:
        """Plot 3 subplots: RGB, GT mask, overlay with metrics."""
        axes[0].imshow(rgb, aspect="auto")
        axes[0].set_title("RGB Input")
        axes[0].set_xlabel("Width")
        axes[0].set_ylabel("Height")

        axes[1].imshow(gt, cmap="gray", aspect="auto")
        axes[1].set_title("Ground Truth Mask")
        axes[1].set_xlabel("Width")
        axes[1].set_ylabel("Height")

        overlay = self._create_overlay(pred, gt)
        axes[2].imshow(rgb, aspect="auto")
        axes[2].imshow(overlay, aspect="auto")

        title = f"Overlay (RGB) - IoU: {metrics['iou']:.3f}"
        if per_image_ap is not None:
            title += f" | AP: {per_image_ap:.3f}"
        title += "\nGreen=TP, Red=FP, Yellow=FN"
        axes[2].set_title(title, fontsize=9)
        axes[2].set_xlabel("Width")
        axes[2].set_ylabel("Height")

    def _plot_no_gt(self, axes, rgb: np.ndarray, pred: np.ndarray) -> None:
        """Plot 2 subplots: RGB, RGB with overlay; add stats box."""
        axes[0].imshow(rgb, aspect="auto")
        axes[0].set_title("RGB Input")
        axes[0].set_xlabel("Width")
        axes[0].set_ylabel("Height")

        overlay = self._create_overlay(pred, None)
        axes[1].imshow(rgb, aspect="auto")
        axes[1].imshow(overlay, aspect="auto")
        axes[1].set_title("Prediction Overlay (RGB)\nCyan=Predicted Anomalies")
        axes[1].set_xlabel("Width")
        axes[1].set_ylabel("Height")

    def forward(
        self,
        decisions: torch.Tensor,
        rgb_image: torch.Tensor,
        mask: torch.Tensor | None = None,
        context: Context | None = None,
        scores: torch.Tensor | None = None,
    ) -> dict:
        """Create anomaly mask visualizations with GT/pred comparison on RGB images.

        Parameters
        ----------
        decisions : torch.Tensor
            Binary anomaly decisions [B, H, W, 1]
        rgb_image : torch.Tensor
            RGB image [B, H, W, 3] for visualization
        mask : torch.Tensor | None
            Ground truth anomaly mask [B, H, W, 1] (optional)
        context : Context | None
            Execution context with stage, epoch, batch_idx
        scores : torch.Tensor | None
            Optional anomaly logits/scores [B, H, W, 1]

        Returns
        -------
        dict
            Dictionary with "artifacts" key containing list of Artifact objects
        """
        if context is None:
            raise ValueError("RGBAnomalyMask.forward() requires a Context object")

        # Convert to numpy only at this point (keep on device until last moment)
        pred_mask_np: np.ndarray = tensor_to_numpy(decisions.float().squeeze(-1))  # [B, H, W]
        rgb_np: np.ndarray = tensor_to_numpy(rgb_image)  # [B, H, W, 3]

        # Normalize RGB to [0, 1]
        if rgb_np.max() > 1.0:
            rgb_np = rgb_np / 255.0
        rgb_np = np.clip(rgb_np, 0.0, 1.0)

        # Check if GT available and valid
        use_gt = (
            mask is not None and context.stage != ExecutionStage.INFERENCE and mask.any().item()
        )

        # Validate and convert GT if available
        gt_mask_np: np.ndarray | None = None
        batch_iou: float | None = None
        if use_gt:
            assert mask is not None
            gt_mask_np = tensor_to_numpy(mask.squeeze(-1))  # [B, H, W]
            unique_values = np.unique(gt_mask_np)
            if not np.all(np.isin(unique_values, [0, 1, True, False])):
                raise ValueError(f"RGBAnomalyMask expects binary masks, found: {unique_values}")
            # Compute batch IoU
            batch_pred = pred_mask_np > 0.5
            batch_gt = gt_mask_np > 0.5
            tp = np.logical_and(batch_pred, batch_gt).sum()
            batch_iou = float(
                tp
                / (
                    tp
                    + np.logical_and(batch_pred, ~batch_gt).sum()
                    + np.logical_and(~batch_pred, batch_gt).sum()
                    + 1e-8
                )
            )

        batch_size = pred_mask_np.shape[0]
        up_to_batch = min(batch_size, self.up_to or batch_size)
        artifacts = []

        # Loop through images and visualize
        for i in range(up_to_batch):
            pred = pred_mask_np[i] > 0.5
            rgb_img = rgb_np[i]
            gt = gt_mask_np[i] > 0.5 if gt_mask_np is not None else None

            # Compute metrics and AP if GT available
            metrics: dict | None = None
            per_image_ap: float | None = None
            if gt is not None:
                metrics = self._compute_metrics(pred, gt)
                if scores is not None and mask is not None:
                    raw_scores = scores[i, ..., 0]
                    probs = torch.sigmoid(raw_scores).flatten()
                    target_tensor = mask[i, ..., 0].flatten().to(dtype=torch.long)
                    if probs.numel() == target_tensor.numel():
                        per_image_ap = binary_average_precision(probs, target_tensor).item()

            # Create figure and plot
            ncols = 3 if gt is not None else 2
            fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
            if ncols == 1:
                axes = [axes]

            if gt is not None and metrics is not None and batch_iou is not None:
                self._plot_with_gt(
                    axes, rgb_img, pred, gt, metrics, batch_iou, batch_size, per_image_ap
                )
                log_msg = (
                    f"Created RGB anomaly mask ({i + 1}/{up_to_batch}): IoU={metrics['iou']:.3f}"
                )
            else:
                self._plot_no_gt(axes, rgb_img, pred)
                log_msg = f"Created RGB anomaly mask ({i + 1}/{up_to_batch}) (no GT)"

            plt.tight_layout()
            img_array = fig_to_array(fig, dpi=150)
            plt.close(fig)

            artifact = Artifact(
                name=f"rgb_anomaly_mask_img{i:02d}",
                value=img_array,
                el_id=i,
                desc=log_msg,
                type=ArtifactType.IMAGE,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
            artifacts.append(artifact)

        return {"artifacts": artifacts}


class ChannelSelectorFalseRGBViz(ImageArtifactVizBase):
    """Visualize false RGB output from channel selectors with optional mask overlay.

    Produces per-sample image artifacts:
    - ``false_rgb_sample_{b}``: Normalized false RGB image [H, W, 3]
    - ``mask_overlay_sample_{b}``: False RGB with red alpha-blend on foreground pixels (if mask provided)

    Parameters
    ----------
    mask_overlay_alpha : float, optional
        Alpha value for red mask overlay on foreground pixels (default: 0.4).
    max_samples : int, optional
        Maximum number of batch elements to visualize (default: 4).
    log_every_n_batches : int, optional
        Log every N-th batch (default: 1).
    """

    INPUT_SPECS = {
        "rgb_output": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="False RGB output from channel selector [B, H, W, 3]",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Segmentation mask [B, H, W] where >0 is foreground",
            optional=True,
        ),
    }

    def __init__(
        self,
        mask_overlay_alpha: float = 0.4,
        max_samples: int = 4,
        log_every_n_batches: int = 1,
        **kwargs,
    ) -> None:
        self.mask_overlay_alpha = mask_overlay_alpha
        super().__init__(
            max_samples=max_samples,
            log_every_n_batches=log_every_n_batches,
            mask_overlay_alpha=mask_overlay_alpha,
            **kwargs,
        )

    @staticmethod
    def _create_mask_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
        """Alpha-blend a red tint on foreground pixels.

        Parameters
        ----------
        rgb : np.ndarray
            Normalized RGB image [H, W, 3] in [0, 1].
        mask : np.ndarray
            Segmentation mask [H, W] where >0 is foreground.
        alpha : float
            Blend factor for the red overlay.

        Returns
        -------
        np.ndarray
            RGB image with red overlay on foreground, shape [H, W, 3].
        """
        overlay = rgb.copy()
        fg = mask > 0
        red_tint = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        overlay[fg] = (1.0 - alpha) * overlay[fg] + alpha * red_tint
        return np.clip(overlay, 0.0, 1.0).astype(np.float32)

    def forward(
        self,
        rgb_output: torch.Tensor,
        context: Context,
        mask: torch.Tensor | None = None,
    ) -> dict[str, list[Artifact]]:
        """Generate false RGB and mask overlay artifacts.

        Parameters
        ----------
        rgb_output : torch.Tensor
            False RGB tensor [B, H, W, 3].
        context : Context
            Execution context with stage, epoch, batch_idx.
        mask : torch.Tensor | None
            Optional segmentation mask [B, H, W].

        Returns
        -------
        dict[str, list[Artifact]]
            Dictionary with "artifacts" key containing image artifacts.
        """
        if not self._should_log():
            return {"artifacts": []}

        batch_size = min(rgb_output.shape[0], self.max_samples)
        artifacts = []

        for b in range(batch_size):
            # Normalized false RGB
            rgb_np = self._normalize_image(rgb_output[b].detach().cpu().numpy())

            artifact_rgb = Artifact(
                name=f"false_rgb_sample_{b}",
                value=rgb_np,
                el_id=b,
                desc=f"False RGB for sample {b}",
                type=ArtifactType.IMAGE,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
            artifacts.append(artifact_rgb)

            # Mask overlay (if mask provided and has foreground)
            if mask is not None:
                mask_np = mask[b].detach().cpu().numpy()
                if mask_np.any():
                    overlay = self._create_mask_overlay(rgb_np, mask_np, self.mask_overlay_alpha)
                    artifact_overlay = Artifact(
                        name=f"mask_overlay_sample_{b}",
                        value=overlay,
                        el_id=b,
                        desc=f"False RGB with mask overlay for sample {b}",
                        type=ArtifactType.IMAGE,
                        stage=context.stage,
                        epoch=context.epoch,
                        batch_idx=context.batch_idx,
                    )
                    artifacts.append(artifact_overlay)

        return {"artifacts": artifacts}


__all__ = [
    "ImageArtifactVizBase",
    "AnomalyMask",
    "RGBAnomalyMask",
    "ScoreHeatmapVisualizer",
    "ChannelSelectorFalseRGBViz",
]
