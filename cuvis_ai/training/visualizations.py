"""Visualization leaf nodes for monitoring training progress."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from cuvis_ai.training.leaf_nodes import VisualizationNode


class PCAVisualization(VisualizationNode):
    """Visualize high-dimensional data projected onto principal components.
    
    Creates 2D or 3D scatter plots showing data projected onto the first
    principal components, with explained variance percentages annotated.
    
    Parameters
    ----------
    n_components : int
        Number of principal components (2 or 3 for visualization)
    log_every_n_epochs : int
        Generate visualizations every N epochs
    max_samples : int
        Maximum number of samples to visualize (for performance)
    weight : float
        Weighting factor (unused for visualizations)
        
    Examples
    --------
    >>> pca_viz = PCAVisualization(n_components=2, log_every_n_epochs=5)
    >>> graph.add_leaf_node(pca_viz, parent=normalizer_node)
    """

    def __init__(
        self,
        n_components: int = 2,
        log_every_n_epochs: int = 1,
        max_samples: int = 1000,
        weight: float = 1.0
    ):
        super().__init__(log_every_n_epochs, weight)

        if n_components not in (2, 3):
            raise ValueError("n_components must be 2 or 3 for visualization")

        self.n_components = n_components
        self.max_samples = max_samples

    def visualize(
        self,
        parent_output: torch.Tensor,
        *,
        batch: dict[str, Any] | None = None,
        logger: Any | None = None,
        current_epoch: int | None = None,
        labels: Any | None = None,
        metadata: dict[str, Any] | None = None,
        stage: str = "train",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create PCA projection visualization.
        
        Parameters
        ----------
        parent_output : torch.Tensor
            Output tensor from parent node (B, C, H, W) or (B, C)
        batch : dict[str, Any], optional
            Original batch dictionary
        logger : Any, optional
            Lightning logger instance
        current_epoch : int, optional
            Current epoch number if available
        labels : Any, optional
            Labels associated with parent output (unused)
        metadata : dict[str, Any], optional
            Additional metadata propagated through the graph
        stage : str
            Training stage identifier
        **kwargs : Any
            Additional visualization-specific parameters
            
        Returns
        -------
        dict[str, Any]
            Dictionary with 'figure' key containing matplotlib figure
        """
        metadata_dict = metadata if isinstance(metadata, dict) else {}

        epoch_value = current_epoch
        if epoch_value is None:
            epoch_value = metadata_dict.get("epoch")
        if epoch_value is None:
            epoch_value = kwargs.get("epoch")

        if epoch_value is not None:
            title_suffix = f"Epoch {epoch_value}"
        elif stage:
            title_suffix = stage.capitalize()
        else:
            title_suffix = ""

        # Convert to numpy and reshape
        data = parent_output.detach().cpu().numpy()

        # Handle different input shapes
        if data.ndim == 4:
            # (B, C, H, W) -> flatten spatial dimensions
            B, C, H, W = data.shape
            data = data.reshape(B, C, H * W).transpose(0, 2, 1)  # (B, H*W, C)
            data = data.reshape(-1, C)  # (B*H*W, C)
        elif data.ndim == 3:
            # (B, H, W) -> add channel dimension
            data = data.reshape(-1, 1)
        elif data.ndim == 2:
            # (B, C) -> already in correct shape
            pass
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

        # Subsample if too many points
        if len(data) > self.max_samples:
            indices = np.random.choice(len(data), self.max_samples, replace=False)
            data = data[indices]

        # Fit PCA
        pca = PCA(n_components=self.n_components)
        projected = pca.fit_transform(data)

        # Create figure
        if self.n_components == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(
                projected[:, 0],
                projected[:, 1],
                alpha=0.6,
                s=20,
                c=np.arange(len(projected)),
                cmap='viridis'
            )
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            if title_suffix:
                ax.set_title(f'PCA Projection - {title_suffix}')
            else:
                ax.set_title('PCA Projection')
            plt.colorbar(scatter, ax=ax, label='Sample Index')
            ax.grid(True, alpha=0.3)

        else:  # 3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                projected[:, 0],
                projected[:, 1],
                projected[:, 2],
                alpha=0.6,
                s=20,
                c=np.arange(len(projected)),
                cmap='viridis'
            )
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
            if title_suffix:
                ax.set_title(f'PCA Projection - {title_suffix}')
            else:
                ax.set_title('PCA Projection')
            plt.colorbar(scatter, ax=ax, label='Sample Index', pad=0.1)

        plt.tight_layout()

        return {
            'figure': fig,
            'type': 'pca_projection',
            'n_components': self.n_components,
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'total_variance': float(pca.explained_variance_ratio_.sum())
        }


class AnomalyHeatmap(VisualizationNode):
    """Visualize anomaly detection scores as spatial heatmaps.
    
    Creates color-mapped images showing anomaly scores across spatial dimensions,
    useful for understanding where anomalies are detected.
    
    Parameters
    ----------
    log_every_n_epochs : int
        Generate visualizations every N epochs
    cmap : str
        Matplotlib colormap name
    vmin : float, optional
        Minimum value for colormap scaling
    vmax : float, optional
        Maximum value for colormap scaling
    weight : float
        Weighting factor (unused for visualizations)
        
    Examples
    --------
    >>> heatmap_viz = AnomalyHeatmap(log_every_n_epochs=2, cmap='hot')
    >>> graph.add_leaf_node(heatmap_viz, parent=rx_detector_node)
    """

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        cmap: str = 'hot',
        vmin: float | None = None,
        vmax: float | None = None,
        weight: float = 1.0
    ):
        super().__init__(log_every_n_epochs, weight)
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax

    def visualize(
        self,
        parent_output: torch.Tensor,
        *,
        batch: dict[str, Any] | None = None,
        logger: Any | None = None,
        current_epoch: int | None = None,
        labels: Any | None = None,
        metadata: dict[str, Any] | None = None,
        stage: str = "train",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create anomaly heatmap visualization with overlay on original cube.
        
        Parameters
        ----------
        parent_output : torch.Tensor
            Anomaly scores from parent node (B, H, W) or (B, 1, H, W)
        batch : dict[str, Any], optional
            Original batch dictionary (must contain 'cube' or 'x')
        logger : Any, optional
            Lightning logger instance
        current_epoch : int, optional
            Current epoch number if provided
        labels : Any, optional
            Labels associated with parent output (unused)
        metadata : dict[str, Any], optional
            Additional metadata propagated through the graph
        stage : str
            Training stage identifier
        **kwargs : Any
            Additional visualization-specific parameters
            
        Returns
        -------
        dict[str, Any]
            Dictionary with 'figure' key containing matplotlib figure
        """
        batch_dict = batch if isinstance(batch, dict) else {}
        metadata_dict = metadata if isinstance(metadata, dict) else {}

        epoch_value = current_epoch
        if epoch_value is None:
            epoch_value = metadata_dict.get("epoch")
        if epoch_value is None:
            epoch_value = kwargs.get("epoch")

        if epoch_value is not None:
            title_suffix = f"Epoch {epoch_value}"
        elif stage:
            title_suffix = stage.capitalize()
        else:
            title_suffix = ""

        # Convert scores to numpy
        scores = parent_output.detach().cpu().numpy()

        # Handle different shapes
        if scores.ndim == 4:
            # Check if channel dimension is 1 (could be at index 1 or 3)
            if scores.shape[1] == 1:
                scores = scores[:, 0, :, :]  # (B, 1, H, W) -> (B, H, W)
            elif scores.shape[3] == 1:
                scores = scores[:, :, :, 0]  # (B, H, W, 1) -> (B, H, W)
            else:
                raise ValueError(f"Expected channel dim to be 1, got shape {scores.shape}")
        elif scores.ndim != 3:
            raise ValueError(f"Expected (B, H, W) or (B, C, H, W) with C=1, got {scores.shape}")

        # Take first batch item for visualization
        score_map = scores[0]

        # Get original cube from batch
        cube = batch_dict.get("cube")
        if cube is None:
            cube = batch_dict.get("x")
        if cube is None:
            cube = metadata_dict.get("cube") or metadata_dict.get("x")
        if cube is not None:
            cube_tensor = cube.detach() if hasattr(cube, "detach") else cube
            cube_np = cube_tensor.cpu().numpy() if hasattr(cube_tensor, "cpu") else np.asarray(cube_tensor)
            # Take first batch item: (B, C, H, W) -> (C, H, W)
            cube_img = cube_np[0]
            # Select middle channel for visualization
            channel_idx = cube_img.shape[0] // 2
            cube_channel = cube_img[channel_idx]
        else:
            cube_channel = None

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Subplot 1: Heatmap only
        im0 = axes[0].imshow(
            score_map,
            cmap=self.cmap,
            aspect='auto',
            vmin=self.vmin if self.vmin is not None else score_map.min(),
            vmax=self.vmax if self.vmax is not None else score_map.max()
        )
        if title_suffix:
            axes[0].set_title(f'Anomaly Score Heatmap - {title_suffix}')
        else:
            axes[0].set_title('Anomaly Score Heatmap')
        axes[0].set_xlabel('Width')
        axes[0].set_ylabel('Height')
        plt.colorbar(im0, ax=axes[0], label='Anomaly Score')

        # Subplot 2: Overlay on cube channel (if available)
        threshold = score_map.mean() + 2 * score_map.std()
        binary_mask = score_map > threshold

        if cube_channel is not None:
            # Normalize cube channel to [0, 1] for display
            cube_norm = (cube_channel - cube_channel.min()) / (cube_channel.max() - cube_channel.min() + 1e-8)

            # Display cube channel as grayscale
            axes[1].imshow(cube_norm, cmap='gray', aspect='auto')

            # Overlay anomaly mask in red with transparency
            overlay = np.zeros((*binary_mask.shape, 4))
            overlay[binary_mask] = [1, 0, 0, 0.5]  # Red with 50% transparency
            axes[1].imshow(overlay, aspect='auto')

            suffix = f' - {title_suffix}' if title_suffix else ''
            axes[1].set_title(f'Anomaly Overlay (Channel {channel_idx}){suffix}')
            axes[1].set_xlabel('Width')
            axes[1].set_ylabel('Height')
        else:
            # Fallback: just show binary mask
            axes[1].imshow(binary_mask, cmap='gray', aspect='auto')
            axes[1].set_title('Anomaly Mask (no cube available)')
            axes[1].set_xlabel('Width')
            axes[1].set_ylabel('Height')

        # Subplot 3: Binary mask
        axes[2].imshow(binary_mask, cmap='gray', aspect='auto')
        axes[2].set_title(f'Binary Mask (threshold={threshold:.3f})')
        axes[2].set_xlabel('Width')
        axes[2].set_ylabel('Height')

        # Add statistics as text
        stats_text = (
            f"Min: {score_map.min():.3f}\n"
            f"Max: {score_map.max():.3f}\n"
            f"Mean: {score_map.mean():.3f}\n"
            f"Std: {score_map.std():.3f}\n"
            f"Threshold: {threshold:.3f}\n"
            f"Anomalies: {binary_mask.sum()}/{binary_mask.size}\n"
            f"Rate: {binary_mask.sum()/binary_mask.size*100:.2f}%"
        )
        if cube_channel is not None:
            stats_text += f"\nChannel: {channel_idx}/{cube_img.shape[0]}"

        fig.text(0.98, 0.5, stats_text, ha='left', va='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        return {
            'figure': fig,
            'type': 'anomaly_heatmap',
            'statistics': {
                'min': float(score_map.min()),
                'max': float(score_map.max()),
                'mean': float(score_map.mean()),
                'std': float(score_map.std()),
                'threshold': float(threshold),
                'anomaly_count': int(binary_mask.sum()),
                'total_pixels': int(binary_mask.size),
                'anomaly_rate': float(binary_mask.sum() / binary_mask.size * 100),
                'cube_channel': int(channel_idx) if cube_channel is not None else None
            }
        }


class ScoreHistogram(VisualizationNode):
    """Visualize distribution of anomaly scores with statistics.
    
    Creates histogram plots showing the distribution of scores with
    mean, standard deviation, and threshold annotations.
    
    Parameters
    ----------
    log_every_n_epochs : int
        Generate visualizations every N epochs
    bins : int
        Number of histogram bins
    weight : float
        Weighting factor (unused for visualizations)
        
    Examples
    --------
    >>> hist_viz = ScoreHistogram(log_every_n_epochs=1, bins=50)
    >>> graph.add_leaf_node(hist_viz, parent=rx_detector_node)
    """

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        bins: int = 50,
        weight: float = 1.0
    ):
        super().__init__(log_every_n_epochs, weight)
        self.bins = bins

    def visualize(
        self,
        parent_output: torch.Tensor,
        *,
        batch: dict[str, Any] | None = None,
        logger: Any | None = None,
        current_epoch: int | None = None,
        labels: Any | None = None,
        metadata: dict[str, Any] | None = None,
        stage: str = "train",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create score histogram visualization.
        
        Parameters
        ----------
        parent_output : torch.Tensor
            Scores from parent node (any shape)
        batch : dict[str, Any], optional
            Original batch dictionary (unused)
        logger : Any, optional
            Lightning logger instance (unused)
        current_epoch : int, optional
            Current epoch number if provided
        labels : Any, optional
            Labels associated with parent output (unused)
        metadata : dict[str, Any], optional
            Additional metadata propagated through the graph
        stage : str
            Training stage identifier
        **kwargs : Any
            Additional visualization-specific parameters
            
        Returns
        -------
        dict[str, Any]
            Dictionary with 'figure' key containing matplotlib figure
        """
        metadata_dict = metadata if isinstance(metadata, dict) else {}

        epoch_value = current_epoch
        if epoch_value is None:
            epoch_value = metadata_dict.get("epoch")
        if epoch_value is None:
            epoch_value = kwargs.get("epoch")

        if epoch_value is not None:
            title_suffix = f"Epoch {epoch_value}"
        elif stage:
            title_suffix = stage.capitalize()
        else:
            title_suffix = ""

        # Convert to numpy and flatten
        scores = parent_output.detach().cpu().numpy().flatten()

        # Compute statistics
        mean_score = scores.mean()
        std_score = scores.std()
        threshold = mean_score + 2 * std_score

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        n, bins_edges, patches = ax.hist(
            scores,
            bins=self.bins,
            alpha=0.7,
            color='blue',
            edgecolor='black',
            label='Score Distribution'
        )

        # Add vertical lines for statistics
        ax.axvline(mean_score, color='green', linestyle='--',
                   linewidth=2, label=f'Mean: {mean_score:.3f}')
        ax.axvline(mean_score + std_score, color='orange', linestyle='--',
                   linewidth=1.5, label=f'Mean + 1σ: {mean_score + std_score:.3f}')
        ax.axvline(threshold, color='red', linestyle='--',
                   linewidth=2, label=f'Threshold (Mean + 2σ): {threshold:.3f}')

        # Labels and title
        ax.set_xlabel('Score Value')
        ax.set_ylabel('Frequency')
        if title_suffix:
            ax.set_title(f'Score Distribution - {title_suffix}')
        else:
            ax.set_title('Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = (
            f"N = {len(scores)}\n"
            f"μ = {mean_score:.4f}\n"
            f"σ = {std_score:.4f}\n"
            f"Min = {scores.min():.4f}\n"
            f"Max = {scores.max():.4f}\n"
            f"Anomalies = {(scores > threshold).sum()} ({(scores > threshold).sum() / len(scores) * 100:.2f}%)"
        )
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontfamily='monospace')

        plt.tight_layout()

        return {
            'figure': fig,
            'type': 'score_histogram',
            'statistics': {
                'count': int(len(scores)),
                'mean': float(mean_score),
                'std': float(std_score),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'threshold': float(threshold),
                'anomaly_count': int((scores > threshold).sum()),
                'anomaly_percentage': float((scores > threshold).sum() / len(scores) * 100)
            }
        }


__all__ = [
    'PCAVisualization',
    'AnomalyHeatmap',
    'ScoreHistogram',
]
