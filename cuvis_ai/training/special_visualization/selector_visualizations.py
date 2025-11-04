"""Selector-specific visualization nodes for monitoring channel selection during training.

This module provides visualization leaf nodes specifically designed for SoftChannelSelector,
tracking temperature annealing, channel selection patterns, and selection stability over time.
"""

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.use('Agg')  # Non-interactive backend
import numpy as np

from cuvis_ai.node import LabelLike, MetaLike
from cuvis_ai.training.leaf_nodes import VisualizationNode


class SelectorTemperaturePlot(VisualizationNode):
    """Visualize selector temperature annealing over training steps.
    
    This visualization tracks the temperature parameter of SoftChannelSelector
    over time, showing the annealing schedule and current temperature value.
    
    Parameters
    ----------
    log_frequency : int, default=1
        How often to generate visualization (every N validation steps)
    history_size : int, default=1000
        Maximum number of steps to keep in history
    
    Attributes
    ----------
    temperature_history : list
        History of (step, temperature) tuples
    step_counter : int
        Current step counter
    
    Examples
    --------
    >>> from cuvis_ai.node.selector import SoftChannelSelector
    >>> selector = SoftChannelSelector(n_select=15, trainable=True)
    >>> temp_viz = SelectorTemperaturePlot(log_frequency=1)
    >>> graph.add_leaf_node(temp_viz, parent=selector)
    """

    compatible_parent_types = ()  # Accept any parent with temperature attribute
    required_parent_attributes = ("temperature",)

    def __init__(
        self,
        log_frequency: int = 1,
        history_size: int = 1000,
    ):
        super().__init__()
        self.log_frequency = log_frequency
        self.history_size = history_size
        self.temperature_history = []
        self.step_counter = 0

    def visualize(
        self,
        parent_output: torch.Tensor,
        labels: LabelLike = None,
        metadata: MetaLike = None,
        stage: str = "train",
    ) -> dict[str, Any]:
        """Generate temperature annealing curve visualization.
        
        Parameters
        ----------
        parent_output : torch.Tensor
            Output from parent selector (not used directly)
        labels : optional
            Labels (not used)
        metadata : optional
            Metadata (not used)
        stage : str
            Training stage ('train' or 'val')
        
        Returns
        -------
        dict
            Artifact dictionary with figure and metadata
        """
        # Only log during validation and at specified frequency
        if stage != "val" or self.step_counter % self.log_frequency != 0:
            self.step_counter += 1
            return {}

        # Get current temperature from parent
        current_temp = float(self.parent.temperature)

        # Update history
        self.temperature_history.append((self.step_counter, current_temp))

        # Limit history size
        if len(self.temperature_history) > self.history_size:
            self.temperature_history = self.temperature_history[-self.history_size:]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        if len(self.temperature_history) > 1:
            steps, temps = zip(*self.temperature_history)
            ax.plot(steps, temps, 'b-', linewidth=2, label='Temperature')
            ax.axhline(y=self.parent.temperature_min, color='r', linestyle='--',
                      linewidth=1, label=f'Min Temperature ({self.parent.temperature_min:.3f})')
            ax.axhline(y=self.parent.temperature_init, color='g', linestyle='--',
                      linewidth=1, label=f'Init Temperature ({self.parent.temperature_init:.3f})')

        ax.set_xlabel('Validation Step', fontsize=12)
        ax.set_ylabel('Temperature', fontsize=12)
        ax.set_title(f'Selector Temperature Annealing (Current: {current_temp:.4f})', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        if len(self.temperature_history) > 1:
            stats_text = (
                f"Steps: {len(self.temperature_history)}\n"
                f"Decay: {self.parent.temperature_decay:.4f}\n"
                f"Range: [{self.parent.temperature_min:.3f}, {self.parent.temperature_init:.3f}]"
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        self.step_counter += 1

        return {
            "figure": fig,
            "data": {
                "temperature": current_temp,
                "step": self.step_counter,
                "history_length": len(self.temperature_history),
            }
        }


class SelectorChannelMaskPlot(VisualizationNode):
    """Visualize channel selection probabilities and hard selection mask.
    
    This visualization shows:
    1. Soft selection probabilities for all channels
    2. Hard selection mask (top-k selected channels)
    3. Channel indices sorted by importance
    
    Parameters
    ----------
    log_frequency : int, default=5
        How often to generate visualization (every N validation steps)
    max_channels_display : int, default=50
        Maximum number of channels to display (truncates if more)
    
    Examples
    --------
    >>> selector = SoftChannelSelector(n_select=15, trainable=True)
    >>> mask_viz = SelectorChannelMaskPlot(log_frequency=5)
    >>> graph.add_leaf_node(mask_viz, parent=selector)
    """

    compatible_parent_types = ()  # Accept any parent with get_selection_weights
    required_parent_attributes = ("get_selection_weights", "get_top_k_channels", "n_select")

    def __init__(
        self,
        log_frequency: int = 5,
        max_channels_display: int = 50,
    ):
        super().__init__()
        self.log_frequency = log_frequency
        self.max_channels_display = max_channels_display
        self.step_counter = 0

    def visualize(
        self,
        parent_output: torch.Tensor,
        labels: LabelLike = None,
        metadata: MetaLike = None,
        stage: str = "train",
    ) -> dict[str, Any]:
        """Generate channel selection visualization.
        
        Parameters
        ----------
        parent_output : torch.Tensor
            Output from parent selector
        labels : optional
            Labels (not used)
        metadata : optional
            Metadata (not used)
        stage : str
            Training stage ('train' or 'val')
        
        Returns
        -------
        dict
            Artifact dictionary with figure and metadata
        """
        # Only log during validation and at specified frequency
        if stage != "val" or self.step_counter % self.log_frequency != 0:
            self.step_counter += 1
            return {}

        with torch.no_grad():
            # Get selection weights (probabilities)
            weights = self.parent.get_selection_weights().cpu().numpy()

            # Get top-k selected channels
            top_k_indices = self.parent.get_top_k_channels().cpu().numpy()

            n_channels = len(weights)
            n_select = self.parent.n_select

            # Limit display if too many channels
            if n_channels > self.max_channels_display:
                # Show top channels by weight
                sorted_indices = np.argsort(weights)[::-1][:self.max_channels_display]
                display_weights = weights[sorted_indices]
                display_indices = sorted_indices
                truncated = True
            else:
                display_weights = weights
                display_indices = np.arange(n_channels)
                truncated = False

            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Plot 1: Bar chart of selection weights
            colors = ['red' if idx in top_k_indices else 'blue' for idx in display_indices]
            ax1.bar(range(len(display_weights)), display_weights, color=colors, alpha=0.7)
            ax1.axhline(y=1.0/n_channels, color='green', linestyle='--', linewidth=1,
                       label=f'Uniform ({1.0/n_channels:.4f})')
            ax1.set_xlabel('Channel Index', fontsize=12)
            ax1.set_ylabel('Selection Probability', fontsize=12)
            title = f'Channel Selection Weights (Top {n_select} selected in red)'
            if truncated:
                title += f'\n(Showing top {self.max_channels_display} of {n_channels} channels)'
            ax1.set_title(title, fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')

            # Set x-axis labels
            if len(display_indices) <= 30:
                ax1.set_xticks(range(len(display_indices)))
                ax1.set_xticklabels(display_indices, rotation=45, ha='right')

            # Plot 2: Heatmap of selection mask
            selection_mask = np.zeros(n_channels)
            selection_mask[top_k_indices] = 1.0

            # Display as heatmap
            mask_2d = selection_mask.reshape(1, -1)
            if truncated:
                mask_2d = mask_2d[:, sorted_indices]

            im = ax2.imshow(mask_2d, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            ax2.set_xlabel('Channel Index', fontsize=12)
            ax2.set_ylabel('Selection', fontsize=12)
            ax2.set_title(f'Hard Selection Mask ({n_select}/{n_channels} channels)', fontsize=14)
            ax2.set_yticks([0])
            ax2.set_yticklabels(['Selected'])

            if truncated:
                if len(sorted_indices) <= 30:
                    ax2.set_xticks(range(len(sorted_indices)))
                    ax2.set_xticklabels(sorted_indices, rotation=45, ha='right')
            else:
                if n_channels <= 30:
                    ax2.set_xticks(range(n_channels))
                    ax2.set_xticklabels(range(n_channels), rotation=45, ha='right')

            plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.1)

            # Add statistics text box
            entropy = self.parent.compute_entropy().item()
            diversity = self.parent.compute_diversity_loss().item()
            stats_text = (
                f"Selected Channels: {sorted(top_k_indices.tolist())}\n"
                f"Entropy: {entropy:.4f}\n"
                f"Diversity: {diversity:.4f}\n"
                f"Max Weight: {weights.max():.4f}\n"
                f"Min Weight: {weights.min():.4f}"
            )
            fig.text(0.02, 0.98, stats_text, transform=fig.transFigure,
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout(rect=[0, 0, 1, 0.96])

        self.step_counter += 1

        return {
            "figure": fig,
            "data": {
                "selected_channels": top_k_indices.tolist(),
                "selection_weights": weights.tolist(),
                "entropy": entropy,
                "diversity": diversity,
                "step": self.step_counter,
            }
        }


class SelectorStabilityPlot(VisualizationNode):
    """Track stability of channel selection over time.
    
    This visualization monitors how the selected channels change across
    validation steps, helping identify when selection has stabilized.
    
    Parameters
    ----------
    log_frequency : int, default=1
        How often to update tracking (every N validation steps)
    history_size : int, default=100
        Number of selection snapshots to keep in history
    
    Examples
    --------
    >>> selector = SoftChannelSelector(n_select=15, trainable=True)
    >>> stability_viz = SelectorStabilityPlot(log_frequency=1)
    >>> graph.add_leaf_node(stability_viz, parent=selector)
    """

    compatible_parent_types = ()
    required_parent_attributes = ("get_top_k_channels", "n_select")

    def __init__(
        self,
        log_frequency: int = 1,
        history_size: int = 100,
    ):
        super().__init__()
        self.log_frequency = log_frequency
        self.history_size = history_size
        self.selection_history = []  # List of (step, selected_indices_set)
        self.step_counter = 0

    def visualize(
        self,
        parent_output: torch.Tensor,
        labels: LabelLike = None,
        metadata: MetaLike = None,
        stage: str = "train",
    ) -> dict[str, Any]:
        """Generate selection stability visualization.
        
        Parameters
        ----------
        parent_output : torch.Tensor
            Output from parent selector
        labels : optional
            Labels (not used)
        metadata : optional
            Metadata (not used)
        stage : str
            Training stage ('train' or 'val')
        
        Returns
        -------
        dict
            Artifact dictionary with figure and metadata
        """
        # Only log during validation and at specified frequency
        if stage != "val" or self.step_counter % self.log_frequency != 0:
            self.step_counter += 1
            return {}

        with torch.no_grad():
            # Get current selection
            top_k = self.parent.get_top_k_channels().cpu().numpy()
            current_selection = set(top_k.tolist())

            # Update history
            self.selection_history.append((self.step_counter, current_selection))

            # Limit history size
            if len(self.selection_history) > self.history_size:
                self.selection_history = self.selection_history[-self.history_size:]

            # Create figure only if we have enough history
            if len(self.selection_history) < 2:
                self.step_counter += 1
                return {}

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Compute stability metrics
            steps = [step for step, _ in self.selection_history]
            stability_scores = []  # Jaccard similarity with previous step
            change_counts = []  # Number of channels changed from previous

            for i in range(1, len(self.selection_history)):
                prev_set = self.selection_history[i-1][1]
                curr_set = self.selection_history[i][1]

                # Jaccard similarity
                intersection = len(prev_set & curr_set)
                union = len(prev_set | curr_set)
                jaccard = intersection / union if union > 0 else 0.0
                stability_scores.append(jaccard)

                # Count changes
                changes = len(prev_set ^ curr_set)  # Symmetric difference
                change_counts.append(changes)

            # Plot 1: Stability score over time
            ax1.plot(steps[1:], stability_scores, 'b-', linewidth=2, marker='o', markersize=4)
            ax1.axhline(y=1.0, color='g', linestyle='--', linewidth=1, label='Perfect Stability')
            ax1.set_xlabel('Validation Step', fontsize=12)
            ax1.set_ylabel('Jaccard Similarity', fontsize=12)
            ax1.set_title('Selection Stability (compared to previous step)', fontsize=14)
            ax1.set_ylim([-0.05, 1.05])
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Number of channel changes
            ax2.plot(steps[1:], change_counts, 'r-', linewidth=2, marker='s', markersize=4)
            ax2.axhline(y=0, color='g', linestyle='--', linewidth=1, label='No Changes')
            ax2.set_xlabel('Validation Step', fontsize=12)
            ax2.set_ylabel('Number of Changes', fontsize=12)
            ax2.set_title('Channel Changes from Previous Step', fontsize=14)
            ax2.set_ylim([min(change_counts) - 1, max(change_counts) + 1])
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Add statistics
            if len(stability_scores) > 0:
                mean_stability = np.mean(stability_scores)
                recent_stability = np.mean(stability_scores[-10:]) if len(stability_scores) >= 10 else mean_stability

                stats_text = (
                    f"Current Selection: {sorted(list(current_selection))}\n"
                    f"Mean Stability: {mean_stability:.4f}\n"
                    f"Recent Stability (last 10): {recent_stability:.4f}\n"
                    f"Total Steps: {len(self.selection_history)}\n"
                    f"Last Change: {change_counts[-1]} channels"
                )
                fig.text(0.02, 0.98, stats_text, transform=fig.transFigure,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout(rect=[0, 0, 1, 0.96])

        self.step_counter += 1

        return {
            "figure": fig,
            "data": {
                "selected_channels": list(current_selection),
                "stability_scores": stability_scores[-10:] if len(stability_scores) >= 10 else stability_scores,
                "mean_stability": mean_stability if len(stability_scores) > 0 else 0.0,
                "step": self.step_counter,
            }
        }
