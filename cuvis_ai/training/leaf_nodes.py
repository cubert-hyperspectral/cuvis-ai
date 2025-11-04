"""Leaf node infrastructure for losses, metrics, visualizations, and monitoring.

Leaf nodes attach to parent nodes in the graph to provide:
- Loss computation for training
- Metric computation for evaluation
- Visualization generation
- Monitoring/logging to external services
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from cuvis_ai.node import Node


class LeafNode(nn.Module, ABC):
    """Base class for all leaf nodes that attach to graph nodes.
    
    Leaf nodes extend the graph with auxiliary functionality without being
    part of the main computational path. They validate their parent at
    registration time to ensure type safety.
    
    Parameters
    ----------
    weight : float
        Weighting factor for this leaf's contribution (used primarily by LossNode)
    
    Attributes
    ----------
    compatible_parent_types : tuple
        Tuple of parent node classes this leaf can attach to
    required_parent_attributes : tuple
        Tuple of attribute names the parent must have
    """

    compatible_parent_types: tuple = tuple()
    required_parent_attributes: tuple = tuple()

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def validate_parent(self, parent_node: Node) -> None:
        """Ensure the parent node satisfies the compatibility contract.
        
        Parameters
        ----------
        parent_node : Node
            Graph node instance this leaf will attach to
            
        Raises
        ------
        TypeError
            If the parent does not match the declared contract
        """
        # Check parent type compatibility
        if self.compatible_parent_types:
            is_compatible = False
            for expected in self.compatible_parent_types:
                if isinstance(parent_node, expected):
                    is_compatible = True
                    break

            if not is_compatible:
                expected_names = ", ".join(
                    expected.__name__
                    for expected in self.compatible_parent_types
                )
                raise TypeError(
                    f"{self.__class__.__name__} requires parent types "
                    f"[{expected_names}] but received {parent_node.__class__.__name__}"
                )

        # Check required attributes
        missing_attrs = [
            attr for attr in self.required_parent_attributes
            if not hasattr(parent_node, attr)
        ]
        if missing_attrs:
            raise TypeError(
                f"{self.__class__.__name__} requires parent to expose attributes "
                f"{missing_attrs}, but {parent_node.__class__.__name__} is missing them"
            )


class LossNode(LeafNode):
    """Base class for loss leaf nodes.
    
    Loss nodes compute differentiable loss values based on their parent's
    output. Multiple loss nodes can be aggregated during training.
    """

    @abstractmethod
    def compute_loss(
        self,
        parent_output: torch.Tensor,
        batch: dict[str, Any],
        graph_output: torch.Tensor,
        y_out: Any,
        m_out: dict[str, Any]
    ) -> torch.Tensor:
        """Compute loss based on parent node output.
        
        Parameters
        ----------
        parent_output : torch.Tensor
            Output tensor from the parent node
        batch : dict[str, Any]
            Original batch dictionary from dataloader
        graph_output : torch.Tensor
            Final output from the graph forward pass
        y_out : Any
            Labels after graph forward pass
        m_out : dict[str, Any]
            Metadata after graph forward pass (includes parent_node reference)
            
        Returns
        -------
        torch.Tensor
            Scalar loss tensor
        """
        pass


class VisualizationNode(LeafNode):
    """Base class for visualization leaf nodes.
    
    Visualization nodes generate plots, images, or other visual artifacts
    for monitoring training progress.
    
    Parameters
    ----------
    log_every_n_epochs : int
        Generate visualizations every N epochs
    weight : float
        Weighting factor (unused for visualizations, kept for consistency)
    """

    def __init__(self, log_every_n_epochs: int = 1, weight: float = 1.0):
        super().__init__(weight)
        self.log_every_n_epochs = log_every_n_epochs
        self._last_logged_epoch = -1

    def should_log(self, current_epoch: int) -> bool:
        """Check if should log this epoch based on frequency.
        
        Parameters
        ----------
        current_epoch : int
            Current training epoch
            
        Returns
        -------
        bool
            True if visualization should be generated this epoch
        """
        # First epoch (when _last_logged_epoch is -1) should always log
        if self._last_logged_epoch == -1:
            self._last_logged_epoch = current_epoch
            return True

        if current_epoch - self._last_logged_epoch >= self.log_every_n_epochs:
            self._last_logged_epoch = current_epoch
            return True
        return False

    @abstractmethod
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
        """Create visualization from parent node output.
        
        Parameters
        ----------
        parent_output : torch.Tensor
            Output tensor from the parent node
        batch : dict[str, Any], optional
            Original batch dictionary from dataloader (if available)
        logger : Any, optional
            Lightning logger (WandB, TensorBoard, etc.)
        current_epoch : int, optional
            Current epoch number if provided
        labels : Any, optional
            Labels associated with parent output
        metadata : dict[str, Any], optional
            Additional metadata propagated through the graph
        stage : str
            Training stage identifier (e.g., "train", "val")
        **kwargs : Any
            Additional visualization-specific keyword arguments
            
        Returns
        -------
        dict[str, Any]
            Dictionary mapping visualization names to artifacts (figures, images, etc.)
        """
        pass


class MetricNode(LeafNode):
    """Base class for metric leaf nodes.
    
    Metric nodes compute evaluation metrics (accuracy, F1, etc.) based on
    their parent's output. Metrics are typically logged but not backpropagated.
    """

    @abstractmethod
    def compute_metric(
        self,
        parent_output: torch.Tensor,
        batch: dict[str, Any],
        graph_output: torch.Tensor,
        y_out: Any,
        m_out: dict[str, Any]
    ) -> dict[str, float]:
        """Compute metrics based on parent node output.
        
        Parameters
        ----------
        parent_output : torch.Tensor
            Output tensor from the parent node
        batch : dict[str, Any]
            Original batch dictionary from dataloader
        graph_output : torch.Tensor
            Final output from the graph forward pass
        y_out : Any
            Labels after graph forward pass
        m_out : dict[str, Any]
            Metadata after graph forward pass (includes parent_node reference)
            
        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to values
        """
        pass


class MonitoringNode(LeafNode):
    """Protocol-style leaf that forwards events to external monitoring backends.
    
    Monitoring nodes don't attach to specific graph nodes but instead receive
    aggregated metrics and artifacts from the training loop to forward to
    external services (WandB, TensorBoard, MLflow, etc.).
    """

    compatible_parent_types = tuple()  # Not wired to a graph node

    def setup(self, trainer: Any) -> None:
        """Optional hook when the Trainer is created/configured.
        
        Parameters
        ----------
        trainer : Any
            PyTorch Lightning Trainer instance
        """
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        *,
        stage: str
    ) -> None:
        """Receive aggregated scalar metrics.
        
        Parameters
        ----------
        metrics : dict[str, float]
            Dictionary of metric names to values
        step : int
            Global training step
        stage : str
            Training stage ('train', 'val', 'test')
        """
        pass

    def log_artifacts(
        self,
        artifacts: dict[str, Any],
        *,
        stage: str,
        step: int
    ) -> None:
        """Receive arbitrary artifacts (figures, tables, media).
        
        Parameters
        ----------
        artifacts : dict[str, Any]
            Dictionary of artifact names to objects
        stage : str
            Training stage ('train', 'val', 'test')
        step : int
            Global training step
        """
        pass

    def teardown(self) -> None:
        """Optional cleanup hook called after training completes."""
        pass


__all__ = [
    "LeafNode",
    "LossNode",
    "VisualizationNode",
    "MetricNode",
    "MonitoringNode",
]
