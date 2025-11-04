"""Training infrastructure for cuvis.ai PyTorch Lightning integration.

This module provides:
- Training configuration dataclasses with Hydra support
- GraphDataModule base class for data loading
- Leaf node infrastructure for losses, metrics, and visualizations
- Internal Lightning module for training orchestration
"""

from cuvis_ai.training.config import (
    OptimizerConfig,
    TrainerConfig,
    TrainingConfig,
    TrainingConfigSerializable,
    as_dict,
    from_dict_config,
    override_from_iterable,
    register_training_config,
    to_dict_config,
)
from cuvis_ai.training.datamodule import GraphDataModule
from cuvis_ai.training.leaf_nodes import (
    LeafNode,
    LossNode,
    MetricNode,
    MonitoringNode,
    VisualizationNode,
)
from cuvis_ai.training.lightning_module import CuvisLightningModule
from cuvis_ai.training.losses import (
    AnomalyBCEWithLogits,
    MSEReconstructionLoss,
    OrthogonalityLoss,
    WeightedMultiLoss,
)
from cuvis_ai.training.metrics import (
    AnomalyDetectionMetrics,
    ComponentOrthogonalityMetric,
    ExplainedVarianceMetric,
    ScoreStatisticsMetric,
)
from cuvis_ai.training.monitors import DummyMonitor, TensorBoardMonitor, WandBMonitor
from cuvis_ai.training.special_visualization import (
    SelectorChannelMaskPlot,
    SelectorStabilityPlot,
    SelectorTemperaturePlot,
)
from cuvis_ai.training.visualizations import (
    AnomalyHeatmap,
    PCAVisualization,
    ScoreHistogram,
)

__all__ = [
    # Configuration
    "TrainerConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "TrainingConfigSerializable",
    "register_training_config",
    "as_dict",
    "override_from_iterable",
    "to_dict_config",
    "from_dict_config",
    # Data Module
    "GraphDataModule",
    # Leaf Nodes
    "LeafNode",
    "LossNode",
    "MetricNode",
    "VisualizationNode",
    "MonitoringNode",
    # Loss Leaves
    "OrthogonalityLoss",
    "AnomalyBCEWithLogits",
    "MSEReconstructionLoss",
    "WeightedMultiLoss",
    # Metric Leaves
    "ExplainedVarianceMetric",
    "AnomalyDetectionMetrics",
    "ScoreStatisticsMetric",
    "ComponentOrthogonalityMetric",
    # Visualization Leaves
    "PCAVisualization",
    "AnomalyHeatmap",
    "ScoreHistogram",
    # Monitoring Adapters
    "DummyMonitor",
    "WandBMonitor",
    "TensorBoardMonitor",
    # Lightning Module (internal)
    "CuvisLightningModule",
]
