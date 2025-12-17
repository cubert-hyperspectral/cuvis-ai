"""Training infrastructure for cuvis.ai PyTorch Lightning integration.

This module provides:
- Training configuration dataclasses with Hydra support
- GraphDataModule base class for data loading
- Port-based loss and metric nodes for training
- Internal Lightning module for training orchestration
"""

from cuvis_ai.node.losses import (
    AnomalyBCEWithLogits,
    MSEReconstructionLoss,
    OrthogonalityLoss,
    SelectorDiversityRegularizer,
    SelectorEntropyRegularizer,
)
from cuvis_ai.node.metrics import (
    AnomalyDetectionMetrics,
    ComponentOrthogonalityMetric,
    ExplainedVarianceMetric,
    ScoreStatisticsMetric,
)
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.visualizations import AnomalyMask, PCAVisualization
from cuvis_ai.training.config import (
    CallbacksConfig,
    DataConfig,
    OptimizerConfig,
    PipelineConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingConfig,
    TrainRunConfig,
)
from cuvis_ai.training.datamodule import CuvisDataModule
from cuvis_ai.training.trainers import GradientTrainer, StatisticalTrainer
from cuvis_ai.utils.types import Context

__all__ = [
    # Configuration
    "TrainerConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "SchedulerConfig",
    "CallbacksConfig",
    "DataConfig",
    "PipelineConfig",
    "TrainRunConfig",
    # Data Module
    "CuvisDataModule",
    # Context
    "Context",
    # External Trainers (Phase 4.7)
    "GradientTrainer",
    "StatisticalTrainer",
    # Loss Nodes (port-based)
    "OrthogonalityLoss",
    "AnomalyBCEWithLogits",
    "MSEReconstructionLoss",
    "SelectorEntropyRegularizer",
    "SelectorDiversityRegularizer",
    # Metric Nodes (port-based)
    "ExplainedVarianceMetric",
    "AnomalyDetectionMetrics",
    "ScoreStatisticsMetric",
    "ComponentOrthogonalityMetric",
    # Monitoring Adapters
    "TensorBoardMonitorNode",
    # Lightning Module (internal)
    "PCAVisualization",
    "AnomalyMask",
]
