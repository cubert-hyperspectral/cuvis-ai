"""Convenience exports for node base classes and marker mixins."""

from cuvis_ai.node.anomaly_visualization import AnomalyMask, RGBAnomalyMask, ScoreHeatmapVisualizer
from cuvis_ai.node.channel_mixer import ConcreteChannelMixer, LearnableChannelMixer
from cuvis_ai.node.channel_selector import (
    ChannelSelectorBase,
    CIRSelector,
    FixedWavelengthSelector,
    HighContrastSelector,
    SoftChannelSelector,
    SupervisedCIRSelector,
    SupervisedFullSpectrumSelector,
    SupervisedSelectorBase,
    SupervisedWindowedSelector,
    TopKIndices,
)
from cuvis_ai.node.dimensionality_reduction import TrainablePCA
from cuvis_ai.node.labels import BinaryAnomalyLabelMapper
from cuvis_ai.node.normalization import IdentityNormalizer, MinMaxNormalizer, SigmoidNormalizer
from cuvis_ai.node.pipeline_visualization import (
    CubeRGBVisualizer,
    PCAVisualization,
    PipelineComparisonVisualizer,
)

__all__ = [
    "AnomalyMask",
    "BinaryAnomalyLabelMapper",
    "ChannelSelectorBase",
    "CIRSelector",
    "ConcreteChannelMixer",
    "CubeRGBVisualizer",
    "FixedWavelengthSelector",
    "HighContrastSelector",
    "IdentityNormalizer",
    "LearnableChannelMixer",
    "MinMaxNormalizer",
    "PCAVisualization",
    "PipelineComparisonVisualizer",
    "RGBAnomalyMask",
    "ScoreHeatmapVisualizer",
    "SigmoidNormalizer",
    "SoftChannelSelector",
    "SupervisedCIRSelector",
    "SupervisedFullSpectrumSelector",
    "SupervisedSelectorBase",
    "SupervisedWindowedSelector",
    "TopKIndices",
    "TrainablePCA",
]
