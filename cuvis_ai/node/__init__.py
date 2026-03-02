"""Convenience exports for node base classes and marker mixins."""

from cuvis_ai.node.anomaly_visualization import (
    AnomalyMask,
    BBoxesOverlayNode,
    ChannelSelectorFalseRGBViz,
    ChannelWeightsViz,
    ImageArtifactVizBase,
    MaskOverlayNode,
    RGBAnomalyMask,
    ScoreHeatmapVisualizer,
    TrackingOverlayNode,
)
from cuvis_ai.node.channel_mixer import ConcreteChannelMixer, LearnableChannelMixer
from cuvis_ai.node.channel_selector import (
    CameraEmulationFalseRGBSelector,
    ChannelSelectorBase,
    CIETristimulusFalseRGBSelector,
    CIRSelector,
    FixedWavelengthSelector,
    HighContrastSelector,
    NormMode,
    RangeAverageFalseRGBSelector,
    SoftChannelSelector,
    SupervisedCIRSelector,
    SupervisedFullSpectrumSelector,
    SupervisedSelectorBase,
    SupervisedWindowedSelector,
    TopKIndices,
)
from cuvis_ai.node.dimensionality_reduction import TrainablePCA
from cuvis_ai.node.json_writer import DetectionCocoJsonNode, TrackingCocoJsonNode
from cuvis_ai.node.labels import BinaryAnomalyLabelMapper
from cuvis_ai.node.losses import DistinctnessLoss, ForegroundContrastLoss
from cuvis_ai.node.normalization import IdentityNormalizer, MinMaxNormalizer, SigmoidNormalizer
from cuvis_ai.node.pipeline_visualization import (
    CubeRGBVisualizer,
    PCAVisualization,
    PipelineComparisonVisualizer,
)
from cuvis_ai.node.preprocessors import BandpassByWavelength, SpatialRotateNode
from cuvis_ai.node.video import ToVideoNode

__all__ = [
    "AnomalyMask",
    "BandpassByWavelength",
    "BBoxesOverlayNode",
    "BinaryAnomalyLabelMapper",
    "CameraEmulationFalseRGBSelector",
    "ChannelSelectorBase",
    "ChannelSelectorFalseRGBViz",
    "ChannelWeightsViz",
    "CIETristimulusFalseRGBSelector",
    "CIRSelector",
    "ConcreteChannelMixer",
    "CubeRGBVisualizer",
    "DetectionCocoJsonNode",
    "DistinctnessLoss",
    "FixedWavelengthSelector",
    "ForegroundContrastLoss",
    "HighContrastSelector",
    "IdentityNormalizer",
    "ImageArtifactVizBase",
    "LearnableChannelMixer",
    "MaskOverlayNode",
    "MinMaxNormalizer",
    "NormMode",
    "PCAVisualization",
    "PipelineComparisonVisualizer",
    "RangeAverageFalseRGBSelector",
    "RGBAnomalyMask",
    "ScoreHeatmapVisualizer",
    "SigmoidNormalizer",
    "SoftChannelSelector",
    "SpatialRotateNode",
    "SupervisedCIRSelector",
    "SupervisedFullSpectrumSelector",
    "SupervisedSelectorBase",
    "SupervisedWindowedSelector",
    "ToVideoNode",
    "TopKIndices",
    "TrackingCocoJsonNode",
    "TrackingOverlayNode",
    "TrainablePCA",
]
