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
    TrackingPointerOverlayNode,
)
from cuvis_ai.node.channel_mixer import ConcreteChannelMixer, LearnableChannelMixer
from cuvis_ai.node.channel_selector import (
    CameraEmulationFalseRGBSelector,
    ChannelSelectorBase,
    CIETristimulusFalseRGBSelector,
    CIRSelector,
    FastRGBSelector,
    FixedWavelengthSelector,
    HighContrastSelector,
    NDVISelector,
    NormMode,
    RangeAverageFalseRGBSelector,
    SoftChannelSelector,
    SupervisedCIRSelector,
    SupervisedFullSpectrumSelector,
    SupervisedSelectorBase,
    SupervisedWindowedSelector,
    TopKIndices,
)
from cuvis_ai.node.colormap import ScalarHSVColormapNode
from cuvis_ai.node.conversion import DecisionToMask
from cuvis_ai.node.dimensionality_reduction import PCA, TrainablePCA
from cuvis_ai.node.json_reader import DetectionJsonReader, TrackingResultsReader
from cuvis_ai.node.json_writer import (
    CocoTrackBBoxWriter,
    CocoTrackMaskWriter,
    DetectionCocoJsonNode,
)
from cuvis_ai.node.labels import BinaryAnomalyLabelMapper
from cuvis_ai.node.losses import DistinctnessLoss, ForegroundContrastLoss
from cuvis_ai.node.normalization import IdentityNormalizer, MinMaxNormalizer, SigmoidNormalizer
from cuvis_ai.node.numpy_reader import NpyReader
from cuvis_ai.node.numpy_writer import NumpyFeatureWriterNode
from cuvis_ai.node.occlusion import (
    OcclusionNodeBase,
    PoissonCubeOcclusionNode,
    PoissonOcclusionNode,
    SolidOcclusionNode,
)
from cuvis_ai.node.pipeline_visualization import (
    CubeRGBVisualizer,
    PCAVisualization,
    PipelineComparisonVisualizer,
)
from cuvis_ai.node.preprocessors import (
    BandpassByWavelength,
    BBoxRoiCropNode,
    ChannelNormalizeNode,
    SpatialRotateNode,
)
from cuvis_ai.node.spectral_angle_mapper import SpectralAngleMapper
from cuvis_ai.node.spectral_extractor import BBoxSpectralExtractor
from cuvis_ai.node.prompts import BBoxPrompt, MaskPrompt, TextPrompt
from cuvis_ai.node.video import (
    ToVideoNode,
    VideoFrameDataModule,
    VideoFrameDataset,
    VideoFrameNode,
    VideoIterator,
)

__all__ = [
    "AnomalyMask",
    "BandpassByWavelength",
    "BBoxesOverlayNode",
    "BBoxRoiCropNode",
    "BBoxSpectralExtractor",
    "BBoxPrompt",
    "BinaryAnomalyLabelMapper",
    "ChannelNormalizeNode",
    "CameraEmulationFalseRGBSelector",
    "ChannelSelectorBase",
    "ChannelSelectorFalseRGBViz",
    "ChannelWeightsViz",
    "CIETristimulusFalseRGBSelector",
    "CIRSelector",
    "FastRGBSelector",
    "ConcreteChannelMixer",
    "CubeRGBVisualizer",
    "DetectionCocoJsonNode",
    "DetectionJsonReader",
    "DecisionToMask",
    "DistinctnessLoss",
    "FixedWavelengthSelector",
    "ForegroundContrastLoss",
    "HighContrastSelector",
    "IdentityNormalizer",
    "ImageArtifactVizBase",
    "LearnableChannelMixer",
    "MaskOverlayNode",
    "MaskPrompt",
    "TextPrompt",
    "MinMaxNormalizer",
    "NDVISelector",
    "NormMode",
    "NumpyFeatureWriterNode",
    "NpyReader",
    "OcclusionNodeBase",
    "PoissonCubeOcclusionNode",
    "PoissonOcclusionNode",
    "PCA",
    "PCAVisualization",
    "PipelineComparisonVisualizer",
    "RangeAverageFalseRGBSelector",
    "RGBAnomalyMask",
    "ScoreHeatmapVisualizer",
    "ScalarHSVColormapNode",
    "SigmoidNormalizer",
    "SolidOcclusionNode",
    "SoftChannelSelector",
    "SpatialRotateNode",
    "SpectralAngleMapper",
    "SupervisedCIRSelector",
    "SupervisedFullSpectrumSelector",
    "SupervisedSelectorBase",
    "SupervisedWindowedSelector",
    "ToVideoNode",
    "TopKIndices",
    "VideoFrameDataModule",
    "VideoFrameDataset",
    "VideoFrameNode",
    "VideoIterator",
    "CocoTrackMaskWriter",
    "CocoTrackBBoxWriter",
    "TrackingPointerOverlayNode",
    "TrackingOverlayNode",
    "TrackingResultsReader",
    "TrainablePCA",
]
