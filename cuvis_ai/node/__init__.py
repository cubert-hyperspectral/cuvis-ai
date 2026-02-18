"""Convenience exports for node base classes and marker mixins."""

from cuvis_ai.node.band_selection import (
    BandSelectorBase,
    BaselineFalseRGBSelector,
    CIRFalseColorSelector,
    HighContrastBandSelector,
    RangeAverageFalseRGBSelector,
    SupervisedBandSelectorBase,
    SupervisedCIRBandSelector,
    SupervisedFullSpectrumBandSelector,
    SupervisedWindowedFalseRGBSelector,
)
from cuvis_ai.node.channel_mixer import LearnableChannelMixer
from cuvis_ai.node.labels import BinaryAnomalyLabelMapper
from cuvis_ai.node.losses import DistinctnessLoss, ForegroundContrastLoss
from cuvis_ai.node.normalization import IdentityNormalizer, MinMaxNormalizer, SigmoidNormalizer
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.node.preprocessors import BandpassByWavelength, SpatialRotateNode
from cuvis_ai.node.selector import SoftChannelSelector, TopKIndices
from cuvis_ai.node.video import ToVideoNode
from cuvis_ai.node.visualizations import (
    ChannelSelectorFalseRGBViz,
    ChannelWeightsViz,
    ImageArtifactVizBase,
    MaskOverlayNode,
)

__all__ = [
    "BandpassByWavelength",
    "BandSelectorBase",
    "BaselineFalseRGBSelector",
    "BinaryAnomalyLabelMapper",
    "ChannelSelectorFalseRGBViz",
    "ChannelWeightsViz",
    "CIRFalseColorSelector",
    "DistinctnessLoss",
    "ForegroundContrastLoss",
    "HighContrastBandSelector",
    "IdentityNormalizer",
    "ImageArtifactVizBase",
    "LearnableChannelMixer",
    "MaskOverlayNode",
    "MinMaxNormalizer",
    "RangeAverageFalseRGBSelector",
    "SigmoidNormalizer",
    "SoftChannelSelector",
    "SpatialRotateNode",
    "SupervisedBandSelectorBase",
    "SupervisedCIRBandSelector",
    "SupervisedFullSpectrumBandSelector",
    "SupervisedWindowedFalseRGBSelector",
    "ToVideoNode",
    "TopKIndices",
    "TrainablePCA",
]
