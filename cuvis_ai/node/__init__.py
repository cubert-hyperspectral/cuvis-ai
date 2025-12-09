"""Convenience exports for node base classes and marker mixins."""

from cuvis_ai.node.adaclip_node import AdaCLIPDetector
from cuvis_ai.node.band_selection import (
    BandSelectorBase,
    BaselineFalseRGBSelector,
    CIRFalseColorSelector,
    HighContrastBandSelector,
    SupervisedBandSelectorBase,
    SupervisedCIRBandSelector,
    SupervisedFullSpectrumBandSelector,
    SupervisedWindowedFalseRGBSelector,
)
from cuvis_ai.node.labels import BinaryAnomalyLabelMapper
from cuvis_ai.node.node import Node
from cuvis_ai.node.normalization import IdentityNormalizer, MinMaxNormalizer, SigmoidNormalizer
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.node.selector import SoftChannelSelector, TopKIndices
from cuvis_ai.utils.types import ExecutionStage

__all__ = [
    "AdaCLIPDetector",
    "BandSelectorBase",
    "BaselineFalseRGBSelector",
    "BinaryAnomalyLabelMapper",
    "CIRFalseColorSelector",
    "ExecutionStage",
    "HighContrastBandSelector",
    "IdentityNormalizer",
    "MinMaxNormalizer",
    "Node",
    "SigmoidNormalizer",
    "SoftChannelSelector",
    "SupervisedBandSelectorBase",
    "SupervisedCIRBandSelector",
    "SupervisedFullSpectrumBandSelector",
    "SupervisedWindowedFalseRGBSelector",
    "TopKIndices",
    "TrainablePCA",
]
