"""Convenience exports for node base classes and marker mixins."""

# Keep existing exports for backward compatibility
from cuvis_ai.node.labels import BinaryAnomalyLabelMapper
from cuvis_ai.node.node import Node
from cuvis_ai.node.normalization import IdentityNormalizer, MinMaxNormalizer, SigmoidNormalizer
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.node.selector import SoftChannelSelector
from cuvis_ai.utils.types import ExecutionStage

__all__ = [
    "Node",
    "ExecutionStage",
    "BinaryAnomalyLabelMapper",
    "TrainablePCA",
    "SoftChannelSelector",
    "IdentityNormalizer",
    "MinMaxNormalizer",
    "SigmoidNormalizer",
]

# Auto-register all built-in nodes from this package
# This must be done AFTER all imports to avoid circular import issues
from cuvis_ai.utils.node_registry import NodeRegistry

NodeRegistry.auto_register_package("cuvis_ai.node")
