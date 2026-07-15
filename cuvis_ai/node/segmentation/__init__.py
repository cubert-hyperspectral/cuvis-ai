"""
Segmentation Nodes.

This module provides nodes that turn hyperspectral cubes into per-pixel
foreground/background masks using simple, stateless rules.

See Also
--------
cuvis_ai.node.segmentation.intensity : Intensity-threshold segmentation
"""

from cuvis_ai.node.segmentation.intensity import IntensityThresholdSegmenter

__all__ = ["IntensityThresholdSegmenter"]
