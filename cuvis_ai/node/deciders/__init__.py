"""
Binary Decision Nodes.

This module provides nodes that convert anomaly scores into binary
decisions (anomaly/normal) using adaptive thresholding strategies.
These nodes are typically used at the end of detection pipelines to
produce final classification results.

See Also
--------
cuvis_ai.node.deciders.binary_decider : Simple threshold-based binary decisions
cuvis_ai.node.deciders.frame_score_gate : Gate a score map on its per-frame top-k score
cuvis_ai.node.deciders.multi_range_decider : Multi-range bucketing of score maps
cuvis_ai.node.deciders.two_stage_decider : Two-stage adaptive threshold decision
"""

from cuvis_ai.node.deciders.binary_decider import BinaryDecider, QuantileBinaryDecider
from cuvis_ai.node.deciders.frame_score_gate import FrameScoreGate
from cuvis_ai.node.deciders.multi_range_decider import MultiRangeSlicer
from cuvis_ai.node.deciders.two_stage_decider import TwoStageBinaryDecider

__all__ = [
    "BinaryDecider",
    "FrameScoreGate",
    "MultiRangeSlicer",
    "QuantileBinaryDecider",
    "TwoStageBinaryDecider",
]
