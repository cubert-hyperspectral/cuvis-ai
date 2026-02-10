"""
Binary Decision Nodes.

This module provides nodes that convert anomaly scores into binary
decisions (anomaly/normal) using adaptive thresholding strategies.
These nodes are typically used at the end of detection pipelines to
produce final classification results.

See Also
--------
cuvis_ai.deciders.binary_decider : Simple threshold-based binary decisions
cuvis_ai.deciders.two_stage_decider : Two-stage adaptive threshold decision
"""

from cuvis_ai.deciders.two_stage_decider import TwoStageBinaryDecider

__all__ = ["TwoStageBinaryDecider"]
