"""
Anomaly Detection Nodes.

This module provides anomaly detection nodes for hyperspectral image analysis,
including both statistical methods (RX, LAD) and deep learning approaches
(Deep SVDD). These nodes detect anomalies by comparing test samples against
normal/background reference data.

The module includes:
- Statistical detectors: RX (Reed-Xiaoli) and LAD (Local Anomaly Detector)
- Deep learning: Deep SVDD with projection networks
- Preprocessing: Z-score normalization for statistical stability
- Conversion: Score to logit transformation

See Also
--------
cuvis_ai.anomaly.rx_detector : RX anomaly detection
cuvis_ai.anomaly.lad_detector : LAD anomaly detection
cuvis_ai.anomaly.deep_svdd : Deep SVDD neural network detector
"""

from cuvis_ai.anomaly.deep_svdd import DeepSVDDProjection, ZScoreNormalizerGlobal
from cuvis_ai.anomaly.lad_detector import LADGlobal
from cuvis_ai.anomaly.rx_detector import RXGlobal, RXPerBatch
from cuvis_ai.node.conversion import ScoreToLogit

__all__ = [
    "ZScoreNormalizerGlobal",
    "DeepSVDDProjection",
    "LADGlobal",
    "RXGlobal",
    "RXPerBatch",
    "ScoreToLogit",
]
