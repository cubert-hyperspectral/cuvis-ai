"""Deprecated: use cuvis_ai.node.anomaly instead. Removed in v0.8."""

import warnings

warnings.warn(
    "'cuvis_ai.anomaly' is deprecated and will be removed in v0.8. "
    "Use 'cuvis_ai.node.anomaly' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cuvis_ai.node.anomaly import (  # noqa: E402, F401
    DeepSVDDProjection,
    LADGlobal,
    RXGlobal,
    RXPerBatch,
    ScoreToLogit,
    ZScoreNormalizerGlobal,
    __all__,
)
