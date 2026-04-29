"""Deprecated: use cuvis_ai.node.anomaly.lad_detector instead. Removed in v0.8."""

import warnings

warnings.warn(
    "'cuvis_ai.anomaly.lad_detector' is deprecated and will be removed in v0.8. "
    "Use 'cuvis_ai.node.anomaly.lad_detector' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cuvis_ai.node.anomaly.lad_detector import *  # noqa: E402, F401, F403
from cuvis_ai.node.anomaly.lad_detector import __all__  # noqa: E402, F401
