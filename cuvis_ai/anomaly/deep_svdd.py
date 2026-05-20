"""Deprecated: use cuvis_ai.node.anomaly.deep_svdd instead. Removed in v0.8."""

import warnings

warnings.warn(
    "'cuvis_ai.anomaly.deep_svdd' is deprecated and will be removed in v0.8. "
    "Use 'cuvis_ai.node.anomaly.deep_svdd' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cuvis_ai.node.anomaly.deep_svdd import *  # noqa: E402, F401, F403
from cuvis_ai.node.anomaly.deep_svdd import __all__  # noqa: E402, F401
