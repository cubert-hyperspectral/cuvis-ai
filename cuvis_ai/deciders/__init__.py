"""Deprecated: use cuvis_ai.node.deciders instead. Removed in v0.8."""

import warnings

warnings.warn(
    "'cuvis_ai.deciders' is deprecated and will be removed in v0.8. "
    "Use 'cuvis_ai.node.deciders' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cuvis_ai.node.deciders import (  # noqa: E402, F401
    BinaryDecider,
    QuantileBinaryDecider,
    TwoStageBinaryDecider,
    __all__,
)
