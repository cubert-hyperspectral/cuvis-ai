"""Deprecated: use cuvis_ai.node.deciders.two_stage_decider instead. Removed in v0.8."""

import warnings

warnings.warn(
    "'cuvis_ai.deciders.two_stage_decider' is deprecated and will be removed in v0.8. "
    "Use 'cuvis_ai.node.deciders.two_stage_decider' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cuvis_ai.node.deciders.two_stage_decider import *  # noqa: E402, F401, F403
