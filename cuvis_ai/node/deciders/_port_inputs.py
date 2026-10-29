"""Validation of threshold values that reach a decider as hparams or ``[1]`` input ports.

A pipeline yaml or a GUI can hand a decider a string, a bool, a NaN or a tensor of the wrong
shape. Silent coercion would decide at a number nobody chose, so every refusal names the
hparam or the port, and a GUI learns which control sent the bad value.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, overload

from torch import Tensor


def optional_finite(name: str, value: Any) -> float | None:
    """Return ``value`` as a float, pass ``None`` through, refuse anything else by name."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number or None, got {value!r}")
    return float(value)


@overload
def call_threshold(name: str, value: Tensor | None, fallback: float) -> float: ...


@overload
def call_threshold(name: str, value: Tensor | None, fallback: float | None) -> float | None: ...


def call_threshold(name: str, value: Tensor | None, fallback: float | None) -> float | None:
    """The threshold for this call: the ``[1]`` input when one is given, else ``fallback``.

    An input of another shape or a non-finite value is refused by port name.
    """
    if value is None:
        return fallback
    if tuple(value.shape) != (1,):
        raise ValueError(f"{name} input must have shape [1], got {list(value.shape)}")
    return optional_finite(name, float(value.item()))
