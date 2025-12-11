"""Utility functions for AdaCLIP.

Adapted from: https://github.com/caoyunkang/AdaCLIP
"""

from __future__ import annotations

import collections.abc
from collections.abc import Callable
from itertools import repeat
from typing import Any

from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d


def freeze_batch_norm_2d(
    module: nn.Module, module_match: dict | None = None, name: str = ""
) -> nn.Module:
    """Convert BatchNorm2d and SyncBatchNorm layers to FrozenBatchNorm2d.

    Args:
        module: Any PyTorch module.
        module_match: Dictionary of full module names to freeze (all if empty)
        name: Full module name (prefix)

    Returns:
        Module with frozen batch norm layers.
    """
    if module_match is None:
        module_match = {}

    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(
        module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)
    ):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = ".".join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


def _ntuple(n: int) -> Callable[[Any], tuple]:
    """Create a function that converts input to n-tuple."""

    def parse(x) -> tuple:
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)


def to_ntuple(n: int, x) -> tuple:
    """Convert x to n-tuple."""
    return _ntuple(n)(x)


__all__ = [
    "freeze_batch_norm_2d",
    "to_1tuple",
    "to_2tuple",
    "to_3tuple",
    "to_4tuple",
    "to_ntuple",
]
