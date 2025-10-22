import inspect
from types import ModuleType

import sklearn
import torch
import torchvision
import torchvision.transforms.v2

from cuvis_ai.node.sklearn import _wrap_sklearn_class, _wrap_sklearn_instance
from cuvis_ai.node.skorch import _wrap_torch_class, _wrap_torch_instance
from cuvis_ai.node.torchvision import _wrap_torchvision_class, _wrap_torchvision_instance


def _wrap_class(cls):
    if issubclass(cls, sklearn.base.BaseEstimator):
        return _wrap_sklearn_class(cls)
    elif issubclass(cls, torchvision.transforms.v2.Transform):
        return _wrap_torchvision_class(cls)
    elif issubclass(cls, torch.nn.Module):
        return _wrap_torch_class(cls)
    else:
        raise ValueError("Called on unsupported class")


def _wrap_instance(obj):
    if isinstance(obj, sklearn.base.BaseEstimator):
        return _wrap_sklearn_instance(obj)
    elif isinstance(obj, torchvision.transforms.v2.Transform):
        return _wrap_torchvision_instance(obj)
    elif isinstance(obj, torch.nn.Module):
        return _wrap_torch_instance(obj)
    else:
        raise ValueError("Called on unsupported object")


def make_node(wrapped):
    """Node Wrapper / Decorator. Use to wrap a specific module into a node."""

    if isinstance(wrapped, ModuleType):
        raise NotImplementedError("Currently cannot be wrapped")

    if inspect.isclass(wrapped):
        return _wrap_class(wrapped)
    if isinstance(wrapped, object):
        return _wrap_instance(wrapped)
