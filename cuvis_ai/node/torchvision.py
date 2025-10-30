import functools
from typing import Any

import torch
import torchvision
import torchvision.transforms.v2
from torch import Tensor

from cuvis_ai.node import Node
from cuvis_ai.node.base import BaseTransformation


def _wrap_torchvision_transform(cls):
    class WrappedTorchVisionTransformation(Node, BaseTransformation):
        __doc__ = cls.__doc__
        __module__ = cls.__module__

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.tv_transform = cls(*args, **kwargs)
            self.initialized = self.tv_transform is not None

        def forward(
            self,
            x: Tensor,
            y: Tensor | None = None,
            m: Any = None,
            **_: Any,
        ) -> Tensor:
            if not torch.is_tensor(x):
                raise TypeError(
                    f"{type(self).__name__} expects torch.Tensor input, received {type(x)!r}."
                )

            if x.dim() != 4:
                raise ValueError(
                    f"{type(self).__name__} expects BHWC tensors; received shape {tuple(x.shape)}"
                )

            channels_first = x.movedim(-1, 1)
            transformed = self.tv_transform(channels_first)
            return transformed.movedim(1, -1)

        def fit(self, X: tuple | Tensor):
            pass

        @Node.output_dim.getter
        def output_dim(self) -> tuple[int, int, int]:
            return (-1, -1, -1)

        @Node.input_dim.getter
        def input_dim(self) -> tuple[int, int, int]:
            return (-1, -1, -1)

        def serialize(self, serial_dir: str) -> dict[str, Any]:
            """Serialize this node."""
            if not self.initialized:
                print("Module not fully initialized, skipping output!")
                return dict()

            data_independent = {}
            for name, value in self.tv_transform.__dict__.items():
                if name.startswith("_") or name == "training":
                    continue

                # if not isinstance(value, (bool, int, float, str, tuple, list, enum.Enum)):
                if not isinstance(value, (bool, int, float, str, tuple, list)):
                    continue

                data_independent[name] = value
            return {"params": data_independent}

        def load(self, params: dict, serial_dir: str):
            """Load this node from a serialized graph."""
            self.id = params.get("id")
            self.tv_transform = cls(**params["params"])
            self.initialized = True

    WrappedTorchVisionTransformation.__name__ = cls.__name__
    functools.update_wrapper(WrappedTorchVisionTransformation.__init__, cls.__init__)
    return WrappedTorchVisionTransformation


def _wrap_torchvision_class(cls):
    if issubclass(cls, torchvision.transforms.v2.Transform):
        return _wrap_torchvision_transform(cls)
    else:
        raise ValueError("Called on unsupported class")


def _wrap_torchvision_instance(obj):
    cls = _wrap_torchvision_class(obj.__class__)

    params = obj.get_params()

    return cls(**params)
