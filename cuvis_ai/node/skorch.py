import functools
import uuid
from pathlib import Path

import numpy as np
import skorch
import torch
import torch.nn as nn

from cuvis_ai.node.base import BaseSupervised, BaseUnsupervised
from cuvis_ai.node.node import Node
from cuvis_ai.utils.dict import remove_prefix
from cuvis_ai.utils.numpy import *
from cuvis_ai.utils.torch import InputDimension, extract_state_dict, guess_input_dimensionalty


class SkorchWrapped:
    pass


def _serialize_skorch_model(obj, cls, data_dir: Path) -> dict:
    data_independent = obj.model_args_no_prefix.copy()
    if not obj.initialized:
        return {"params": data_independent, "code": cls}

    f_params = f"{uuid.uuid4()}.pth"

    sd = obj.net.module_.state_dict()
    torch.save(sd, Path(data_dir) / f_params)
    data_dependend = {"weights": f_params}

    return {"params": data_independent, "state": data_dependend, "code": cls}


def _load_skorch_model(obj, cls, params: dict, data_dir: Path) -> None:
    data_independent_keys = set(remove_prefix(obj.net.get_params(obj), "module__", True).keys())

    params_independent = {key: params["params"][key] for key in data_independent_keys}

    params_independent_with_prefix = {f"module__{k}": v for k, v in params_independent.items()}

    obj.net.set_params(**params_independent_with_prefix)
    obj.model_args = params_independent
    obj.model_args_no_prefix = params_independent_with_prefix

    if "state" not in params.keys():
        return

    data_dependent_keys = {key for key in params["state"].keys()}

    params_dependent = {key: params["state"][key] for key in data_dependent_keys}

    weights = params_dependent["weights"]

    loaded_weights = extract_state_dict(torch.load(weights, weights_only=True), format="torch")

    obj.net.module_.load_state_dict(loaded_weights)
    obj.initialized = True


def _wrap_preprocessor_class(cls):
    pass


def _wrap_supervised_class(cls):
    class SkorchWrappedSupervised(Node, BaseSupervised, SkorchWrapped):
        __doc__ = cls.__doc__
        __module__ = cls.__module__

        def __init__(self, *args, criterion=None, **kwargs):
            super(SkorchWrappedSupervised, self).__init__()
            self.input_size = (-1, -1, -1)
            self.output_size = (-1, -1, -1)

            self.model_args = {f"module__{k}": v for k, v in kwargs.items()}

            self.model_args_no_prefix = {k: v for k, v in kwargs.items()}

            self.criterion = torch.nn.NLLLoss

            self.net = skorch.NeuralNetClassifier(
                module=cls, criterion=self.criterion, train_split=None, **self.model_args
            )
            self.net.initialize()

            self.expected_dim = guess_input_dimensionalty(self.net.module_)

        @Node.input_dim.getter
        def input_dim(self):
            return self.input_size

        @Node.output_dim.getter
        def output_dim(self):
            return self.output_size

        def fit(self, X: np.ndarray, Y: np.ndarray, warm_start=False):
            if self.expected_dim == InputDimension.One:
                flattened_data = flatten_batch_and_spatial(X)
                flattened_label = flatten_labels(Y)
            elif self.expected_dim == InputDimension.Two:
                flattened_data = flatten_spatial(X)
                flattened_label = flatten_labels(Y)

            if warm_start:
                self.net.partial_fit(flattened_data, flattened_label, warm_start=warm_start)
            else:
                self.net.fit(flattened_data, flattened_label, warm_start=warm_start)

            self._input_size = X.shape[-3:]
            self._output_size = X.shape[-3:]
            self.initialized = True

        def forward(self, X: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)
            transformed_data = self.net.forward(self, flattened_data)
            return unflatten_batch_and_spatial(transformed_data, X.shape)

        def serialize(self, data_dir: Path) -> dict:
            return _serialize_skorch_model(self, cls, data_dir)

        def load(self, params: dict, data_dir: Path) -> None:
            return _load_skorch_model(self, cls, params, data_dir)

    SkorchWrappedSupervised.__name__ = cls.__name__
    functools.update_wrapper(SkorchWrappedSupervised.__init__, cls.__init__)
    return SkorchWrappedSupervised


def _wrap_unsupervised_class(cls):
    class SkorchWrappedUnsupervised(Node, BaseUnsupervised, SkorchWrapped):
        __doc__ = cls.__doc__
        __module__ = cls.__module__

        def __init__(self, *args, criterion=None, **kwargs):
            super(SkorchWrappedUnsupervised, self).__init__()
            self._input_size = (-1, -1, -1)
            self._output_size = (-1, -1, -1)

            self.model_args = {f"module__{k}": v for k, v in kwargs.items()}

            self.model_args_no_prefix = {k: v for k, v in kwargs.items()}

            self.criterion = torch.nn.NLLLoss

            self.net = skorch.NeuralNet(
                module=cls,
                criterion=self.criterion,
                train_split=None,
                device="cuda",
                **self.model_args,
            )
            self.net.initialize()

            self.expected_dim = guess_input_dimensionalty(self.net.module_)

        @Node.input_dim.getter
        def input_dim(self):
            return self._input_size

        @Node.output_dim.getter
        def output_dim(self):
            return self._output_size

        @property
        def model(self):
            return self.net.module_

        def fit(self, X: np.ndarray, warm_start=False):
            if self.expected_dim == InputDimension.One:
                flattened_data = flatten_batch_and_spatial(X)
            elif self.expected_dim == InputDimension.Three:
                flattened_data = np.moveaxis(X, -1, -3)
            else:
                raise RuntimeError("Could not estimate needed input Dimension")

            if warm_start:
                self.net.partial_fit(flattened_data, flattened_data)
            else:
                self.net.fit(flattened_data, flattened_data)

            self._input_size = X.shape[-3:]
            self._output_size = X.shape[-3:]
            self.initialized = True

        def forward(self, X: np.ndarray):
            import time

            if self.expected_dim == InputDimension.One:
                flattened_data = flatten_batch_and_spatial(X)
            elif self.expected_dim == InputDimension.Three:
                flattened_data = np.moveaxis(X, -1, -3)
            flattened_data = torch.from_numpy(flattened_data)
            start_time = time.time()
            transformed_data = self.net.forward(flattened_data)
            print("--- %s seconds ---" % (time.time() - start_time))
            transformed_data = transformed_data.to("cpu").numpy()
            if self.expected_dim == InputDimension.One:
                return unflatten_batch_and_spatial(transformed_data, X.shape)
            elif self.expected_dim == InputDimension.Three:
                return np.moveaxis(transformed_data, -3, -1)

        def serialize(self, data_dir: Path) -> dict:
            return _serialize_skorch_model(self, cls, data_dir)

        def load(self, params: dict, data_dir: Path) -> None:
            return _load_skorch_model(self, cls, params, data_dir)

    SkorchWrappedUnsupervised.__name__ = cls.__name__
    functools.update_wrapper(SkorchWrappedUnsupervised.__init__, cls.__init__)
    return SkorchWrappedUnsupervised


def _wrap_torch_class(cls):
    if issubclass(cls, nn.Module):
        return _wrap_unsupervised_class(cls)
    else:
        raise ValueError("Called on unsupported class")


def _wrap_torch_instance(obj):
    cls = _wrap_torch_class(obj.__class__)

    params = obj.get_params()

    return cls(**params)
