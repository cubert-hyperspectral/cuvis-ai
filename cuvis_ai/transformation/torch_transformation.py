import pickle as pk
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cuvis_ai.node import Node
from cuvis_ai.node.base import BaseTransformation


class TorchTransformation(Node, BaseTransformation):
    """Node representing a transformation of data using a pytorch function.

    Parameters
    ---------
    function_name : str,optional
        The name of the pytorch function to use. Almost any function available from the torch module can be used.
    operand_b : Any,optional
        A constant value to pass into the function alongside the regular input data.
    kwargs : Dict
        Any additional keyword arguments will be passed to the pytorch function anytime it is called.
    """

    def __init__(self, function_name: str | None = None, *, operand_b: Any | None = None, **kwargs):
        super().__init__()
        self.op_name = function_name
        self.b = operand_b
        self.fun_kwargs = kwargs

        if self.op_name is not None:
            self.initialized = True
            self.fun = getattr(torch, self.op_name)
        else:
            self.fun = None
            self.initialized = False

    def forward(self, X: np.ndarray, Y: np.ndarray | None = None):
        """Apply the pytorch method :arg:`function_name` on :arg:`X`.
        This node basically runs `torch.<function_name>(X, Y)`.

        Parameters
        ----------
        X : np.ndarray
            The first operand for the pytorch method.
        Y : np.ndarray, optional
            The second operand for the pytorch method.

        Returns
        -------
        Any, np.ndarray
            Returns the result of the pytorch method and any additional labels or metadata passed along with :arg:`X`
        """
        if Y is not None and self.b is not None:
            raise ValueError(
                f"TorchTransformation with operation '{self.op_name}' was given a constant value and a second operand!"
                "\nTorchTransformation can have none or one of either, but must not have both."
            )

        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X)

        if isinstance(Y, np.ndarray):
            Y = torch.as_tensor(Y)

        try:
            if Y is not None:
                return self.fun(X, Y, **self.fun_kwargs).numpy()
            elif self.b is not None:
                return self.fun(X, self.b, **self.fun_kwargs).numpy()
            else:
                return self.fun(X, **self.fun_kwargs).numpy()
        except RuntimeError as re:
            raise RuntimeError(
                f"TorchTransformation with operation '{self.op_name}' was called with non-matching input and "
                f"{'constant ' if self.b is not None else ''}second operand shapes!\nPyTorch reports: '{re}'"
            )

    def check_output_dim(self, X: Any, Y: Any | None = None):
        pass

    def check_input_dim(self, X: Iterable, Y: Iterable | None = None):
        try:
            self.forward(X, Y)
        except RuntimeError:
            assert False

    @Node.output_dim.getter
    def output_dim(self) -> tuple[int, int, int]:
        return -1, -1, -1

    @Node.input_dim.getter
    def input_dim(self) -> tuple[int, int, int]:
        return -1, -1, -1

    def serialize(self, serial_dir: str) -> str:
        """Serialize this node and save to :arg:`serial_dir`."""
        if not self.initialized:
            print("Module not fully initialized, skipping output!")
            return

        blob = (self.b, self.fun_kwargs)
        blobfile_name = f"{hash(str(blob))}_torchtransformation.pkl"
        with open(Path(serial_dir) / blobfile_name, "wb") as f:
            pk.dump(blob, f)

        data = {
            "id": self.id,
            "op_name": self.op_name,
            "transformation_blob": blobfile_name,
        }
        return data

    def load(self, params: dict, serial_dir: str):
        """Load this node from a serialized graph."""
        self.id = params.get("id")
        with open(Path(serial_dir) / params.get("transformation_blob"), "rb") as f:
            self.b, self.fun_kwargs = pk.load(f)
        self.op_name = params.get("op_name")
        self.fun = getattr(torch, self.op_name)
        self.initialized = True
