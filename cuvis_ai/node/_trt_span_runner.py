"""Run a serialized TensorRT engine with several inputs on torch CUDA tensors.

:class:`cuvis_ai_core.utils.trt_engine.TensorRTEngine` assumes one engine input; an
exported span has one per span input plus one per baked constant. :class:`TrtSpanRunner`
binds every graph input and output by name instead of assuming a single one, keeps its
own output buffers and copies them out on every call so a caller never sees a buffer the
next call will overwrite.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

from cuvis_ai_core.utils.trt_engine import import_tensorrt, trt_logger

_DTYPE_NAMES: tuple[tuple[str, torch.dtype], ...] = (
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
    ("int32", torch.int32),
    ("int64", torch.int64),
    ("int8", torch.int8),
    ("uint8", torch.uint8),
    ("bool", torch.bool),
)


class TrtSpanRunner:
    """Run a multi-input, multi-output TensorRT engine on torch CUDA tensors.

    The engine runs on its own CUDA stream (TensorRT adds a host synchronisation to
    every call on the default stream), ordered after the work already queued on
    torch's current stream, and torch's current stream waits for it, so torch ops
    that follow see the results in order. Each call copies the output buffers into
    fresh tensors before returning them, so the next call's writes never reach a
    tensor a caller still holds.

    Parameters
    ----------
    path : str or os.PathLike
        The engine file.
    device : torch.device or str
        The CUDA device it runs on; ``"cuda"`` without an index is the current
        device at construction, and the engine stays there.
    rebuild_hint : str
        Appended to the error when the engine does not load on this machine (for
        example the command that builds it by hand).
    """

    def __init__(
        self,
        path: str | os.PathLike,
        device: torch.device | str = "cuda",
        rebuild_hint: str = "",
    ) -> None:
        trt = import_tensorrt()
        self.path = str(path)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        dtypes = {getattr(trt, name): dtype for name, dtype in _DTYPE_NAMES if hasattr(trt, name)}
        self._runtime = trt.Runtime(trt_logger(trt))
        with open(self.path, "rb") as f, torch.cuda.device(self.device):
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
            self._stream = torch.cuda.Stream(self.device)
        if self._engine is None:
            raise RuntimeError(
                f"TensorRT could not load {self.path}: an engine only runs on the GPU and "
                f"TensorRT version it was built with; rebuild it on this machine. "
                f"{rebuild_hint}".strip()
            )
        self._context = self._engine.create_execution_context()
        self.input_specs: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
        self._outputs: dict[str, Tensor] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            dtype = dtypes[self._engine.get_tensor_dtype(name)]
            shape = tuple(self._engine.get_tensor_shape(name))
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_specs[name] = (dtype, shape)
            else:
                buf = torch.empty(shape, dtype=dtype, device=self.device)
                self._outputs[name] = buf
                self._context.set_tensor_address(name, buf.data_ptr())
        self._held: dict[str, Tensor] = {}

    def bind_constant(self, name: str, value: Tensor) -> None:
        """Bind a graph input once, for the life of the runner (a baked constant).

        A constant the builder folded into the graph at compile time (TensorRT, like
        the ONNX exporter, may do this for a value that never changes) has no tensor
        of this name on the engine; there is nothing to bind, and this is a no-op.

        Parameters
        ----------
        name : str
            The engine's tensor name for this input.
        value : Tensor
            Its value; moved to the engine's device and dtype and kept alive.
        """
        if name not in self.input_specs:
            return
        dtype, shape = self.input_specs[name]
        tensor = value.detach().to(device=self.device, dtype=dtype).contiguous()
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{self.path}: constant {name!r} takes shape {shape}, got {tuple(tensor.shape)}."
            )
        self._context.set_tensor_address(name, tensor.data_ptr())
        self._held[name] = tensor  # the engine reads it on every call; keep it alive

    def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the engine on the named inputs; constants bound earlier stay bound.

        Parameters
        ----------
        inputs : dict of str to Tensor
            One entry per non-constant graph input, by its engine tensor name, at
            the engine's static shape.

        Returns
        -------
        dict of str to Tensor
            One fresh tensor per graph output, by its engine tensor name; safe to
            keep across the next call.
        """
        bound: dict[str, Tensor] = {}
        for name, value in inputs.items():
            if name not in self.input_specs:
                raise ValueError(
                    f"{self.path}: no engine input named {name!r}; the engine has "
                    f"{sorted(self.input_specs)}."
                )
            dtype, shape = self.input_specs[name]
            tensor = value.detach().to(device=self.device, dtype=dtype).contiguous()
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{self.path}: input {name!r} takes shape {shape}, got {tuple(tensor.shape)}."
                )
            self._context.set_tensor_address(name, tensor.data_ptr())
            bound[name] = tensor
        current = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)  # inputs written, previous outputs consumed
        with torch.cuda.device(self.device):
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError(f"TensorRT execution failed for {self.path}.")
        current.wait_stream(self._stream)  # everything queued after this sees the outputs
        self._held.update(bound)  # the engine read these asynchronously; keep until reused
        return {name: buf.clone() for name, buf in self._outputs.items()}

    @property
    def output_names(self) -> tuple[str, ...]:
        """The engine's output tensor names, in no particular order."""
        return tuple(self._outputs)
