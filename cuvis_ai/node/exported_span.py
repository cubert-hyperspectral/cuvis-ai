"""Run a span that ``export-pipeline`` wrote to ONNX in place of the nodes it stands for."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_core.node import Node
from cuvis_ai_core.utils.trt_engine import cached_file_md5

#: A relative ``onnx_path`` is looked up under this folder first.
DEPLOY_DIR_ENV = "CUVIS_AI_DEPLOY_DIR"
#: The sidecar format this node reads (``export-pipeline`` writes it).
SIDECAR_FORMAT = 1

_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}
_NUMPY: dict[torch.dtype, type] = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.bool: np.bool_,
}


def _onnxruntime() -> ModuleType:
    """The ``onnxruntime`` module, or an ImportError that names the extra."""
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError(
            "ExportedSpan needs ONNX Runtime: run `uv sync --extra onnx` in a cuvis-ai "
            "checkout, or list the `cuvis_ai_onnx` plugin in the pipeline so a composed "
            "environment installs cuvis-ai[onnx]."
        ) from exc
    return onnxruntime


def _pipeline_source_dir() -> Path | None:
    """The folder of the pipeline file being loaded, when the loader says so."""
    try:
        from cuvis_ai_core.pipeline.factory import pipeline_source_dir
    except ImportError:
        return None
    return pipeline_source_dir()


def resolve_deploy_path(onnx_path: str | os.PathLike[str]) -> Path:
    """Where ``onnx_path`` is.

    An absolute path is taken as given. A relative one is looked up under
    ``CUVIS_AI_DEPLOY_DIR`` when that variable is set, then under the folder of the
    pipeline file being loaded (the loaders of cuvis-ai-core say where that is), then
    under the working directory. The error names what was tried.
    """
    path = Path(onnx_path)
    if path.is_absolute():
        if path.is_file():
            return path
        raise FileNotFoundError(f"ExportedSpan: {path} does not exist.")
    bases: list[tuple[str, Path]] = []
    env = os.environ.get(DEPLOY_DIR_ENV)
    if env:
        bases.append((DEPLOY_DIR_ENV, Path(env)))
    source = _pipeline_source_dir()
    if source is not None:
        bases.append(("the pipeline file's folder", Path(source)))
    bases.append(("the working directory", Path.cwd()))
    for _, base in bases:
        candidate = base / path
        if candidate.is_file():
            return candidate.resolve()
    tried = "; ".join(f"{label} ({base})" for label, base in bases)
    raise FileNotFoundError(
        f"ExportedSpan: {path} was not found under {tried}. Set {DEPLOY_DIR_ENV} to the "
        "deploy folder, load the deploy yaml from its folder, or write an absolute onnx_path."
    )


def read_sidecar(onnx_path: Path) -> dict[str, Any]:
    """The sidecar ``<onnx_path>.json`` next to the file, checked for its format."""
    sidecar_path = onnx_path.with_name(onnx_path.name + ".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"ExportedSpan: the sidecar {sidecar_path.name} is missing next to {onnx_path}; "
            "export-pipeline writes it with the ONNX file."
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    fmt = sidecar.get("format")
    if fmt != SIDECAR_FORMAT:
        raise ValueError(
            f"ExportedSpan: {sidecar_path.name} has sidecar format {fmt}; this cuvis-ai "
            f"reads format {SIDECAR_FORMAT}. Export with a matching cuvis-ai-core or update "
            "cuvis-ai."
        )
    return sidecar


def _batched_spec(entry: dict[str, Any]) -> PortSpec:
    """A port whose first dimension is the batch; the rest is fixed by the export."""
    shape = list(entry["shape"])
    return PortSpec(
        dtype=_DTYPES[entry["dtype"]],
        shape=(-1, *shape[1:]) if shape else (),
        description=f"{entry['port']} of the exported span, {entry['dtype']} {shape}",
    )


def _fixed_spec(entry: dict[str, Any]) -> PortSpec:
    """A constant's port: the value baked into the graph, checked at run time."""
    return PortSpec(
        dtype=_DTYPES[entry["dtype"]],
        shape=tuple(entry["shape"]),
        description=f"{entry['port']}, baked into the exported span; must equal that value",
    )


class ExportedSpan(Node):
    """Run a span exported to ONNX in place of the nodes it was cut from.

    ``export-pipeline`` (cuvis-ai-core) writes ``<name>.onnx``, the sidecar
    ``<name>.onnx.json`` and a deploy yaml in which this node stands where the span's
    nodes were. The ports come from the sidecar: one input per span input, one per
    constant baked into the graph (its run-time value must equal the baked one), one
    output per span output. Shapes are fixed at the export's example shapes with a free
    batch dimension; the node runs the graph once per batch element. The ONNX file must
    hash to the sidecar's ``onnx_md5``.

    ONNX Runtime runs on the device the pipeline moved the node to (``device`` overrides
    it). On CUDA the torch tensors are bound to the session directly, and a CUDA request
    whose session lands on the CPU provider is refused. Inference, validation and test
    stages run; a training stage is refused: train the canonical yaml and export again.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset({NodeTag.INFERENCE})

    def __init__(
        self,
        onnx_path: str,
        backend: str = "onnxruntime",
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        if backend == "tensorrt":
            raise NotImplementedError(
                "ExportedSpan backend 'tensorrt' is not available yet; use 'onnxruntime'."
            )
        if backend != "onnxruntime":
            raise ValueError(f"ExportedSpan backend must be 'onnxruntime', got {backend!r}.")
        resolved = resolve_deploy_path(onnx_path)
        sidecar = read_sidecar(resolved)
        span = sidecar["span"]
        # The ports are per file, so the instance shadows the empty class-level dicts
        # before the base class creates the port proxies.
        self.INPUT_SPECS = {
            **{entry["name"]: _batched_spec(entry) for entry in span["inputs"]},
            **{entry["name"]: _fixed_spec(entry) for entry in span["constants"]},
        }
        self.OUTPUT_SPECS = {entry["name"]: _batched_spec(entry) for entry in span["outputs"]}
        super().__init__(onnx_path=onnx_path, backend=backend, device=device, **kwargs)
        self.onnx_path = onnx_path
        self.backend = backend
        self.device_override = device
        self.resolved_onnx_path = resolved
        self.sidecar = sidecar
        self._inputs: list[dict[str, Any]] = list(span["inputs"])
        self._outputs: list[dict[str, Any]] = list(span["outputs"])
        self._constants: dict[str, tuple[str, Tensor]] = {
            entry["name"]: (
                entry["port"],
                torch.tensor(entry["value"], dtype=_DTYPES[entry["dtype"]]).reshape(entry["shape"]),
            )
            for entry in span["constants"]
        }
        # Follows ``pipeline.to(device)``; not part of the state dict.
        self.register_buffer("_device_anchor", torch.zeros(0), persistent=False)
        self.session: Any = None
        self._session_device: torch.device | None = None

    # ------------------------------------------------------------------ lifecycle

    def prepare_inference(self) -> None:
        """Check the file against the sidecar and open the session on the node's device."""
        self._check_files()
        device = self._target_device()
        if self.session is not None and self._session_device == device:
            return
        ort = _onnxruntime()
        if device.type == "cuda":
            # The CUDA and cuDNN libraries come with torch; ONNX Runtime finds them once
            # torch is imported, and preload_dlls() makes that lookup explicit.
            preload = getattr(ort, "preload_dlls", None)
            if preload is not None:
                preload(cuda=True, cudnn=True)
            providers: list[Any] = [("CUDAExecutionProvider", {"device_id": device.index})]
        else:
            providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(self.resolved_onnx_path), providers=providers)
        active = list(session.get_providers())
        if device.type == "cuda" and (not active or active[0] != "CUDAExecutionProvider"):
            raise RuntimeError(
                f"Node {self.name!r} was asked to run on {device} but ONNX Runtime opened "
                f"the session with {active}: the CUDA provider is not available (it needs "
                "onnxruntime-gpu with its CUDA libraries). Set device='cpu' on the node to "
                "run on the CPU on purpose."
            )
        self.session = session
        self._session_device = device

    def cleanup(self) -> None:
        """Drop the session."""
        self.session = None
        self._session_device = None

    def _target_device(self) -> torch.device:
        """The device the session runs on: the override, else where the pipeline moved us."""
        device = (
            torch.device(self.device_override)
            if self.device_override
            else self._device_anchor.device
        )
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", 0)
        return device

    def _check_files(self) -> None:
        """The ONNX file (and its external data) must hash to the sidecar's values."""
        checks = [(self.resolved_onnx_path, "onnx_md5")]
        data_name = self.sidecar.get("onnx_data")
        if data_name:
            checks.append((self.resolved_onnx_path.with_name(data_name), "onnx_data_md5"))
        for path, key in checks:
            expected = self.sidecar.get(key)
            actual = cached_file_md5(path)
            if actual != expected:
                raise RuntimeError(
                    f"Node {self.name!r}: {path.name} hashes to {actual}, the sidecar's {key} "
                    f"is {expected}; the file was edited or replaced after the export. "
                    "Export the pipeline again."
                )

    # ------------------------------------------------------------------ forward

    def forward(self, *, context: Context | None = None, **ports: Any) -> dict[str, Tensor]:
        """Run the exported graph once per batch element."""
        if context is not None and context.stage == ExecutionStage.TRAIN:
            raise RuntimeError(
                f"Node {self.name!r} runs an exported span and cannot train. Train the "
                "canonical pipeline yaml and export it again."
            )
        self._check_constants(ports)
        tensors = self._check_inputs(ports)
        if self.session is None or self._session_device != self._target_device():
            self.prepare_inference()
        device = self._session_device
        assert device is not None
        batch = tensors[0].shape[0] if tensors else 1
        runs = [self._run_one([t[i : i + 1] for t in tensors], device) for i in range(batch)]
        return {
            entry["name"]: torch.cat([run[k] for run in runs], dim=0)
            for k, entry in enumerate(self._outputs)
        }

    def _check_constants(self, ports: dict[str, Any]) -> None:
        """Every baked constant must arrive with the value it was baked with."""
        for name, (port, baked) in self._constants.items():
            value = ports.get(name)
            if not isinstance(value, Tensor):
                raise ValueError(
                    f"Node {self.name!r}: input {name!r} ({port}) must be a tensor, got "
                    f"{type(value).__name__}."
                )
            got = value.detach().cpu()
            if tuple(got.shape) != tuple(baked.shape) or not torch.equal(
                got.to(baked.dtype), baked
            ):
                raise ValueError(
                    f"Node {self.name!r}: input {name!r} differs from the value the span was "
                    f"exported with ({port} = {baked.flatten()[:8].tolist()}, shape "
                    f"{tuple(baked.shape)}); this run feeds {got.flatten()[:8].tolist()}, shape "
                    f"{tuple(got.shape)}. Export the pipeline again for this configuration."
                )

    def _check_inputs(self, ports: dict[str, Any]) -> list[Tensor]:
        """The span inputs, at the exported dtypes and shapes (batch free)."""
        tensors: list[Tensor] = []
        for entry in self._inputs:
            name = entry["name"]
            value = ports.get(name)
            if not isinstance(value, Tensor):
                raise ValueError(
                    f"Node {self.name!r}: input {name!r} ({entry['port']}) must be a tensor, "
                    f"got {type(value).__name__}."
                )
            expected_dtype = _DTYPES[entry["dtype"]]
            if value.dtype != expected_dtype:
                raise ValueError(
                    f"Node {self.name!r}: input {name!r} has dtype {value.dtype}; the span was "
                    f"exported with {entry['dtype']}."
                )
            exported = tuple(entry["shape"])
            if tuple(value.shape[1:]) != exported[1:]:
                raise ValueError(
                    f"Node {self.name!r}: input {name!r} has shape {tuple(value.shape)}; the "
                    f"span was exported for {exported} (batch free). Export the pipeline "
                    "again at this resolution."
                )
            tensors.append(value)
        return tensors

    def _run_one(self, parts: list[Tensor], device: torch.device) -> list[Tensor]:
        """One batch element through the session, outputs on the input tensors' device."""
        if device.type == "cuda":
            return self._run_bound(parts, device)
        feeds = {
            entry["tensor"]: part.detach().cpu().contiguous().numpy()
            for entry, part in zip(self._inputs, parts, strict=True)
        }
        outputs = self.session.run([entry["tensor"] for entry in self._outputs], feeds)
        home = parts[0].device if parts else torch.device("cpu")
        return [torch.from_numpy(np.ascontiguousarray(out)).to(home) for out in outputs]

    def _run_bound(self, parts: list[Tensor], device: torch.device) -> list[Tensor]:
        """One batch element through the session with the tensors bound on the device."""
        binding = self.session.io_binding()
        bound_inputs = []
        for entry, part in zip(self._inputs, parts, strict=True):
            tensor = part.detach().to(device).contiguous()
            bound_inputs.append(tensor)
            binding.bind_input(
                name=entry["tensor"],
                device_type="cuda",
                device_id=device.index,
                element_type=_NUMPY[tensor.dtype],
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )
        outputs = []
        for entry in self._outputs:
            tensor = torch.empty(
                tuple(entry["shape"]), dtype=_DTYPES[entry["dtype"]], device=device
            )
            outputs.append(tensor)
            binding.bind_output(
                name=entry["tensor"],
                device_type="cuda",
                device_id=device.index,
                element_type=_NUMPY[tensor.dtype],
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )
        self.session.run_with_iobinding(binding)
        return outputs
