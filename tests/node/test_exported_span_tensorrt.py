"""ExportedSpan's TensorRT backend: a fake ``tensorrt`` module stands in for the real one.

The engine is never actually compiled here; ``FakeEngine`` writes known values into the
tensor addresses the node binds, the way a real TensorRT context would, so the tests check
the plumbing (which inputs and constants get bound, that the engine is built once and
reused, that outputs are copied out of reused buffers) without CUDA or a TensorRT install.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from torch import nn

from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.utils import trt_engine

pytestmark = pytest.mark.unit

WAVELENGTHS = [450.0, 550.0, 650.0]


class _Span(nn.Module):
    """The span the deploy folder stands for: scale per band, mean over bands, max."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor([0.5, 1.0, 2.0]))

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = (data * self.scale).mean(dim=-1, keepdim=True)
        return scores, scores.flatten(1).max(dim=1).values


def _write_deploy(folder: Path, name: str = "span") -> dict:
    """An ONNX file plus a format 1 sidecar for ``_Span``, like export-pipeline writes."""
    folder.mkdir(parents=True, exist_ok=True)
    onnx_path = folder / f"{name}.onnx"
    torch.onnx.export(
        _Span().eval(),
        (torch.rand(1, 4, 4, 3),),
        str(onnx_path),
        dynamo=True,
        opset_version=18,
        input_names=["norm.data"],
        output_names=["det.scores", "det.anomaly_score"],
        verbose=False,
    )
    sidecar = {
        "format": 1,
        "pipeline": "tiny",
        "span": {
            "node_name": "exported_span",
            "nodes": [
                {"name": "norm", "class_name": "tests.Norm"},
                {"name": "sel", "class_name": "tests.Select"},
                {"name": "det", "class_name": "tests.Detect"},
            ],
            "inputs": [
                {
                    "port": "norm.data",
                    "name": "data",
                    "tensor": "norm.data",
                    "dtype": "float32",
                    "shape": [1, 4, 4, 3],
                }
            ],
            "outputs": [
                {
                    "port": "det.scores",
                    "name": "scores",
                    "tensor": "det.scores",
                    "dtype": "float32",
                    "shape": [1, 4, 4, 1],
                },
                {
                    "port": "det.anomaly_score",
                    "name": "anomaly_score",
                    "tensor": "det.anomaly_score",
                    "dtype": "float32",
                    "shape": [1],
                },
            ],
            "constants": [
                {
                    "port": "sel.wavelengths",
                    "name": "wavelengths",
                    "tensor": "onnx::wavelengths_const",
                    "dtype": "float32",
                    "shape": [3],
                    "value": WAVELENGTHS,
                }
            ],
        },
        "exporter": {
            "kind": "dynamo",
            "opset": 18,
            "torch": torch.__version__,
            "cuvis_ai_core": "0.19.0",
        },
        "fingerprint": "0123456789abcdef",
        "onnx_md5": hashlib.md5(onnx_path.read_bytes()).hexdigest(),  # noqa: S324
        "parity": {"examples": 1, "atol": 1e-5, "rtol": 1e-4, "max_abs_diff": {}},
    }
    (folder / f"{name}.onnx.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


@pytest.fixture
def deploy(tmp_path) -> Path:
    folder = tmp_path / "deploy"
    _write_deploy(folder)
    return folder


def _node(folder: Path, **hparams):
    from cuvis_ai.node.exported_span import ExportedSpan

    hparams.setdefault("device", "cuda")
    return ExportedSpan(
        onnx_path=str(folder / "span.onnx"), name="exported_span", backend="tensorrt", **hparams
    )


def _inference() -> Context:
    return Context(stage=ExecutionStage.INFERENCE)


def _wavelengths() -> torch.Tensor:
    return torch.tensor(WAVELENGTHS)


# ------------------------------------------------------------------ the fake tensorrt


class _BuilderFlag:
    TF32, FP16 = "TF32", "FP16"


class _Logger:
    WARNING = 1

    def __init__(self, level):
        self.level = level


class _TensorIOMode:
    INPUT, OUTPUT = "INPUT", "OUTPUT"


class FakeEngine:
    """Stands in for a deserialized TensorRT engine: fixed I/O tensors, known outputs.

    ``scores`` is written as ``data`` scaled by the node's buffer and reduced over the
    last axis; ``anomaly_score`` as its flattened max. Reading the tensor addresses the
    runner bound lets the test check the plumbing bit for bit, without a real engine.
    """

    def __init__(self) -> None:
        self.io = [
            ("norm.data", _TensorIOMode.INPUT, torch.float32, (1, 4, 4, 3)),
            ("onnx::wavelengths_const", _TensorIOMode.INPUT, torch.float32, (3,)),
            ("det.scores", _TensorIOMode.OUTPUT, torch.float32, (1, 4, 4, 1)),
            ("det.anomaly_score", _TensorIOMode.OUTPUT, torch.float32, (1,)),
        ]
        self.num_io_tensors = len(self.io)
        self.execute_calls = 0

    def get_tensor_name(self, i):
        return self.io[i][0]

    def get_tensor_mode(self, name):
        return next(mode for n, mode, _, _ in self.io if n == name)

    def get_tensor_dtype(self, name):
        dtype = next(dtype for n, _, dtype, _ in self.io if n == name)
        return {torch.float32: "float32"}[dtype]

    def get_tensor_shape(self, name):
        return next(shape for n, _, _, shape in self.io if n == name)

    def create_execution_context(self):
        return FakeContext(self)


class FakeContext:
    """Stands in for an execution context: records bound addresses, fills outputs on run."""

    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine
        self.addresses: dict[str, int] = {}
        self._tensor_by_ptr: dict[int, torch.Tensor] = {}

    def set_tensor_address(self, name, ptr):
        self.addresses[name] = ptr

    def _read(self, name: str) -> torch.Tensor:
        shape = next(s for n, _, _, s in self.engine.io if n == name)
        ptr = self.addresses[name]
        # The pointer identifies the live tensor through the test's own bookkeeping.
        return _PTR_REGISTRY[ptr].reshape(shape)

    def execute_async_v3(self, stream):
        self.engine.execute_calls += 1
        data = self._read("norm.data")
        scale = torch.tensor([0.5, 1.0, 2.0])
        scores = (data * scale).mean(dim=-1, keepdim=True)
        anomaly_score = scores.flatten(1).max(dim=1).values
        _PTR_REGISTRY[self.addresses["det.scores"]].copy_(scores.reshape(-1))
        _PTR_REGISTRY[self.addresses["det.anomaly_score"]].copy_(anomaly_score.reshape(-1))
        return True


#: ``data_ptr() -> flat tensor view``, refreshed by a hook on every bind so FakeContext
#: can read and write through the same addresses the runner uses.
_PTR_REGISTRY: dict[int, torch.Tensor] = {}


@pytest.fixture
def fake_cuda(monkeypatch):
    """CUDA calls the runner needs, without a GPU: tensors tagged "cuda" live on the CPU.

    This machine runs its tests with ``CUDA_VISIBLE_DEVICES=-1``, so no real CUDA tensor can
    be allocated; ``torch.empty`` and ``Tensor.to`` are wrapped to allocate on the CPU
    whenever a "cuda" device is asked for, while the node and the runner still see and
    branch on ``device.type == "cuda"`` exactly as they would on a real GPU.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda *_a, **_k: _NullCtx())
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Fake GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (0, 0))

    orig_empty = torch.empty
    orig_to = torch.Tensor.to

    def fake_empty(*args, **kwargs):
        if kwargs.get("device") is not None and torch.device(kwargs["device"]).type == "cuda":
            kwargs["device"] = "cpu"
        out = orig_empty(*args, **kwargs)
        _register_ptr(out)
        return out

    def fake_to(self, *args, **kwargs):
        if args and isinstance(args[0], (torch.device, str)):
            if torch.device(args[0]).type == "cuda":
                args = ("cpu", *args[1:])
        if kwargs.get("device") is not None and torch.device(kwargs["device"]).type == "cuda":
            kwargs["device"] = "cpu"
        out = orig_to(self, *args, **kwargs)
        _register_ptr(out)
        return out

    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(torch.Tensor, "to", fake_to)

    class _FakeStream:
        def __init__(self, device=None):
            self.device = device

        def wait_stream(self, other):
            pass

        @property
        def cuda_stream(self):
            return 0

    class _FakeCurrentStream:
        def wait_stream(self, other):
            pass

    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _FakeCurrentStream())
    return None


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _register_ptr(tensor: torch.Tensor) -> None:
    _PTR_REGISTRY[tensor.data_ptr()] = tensor.reshape(-1)


@pytest.fixture
def fake_trt(monkeypatch, fake_cuda):
    """A minimal ``tensorrt`` module: ``Runtime.deserialize_cuda_engine`` returns ``FakeEngine``."""
    _PTR_REGISTRY.clear()
    built = {"count": 0}

    class _Runtime:
        def __init__(self, logger):
            pass

        def deserialize_cuda_engine(self, data):
            built["count"] += 1
            return FakeEngine()

    class _Builder:
        def __init__(self, logger):
            pass

        def create_network(self, flags):
            return object()

        def create_builder_config(self):
            return _Config()

        def build_serialized_network(self, network, config):
            return b"engine-bytes"

    class _Config:
        def __init__(self):
            self.flags = {_BuilderFlag.TF32}

        def set_flag(self, flag):
            self.flags.add(flag)

        def clear_flag(self, flag):
            self.flags.discard(flag)

        def get_flag(self, flag):
            return flag in self.flags

    class _OnnxParser:
        def __init__(self, network, logger):
            self.num_errors = 0

        def parse_from_file(self, path):
            return True

        def get_error(self, i):
            return "bad node"

    trt = types.SimpleNamespace(
        __version__="10.15.1.29",
        BuilderFlag=_BuilderFlag,
        Logger=_Logger,
        Builder=_Builder,
        OnnxParser=_OnnxParser,
        Runtime=_Runtime,
        TensorIOMode=_TensorIOMode,
        float32="float32",
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(trt_engine, "_LOGGER", None)
    trt.built = built
    return trt


@pytest.fixture
def engine_dir(tmp_path, monkeypatch):
    path = tmp_path / "engines"
    monkeypatch.setenv(trt_engine.ENGINE_DIR_ENV, str(path))
    return path


# ----- backend selection and the not-yet-implemented message is gone -------------------


def test_the_onnxruntime_backend_is_unaffected(deploy):
    from cuvis_ai.node.exported_span import ExportedSpan

    node = ExportedSpan(onnx_path=str(deploy / "span.onnx"), name="exported_span", device="cpu")
    assert node.backend == "onnxruntime"


def test_an_unknown_backend_is_still_refused(deploy):
    from cuvis_ai.node.exported_span import ExportedSpan

    with pytest.raises(ValueError, match="backend"):
        ExportedSpan(onnx_path=str(deploy / "span.onnx"), name="exported_span", backend="tvm")


# ----- refusals that do not need a real engine ------------------------------------------


def test_cpu_device_is_refused_for_the_tensorrt_backend(deploy):
    with pytest.raises(RuntimeError, match="cpu"):
        _node(deploy, device="cpu").prepare_inference()


def test_a_missing_tensorrt_package_names_the_consumer(deploy, monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    node = _node(deploy)

    with pytest.raises(ImportError, match="ExportedSpan 'exported_span' backend tensorrt"):
        node.prepare_inference()


def test_the_train_stage_is_refused(deploy, fake_trt, engine_dir):
    node = _node(deploy)
    node.prepare_inference()
    data = torch.rand(1, 4, 4, 3)

    with pytest.raises(RuntimeError, match="canonical"):
        node(
            data=data,
            wavelengths=_wavelengths(),
            context=Context(stage=ExecutionStage.TRAIN),
        )


def test_a_changed_onnx_file_is_refused_before_building(deploy, fake_trt, engine_dir):
    node = _node(deploy)
    with (deploy / "span.onnx").open("ab") as handle:
        handle.write(b"\0")

    with pytest.raises(RuntimeError, match="onnx_md5"):
        node.prepare_inference()
    assert fake_trt.built["count"] == 0


def test_a_constant_that_differs_from_the_baked_value_is_refused(
    deploy, fake_trt, engine_dir, monkeypatch
):
    node = _node(deploy)
    node.prepare_inference()

    with pytest.raises(ValueError, match="wavelengths"):
        node(data=torch.rand(1, 4, 4, 3), wavelengths=_wavelengths() + 1.0, context=_inference())


def test_another_shape_is_refused(deploy, fake_trt, engine_dir, monkeypatch):
    node = _node(deploy)
    node.prepare_inference()

    with pytest.raises(ValueError, match=r"\(1, 4, 4, 3\)"):
        node(data=torch.rand(1, 5, 5, 3), wavelengths=_wavelengths(), context=_inference())


# ----- the runner module directly ---------------------------------------------------


def test_bind_constant_is_a_noop_for_a_value_the_engine_folded_away(
    fake_cuda, monkeypatch, tmp_path
):
    """A constant that never reaches the engine as a named input (TensorRT folded it
    into the graph at compile time) is not an error: there is nothing left to bind."""
    from cuvis_ai.node._trt_span_runner import TrtSpanRunner

    class _OneInputEngine(FakeEngine):
        def __init__(self) -> None:
            self.io = [("x", _TensorIOMode.INPUT, torch.float32, (1,))]
            self.num_io_tensors = len(self.io)

    class _Runtime:
        def __init__(self, logger):
            pass

        def deserialize_cuda_engine(self, data):
            return _OneInputEngine()

    trt = types.SimpleNamespace(
        __version__="10.15.1.29",
        Logger=_Logger,
        Runtime=_Runtime,
        TensorIOMode=_TensorIOMode,
        float32="float32",
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(trt_engine, "_LOGGER", None)
    engine_path = tmp_path / "fake.engine"
    engine_path.write_bytes(b"engine-bytes")

    runner = TrtSpanRunner(engine_path, device="cuda")

    runner.bind_constant("wavelengths_not_an_engine_input", torch.tensor([1.0, 2.0, 3.0]))

    assert "wavelengths_not_an_engine_input" not in runner.input_specs


# ----- building and reusing the engine ---------------------------------------------


def test_the_engine_is_built_once_and_reused(deploy, fake_trt, engine_dir, monkeypatch):
    node = _node(deploy)

    node.prepare_inference()
    assert fake_trt.built["count"] == 1
    first_engine_files = list(engine_dir.rglob("*.engine"))
    assert len(first_engine_files) == 1

    node.cleanup()
    node.prepare_inference()  # a second prepare on the same node: engine stays on disk

    assert fake_trt.built["count"] == 2  # the fake re-deserializes; the build itself ran once
    assert list(engine_dir.rglob("*.engine")) == first_engine_files


def test_a_second_node_with_the_same_sidecar_reuses_the_engine_file(
    deploy, fake_trt, engine_dir, monkeypatch
):
    node_a = _node(deploy)
    node_a.prepare_inference()
    built_after_first = len(list(engine_dir.rglob("*.engine")))

    node_b = _node(deploy)
    node_b.prepare_inference()

    assert len(list(engine_dir.rglob("*.engine"))) == built_after_first  # no second build dir


def test_forward_reproduces_the_span(deploy, fake_trt, engine_dir, monkeypatch):
    node = _node(deploy)
    node.prepare_inference()
    torch.manual_seed(0)
    data = torch.rand(1, 4, 4, 3)

    out = node(data=data, wavelengths=_wavelengths(), context=_inference())

    scores, anomaly_score = _Span().eval()(data)
    torch.testing.assert_close(out["scores"].cpu(), scores, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out["anomaly_score"].cpu(), anomaly_score, atol=1e-5, rtol=1e-4)


def test_mutating_a_returned_output_does_not_change_the_next_call(
    deploy, fake_trt, engine_dir, monkeypatch
):
    node = _node(deploy)
    node.prepare_inference()

    first = node(data=torch.rand(1, 4, 4, 3), wavelengths=_wavelengths(), context=_inference())
    held = first["scores"].clone()
    first["scores"].add_(1000.0)  # mutate the tensor the node handed back

    second = node(data=torch.rand(1, 4, 4, 3), wavelengths=_wavelengths(), context=_inference())

    assert not torch.equal(second["scores"].cpu(), held + 1000.0)


def test_a_numpy_constant_and_input_are_accepted(deploy, fake_trt, engine_dir, monkeypatch):
    node = _node(deploy)
    node.prepare_inference()
    data = torch.rand(1, 4, 4, 3)

    out = node(
        data=data.numpy(),
        wavelengths=np.asarray(WAVELENGTHS, dtype=np.float32),
        context=_inference(),
    )

    scores, _ = _Span().eval()(data)
    torch.testing.assert_close(out["scores"].cpu(), scores, atol=1e-5, rtol=1e-4)


def test_cleanup_drops_the_runner(deploy, fake_trt, engine_dir, monkeypatch):
    node = _node(deploy)
    node.prepare_inference()
    assert node.session is not None

    node.cleanup()

    assert node.session is None


# ----- a deploy yaml with backend: tensorrt as an hparam ----------------------------


def _write_deploy_yaml(folder: Path) -> tuple[Path, Path]:
    torch.manual_seed(1)
    sample_path = folder / "sample.pt"
    torch.save({"cube": torch.rand(1, 4, 4, 3), "wavelengths": _wavelengths()}, sample_path)
    module = "tests.node.test_exported_span"
    config = {
        "metadata": {"name": "tiny_deploy"},
        "plugins": None,
        "nodes": [
            {
                "name": "source",
                "class_name": f"{module}._Source",
                "hparams": {"sample_path": str(sample_path)},
            },
            {
                "name": "exported_span",
                "class_name": "cuvis_ai.node.exported_span.ExportedSpan",
                "hparams": {"onnx_path": "span.onnx", "backend": "tensorrt", "device": "cuda"},
            },
        ],
        "connections": [
            {"source": "source.outputs.cube", "target": "exported_span.inputs.data"},
            {
                "source": "source.outputs.wavelengths",
                "target": "exported_span.inputs.wavelengths",
            },
        ],
    }
    yaml_path = folder / "tiny_deploy.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return yaml_path, sample_path


def test_the_backend_hparam_round_trips_through_a_deploy_yaml(
    deploy, fake_trt, engine_dir, monkeypatch
):
    from cuvis_ai.node.exported_span import DEPLOY_DIR_ENV

    yaml_path, sample_path = _write_deploy_yaml(deploy)
    monkeypatch.setenv(DEPLOY_DIR_ENV, str(deploy))

    pipeline = CuvisPipeline.load_pipeline(yaml_path, device="cpu")
    out = pipeline.forward(batch={}, stage=ExecutionStage.INFERENCE)

    cube = torch.load(sample_path, weights_only=True)["cube"]
    scores, _ = _Span().eval()(cube)
    torch.testing.assert_close(out[("exported_span", "scores")].cpu(), scores, atol=1e-5, rtol=1e-4)
