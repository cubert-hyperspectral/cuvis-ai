"""ExportedSpan runs a span that export-pipeline wrote to ONNX in place of its nodes.

The deploy folder here is written by hand with torch.onnx.export and a sidecar in the
format cuvis-ai-core's exporter writes (format 1), so the node's contract is tested
without the exporter. The last test runs the real exporter when its module is installed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
import yaml
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from torch import nn

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

pytestmark = pytest.mark.unit
pytest.importorskip("onnxruntime")

WAVELENGTHS = [450.0, 550.0, 650.0]
_CUBE = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))
_VEC = PortSpec(dtype=torch.float32, shape=(-1,))


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


@pytest.fixture(scope="module")
def deploy(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("deploy")
    _write_deploy(folder)
    return folder


def _node(folder: Path, **hparams):
    from cuvis_ai.node.exported_span import ExportedSpan

    return ExportedSpan(onnx_path=str(folder / "span.onnx"), name="exported_span", **hparams)


def _inference() -> Context:
    return Context(stage=ExecutionStage.INFERENCE)


def _wavelengths() -> torch.Tensor:
    return torch.tensor(WAVELENGTHS)


# ----- ports and the sidecar ------------------------------------------


def test_ports_come_from_the_sidecar(deploy):
    node = _node(deploy)

    assert set(node.INPUT_SPECS) == {"data", "wavelengths"}
    assert set(node.OUTPUT_SPECS) == {"scores", "anomaly_score"}
    assert node.INPUT_SPECS["data"].dtype == torch.float32
    assert tuple(node.INPUT_SPECS["data"].shape) == (-1, 4, 4, 3)
    assert tuple(node.INPUT_SPECS["wavelengths"].shape) == (3,)
    assert tuple(node.OUTPUT_SPECS["scores"].shape) == (-1, 4, 4, 1)
    assert tuple(node.OUTPUT_SPECS["anomaly_score"].shape) == (-1,)
    assert {"data", "wavelengths"} <= set(vars(node.inputs))
    assert {"scores", "anomaly_score"} <= set(vars(node.outputs))


def test_hparams_round_trip(deploy):
    node = _node(deploy, device="cpu")

    assert node.hparams == {
        "onnx_path": str(deploy / "span.onnx"),
        "backend": "onnxruntime",
        "device": "cpu",
    }
    assert node.state_dict() == {}


def test_a_sidecar_of_another_format_is_refused(tmp_path):
    _write_deploy(tmp_path)
    sidecar_path = tmp_path / "span.onnx.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["format"] = 2
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="format 2"):
        _node(tmp_path)


# ----- running the span -------------------------------------------------


def test_forward_reproduces_the_span_over_a_batch(deploy):
    node = _node(deploy, device="cpu")
    node.prepare_inference()
    torch.manual_seed(0)
    data = torch.rand(3, 4, 4, 3)

    out = node(data=data, wavelengths=_wavelengths(), context=_inference())

    scores, anomaly_score = _Span().eval()(data)
    torch.testing.assert_close(out["scores"], scores, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out["anomaly_score"], anomaly_score, atol=1e-5, rtol=1e-4)
    assert out["scores"].shape == (3, 4, 4, 1)
    assert out["anomaly_score"].shape == (3,)


def test_forward_prepares_the_session_when_the_loader_did_not(deploy):
    node = _node(deploy, device="cpu")
    data = torch.rand(1, 4, 4, 3)

    out = node(data=data, wavelengths=_wavelengths(), context=_inference())

    torch.testing.assert_close(out["scores"], _Span().eval()(data)[0], atol=1e-5, rtol=1e-4)


def test_evaluation_stages_run_and_training_is_refused(deploy):
    node = _node(deploy, device="cpu")
    data = torch.rand(1, 4, 4, 3)

    for stage in (ExecutionStage.VAL, ExecutionStage.TEST):
        assert "scores" in node(data=data, wavelengths=_wavelengths(), context=Context(stage=stage))
    with pytest.raises(RuntimeError, match="canonical"):
        node(data=data, wavelengths=_wavelengths(), context=Context(stage=ExecutionStage.TRAIN))


def test_another_shape_is_refused_and_the_exported_one_named(deploy):
    node = _node(deploy, device="cpu")

    with pytest.raises(ValueError, match=r"\(1, 4, 4, 3\)"):
        node(data=torch.rand(1, 5, 5, 3), wavelengths=_wavelengths(), context=_inference())


def test_another_dtype_is_refused(deploy):
    node = _node(deploy, device="cpu")

    with pytest.raises(ValueError, match="float64"):
        node(
            data=torch.rand(1, 4, 4, 3, dtype=torch.float64),
            wavelengths=_wavelengths(),
            context=_inference(),
        )


def test_a_constant_that_differs_from_the_baked_value_is_refused(deploy):
    node = _node(deploy, device="cpu")

    with pytest.raises(ValueError, match="wavelengths"):
        node(data=torch.rand(1, 4, 4, 3), wavelengths=_wavelengths() + 1.0, context=_inference())


# ----- the file and the environment ------------------------------------


def test_a_changed_onnx_file_is_refused(tmp_path):
    _write_deploy(tmp_path)
    node = _node(tmp_path, device="cpu")
    with (tmp_path / "span.onnx").open("ab") as handle:
        handle.write(b"\0")

    with pytest.raises(RuntimeError, match="onnx_md5"):
        node.prepare_inference()


def test_a_relative_path_needs_a_base(tmp_path, monkeypatch):
    from cuvis_ai.node.exported_span import DEPLOY_DIR_ENV, ExportedSpan

    monkeypatch.delenv(DEPLOY_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match=DEPLOY_DIR_ENV):
        ExportedSpan(onnx_path="span.onnx", name="exported_span")


def test_a_relative_path_resolves_against_the_deploy_dir_variable(deploy, monkeypatch, tmp_path):
    from cuvis_ai.node.exported_span import DEPLOY_DIR_ENV, ExportedSpan

    monkeypatch.setenv(DEPLOY_DIR_ENV, str(deploy))
    monkeypatch.chdir(tmp_path)

    node = ExportedSpan(onnx_path="span.onnx", name="exported_span", device="cpu")

    assert node.resolved_onnx_path == deploy / "span.onnx"


def test_the_tensorrt_backend_is_not_available_yet(deploy):
    with pytest.raises(NotImplementedError, match="tensorrt"):
        _node(deploy, backend="tensorrt")


def test_an_unknown_backend_is_refused(deploy):
    with pytest.raises(ValueError, match="backend"):
        _node(deploy, backend="tvm")


def test_a_cuda_session_that_lands_on_the_cpu_provider_is_refused(deploy, monkeypatch):
    import onnxruntime as ort

    class _CpuSession:
        def __init__(self, *_, **__):
            pass

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(ort, "InferenceSession", _CpuSession)
    node = _node(deploy, device="cuda")

    with pytest.raises(RuntimeError, match="CPUExecutionProvider"):
        node.prepare_inference()


def test_a_missing_onnxruntime_names_the_extra(deploy, monkeypatch):
    node = _node(deploy, device="cpu")
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    with pytest.raises(ImportError, match="uv sync --extra onnx") as info:
        node.prepare_inference()
    assert "cuvis_ai_onnx" in str(info.value)


def test_cleanup_drops_the_session(deploy):
    node = _node(deploy, device="cpu")
    node.prepare_inference()
    assert node.session is not None

    node.cleanup()

    assert node.session is None


@pytest.mark.gpu
def test_cuda_outputs_stay_on_the_device(deploy):
    import onnxruntime as ort

    if (
        not torch.cuda.is_available()
        or "CUDAExecutionProvider" not in ort.get_available_providers()
    ):
        pytest.skip("needs a CUDA device and the ONNX Runtime CUDA provider")
    node = _node(deploy, device="cuda")
    node.prepare_inference()
    data = torch.rand(2, 4, 4, 3)

    out = node(data=data.cuda(), wavelengths=_wavelengths().cuda(), context=_inference())

    assert out["scores"].device.type == "cuda"
    torch.testing.assert_close(out["scores"].cpu(), _Span().eval()(data)[0], atol=1e-5, rtol=1e-4)


# ----- a deploy yaml through the pipeline loader ----------------------


class _Source(Node):
    """Reads a cube and the wavelength grid from a ``.pt`` file (pure tensors)."""

    OUTPUT_SPECS = {"cube": _CUBE, "wavelengths": _VEC}

    def __init__(self, sample_path: str, **kwargs) -> None:
        super().__init__(sample_path=sample_path, **kwargs)
        self.sample_path = sample_path

    def forward(self, **_) -> dict[str, torch.Tensor]:
        sample = torch.load(self.sample_path, weights_only=True)
        return {"cube": sample["cube"], "wavelengths": sample["wavelengths"]}


class _Decide(Node):
    """``decisions = scores > threshold``."""

    INPUT_SPECS = {"scores": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 1))}
    OUTPUT_SPECS = {"decisions": PortSpec(dtype=torch.bool, shape=(-1, -1, -1, 1))}

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = threshold

    def forward(self, scores: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        return {"decisions": scores > self.threshold}


def _write_deploy_yaml(folder: Path, *, onnx_path: str = "span.onnx") -> tuple[Path, Path]:
    """A deploy yaml next to the ONNX file: source, the span node, a decider."""
    torch.manual_seed(1)
    sample_path = folder / "sample.pt"
    torch.save({"cube": torch.rand(2, 4, 4, 3), "wavelengths": _wavelengths()}, sample_path)
    module = __name__ if __name__ != "__main__" else "tests.node.test_exported_span"
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
                "hparams": {"onnx_path": onnx_path, "backend": "onnxruntime"},
            },
            {"name": "decide", "class_name": f"{module}._Decide", "hparams": {"threshold": 0.5}},
        ],
        "connections": [
            {"source": "source.outputs.cube", "target": "exported_span.inputs.data"},
            {"source": "source.outputs.wavelengths", "target": "exported_span.inputs.wavelengths"},
            {"source": "exported_span.outputs.scores", "target": "decide.inputs.scores"},
        ],
    }
    yaml_path = folder / "tiny_deploy.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return yaml_path, sample_path


def _expected(sample_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    cube = torch.load(sample_path, weights_only=True)["cube"]
    scores, _ = _Span().eval()(cube)
    return scores, scores > 0.5


def test_a_deploy_yaml_loads_and_runs_with_the_deploy_dir_variable(tmp_path, monkeypatch):
    from cuvis_ai.node.exported_span import DEPLOY_DIR_ENV

    _write_deploy(tmp_path)
    yaml_path, sample_path = _write_deploy_yaml(tmp_path)
    monkeypatch.setenv(DEPLOY_DIR_ENV, str(tmp_path))

    pipeline = CuvisPipeline.load_pipeline(yaml_path, device="cpu")
    out = pipeline.forward(batch={}, stage=ExecutionStage.INFERENCE)

    scores, decisions = _expected(sample_path)
    torch.testing.assert_close(out[("exported_span", "scores")], scores, atol=1e-5, rtol=1e-4)
    assert torch.equal(out[("decide", "decisions")], decisions)


def test_a_deploy_yaml_finds_the_onnx_file_next_to_itself(tmp_path, monkeypatch):
    """A relative ``onnx_path`` resolves against the yaml's folder when the loader says
    where the yaml is (``pipeline_source_dir`` in cuvis-ai-core)."""
    factory = pytest.importorskip("cuvis_ai_core.pipeline.factory")
    if not hasattr(factory, "pipeline_source_dir"):
        pytest.skip("needs cuvis-ai-core with pipeline_source_dir (0.19)")
    from cuvis_ai.node.exported_span import DEPLOY_DIR_ENV

    _write_deploy(tmp_path)
    yaml_path, sample_path = _write_deploy_yaml(tmp_path)
    monkeypatch.delenv(DEPLOY_DIR_ENV, raising=False)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    pipeline = CuvisPipeline.load_pipeline(yaml_path, device="cpu")
    out = pipeline.forward(batch={}, stage=ExecutionStage.INFERENCE)

    torch.testing.assert_close(
        out[("exported_span", "scores")], _expected(sample_path)[0], atol=1e-5, rtol=1e-4
    )


def test_an_absolute_onnx_path_needs_no_base(tmp_path, monkeypatch):
    from cuvis_ai.node.exported_span import DEPLOY_DIR_ENV

    _write_deploy(tmp_path)
    yaml_path, sample_path = _write_deploy_yaml(tmp_path, onnx_path=str(tmp_path / "span.onnx"))
    monkeypatch.delenv(DEPLOY_DIR_ENV, raising=False)
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy(yaml_path, moved / yaml_path.name)
    monkeypatch.chdir(moved)

    pipeline = CuvisPipeline.load_pipeline(moved / yaml_path.name, device="cpu")
    out = pipeline.forward(batch={}, stage=ExecutionStage.INFERENCE)

    torch.testing.assert_close(
        out[("exported_span", "scores")], _expected(sample_path)[0], atol=1e-5, rtol=1e-4
    )


# ----- the real exporter, when installed ---------------------------------


class _Scale(Node):
    """``scaled = data * scale`` with a fitted per-band scale."""

    INPUT_SPECS = {"data": _CUBE}
    OUTPUT_SPECS = {"scaled": _CUBE}

    def __init__(self, bands: int = 3, **kwargs) -> None:
        super().__init__(bands=bands, **kwargs)
        self.register_buffer("scale", torch.ones(bands))

    def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        return {"scaled": data * self.scale}


class _Score(Node):
    """Mean over bands, kept as ``[B, H, W, 1]``."""

    INPUT_SPECS = {"data": _CUBE}
    OUTPUT_SPECS = {"scores": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 1))}

    def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        return {"scores": data.mean(dim=-1, keepdim=True)}


def test_the_exporter_output_runs_through_the_node(tmp_path, monkeypatch):
    export = pytest.importorskip("cuvis_ai_core.pipeline.export")
    from cuvis_ai.node.exported_span import DEPLOY_DIR_ENV

    torch.manual_seed(2)
    sample_path = tmp_path / "sample.pt"
    torch.save({"cube": torch.rand(1, 4, 4, 3), "wavelengths": _wavelengths()}, sample_path)
    source = _Source(sample_path=str(sample_path), name="source")
    scale, score, decide = _Scale(name="scale"), _Score(name="score"), _Decide(name="decide")
    scale.scale.copy_(torch.tensor([0.5, 1.0, 2.0]))
    pipeline = CuvisPipeline("tiny")
    pipeline.connect(
        (source.outputs.cube, scale.inputs.data),
        (scale.outputs.scaled, score.inputs.data),
        (score.outputs.scores, decide.inputs.scores),
    )
    examples = [{"scale.data": torch.rand(1, 4, 4, 3)} for _ in range(2)]

    result = export.export_deployment(
        pipeline,
        tmp_path / "deploy",
        inputs=["scale.data"],
        outputs=["score.scores"],
        examples=examples,
    )
    monkeypatch.setenv(DEPLOY_DIR_ENV, str(result.yaml_path.parent))
    deployed = CuvisPipeline.load_pipeline(
        result.yaml_path, weights_path=result.weights_path, device="cpu"
    )
    out = deployed.forward(batch={}, stage=ExecutionStage.INFERENCE)
    reference = pipeline.forward(batch={}, stage=ExecutionStage.INFERENCE)

    assert [n.name for n in deployed.nodes] == ["source", "exported_span", "decide"]
    assert deployed._plugins == ["cuvis_ai_onnx"]
    torch.testing.assert_close(
        out[("exported_span", "scores")], reference[("score", "scores")], atol=1e-5, rtol=1e-4
    )
    assert torch.equal(out[("decide", "decisions")], reference[("decide", "decisions")])
