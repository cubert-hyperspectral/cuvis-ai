"""Registry-wide ONNX export contract: every builtin node tagged NodeTag.ONNX must export
with the dynamo exporter in its registry contract default configuration and ONNX Runtime
must reproduce its eager output.

Marked slow: it builds, fits and exports every tagged node, one process each would be
overkill for a unit run, so it runs in-process but still takes real wall time. Run with
``--runslow``.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeTag
from cuvis_ai_schemas.execution.context import Context

from cuvis_ai_core.utils.node_registry import NodeRegistry
from tests.training.test_node_registry_statistical_contract import _minimal_constructor_kwargs

pytestmark = [pytest.mark.unit, pytest.mark.slow]

B, H, W, C = 1, 16, 16, 5
WAVELENGTHS = [500.0, 600.0, 700.0, 800.0, 900.0]
CTX = Context(stage=ExecutionStage.INFERENCE)


def _dims_for(shape: tuple[int, ...]) -> list[int]:
    """Concrete dims for a PortSpec shape, defaulting -1 axes to the synthetic sizes."""
    rank = len(shape)
    defaults = {4: [B, H, W, C], 3: [B, H, W], 2: [B, C], 1: [C], 0: []}.get(rank, [2] * rank)
    return [int(d) if isinstance(d, int) and d != -1 else defaults[i] for i, d in enumerate(shape)]


def _tensor_for(name: str, spec: Any) -> torch.Tensor:
    shape = _dims_for(spec.shape)
    dt = str(spec.dtype).lower()
    if "wavelength" in name and len(shape) == 1:
        n = shape[0]
        values = (
            WAVELENGTHS[:n]
            if n <= len(WAVELENGTHS)
            else WAVELENGTHS + [1000.0] * (n - len(WAVELENGTHS))
        )
        return torch.tensor(values, dtype=torch.float32)
    if "bool" in dt:
        return torch.rand(shape) > 0.5
    if "int" in dt:
        return torch.randint(0, 2, shape, dtype=torch.int64)
    return torch.rand(shape, dtype=torch.float32)


def _build_inputs(node: Any) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    kwargs: dict[str, Any] = {}
    for name, spec in node.INPUT_SPECS.items():
        if getattr(spec, "optional", False):
            continue
        if getattr(spec, "variadic", False):
            kwargs[name] = [_tensor_for(name, spec), _tensor_for(name, spec)]
        else:
            kwargs[name] = _tensor_for(name, spec)
    return kwargs


def _accepts_context(fn: Any) -> bool:
    params = inspect.signature(fn).parameters
    return "context" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


class _Probe(torch.nn.Module):
    """Flatten a node's kwargs (including variadic lists) to positional tensors."""

    def __init__(self, node: torch.nn.Module, kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.node = node
        self.layout = [(k, len(v) if isinstance(v, list) else None) for k, v in kwargs.items()]
        self.with_context = _accepts_context(node.forward)

    def flat(self, kwargs: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        out: list[torch.Tensor] = []
        for name, count in self.layout:
            out.extend(kwargs[name] if count is not None else [kwargs[name]])
        return tuple(out)

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        kwargs: dict[str, Any] = {}
        cursor = 0
        for name, count in self.layout:
            if count is None:
                kwargs[name] = tensors[cursor]
                cursor += 1
            else:
                kwargs[name] = list(tensors[cursor : cursor + count])
                cursor += count
        if self.with_context:
            kwargs["context"] = CTX
        out = self.node.forward(**kwargs)
        # A numpy-dtype output (e.g. a wavelength grid passed through) still leaves the
        # traced graph as a real output; compare it too, not only the torch.Tensor ones.
        return tuple(
            v if torch.is_tensor(v) else torch.as_tensor(v)
            for v in out.values()
            if torch.is_tensor(v) or isinstance(v, np.ndarray)
        )


def _onnx_tagged_node_names() -> list[str]:
    NodeRegistry.auto_register_package("cuvis_ai.node")
    names = []
    for name in sorted(NodeRegistry.list_builtin_nodes()):
        cls = NodeRegistry.get_builtin_class(name)
        if inspect.isabstract(cls):
            continue
        if NodeTag.ONNX in cls.get_tags():
            names.append(name)
    return names


_ONNX_NODE_NAMES = _onnx_tagged_node_names()


def _export_and_check(class_name: str) -> float:
    """Build, export and check one node; return the max abs diff against eager.

    Shared body for the parametrized test below and the report sweep in
    :func:`test_onnx_export_contract_report`, which needs the diff per node and cannot
    get it from a test function (pytest warns on a non-``None`` return from one).
    """
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")

    torch.manual_seed(0)
    cls = NodeRegistry.get_builtin_class(class_name)
    node = cls(**_minimal_constructor_kwargs(cls))
    node.eval()

    kwargs = _build_inputs(node)
    if not kwargs:
        pytest.skip(f"{class_name} has no tensor inputs")

    if getattr(node, "requires_initial_fit", False):
        node.statistical_initialization(iter([dict(kwargs)]))

    probe = _Probe(node, kwargs).eval()
    args = probe.flat(kwargs)
    with torch.no_grad():
        eager = probe(*args)
    if not eager:
        pytest.skip(f"{class_name} has no tensor outputs")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{class_name}.onnx"
        torch.onnx.export(
            probe,
            args,
            str(path),
            dynamo=True,
            opset_version=18,
            external_data=False,
        )
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feeds = {inp.name: a.numpy() for inp, a in zip(session.get_inputs(), args, strict=True)}
        outs = session.run(None, feeds)

    max_abs_diff = 0.0
    for got, want in zip(outs, eager, strict=True):
        got_f = got.astype(np.float64) if got.dtype.kind == "f" else got
        want_f = want.numpy().astype(np.float64) if want.is_floating_point() else want.numpy()
        np.testing.assert_allclose(got_f, want_f, atol=1e-5, rtol=1e-4)
        if got.dtype.kind == "f":
            max_abs_diff = max(max_abs_diff, float(np.abs(got_f - want_f).max()))
    return max_abs_diff


@pytest.mark.parametrize("class_name", _ONNX_NODE_NAMES)
def test_onnx_tagged_node_exports_and_matches_onnx_runtime(class_name: str) -> None:
    """Every NodeTag.ONNX node, built from its registry contract defaults, exports and
    ONNX Runtime reproduces its eager output within tolerance."""
    _export_and_check(class_name)


#: Set to a file path to have the sweep append one JSON line per node to it.
REPORT_ENV = "CUVIS_ONNX_CONTRACT_REPORT"


def _append_report_line(path: str, record: dict[str, Any]) -> None:
    """Append one JSON line describing a node's outcome to the report file."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def test_onnx_export_contract_report(capsys: pytest.CaptureFixture[str]) -> None:
    """Report the node count and total wall time of the parametrized export sweep above.

    When ``CUVIS_ONNX_CONTRACT_REPORT`` names a file, append one JSON line per node
    (name, outcome, max abs diff, seconds) so a scheduled run can turn the sweep into a
    job summary without reparsing pytest output.
    """
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    report_path = os.environ.get(REPORT_ENV)
    t0 = time.time()
    failures: list[str] = []
    for class_name in _ONNX_NODE_NAMES:
        node_t0 = time.time()
        max_abs_diff: float | None = None
        outcome = "passed"
        error: str | None = None
        try:
            max_abs_diff = _export_and_check(class_name)
        except pytest.skip.Exception:
            outcome = "skipped"
        except Exception as exc:  # noqa: BLE001
            outcome = "failed"
            error = str(exc)
            failures.append(f"{class_name}: {exc}")
        if report_path:
            record = {
                "name": class_name,
                "outcome": outcome,
                "max_abs_diff": max_abs_diff,
                "seconds": round(time.time() - node_t0, 4),
            }
            if error is not None:
                record["error"] = error
            _append_report_line(report_path, record)
    elapsed = time.time() - t0
    with capsys.disabled():
        print(
            f"\nONNX export contract: {len(_ONNX_NODE_NAMES)} nodes, "
            f"{len(failures)} failed, {elapsed:.1f}s total"
        )
        for failure in failures:
            print(f"  FAILED {failure}")
    assert not failures, f"{len(failures)} of {len(_ONNX_NODE_NAMES)} nodes failed to export"


def test_onnx_export_contract_report_hook_writes_one_line(
    tmp_path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the report env var set, the sweep appends one JSON line per node it runs."""
    pytest.importorskip("onnxscript")
    pytest.importorskip("onnxruntime")
    report_path = tmp_path / "report.jsonl"
    monkeypatch.setenv(REPORT_ENV, str(report_path))
    monkeypatch.setattr(sys.modules[__name__], "_ONNX_NODE_NAMES", _ONNX_NODE_NAMES[:1])

    test_onnx_export_contract_report(capsys)

    lines = report_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["name"] == _ONNX_NODE_NAMES[0]
    assert record["outcome"] == "passed"
    assert isinstance(record["max_abs_diff"], float)
    assert isinstance(record["seconds"], float)
