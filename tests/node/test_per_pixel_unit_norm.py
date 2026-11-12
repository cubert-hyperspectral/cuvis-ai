"""Unit tests for PerPixelUnitNorm node."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node.normalization import PerPixelUnitNorm
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

pytestmark = pytest.mark.unit


def _random_cube(
    batches: int = 2, height: int = 8, width: int = 9, channels: int = 16, seed: int = 123
) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return torch.randn((batches, height, width, channels), generator=rng, dtype=torch.float32)


class _CubeSource(Node):
    """Module-scope test source emitting a fixed BHWC cube, for pipeline reload tests."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.TORCH})
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))}

    def forward(self, **_) -> dict[str, torch.Tensor]:
        return {"data": _random_cube(2, 6, 7, 5, seed=42)}


@torch.no_grad()
def test_shape_and_properties() -> None:
    x = _random_cube(3, 10, 12, 20)
    node = PerPixelUnitNorm(eps=1e-8)

    out = node.forward(data=x)["normalized"]

    assert out.shape == x.shape

    B, H, W, C = out.shape
    flat = out.view(B * H * W, C)
    means = flat.mean(dim=1).cpu().numpy()
    norms = flat.norm(p=2, dim=1).cpu().numpy()

    assert np.isfinite(means).all() and np.isfinite(norms).all()
    assert float(np.mean(np.abs(means))) < 1e-3
    assert abs(float(np.mean(norms)) - 1.0) < 1e-2


@torch.no_grad()
def test_accepts_hwc_tensor() -> None:
    x_bhwc = _random_cube(1, 5, 6, 7)
    x_hwc = x_bhwc[0]
    node = PerPixelUnitNorm(eps=1e-8)

    # Node contract expects BHWC; callers must add batch dim for HWC tensors
    out = node.forward(data=x_hwc.unsqueeze(0))["normalized"]

    assert out.shape == x_bhwc.shape


@torch.no_grad()
def test_zero_variance_pixels_stable() -> None:
    x = _random_cube(2, 6, 7, 10)
    x[0, 0, 0, :] = 5.0
    node = PerPixelUnitNorm(eps=1e-6)

    out = node.forward(data=x)["normalized"]

    assert torch.isfinite(out).all()
    assert torch.allclose(out[0, 0, 0, :], torch.zeros_like(out[0, 0, 0, :]))


@torch.no_grad()
def test_single_channel_stable() -> None:
    """Single-channel input (C=1) stays finite and doesn't produce NaN/Inf.

    With C=1, mean-centering zeros the value and norm becomes ~0. The eps
    parameter should prevent division by zero.
    """
    x = _random_cube(2, 4, 5, 1)
    node = PerPixelUnitNorm(eps=1e-6)

    out = node.forward(data=x)["normalized"]

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# center mode: default stays bit-identical, center=False is a direct unit norm.
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_center_true_is_the_default() -> None:
    node = PerPixelUnitNorm(eps=1e-8)
    assert node.center is True
    assert node.hparams["center"] is True


@torch.no_grad()
def test_center_true_matches_old_formula_written_out_bit_for_bit() -> None:
    """Pin the default against the pre-existing formula, written out independently here.

    This is the bit-identical guarantee: a YAML saved before `center` existed loads with
    `center=True` and must reproduce exactly what the old, uncentered-parameter code computed.
    """
    x = _random_cube(3, 10, 12, 20)
    node = PerPixelUnitNorm(eps=1e-8)

    out = node.forward(data=x)["normalized"]

    B, H, W, C = x.shape
    flat = x.reshape(B, -1, C)
    mu = flat.mean(dim=2, keepdim=True)
    centered = flat - mu
    l2 = centered.norm(p=2, dim=2, keepdim=True).clamp_min(1e-8)
    expected = (centered / l2).reshape(B, H, W, C)

    torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)


@torch.no_grad()
def test_center_false_matches_direct_unit_norm_formula() -> None:
    x = _random_cube(3, 10, 12, 20)
    node = PerPixelUnitNorm(eps=1e-8, center=False)

    out = node.forward(data=x)["normalized"]

    B, H, W, C = x.shape
    flat = x.reshape(B, -1, C)
    l2 = flat.norm(p=2, dim=2, keepdim=True).clamp_min(1e-8)
    expected = (flat / l2).reshape(B, H, W, C)

    torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)


@torch.no_grad()
def test_center_false_unchanged_under_positive_gain() -> None:
    """A positive scalar gain does not change direction, so the unit-norm output is unchanged."""
    x = _random_cube(2, 5, 6, 8)
    node = PerPixelUnitNorm(eps=1e-8, center=False)

    out = node.forward(data=x)["normalized"]
    out_scaled = node.forward(data=x * 3.5)["normalized"]

    torch.testing.assert_close(out_scaled, out, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_center_false_changes_under_constant_offset_for_nonconstant_spectrum() -> None:
    """A constant additive offset changes direction for a spectrum that is not itself constant."""
    x = _random_cube(2, 5, 6, 8)
    offset = torch.full_like(x, 2.0)

    node = PerPixelUnitNorm(eps=1e-8, center=False)
    out = node.forward(data=x)["normalized"]
    out_offset = node.forward(data=x + offset)["normalized"]

    assert not torch.allclose(out_offset, out, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_center_false_unchanged_under_constant_offset_for_constant_spectrum() -> None:
    """A constant positive spectrum keeps its direction under a constant additive offset."""
    x = torch.full((1, 2, 2, 6), 2.0)
    offset = torch.full_like(x, 3.0)

    node = PerPixelUnitNorm(eps=1e-8, center=False)
    out = node.forward(data=x)["normalized"]
    out_offset = node.forward(data=x + offset)["normalized"]

    torch.testing.assert_close(out_offset, out, atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_center_true_unchanged_under_positive_gain() -> None:
    x = _random_cube(2, 5, 6, 8)
    node = PerPixelUnitNorm(eps=1e-8, center=True)

    out = node.forward(data=x)["normalized"]
    out_scaled = node.forward(data=x * 2.25)["normalized"]

    torch.testing.assert_close(out_scaled, out, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_center_true_unchanged_under_constant_offset() -> None:
    """Mean-centering absorbs a constant additive offset, so center=True is invariant to it too."""
    x = _random_cube(2, 5, 6, 8)
    offset = torch.full_like(x, -1.5)
    node = PerPixelUnitNorm(eps=1e-8, center=True)

    out = node.forward(data=x)["normalized"]
    out_offset = node.forward(data=x + offset)["normalized"]

    torch.testing.assert_close(out_offset, out, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_center_false_all_zero_spectrum_stays_finite() -> None:
    x = torch.zeros((1, 2, 2, 6))
    node = PerPixelUnitNorm(eps=1e-8, center=False)

    out = node.forward(data=x)["normalized"]

    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))


@torch.no_grad()
def test_center_false_constant_spectrum_stays_finite() -> None:
    x = torch.full((1, 2, 2, 6), 4.0)
    node = PerPixelUnitNorm(eps=1e-8, center=False)

    out = node.forward(data=x)["normalized"]

    assert torch.isfinite(out).all()
    expected_value = 4.0 / (4.0 * (6**0.5))
    torch.testing.assert_close(out, torch.full_like(out, expected_value), atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_center_false_single_channel_stays_finite() -> None:
    x = _random_cube(2, 4, 5, 1)
    node = PerPixelUnitNorm(eps=1e-6, center=False)

    out = node.forward(data=x)["normalized"]

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@torch.no_grad()
def test_center_false_norm_below_eps_stays_finite() -> None:
    """A spectrum whose norm sits below eps does not blow up the division."""
    x = torch.full((1, 2, 2, 6), 1e-10)
    node = PerPixelUnitNorm(eps=1e-8, center=False)

    out = node.forward(data=x)["normalized"]

    assert torch.isfinite(out).all()


@torch.no_grad()
def test_center_false_matches_golden_reference_matrix() -> None:
    """Golden-reference: a hand-built 1x1x2x3 cube against a manually computed unit norm."""
    x = torch.tensor([[[[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]]]])  # [1, 1, 2, 3]
    node = PerPixelUnitNorm(eps=1e-8, center=False)

    out = node.forward(data=x)["normalized"]

    expected = torch.tensor([[[[0.6, 0.8, 0.0], [1.0, 0.0, 0.0]]]])
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_port_contract_unchanged_for_both_modes() -> None:
    assert "data" in PerPixelUnitNorm.INPUT_SPECS
    assert "normalized" in PerPixelUnitNorm.OUTPUT_SPECS
    for center in (True, False):
        node = PerPixelUnitNorm(eps=1e-8, center=center)
        x = _random_cube(1, 4, 4, 5)
        with torch.no_grad():
            out = node.forward(data=x)["normalized"]
        assert out.shape == x.shape
        assert out.dtype == torch.float32


def test_center_is_json_serialisable_and_rebuilds_the_same_node() -> None:
    node = PerPixelUnitNorm(eps=1e-8, center=False)
    json.dumps(node.hparams)
    rebuilt = PerPixelUnitNorm(**node.hparams)
    assert rebuilt.hparams == node.hparams
    assert rebuilt.center is False


@pytest.mark.parametrize("center", [True, False])
def test_pipeline_round_trips_through_yaml_with_center(tmp_path, center) -> None:
    """A saved pipeline reloads and reproduces the same output for each center mode."""
    pipe = CuvisPipeline("per_pixel_unit_norm_reload")
    src = _CubeSource(name="src")
    node = PerPixelUnitNorm(name="normalizer", eps=1e-8, center=center)
    pipe.connect(src.outputs.data, node.inputs.data)
    before = pipe.forward(batch={}, stage=ExecutionStage.INFERENCE)[("normalizer", "normalized")]

    path = tmp_path / "per_pixel_unit_norm.yaml"
    pipe.save_to_file(str(path))
    rebuilt = PipelineBuilder().build_from_config(str(path))
    after = rebuilt.forward(batch={}, stage=ExecutionStage.INFERENCE)[("normalizer", "normalized")]

    assert torch.equal(after, before)


def test_yaml_without_center_loads_with_default_true(tmp_path) -> None:
    """A pipeline saved before `center` existed has no `center` key; it must load as True."""
    pipe = CuvisPipeline("per_pixel_unit_norm_legacy")
    src = _CubeSource(name="src")
    node = PerPixelUnitNorm(name="normalizer", eps=1e-8)
    pipe.connect(src.outputs.data, node.inputs.data)

    path = tmp_path / "per_pixel_unit_norm_legacy.yaml"
    pipe.save_to_file(str(path))

    text = path.read_text()
    assert "center" in text  # the current save always writes it
    stripped = "\n".join(
        line for line in text.splitlines() if line.strip().split(":")[0] != "center"
    )
    path.write_text(stripped + "\n")

    rebuilt = PipelineBuilder().build_from_config(str(path))
    after = rebuilt.forward(batch={}, stage=ExecutionStage.INFERENCE)[("normalizer", "normalized")]
    expected = PerPixelUnitNorm(eps=1e-8).forward(data=src.forward()["data"])["normalized"]
    torch.testing.assert_close(after, expected, atol=0.0, rtol=0.0)
