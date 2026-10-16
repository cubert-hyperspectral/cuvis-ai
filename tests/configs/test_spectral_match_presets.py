"""The Spectral Match presets: a signature from the request, a threshold input, both modes.

Each test builds the shipped yaml with its own hparam values, so the presets' defaults can change
without breaking these assertions. The scene is 8 x 8 pixels: material A on the left half,
material B on the right half, plus single pixels planted where a test needs them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from cuvis_ai_core.grpc import helpers
from cuvis_ai_core.grpc.inference_service import InferenceService
from cuvis_ai_core.grpc.session_manager import SessionManager
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_core.pipeline.factory import PipelineBuilder

pytestmark = pytest.mark.unit

_PRESETS = (
    Path(__file__).resolve().parents[2] / "cuvis_ai" / "configs" / "pipeline" / "spectral_match"
)
PLAIN = _PRESETS / "spectral_match.yaml"
VALIDITY = _PRESETS / "spectral_match_validity.yaml"
BOTH = pytest.mark.parametrize("preset", [PLAIN, VALIDITY], ids=["plain", "validity"])

C = 8
A = torch.linspace(1.0, 2.0, C) * 100.0
B = torch.linspace(2.0, 1.0, C) * 100.0
SPECK = (2, 6)  # one A pixel inside the B half
DARK = (7, 7)  # an all-zero pixel inside the B half


def _scene() -> torch.Tensor:
    cube = torch.empty(1, 8, 8, C)
    cube[0, :, :4] = A
    cube[0, :, 4:] = B
    cube[0, SPECK[0], SPECK[1]] = A
    cube[0, DARK[0], DARK[1]] = 0.0
    return cube


def _batch(cube: torch.Tensor, signature: torch.Tensor, **extra: torch.Tensor) -> dict:
    wavelengths = torch.arange(430, 430 + 8 * C, 8, dtype=torch.int32)[None]
    return {"cube": cube, "wavelengths": wavelengths, "spectral_signature": signature, **extra}


def _build(preset: Path, **hparams: dict):
    config = yaml.safe_load(preset.read_text(encoding="utf-8"))
    for node in config["nodes"]:
        node["hparams"] = {**(node.get("hparams") or {}), **hparams.get(node["name"], {})}
    return PipelineBuilder().build_from_config(config)


def _mask(outputs: dict) -> torch.Tensor:
    return outputs[("mask", "decisions")][0, ..., 0]


def _match(preset: Path, keep: str, threshold: float = 0.05, min_area: int = 0, **hparams):
    return _build(
        preset,
        match={"keep": keep, "threshold": threshold},
        mask={"min_area": min_area, "cell": 1},
        **hparams,
    )


SIG_A = A.view(1, 1, 1, C)


@BOTH
def test_the_signature_and_threshold_are_request_inputs(preset: Path) -> None:
    specs = _build(preset).get_input_specs()
    assert specs["spectral_signature"]["required"]
    assert not specs["threshold"]["required"]
    assert specs["cube"]["required"] and specs["wavelengths"]["required"]


@BOTH
def test_similar_marks_the_signature_material(preset: Path) -> None:
    mask = _mask(_match(preset, "below").forward(batch=_batch(_scene(), SIG_A)))
    expected = torch.zeros(8, 8, dtype=torch.bool)
    expected[:, :4] = True
    expected[SPECK] = True
    assert torch.equal(mask, expected)


@BOTH
def test_min_area_drops_the_single_pixel(preset: Path) -> None:
    mask = _mask(_match(preset, "below", min_area=2).forward(batch=_batch(_scene(), SIG_A)))
    assert mask[:, :4].all()
    assert not mask[SPECK]


@BOTH
def test_the_threshold_input_overrides_the_hparam(preset: Path) -> None:
    pipeline = _match(preset, "below", threshold=0.05)
    assert not _mask(pipeline.forward(batch=_batch(_scene(), SIG_A)))[0, 5]
    a, b = A.double(), B.double()
    angle_ab = torch.arccos(a @ b / (a.norm() * b.norm())).item()
    wide = torch.tensor([angle_ab + 0.1])
    assert _mask(pipeline.forward(batch=_batch(_scene(), SIG_A, threshold=wide)))[0, 5]


@BOTH
def test_several_labels_report_which_one_matched(preset: Path) -> None:
    signature = torch.stack([A, B]).view(2, 1, 1, C)
    out = _match(preset, "below").forward(batch=_batch(_scene(), signature))
    identity = out[("spectral_angle", "identity_mask")][0]
    assert (identity[:, :4] == 1).all()
    assert identity[0, 5].item() == 2
    assert _mask(out)[0, 5]


def test_different_lights_a_dark_pixel_in_the_plain_preset_only() -> None:
    plain = _mask(_match(PLAIN, "above").forward(batch=_batch(_scene(), SIG_A)))
    gated = _mask(_match(VALIDITY, "above").forward(batch=_batch(_scene(), SIG_A)))
    assert plain[DARK] and not gated[DARK]
    assert plain[0, 5] and gated[0, 5]  # material B is different in both
    assert not plain[:, :4].any() and not gated[:, :4].any()


def test_validity_keeps_a_saturated_pixel_off_when_saturation_is_set() -> None:
    cube = _scene()
    cube[0, 5, 5, 3] = 1000.0
    pipeline = _match(VALIDITY, "above", validity={"saturation_value": 900.0})
    out = pipeline.forward(batch=_batch(cube, SIG_A))
    assert not _mask(out)[5, 5]
    assert out[("validity", "invalid_reason")][0, 5, 5, 0].item() == 3


def test_validity_reports_a_frame_below_the_norm_floor() -> None:
    out = _match(VALIDITY, "above").forward(batch=_batch(_scene() * 0.01, SIG_A))
    assert out[("validity", "frame_status")].tolist() == [1]
    assert out[("validity", "valid_fraction")].tolist() == [0.0]
    assert not _mask(out).any()


@BOTH
def test_a_request_without_wavelengths_is_refused(preset: Path) -> None:
    batch = _batch(_scene(), SIG_A)
    del batch["wavelengths"]
    with pytest.raises(Exception, match="wavelengths"):
        _match(preset, "below").forward(batch=batch)


@BOTH
def test_a_signature_with_other_bands_is_refused(preset: Path) -> None:
    with pytest.raises(ValueError, match="channels"):
        _match(preset, "below").forward(batch=_batch(_scene(), SIG_A[..., :-1]))


@BOTH
def test_the_wire_format_reaches_the_ports(preset: Path) -> None:
    """A gRPC InputBatch with a [3, 1, 1, C] signature and a threshold, parsed as the server does."""
    signature = torch.stack([A, B, (A + B) / 2]).view(3, 1, 1, C)
    direct = _batch(_scene(), signature, threshold=torch.tensor([0.05]))
    inputs = cuvis_ai_pb2.InputBatch(
        cube=helpers.tensor_to_proto(direct["cube"]),
        wavelengths=helpers.tensor_to_proto(direct["wavelengths"]),
        extra_inputs={
            "spectral_signature": helpers.tensor_to_proto(signature),
            "threshold": helpers.tensor_to_proto(direct["threshold"]),
        },
    )
    parsed = InferenceService(SessionManager())._parse_input_batch(inputs)
    assert parsed["spectral_signature"].shape == (3, 1, 1, C)
    pipeline = _match(preset, "below")
    via_wire = pipeline.forward(batch=parsed)
    reference = pipeline.forward(batch=direct)
    for key in [("mask", "decisions"), ("spectral_angle", "identity_mask")]:
        assert torch.equal(via_wire[key], reference[key])
