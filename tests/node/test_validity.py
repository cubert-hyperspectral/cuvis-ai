"""Tests for SpectralValidity: port contract and per-pixel/per-frame rules."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.preprocessors import SpectralValidity

pytestmark = pytest.mark.unit


def _cube(b: int, h: int, w: int, c: int, fill: float = 5000.0) -> torch.Tensor:
    return torch.full((b, h, w, c), fill, dtype=torch.float32)


def test_port_contract_shapes_and_dtypes() -> None:
    node = SpectralValidity(n_features=4, norm_floor=1.0)
    cube = _cube(2, 3, 3, 4)
    out = node.forward(cube=cube)
    assert out["valid_mask"].shape == (2, 3, 3, 1)
    assert out["valid_mask"].dtype == torch.bool
    assert out["invalid_reason"].shape == (2, 3, 3, 1)
    assert out["invalid_reason"].dtype == torch.int32
    assert out["frame_status"].shape == (2,)
    assert out["frame_status"].dtype == torch.int32
    assert out["valid_fraction"].shape == (2,)
    assert out["valid_fraction"].dtype == torch.float32


def test_all_valid_pixel_gives_reason_zero() -> None:
    node = SpectralValidity(n_features=4, norm_floor=1.0)
    cube = _cube(1, 2, 2, 4, fill=100.0)
    out = node.forward(cube=cube)
    assert out["valid_mask"].all()
    assert (out["invalid_reason"] == 0).all()
    assert out["frame_status"][0].item() == 0
    assert out["valid_fraction"][0].item() == pytest.approx(1.0)


def test_non_finite_pixel_flagged() -> None:
    node = SpectralValidity(n_features=4, norm_floor=1.0)
    cube = _cube(1, 2, 2, 4, fill=100.0)
    cube[0, 0, 0, 0] = float("nan")
    out = node.forward(cube=cube)
    assert not out["valid_mask"][0, 0, 0, 0]
    assert out["invalid_reason"][0, 0, 0, 0].item() == 1


def test_low_norm_pixel_flagged() -> None:
    node = SpectralValidity(n_features=4, norm_floor=100.0)
    cube = torch.zeros(1, 2, 2, 4)
    cube[0, 0, 0] = torch.tensor([1.0, 1.0, 1.0, 1.0])  # norm = 2.0 < 100
    cube[0, 0, 1] = torch.tensor([1000.0, 0, 0, 0])  # norm = 1000 >= 100
    out = node.forward(cube=cube)
    assert not out["valid_mask"][0, 0, 0, 0]
    assert out["invalid_reason"][0, 0, 0, 0].item() == 2
    assert out["valid_mask"][0, 0, 1, 0]


def test_saturated_band_flagged() -> None:
    node = SpectralValidity(n_features=4, norm_floor=1.0, saturation_value=10000.0)
    cube = _cube(1, 1, 2, 4, fill=100.0)
    cube[0, 0, 0, 2] = 10000.0  # saturated
    out = node.forward(cube=cube)
    assert not out["valid_mask"][0, 0, 0, 0]
    assert out["invalid_reason"][0, 0, 0, 0].item() == 3
    assert out["valid_mask"][0, 0, 1, 0]


def test_saturation_disabled_by_default() -> None:
    node = SpectralValidity(n_features=4, norm_floor=1.0)
    cube = _cube(1, 1, 1, 4, fill=1e9)
    out = node.forward(cube=cube)
    assert out["valid_mask"].all()


def test_roi_margin_flags_border() -> None:
    node = SpectralValidity(
        n_features=4,
        norm_floor=1.0,
        roi_margin_px=1,
        roi_reference_hw=[4, 4],
    )
    cube = _cube(1, 4, 4, 4, fill=100.0)
    out = node.forward(cube=cube)
    # border ring (margin 1) invalid, interior 2x2 valid
    assert not out["valid_mask"][0, 0, 0, 0]
    assert not out["valid_mask"][0, 0, 3, 0]
    assert out["valid_mask"][0, 1, 1, 0]
    assert out["invalid_reason"][0, 0, 0, 0].item() == 4


def test_roi_margin_scales_with_resolution() -> None:
    # reference 4x4 with margin 1 -> at 8x8 the margin should be 2
    node = SpectralValidity(
        n_features=4,
        norm_floor=1.0,
        roi_margin_px=1,
        roi_reference_hw=[4, 4],
    )
    cube = _cube(1, 8, 8, 4, fill=100.0)
    out = node.forward(cube=cube)
    assert not out["valid_mask"][0, 1, 1, 0]  # within scaled margin of 2
    assert out["valid_mask"][0, 2, 2, 0]  # just inside


def test_band_count_mismatch_sets_input_mismatch_and_all_invalid() -> None:
    node = SpectralValidity(n_features=4, norm_floor=1.0)
    cube = _cube(1, 2, 2, 5, fill=100.0)  # wrong channel count
    out = node.forward(cube=cube)
    assert not out["valid_mask"].any()
    assert out["frame_status"][0].item() == 2


def test_wavelength_mismatch_sets_input_mismatch() -> None:
    node = SpectralValidity(
        n_features=3,
        expected_wavelengths=[430.0, 438.0, 446.0],
        wavelength_tol_nm=1.0,
        norm_floor=1.0,
    )
    cube = _cube(1, 2, 2, 3, fill=100.0)
    bad_wavelengths = np.array([430.0, 438.0, 500.0], dtype=np.int32)  # last band way off
    out = node.forward(cube=cube, wavelengths=bad_wavelengths)
    assert not out["valid_mask"].any()
    assert out["frame_status"][0].item() == 2


def test_wavelength_within_tolerance_passes() -> None:
    node = SpectralValidity(
        n_features=3,
        expected_wavelengths=[430.0, 438.0, 446.0],
        wavelength_tol_nm=1.0,
        norm_floor=1.0,
    )
    cube = _cube(1, 2, 2, 3, fill=100.0)
    good_wavelengths = np.array([430.0, 438.4, 446.0], dtype=np.float64)
    out = node.forward(cube=cube, wavelengths=good_wavelengths)
    assert out["valid_mask"].all()
    assert out["frame_status"][0].item() == 0


def test_wavelength_accepts_torch_tensor() -> None:
    node = SpectralValidity(
        n_features=3,
        expected_wavelengths=[430.0, 438.0, 446.0],
        wavelength_tol_nm=1.0,
        norm_floor=1.0,
    )
    cube = _cube(1, 2, 2, 3, fill=100.0)
    good_wavelengths = torch.tensor([430, 438, 446], dtype=torch.int32)
    out = node.forward(cube=cube, wavelengths=good_wavelengths)
    assert out["frame_status"][0].item() == 0


def test_frame_status_invalid_when_valid_fraction_too_low() -> None:
    node = SpectralValidity(n_features=4, norm_floor=100.0, min_valid_fraction=0.9)
    cube = torch.zeros(1, 2, 2, 4)
    cube[0, 0, 0] = torch.tensor([1000.0, 0, 0, 0])  # only 1/4 valid
    out = node.forward(cube=cube)
    assert out["frame_status"][0].item() == 1
    assert out["valid_fraction"][0].item() == pytest.approx(0.25)


def test_rebuild_from_hparams() -> None:
    node = SpectralValidity(
        n_features=4,
        norm_floor=2.0,
        saturation_value=9000.0,
        roi_margin_px=5,
        roi_reference_hw=[1000, 1080],
        min_valid_fraction=0.3,
    )
    rebuilt = type(node)(**node.hparams)
    assert rebuilt.n_features == 4
    assert rebuilt.norm_floor == 2.0
    assert rebuilt.saturation_value == 9000.0
    assert rebuilt.roi_margin_px == 5
    assert rebuilt.min_valid_fraction == 0.3


def test_state_dict_unchanged_by_forward() -> None:
    node = SpectralValidity(n_features=4, norm_floor=1.0)
    before = {k: v.clone() for k, v in node.state_dict().items()}
    node.forward(cube=_cube(1, 2, 2, 4))
    after = node.state_dict()
    assert before.keys() == after.keys()
    for k in before:
        assert torch.equal(before[k], after[k])


def test_wavelengths_port_connects_to_the_cu3s_data_node() -> None:
    """The builtin data node publishes wavelengths as np.int32 [C]; the pipeline must accept it."""
    from cuvis_ai_core.pipeline.factory import PipelineBuilder

    config = {
        "metadata": {"name": "wiring"},
        "nodes": [
            {"name": "src", "class_name": "cuvis_ai.node.data.CU3SDataNode", "hparams": {}},
            {
                "name": "validity",
                "class_name": "cuvis_ai.node.preprocessors.SpectralValidity",
                "hparams": {"n_features": 4},
            },
        ],
        "connections": [
            {"source": "src.outputs.cube", "target": "validity.inputs.cube"},
            {"source": "src.outputs.wavelengths", "target": "validity.inputs.wavelengths"},
        ],
    }
    pipeline = PipelineBuilder().build_from_config(config)
    assert {n.name for n in pipeline.nodes} == {"src", "validity"}


def test_any_band_count_without_n_features() -> None:
    node = SpectralValidity(norm_floor=1.0)
    assert node.n_features is None
    out = node.forward(cube=_cube(1, 2, 2, 164, fill=100.0))
    assert out["valid_mask"].all()
    assert out["frame_status"][0].item() == 0


def test_n_features_still_flags_another_band_count() -> None:
    out = SpectralValidity(n_features=61, norm_floor=1.0).forward(cube=_cube(1, 2, 2, 164))
    assert not out["valid_mask"].any()
    assert out["frame_status"][0].item() == 2


def test_rebuild_from_hparams_without_n_features() -> None:
    node = SpectralValidity(norm_floor=50.0)
    rebuilt = type(node)(**node.hparams)
    assert rebuilt.n_features is None
    assert rebuilt.hparams == node.hparams


def test_bad_n_features_raises() -> None:
    with pytest.raises(ValueError, match="n_features"):
        SpectralValidity(n_features=0)
