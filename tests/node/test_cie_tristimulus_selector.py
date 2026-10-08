"""CIETristimulusRGBSelector: D65 rendering, legacy illuminant E, band info, NIR rejection, caching."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from loguru import logger

from cuvis_ai.node.channel_selector import CIETristimulusRGBSelector
from cuvis_ai_core.pipeline.factory import PipelineBuilder

pytestmark = pytest.mark.unit

# Golden reference inputs: the CIE 1931 CMFs and D65 at 5 nm (CIE 015, CIE S 014-2) as
# tabulated on the node, the sRGB matrix (IEC 61966-2-1) and the published Bradford matrix.
_XMR_WAVELENGTHS = np.linspace(430.0, 910.0, 61)  # XMR band centres, 8 nm spacing
_CIE_WAVELENGTHS = np.arange(380.0, 781.0, 5.0)
_SRGB = CIETristimulusRGBSelector._XYZ_TO_SRGB
_BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]
)
_NODE_CLASS = "cuvis_ai.node.channel_selector.CIETristimulusRGBSelector"
_D65 = CIETristimulusRGBSelector._ILLUMINANTS.index("D65")  # marker codes of the running bounds
_E = CIETristimulusRGBSelector._ILLUMINANTS.index("E")


def _cube_and_wavelengths() -> tuple[torch.Tensor, np.ndarray]:
    wavelengths = np.arange(400.0, 1000.0, 20.0, dtype=np.float32)  # 30 bands, 400 to 980 nm
    cube = torch.rand(2, 4, 5, wavelengths.size, generator=torch.Generator().manual_seed(0))
    return cube, wavelengths


def _reference_xyz_weights(wavelengths: np.ndarray, illuminant: str) -> np.ndarray:
    """Plain numpy XYZ weights per band: CMF times band width, times D65 for D65."""
    node = CIETristimulusRGBSelector
    cmfs = np.stack(
        [
            np.interp(wavelengths, node._CMF_WAVELENGTHS, bar, left=0.0, right=0.0)
            for bar in (node._X_BAR, node._Y_BAR, node._Z_BAR)
        ]
    )
    weights = cmfs * np.gradient(wavelengths)
    if illuminant == "D65":
        weights = weights * np.interp(wavelengths, node._CMF_WAVELENGTHS, node._D65_SPD)
    return weights


def _reference_d65_rgb(reflectance: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Reflectance under D65, white normalised, Bradford adapted to the sRGB white."""
    weights = _reference_xyz_weights(wavelengths, "D65")
    white = weights.sum(axis=1)
    target = np.linalg.solve(_SRGB, np.ones(3))
    gain = np.diag((_BRADFORD @ target) / (_BRADFORD @ white))
    adapt = np.linalg.inv(_BRADFORD) @ gain @ _BRADFORD
    return _SRGB @ adapt @ (weights @ reflectance)


def _raw_rgb(
    node: CIETristimulusRGBSelector, reflectance: np.ndarray, wavelengths: np.ndarray
) -> np.ndarray:
    cube = torch.tensor(reflectance, dtype=torch.float32).reshape(1, 1, 1, -1)
    raw = node._compute_raw_rgb(cube, wavelengths.astype(np.float32))
    return raw[0, 0, 0].double().numpy()


def test_cie_selector_renders_unit_range_rgb_with_band_info() -> None:
    cube, wavelengths = _cube_and_wavelengths()
    node = CIETristimulusRGBSelector(apply_gamma=False)
    out = node.forward(cube=cube, wavelengths=wavelengths)
    rgb = out["rgb_image"]
    assert rgb.shape == (2, 4, 5, 3)
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0

    info = out["band_info"]
    assert info["strategy"] == "cie_tristimulus" and info["illuminant"] == "D65"
    assert info["apply_gamma"] is False
    assert info["sensor_bands_total"] == 30
    assert info["wavelength_range_nm"] == [400.0, 980.0]
    cmf_sum = np.sum(
        [
            np.interp(wavelengths, node._CMF_WAVELENGTHS, bar, left=0.0, right=0.0)
            for bar in (node._X_BAR, node._Y_BAR, node._Z_BAR)
        ],
        axis=0,
    )
    assert info["sensor_bands_visible"] == int((cmf_sum > 1e-6).sum())
    assert 18 <= info["sensor_bands_visible"] <= 20


@pytest.mark.parametrize("illuminant", ["D65", "E"])
def test_cie_selector_gives_no_colour_to_nir_only_energy(illuminant: str) -> None:
    wavelengths = np.arange(400.0, 1000.0, 20.0, dtype=np.float32)
    cube = torch.zeros(1, 2, 2, wavelengths.size)
    cube[..., torch.from_numpy(wavelengths > 800.0)] = 1.0
    raw = CIETristimulusRGBSelector(illuminant=illuminant)._compute_raw_rgb(cube, wavelengths)
    assert torch.equal(raw, torch.zeros_like(raw))


def test_cie_selector_caches_the_cmf_weights_per_wavelength_grid() -> None:
    cube, wavelengths = _cube_and_wavelengths()
    node = CIETristimulusRGBSelector()
    node.forward(cube=cube, wavelengths=wavelengths)
    weights = node._cmf_weights
    assert weights is not None and tuple(weights.shape) == (3, 30)

    node.forward(cube=cube, wavelengths=wavelengths)
    assert node._cmf_weights is weights

    node.forward(cube=cube[..., :15], wavelengths=wavelengths[:15])
    assert node._cmf_weights is not weights and tuple(node._cmf_weights.shape) == (3, 15)
    assert (
        node.forward(cube=cube[..., :15], wavelengths=wavelengths[:15])["band_info"][
            "sensor_bands_total"
        ]
        == 15
    )


def test_d65_table_has_the_d65_white_point() -> None:
    node = CIETristimulusRGBSelector
    xyz = np.array([np.sum(node._D65_SPD * bar) for bar in (node._X_BAR, node._Y_BAR, node._Z_BAR)])
    np.testing.assert_allclose(xyz[:2] / xyz.sum(), [0.3127, 0.3290], atol=2e-4)


@pytest.mark.parametrize("wavelengths", [_XMR_WAVELENGTHS, _CIE_WAVELENGTHS], ids=["xmr", "cie"])
def test_d65_maps_a_perfect_white_to_srgb_white_and_scales_linearly(
    wavelengths: np.ndarray,
) -> None:
    node = CIETristimulusRGBSelector()
    ones = np.ones(wavelengths.size)
    np.testing.assert_allclose(_raw_rgb(node, ones, wavelengths), [1.0, 1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(_raw_rgb(node, 0.5 * ones, wavelengths), [0.5, 0.5, 0.5], atol=1e-5)
    assert float(node._cmf_weights[1].double().sum()) == pytest.approx(1.0, abs=1e-5)  # Y_white


@pytest.mark.parametrize("wavelengths", [_XMR_WAVELENGTHS, _CIE_WAVELENGTHS], ids=["xmr", "cie"])
def test_d65_matches_the_reference_rendering_of_a_coloured_spectrum(
    wavelengths: np.ndarray,
) -> None:
    reflectance = 0.2 + 0.6 / (1.0 + np.exp(-(wavelengths - 580.0) / 20.0))  # orange edge
    expected = np.clip(_reference_d65_rgb(reflectance, wavelengths), 0.0, None)
    got = _raw_rgb(CIETristimulusRGBSelector(), reflectance, wavelengths)
    np.testing.assert_allclose(got, expected, atol=1e-5)
    assert got[0] > got[1] > got[2]


def test_d65_adaptation_is_near_identity_on_the_full_cie_range() -> None:
    weights = _reference_xyz_weights(_CIE_WAVELENGTHS, "D65")
    node = CIETristimulusRGBSelector()
    node.forward(cube=torch.ones(1, 1, 1, _CIE_WAVELENGTHS.size), wavelengths=_CIE_WAVELENGTHS)
    np.testing.assert_allclose(
        node._cmf_weights.double().numpy(), weights / weights[1].sum(), atol=1e-4
    )


def test_illuminant_e_keeps_the_legacy_unnormalised_warm_white() -> None:
    ones = np.ones(_XMR_WAVELENGTHS.size)
    expected = _SRGB @ _reference_xyz_weights(_XMR_WAVELENGTHS, "E").sum(axis=1)
    got = _raw_rgb(CIETristimulusRGBSelector(illuminant="E"), ones, _XMR_WAVELENGTHS)
    np.testing.assert_allclose(got, expected, rtol=1e-5)
    np.testing.assert_allclose(got / got[0], [1.0, 0.811, 0.674], atol=1e-3)


def test_band_info_reports_illuminant_and_white_coverage() -> None:
    cube = torch.rand(1, 3, 4, _XMR_WAVELENGTHS.size, generator=torch.Generator().manual_seed(1))
    info = CIETristimulusRGBSelector().forward(cube=cube, wavelengths=_XMR_WAVELENGTHS)["band_info"]
    assert info["illuminant"] == "D65" and info["white_normalised"] is True
    x, y, z = info["white_xyz_coverage"]
    assert y > 0.99 and x < 0.99 and 0.90 < z < 0.92  # XMR misses z_bar below 430 nm
    assert info["white_adapted"] is True

    legacy = CIETristimulusRGBSelector(illuminant="E")
    info = legacy.forward(cube=cube, wavelengths=_XMR_WAVELENGTHS)["band_info"]
    assert info["illuminant"] == "E" and info["white_normalised"] is False
    assert info["white_xyz_coverage"] is None and info["white_adapted"] is None


def test_grid_too_narrow_to_adapt_only_normalises_y_and_says_so() -> None:
    wavelengths = np.linspace(500.0, 900.0, 51)  # sees about 5 % of the z_bar white
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING")
    try:
        node = CIETristimulusRGBSelector()
        info = node.forward(cube=torch.ones(1, 1, 1, 51), wavelengths=wavelengths)["band_info"]
    finally:
        logger.remove(sink)
    assert info["white_adapted"] is False and info["white_xyz_coverage"][2] < 0.5
    weights = node._cmf_weights.double().numpy()
    np.testing.assert_allclose(weights.sum(axis=1)[1], 1.0, atol=1e-6)  # Y_white = 1
    reference = _reference_xyz_weights(wavelengths, "D65")
    np.testing.assert_allclose(weights, reference / reference[1].sum(), atol=1e-6)
    assert len(messages) == 1 and "not neutral" in messages[0]


def test_partial_band_range_warns_once() -> None:
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING")
    try:
        node = CIETristimulusRGBSelector()
        cube = torch.ones(1, 1, 1, _XMR_WAVELENGTHS.size)
        node.forward(cube=cube, wavelengths=_XMR_WAVELENGTHS)
        node.forward(cube=cube[..., :40], wavelengths=_XMR_WAVELENGTHS[:40])
        full = torch.ones(1, 1, 1, _CIE_WAVELENGTHS.size)
        CIETristimulusRGBSelector().forward(cube=full, wavelengths=_CIE_WAVELENGTHS)
    finally:
        logger.remove(sink)
    assert len(messages) == 1 and "430 to 910 nm" in messages[0]


def test_unknown_illuminant_is_rejected() -> None:
    with pytest.raises(ValueError, match="illuminant"):
        CIETristimulusRGBSelector(illuminant="A")


def test_port_contract_matches_output_specs() -> None:
    out = CIETristimulusRGBSelector().forward(
        cube=torch.rand(2, 3, 4, 61), wavelengths=_XMR_WAVELENGTHS
    )
    spec = CIETristimulusRGBSelector.OUTPUT_SPECS["rgb_image"]
    assert out["rgb_image"].dtype == spec.dtype and out["rgb_image"].shape == (2, 3, 4, 3)
    assert isinstance(out["band_info"], dict)


def _single_node_config(hparams: dict) -> dict:
    return {
        "metadata": {"name": "cie"},
        "nodes": [{"name": "true_rgb", "class_name": _NODE_CLASS, "hparams": hparams}],
        "connections": [],
    }


@pytest.mark.parametrize("illuminant", ["D65", "E"])
def test_illuminant_survives_the_pipeline_yaml_round_trip(tmp_path, illuminant: str) -> None:
    yaml_path = tmp_path / "cie.yaml"
    pipeline = PipelineBuilder().build_from_config(_single_node_config({"illuminant": illuminant}))
    pipeline.save_to_file(yaml_path, save_weights=False)
    rebuilt = PipelineBuilder().build_from_config(yaml_path)
    assert next(n for n in rebuilt.nodes if n.name == "true_rgb").illuminant == illuminant


def test_preset_without_hparams_renders_under_d65() -> None:
    pipeline = PipelineBuilder().build_from_config(_single_node_config({}))
    assert next(n for n in pipeline.nodes if n.name == "true_rgb").illuminant == "D65"


def _legacy_state_dict(node: CIETristimulusRGBSelector) -> dict:
    """A checkpoint written before the illuminant marker existed."""
    return {k: v for k, v in node.state_dict().items() if k != "_bounds_illuminant"}


def _xmr_frames(n: int = 25) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    return 0.2 + 0.6 * torch.rand(n, 1, 4, 5, _XMR_WAVELENGTHS.size, generator=generator)


def _render(node: CIETristimulusRGBSelector, frames: torch.Tensor) -> list[torch.Tensor]:
    return [node.forward(cube=frame, wavelengths=_XMR_WAVELENGTHS)["rgb_image"] for frame in frames]


def _load_with_warnings(node: CIETristimulusRGBSelector, state: dict) -> list[str]:
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING")
    try:
        node.load_state_dict(state, strict=True)
    finally:
        logger.remove(sink)
    return messages


def test_legacy_running_checkpoint_rewarms_under_d65() -> None:
    frames = _xmr_frames()
    legacy = CIETristimulusRGBSelector(illuminant="E", apply_gamma=False)
    _render(legacy, frames)
    assert int(legacy._norm_frame_count.item()) == 25

    restored = CIETristimulusRGBSelector(apply_gamma=False)
    messages = _load_with_warnings(restored, _legacy_state_dict(legacy))
    assert len(messages) == 1 and "fitted under illuminant E" in messages[0]
    assert "re-warming" in messages[0]
    assert torch.isnan(restored.running_min).all() and torch.isnan(restored.running_max).all()
    assert int(restored._norm_frame_count.item()) == 0
    assert restored.illuminant == "D65" and int(restored._bounds_illuminant.item()) == _D65

    fresh = CIETristimulusRGBSelector(apply_gamma=False)
    for got, expected in zip(_render(restored, frames), _render(fresh, frames), strict=True):
        assert torch.equal(got, expected)


def test_legacy_statistical_checkpoint_falls_back_to_per_frame() -> None:
    frames = _xmr_frames(4)
    legacy = CIETristimulusRGBSelector(illuminant="E", norm_mode="statistical", apply_gamma=False)
    legacy.statistical_initialization(
        iter([{"cube": frame, "wavelengths": _XMR_WAVELENGTHS} for frame in frames])
    )
    assert legacy._statistically_initialized

    restored = CIETristimulusRGBSelector(norm_mode="statistical", apply_gamma=False)
    messages = _load_with_warnings(restored, _legacy_state_dict(legacy))
    assert len(messages) == 1 and "re-fitted" in messages[0]
    assert restored._statistically_initialized is False
    assert torch.isnan(restored.running_min).all()

    fresh = CIETristimulusRGBSelector(norm_mode="statistical", apply_gamma=False)
    for got, expected in zip(_render(restored, frames), _render(fresh, frames), strict=True):
        assert torch.equal(got, expected) and not torch.isnan(got).any()


def test_legacy_bounds_in_per_frame_mode_reset_silently() -> None:
    legacy = CIETristimulusRGBSelector(illuminant="E", apply_gamma=False)
    _render(legacy, _xmr_frames(3))
    restored = CIETristimulusRGBSelector(norm_mode="per_frame", apply_gamma=False)
    assert _load_with_warnings(restored, _legacy_state_dict(legacy)) == []
    assert torch.isnan(restored.running_min).all() and restored.illuminant == "D65"


def test_legacy_unfitted_checkpoint_loads_unchanged() -> None:
    restored = CIETristimulusRGBSelector(apply_gamma=False)
    legacy = _legacy_state_dict(CIETristimulusRGBSelector(illuminant="E"))
    assert _load_with_warnings(restored, legacy) == []
    assert int(restored._norm_frame_count.item()) == 0
    assert int(restored._bounds_illuminant.item()) == _D65 and restored.illuminant == "D65"


def test_current_checkpoint_round_trip_keeps_bounds() -> None:
    frames = _xmr_frames(12)
    fitted = CIETristimulusRGBSelector(apply_gamma=False)
    _render(fitted, frames)
    restored = CIETristimulusRGBSelector(apply_gamma=False)
    assert _load_with_warnings(restored, fitted.state_dict()) == []
    assert torch.equal(restored.running_min, fitted.running_min)
    assert torch.equal(restored.running_max, fitted.running_max)
    assert int(restored._norm_frame_count.item()) == 12
    assert torch.equal(
        restored.forward(cube=frames[0], wavelengths=_XMR_WAVELENGTHS)["rgb_image"],
        fitted.forward(cube=frames[0], wavelengths=_XMR_WAVELENGTHS)["rgb_image"],
    )


def test_bounds_fitted_under_another_illuminant_are_discarded() -> None:
    fitted = CIETristimulusRGBSelector(apply_gamma=False)
    _render(fitted, _xmr_frames(3))
    restored = CIETristimulusRGBSelector(illuminant="E", apply_gamma=False)
    messages = _load_with_warnings(restored, fitted.state_dict())
    assert len(messages) == 1 and "fitted under illuminant D65" in messages[0]
    assert torch.isnan(restored.running_min).all()
    assert int(restored._bounds_illuminant.item()) == _E and restored.illuminant == "E"


def test_grid_without_visible_bands_renders_black_and_reports_zero_coverage() -> None:
    wavelengths = np.linspace(900.0, 1700.0, 100)
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING")
    try:
        out = CIETristimulusRGBSelector(apply_gamma=False).forward(
            cube=torch.ones(1, 2, 2, 100), wavelengths=wavelengths
        )
    finally:
        logger.remove(sink)
    assert torch.equal(out["rgb_image"], torch.zeros(1, 2, 2, 3))
    assert out["band_info"]["white_xyz_coverage"] == [0.0, 0.0, 0.0]
    assert out["band_info"]["white_adapted"] is False
    assert out["band_info"]["sensor_bands_visible"] == 0
    assert len(messages) == 1 and "not neutral" in messages[0]
