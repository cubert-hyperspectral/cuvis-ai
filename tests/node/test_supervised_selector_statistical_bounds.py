"""The statistical normalisation bounds of the supervised selectors.

``SupervisedCIRSelector``, ``SupervisedWindowedSelector`` and ``SupervisedFullSpectrumSelector``
fit their band choice from masks. In ``statistical`` mode the same fit also fixes the display
bounds of the three chosen bands (the percentiles over the training cubes), so the render is
finite and frozen; up to cuvis-ai 0.18.3 the bounds stayed NaN and so did the image (issue 145).
A checkpoint without usable bounds keeps its band choice, renders per frame and says so once at
load; a fit after that load ends with usable bounds again.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
import torch
from loguru import logger

from cuvis_ai.node._running_bounds import normalize_with_bounds
from cuvis_ai.node.channel_selector import (
    ChannelSelectorBase,
    CIRSelector,
    SupervisedCIRSelector,
    SupervisedFullSpectrumSelector,
    SupervisedWindowedSelector,
)
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

pytestmark = pytest.mark.unit

N_BANDS = 12
# 12 bands on 450 to 950 nm: band 9 is 859 nm and band 10 is 905 nm (the CIR NIR window),
# bands 4 and 5 are 632 and 677 nm (the windowed red window).
WAVELENGTHS = np.linspace(450.0, 950.0, N_BANDS).astype(np.float32)
SUPERVISED = [SupervisedCIRSelector, SupervisedWindowedSelector, SupervisedFullSpectrumSelector]
# Two bands the fit can tell apart inside one window of each selector.
LIT_BANDS = {
    SupervisedCIRSelector: (9, 10),
    SupervisedWindowedSelector: (4, 5),
    SupervisedFullSpectrumSelector: (7, 3),
}
QUANTILES = (ChannelSelectorBase._NORM_QUANTILE_LOW, ChannelSelectorBase._NORM_QUANTILE_HIGH)


def _fit_data(seed: int, lit_band: int, scale: float = 100.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Two cubes with a positive class that lights one band."""
    torch.manual_seed(seed)
    cube = torch.rand(2, 6, 6, N_BANDS) * scale
    mask = torch.zeros(2, 6, 6, 1)
    mask[:, :3] = 1.0
    cube[:, :3, :, lit_band] += scale / 2
    return cube, mask


def _stream(cube: torch.Tensor, mask: torch.Tensor):
    return iter([{"cube": cube, "mask": mask, "wavelengths": WAVELENGTHS}])


def _fitted(cls: type, lit_band: int, seed: int = 1, scale: float = 100.0, **hparams: Any):
    node = cls(num_spectral_bands=N_BANDS, norm_mode="statistical", **hparams)
    cube, mask = _fit_data(seed, lit_band, scale)
    node.statistical_initialization(_stream(cube, mask))
    return node, cube


def _raw_bands(cube: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return torch.stack([cube[..., i] for i in indices], dim=-1)


def _per_cube_bounds(cube: torch.Tensor, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """What the supervised fit accumulates: per-cube percentiles, min of lows, max of highs."""
    lows, highs = [], []
    for sample in _raw_bands(cube, indices):
        flat = sample.reshape(-1, 3)
        lows.append(torch.quantile(flat, QUANTILES[0], dim=0))
        highs.append(torch.quantile(flat, QUANTILES[1], dim=0))
    return torch.stack(lows).amin(dim=0), torch.stack(highs).amax(dim=0)


@contextmanager
def _captured_warnings() -> Generator[list[str], None, None]:
    messages: list[str] = []
    handler = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(handler)


def _without_bounds(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The checkpoint shape cuvis-ai up to 0.18.3 wrote: the band choice, NaN bounds."""
    state = dict(state)
    state["running_min"] = torch.full((3,), float("nan"))
    state["running_max"] = torch.full((3,), float("nan"))
    return state


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("cls", SUPERVISED)
def test_statistical_fit_fills_the_bounds_of_the_chosen_bands(cls: type) -> None:
    node, cube = _fitted(cls, LIT_BANDS[cls][0])
    indices = node.selected_indices.tolist()

    assert not torch.isnan(node.running_min).any() and not torch.isnan(node.running_max).any()
    low, high = _per_cube_bounds(cube, indices)
    assert torch.allclose(node.running_min, low) and torch.allclose(node.running_max, high)

    out = node.forward(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"]
    expected = node._srgb_gamma(
        normalize_with_bounds(_raw_bands(cube, indices), node.running_min, node.running_max, 1e-8)
    )
    assert torch.equal(out, expected)
    assert out.min() >= 0.0 and out.max() <= 1.0

    other, _ = _fit_data(2, LIT_BANDS[cls][0], scale=400.0)
    later = node.forward(cube=other, wavelengths=WAVELENGTHS)["rgb_image"]
    assert not torch.isnan(later).any()
    assert torch.allclose(node.running_min, low) and torch.allclose(node.running_max, high)


@torch.no_grad()
@pytest.mark.parametrize("cls", SUPERVISED)
@pytest.mark.parametrize("norm_mode", ["running", "per_frame"])
def test_running_and_per_frame_fits_leave_the_bounds_alone(cls: type, norm_mode: str) -> None:
    node = cls(num_spectral_bands=N_BANDS, norm_mode=norm_mode)
    node.statistical_initialization(_stream(*_fit_data(1, LIT_BANDS[cls][0])))
    assert node._statistically_initialized is True
    assert torch.isnan(node.running_min).all() and torch.isnan(node.running_max).all()


@torch.no_grad()
def test_statistical_fit_with_the_switch_off_leaves_the_bounds_alone() -> None:
    node, cube = _fitted(SupervisedCIRSelector, 9, normalize_output=False)
    assert torch.isnan(node.running_min).all()
    out = node.forward(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"]
    assert torch.equal(out, _raw_bands(cube, node.selected_indices.tolist()))


@torch.no_grad()
@pytest.mark.parametrize("cls", SUPERVISED)
def test_a_refit_resets_the_bounds_to_the_new_band_choice(cls: type) -> None:
    first, second = LIT_BANDS[cls]
    node, _ = _fitted(cls, first)
    chosen_first = node.selected_indices.tolist()
    max_first = node.running_max.clone()

    cube, mask = _fit_data(3, second, scale=40.0)
    node.statistical_initialization(_stream(cube, mask))
    chosen_second = node.selected_indices.tolist()
    assert chosen_second != chosen_first

    low, high = _per_cube_bounds(cube, chosen_second)
    assert torch.allclose(node.running_min, low) and torch.allclose(node.running_max, high)
    # A merge with the first fit would have kept the larger maximum of the wider data.
    assert bool((node.running_max < max_first).all())


@pytest.mark.parametrize("cls", SUPERVISED)
def test_compute_raw_rgb_requires_a_fitted_selection(cls: type) -> None:
    node = cls(num_spectral_bands=N_BANDS, norm_mode="statistical")
    with pytest.raises(RuntimeError, match="not fitted"):
        node._compute_raw_rgb(torch.rand(1, 2, 2, N_BANDS), WAVELENGTHS)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("cls", SUPERVISED)
def test_a_checkpoint_without_bounds_renders_per_frame_with_one_warning(cls: type) -> None:
    fitted, cube = _fitted(cls, LIT_BANDS[cls][0])
    restored = cls(num_spectral_bands=N_BANDS, norm_mode="statistical")
    with _captured_warnings() as messages:
        restored.load_state_dict(_without_bounds(fitted.state_dict()))
    assert len(messages) == 1
    assert cls.__name__ in messages[0] and "bounds" in messages[0]

    assert restored._statistically_initialized is True
    assert restored.selected_indices.tolist() == fitted.selected_indices.tolist()
    with _captured_warnings() as messages:
        out = restored.forward(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"]
    assert messages == []
    assert not torch.isnan(out).any()
    raw = _raw_bands(cube, restored.selected_indices.tolist())
    assert torch.equal(out, restored._srgb_gamma(restored._per_frame_normalize(raw)))


@torch.no_grad()
@pytest.mark.parametrize(
    "norm_mode, normalize_output", [("running", True), ("per_frame", True), ("statistical", False)]
)
def test_a_checkpoint_without_bounds_loads_silently_when_they_are_not_read(
    norm_mode: str, normalize_output: bool
) -> None:
    fitted, _ = _fitted(SupervisedCIRSelector, 9)
    restored = SupervisedCIRSelector(
        num_spectral_bands=N_BANDS, norm_mode=norm_mode, normalize_output=normalize_output
    )
    with _captured_warnings() as messages:
        restored.load_state_dict(_without_bounds(fitted.state_dict()))
    assert messages == []
    assert restored._statistically_initialized is True


@torch.no_grad()
@pytest.mark.parametrize("cls", SUPERVISED)
def test_a_fitted_checkpoint_round_trips(cls: type) -> None:
    fitted, cube = _fitted(cls, LIT_BANDS[cls][0])
    restored = cls(num_spectral_bands=N_BANDS, norm_mode="statistical")
    with _captured_warnings() as messages:
        restored.load_state_dict(fitted.state_dict())
    assert messages == []
    assert torch.equal(restored.running_min, fitted.running_min)
    assert torch.equal(restored.running_max, fitted.running_max)
    assert torch.equal(
        restored.forward(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"],
        fitted.forward(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"],
    )


@torch.no_grad()
@pytest.mark.parametrize(
    "broken",
    [
        pytest.param({"running_max": float("nan")}, id="nan-maximum"),
        pytest.param({"running_min": float("-inf")}, id="infinite-minimum"),
    ],
)
def test_unusable_bounds_take_the_per_frame_path(broken: dict[str, float]) -> None:
    fitted, cube = _fitted(SupervisedCIRSelector, 9)
    state = dict(fitted.state_dict())
    for key, value in broken.items():
        state[key] = torch.full((3,), value)
    restored = SupervisedCIRSelector(num_spectral_bands=N_BANDS, norm_mode="statistical")
    with _captured_warnings() as messages:
        restored.load_state_dict(state)
    assert len(messages) == 1
    out = restored.forward(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"]
    assert not torch.isnan(out).any() and not torch.isinf(out).any()


@torch.no_grad()
@pytest.mark.parametrize("cls", SUPERVISED)
def test_a_fit_after_a_load_without_bounds_ends_with_usable_bounds(cls: type) -> None:
    fitted, _ = _fitted(cls, LIT_BANDS[cls][0])
    restored = cls(num_spectral_bands=N_BANDS, norm_mode="statistical")
    with _captured_warnings():
        restored.load_state_dict(_without_bounds(fitted.state_dict()))

    cube, mask = _fit_data(4, LIT_BANDS[cls][1], scale=60.0)
    restored.statistical_initialization(_stream(cube, mask))
    low, high = _per_cube_bounds(cube, restored.selected_indices.tolist())
    assert torch.allclose(restored.running_min, low) and torch.allclose(restored.running_max, high)
    out = restored.forward(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"]
    expected = restored._srgb_gamma(
        normalize_with_bounds(
            _raw_bands(cube, restored.selected_indices.tolist()),
            restored.running_min,
            restored.running_max,
            1e-8,
        )
    )
    assert torch.equal(out, expected)


# ---------------------------------------------------------------------------
# The base fit shares the arithmetic
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_a_base_node_refit_repairs_an_unusable_pair() -> None:
    cube = torch.rand(2, 4, 5, 30) * 500.0 + 10.0
    wavelengths = np.linspace(400.0, 1000.0, 30).astype(np.float32)
    fitted = CIRSelector(norm_mode="statistical")
    fitted.statistical_initialization(iter([{"cube": cube, "wavelengths": wavelengths}]))
    state = dict(fitted.state_dict())
    state["running_max"] = torch.full((3,), float("nan"))

    restored = CIRSelector(norm_mode="statistical")
    restored.load_state_dict(state)
    assert restored._statistically_initialized is False

    restored.statistical_initialization(iter([{"cube": cube, "wavelengths": wavelengths}]))
    assert torch.equal(restored.running_min, fitted.running_min)
    assert torch.equal(restored.running_max, fitted.running_max)
    out = restored.forward(cube=cube, wavelengths=wavelengths)["rgb_image"]
    assert not torch.isnan(out).any()
    assert torch.equal(out, fitted.forward(cube=cube, wavelengths=wavelengths)["rgb_image"])


def test_the_base_fit_with_no_data_raises() -> None:
    node = CIRSelector(norm_mode="statistical")
    with pytest.raises(RuntimeError, match="received no data"):
        node.statistical_initialization(iter([]))
    assert node._statistically_initialized is False


# ---------------------------------------------------------------------------
# Through the pipeline
# ---------------------------------------------------------------------------


@torch.no_grad()
def test_a_one_node_pipeline_round_trips_the_bounds(tmp_path) -> None:
    config = {
        "metadata": {"name": "cir"},
        "nodes": [
            {
                "name": "cir",
                "class_name": "cuvis_ai.node.channel_selector.SupervisedCIRSelector",
                "hparams": {"num_spectral_bands": N_BANDS, "norm_mode": "statistical"},
            }
        ],
        "connections": [],
    }
    pipeline = PipelineBuilder().build_from_config(config)
    node = next(n for n in pipeline.nodes if n.name == "cir")
    cube, mask = _fit_data(1, 9)
    node.statistical_initialization(_stream(cube, mask))
    expected = node.forward(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"]

    yaml_path = tmp_path / "cir.yaml"
    pipeline.save_to_file(yaml_path)
    with _captured_warnings() as messages:
        rebuilt = CuvisPipeline.load_pipeline(yaml_path, yaml_path.with_suffix(".pt"))
    assert messages == []
    restored = next(n for n in rebuilt.nodes if n.name == "cir")
    assert restored._statistically_initialized is True
    assert torch.equal(restored.running_min, node.running_min)
    assert torch.equal(restored.forward(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"], expected)
