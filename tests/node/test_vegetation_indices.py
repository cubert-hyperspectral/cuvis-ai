from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from cuvis_ai.node.channel_selector import (
    CIRedEdgeSelector,
    EVI2Selector,
    EVISelector,
    GNDVISelector,
    MCARISelector,
    MSAVISelector,
    NBRSelector,
    NDRESelector,
    NDWISelector,
    PRISelector,
    SAVISelector,
)

pytestmark = pytest.mark.unit


# Each entry: node factory, wavelengths [C] in nm, and a numpy formula computing
# the expected per-pixel index from the cube's two pixels (float64). The cube is
# built so that each named band wavelength matches a sensor wavelength exactly,
# and each band carries a distinct, known reflectance value.
#
# Reflectance layout shared across cases (pixel 0, pixel 1) per logical band:
#   blue=(0.05, 0.10), green=(0.15, 0.20), red=(0.10, 0.25),
#   red_edge=(0.30, 0.35), nir=(0.60, 0.55), swir=(0.20, 0.40),
#   b531=(0.40, 0.30), b570=(0.35, 0.45)


def _evi(cube: np.ndarray) -> np.ndarray:
    blue, red, nir = cube[..., 0], cube[..., 1], cube[..., 2]
    return 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0)


def _evi2(cube: np.ndarray) -> np.ndarray:
    red, nir = cube[..., 0], cube[..., 1]
    return 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)


def _savi(cube: np.ndarray) -> np.ndarray:
    red, nir = cube[..., 0], cube[..., 1]
    soil = 0.5
    return (1.0 + soil) * (nir - red) / (nir + red + soil)


def _msavi(cube: np.ndarray) -> np.ndarray:
    red, nir = cube[..., 0], cube[..., 1]
    term = 2.0 * nir + 1.0
    return 0.5 * (term - np.sqrt(term * term - 8.0 * (nir - red)))


def _ndwi(cube: np.ndarray) -> np.ndarray:
    green, nir = cube[..., 0], cube[..., 1]
    return (green - nir) / (green + nir)


def _nbr(cube: np.ndarray) -> np.ndarray:
    nir, swir = cube[..., 0], cube[..., 1]
    return (nir - swir) / (nir + swir)


def _gndvi(cube: np.ndarray) -> np.ndarray:
    nir, green = cube[..., 0], cube[..., 1]
    return (nir - green) / (nir + green)


def _ndre(cube: np.ndarray) -> np.ndarray:
    nir, red_edge = cube[..., 0], cube[..., 1]
    return (nir - red_edge) / (nir + red_edge)


def _ci_red_edge(cube: np.ndarray) -> np.ndarray:
    red_edge, nir = cube[..., 0], cube[..., 1]
    return nir / red_edge - 1.0


def _mcari(cube: np.ndarray) -> np.ndarray:
    green, red, red_edge = cube[..., 0], cube[..., 1], cube[..., 2]
    return ((red_edge - red) - 0.2 * (red_edge - green)) * (red_edge / red)


def _pri(cube: np.ndarray) -> np.ndarray:
    b531, b570 = cube[..., 0], cube[..., 1]
    return (b531 - b570) / (b531 + b570)


# Per-case cube reflectance values. Each row is one band's (pixel0, pixel1).
# The wavelengths array lists the sensor wavelengths in the SAME column order,
# so nearest-band resolution lands on the intended column.
_CASES: list[tuple[str, Callable[[], object], np.ndarray, np.ndarray, Callable]] = [
    (
        "EVI",
        EVISelector,
        np.array([460.0, 660.0, 800.0], dtype=np.float32),
        np.array([[0.05, 0.10, 0.60], [0.10, 0.25, 0.55]], dtype=np.float64),
        _evi,
    ),
    (
        "EVI2",
        EVI2Selector,
        np.array([660.0, 800.0], dtype=np.float32),
        np.array([[0.10, 0.60], [0.25, 0.55]], dtype=np.float64),
        _evi2,
    ),
    (
        "SAVI",
        SAVISelector,
        np.array([660.0, 800.0], dtype=np.float32),
        np.array([[0.10, 0.60], [0.25, 0.55]], dtype=np.float64),
        _savi,
    ),
    (
        "MSAVI",
        MSAVISelector,
        np.array([660.0, 800.0], dtype=np.float32),
        np.array([[0.10, 0.60], [0.25, 0.55]], dtype=np.float64),
        _msavi,
    ),
    (
        "NDWI",
        NDWISelector,
        np.array([560.0, 860.0], dtype=np.float32),
        np.array([[0.15, 0.60], [0.20, 0.55]], dtype=np.float64),
        _ndwi,
    ),
    (
        "NBR",
        NBRSelector,
        np.array([850.0, 2200.0], dtype=np.float32),
        np.array([[0.60, 0.20], [0.55, 0.40]], dtype=np.float64),
        _nbr,
    ),
    (
        "GNDVI",
        GNDVISelector,
        np.array([800.0, 550.0], dtype=np.float32),
        np.array([[0.60, 0.15], [0.55, 0.20]], dtype=np.float64),
        _gndvi,
    ),
    (
        "NDRE",
        NDRESelector,
        np.array([800.0, 720.0], dtype=np.float32),
        np.array([[0.60, 0.30], [0.55, 0.35]], dtype=np.float64),
        _ndre,
    ),
    (
        "CIRedEdge",
        CIRedEdgeSelector,
        np.array([720.0, 800.0], dtype=np.float32),
        np.array([[0.30, 0.60], [0.35, 0.55]], dtype=np.float64),
        _ci_red_edge,
    ),
    (
        "MCARI",
        MCARISelector,
        np.array([550.0, 670.0, 700.0], dtype=np.float32),
        np.array([[0.15, 0.10, 0.30], [0.20, 0.25, 0.35]], dtype=np.float64),
        _mcari,
    ),
    (
        "PRI",
        PRISelector,
        np.array([531.0, 570.0], dtype=np.float32),
        np.array([[0.40, 0.35], [0.30, 0.45]], dtype=np.float64),
        _pri,
    ),
]


@pytest.mark.parametrize(
    ("name", "factory", "wavelengths", "reflectance", "formula"),
    _CASES,
    ids=[case[0] for case in _CASES],
)
@torch.no_grad()
def test_vegetation_index_matches_numpy_formula(
    name: str,
    factory: Callable[[], object],
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    formula: Callable[[np.ndarray], np.ndarray],
) -> None:
    # reflectance is [2 pixels, C bands]; reshape to a [1, 1, 2, C] cube.
    cube_np = reflectance.reshape(1, 1, 2, -1)
    cube = torch.tensor(cube_np, dtype=torch.float32)

    node = factory()
    result = node.forward(cube=cube, wavelengths=wavelengths)

    expected = formula(cube_np.astype(np.float64))  # [1, 1, 2]
    expected_index = torch.tensor(expected[..., np.newaxis], dtype=torch.float32)

    index_image = result["index_image"]
    rgb_image = result["rgb_image"]

    assert index_image.shape == (1, 1, 2, 1), f"{name}: index_image shape {index_image.shape}"
    assert rgb_image.shape == (1, 1, 2, 3), f"{name}: rgb_image shape {rgb_image.shape}"
    assert torch.allclose(index_image, expected_index, atol=1e-5), (
        f"{name}: index_image {index_image.flatten().tolist()} "
        f"!= expected {expected_index.flatten().tolist()}"
    )


def _evi_inputs() -> tuple[torch.Tensor, np.ndarray]:
    wavelengths = np.array([460.0, 660.0, 800.0], dtype=np.float32)
    cube = torch.tensor(
        np.array([[0.05, 0.10, 0.60], [0.10, 0.25, 0.55]]).reshape(1, 1, 2, 3),
        dtype=torch.float32,
    )
    return cube, wavelengths


@torch.no_grad()
def test_batched_2d_wavelengths_use_first_row() -> None:
    """A [B, C] wavelength grid resolves bands from its first row."""
    cube, wavelengths = _evi_inputs()
    node = EVISelector()
    from_1d = node.forward(cube=cube, wavelengths=wavelengths)
    from_2d = node.forward(cube=cube, wavelengths=wavelengths[np.newaxis, :])
    assert torch.allclose(from_1d["index_image"], from_2d["index_image"])
    assert from_1d["band_info"]["band_indices"] == from_2d["band_info"]["band_indices"]


@torch.no_grad()
def test_higher_rank_wavelengths_rejected() -> None:
    cube, wavelengths = _evi_inputs()
    with pytest.raises(ValueError, match="1D wavelengths"):
        EVISelector().forward(cube=cube, wavelengths=wavelengths.reshape(1, 1, -1))


@torch.no_grad()
def test_compute_raw_rgb_matches_forward_render() -> None:
    cube, wavelengths = _evi_inputs()
    node = EVISelector()
    rgb = node._compute_raw_rgb(cube, wavelengths)
    assert torch.allclose(rgb, node.forward(cube=cube, wavelengths=wavelengths)["rgb_image"])


@pytest.mark.parametrize(
    ("factory", "kwargs", "match"),
    [
        (NDWISelector, {"colormap_min": 1.0, "colormap_max": 1.0}, "colormap_max"),
        (EVISelector, {"eps": -1.0}, "eps"),
        (EVISelector, {"colormap_min": 2.0, "colormap_max": 1.0}, "colormap_max"),
    ],
)
def test_invalid_constructor_arguments_rejected(
    factory: Callable[..., object],
    kwargs: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        factory(**kwargs)
