from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import ndimage
from skimage.measure import regionprops

from cuvis_ai.node.morphology import ShapeMorphology

pytestmark = pytest.mark.unit


def _build_mask() -> torch.Tensor:
    """Three well-separated filled shapes of known geometry on a [1, H, W] grid."""
    h, w = 60, 80
    mask = np.zeros((h, w), dtype=np.int32)
    # A: small square 6x6.
    mask[4:10, 4:10] = 1
    # B: wide rectangle 8x20 (elongated along x).
    mask[20:28, 30:50] = 1
    # C: filled ellipse, semi-axes (ry=10 along y, rx=6 along x), centered.
    cy, cx, ry, rx = 45, 60, 10, 6
    ys, xs = np.ogrid[:h, :w]
    ellipse = ((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2 <= 1.0
    mask[ellipse] = 1
    return torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]


def _region_pixel_sets(label_map: np.ndarray) -> set[frozenset[tuple[int, int]]]:
    """Set of per-region pixel-coordinate sets, ignoring label values."""
    sets = set()
    for value in np.unique(label_map):
        if value == 0:
            continue
        coords = np.argwhere(label_map == value)
        sets.add(frozenset((int(y), int(x)) for y, x in coords))
    return sets


def test_shape_morphology_labels_match_scipy_components() -> None:
    mask = _build_mask()
    binary = (mask[0].numpy() != 0).astype(np.int32)
    structure = np.ones((3, 3), dtype=np.int32)  # 8-connectivity
    scipy_labels, n_scipy = ndimage.label(binary, structure=structure)

    node = ShapeMorphology()
    out = node.forward(mask=mask)

    identity = out["identity_mask"][0].numpy()
    assert int(identity.max()) == n_scipy == 3
    # Labelings match up to a relabel: compare the set of region pixel-sets.
    assert _region_pixel_sets(identity) == _region_pixel_sets(scipy_labels)

    # valid marks exactly N=3 objects true, the rest false (padding).
    valid = out["valid"][0]
    assert valid.dtype == torch.bool
    assert int(valid.sum().item()) == 3
    assert bool(valid[:3].all())
    assert not bool(valid[3:].any())


def test_shape_morphology_descriptors_match_regionprops() -> None:
    mask = _build_mask()
    node = ShapeMorphology()
    out = node.forward(mask=mask)

    identity = out["identity_mask"][0].numpy()
    props = out["properties"][0].numpy()
    prop_names = node.properties

    regions = {r.label: r for r in regionprops(identity)}
    assert len(regions) == 3

    for label, region in regions.items():
        row = props[label - 1]  # identity labels are 1-based, dense
        col = {name: row[i] for i, name in enumerate(prop_names)}

        assert col["area"] == pytest.approx(float(region.area), abs=1e-3)
        assert col["centroid_y"] == pytest.approx(float(region.centroid[0]), abs=1e-3)
        assert col["centroid_x"] == pytest.approx(float(region.centroid[1]), abs=1e-3)
        assert col["bbox_area"] == pytest.approx(float(region.area_bbox), abs=1e-3)
        assert col["major_axis"] == pytest.approx(float(region.axis_major_length), abs=1e-3)
        assert col["minor_axis"] == pytest.approx(float(region.axis_minor_length), abs=1e-3)
        assert col["eccentricity"] == pytest.approx(float(region.eccentricity), abs=1e-3)
        # Orientation is undefined for a near-circular region (equal eigenvalues),
        # so only check it where the shape has a meaningful principal axis.
        if float(region.eccentricity) > 1e-3:
            # Orientation is defined modulo pi; compare the principal-axis direction.
            ours = float(col["orientation"]) % np.pi
            ref = float(region.orientation) % np.pi
            delta = abs(ours - ref)
            assert min(delta, np.pi - delta) == pytest.approx(0.0, abs=1e-3)


def test_shape_morphology_area_matches_pixel_counts() -> None:
    mask = _build_mask()
    node = ShapeMorphology()
    out = node.forward(mask=mask)

    identity = out["identity_mask"][0]
    props = out["properties"][0]
    area_col = node.properties.index("area")
    for label in range(1, int(identity.max().item()) + 1):
        pixel_count = int((identity == label).sum().item())
        assert props[label - 1, area_col].item() == pytest.approx(pixel_count, abs=1e-3)


def test_shape_morphology_already_labeled_passthrough() -> None:
    # Two pre-labeled regions with non-contiguous label values.
    mask = torch.zeros((1, 10, 10), dtype=torch.int32)
    mask[0, 1:3, 1:3] = 7
    mask[0, 6:9, 6:9] = 4
    node = ShapeMorphology(binarize=False)
    out = node.forward(mask=mask)

    identity = out["identity_mask"][0]
    assert int(identity.max().item()) == 2  # densified to 1..2
    valid = out["valid"][0]
    assert int(valid.sum().item()) == 2

    # Areas of the two densified regions: 4 (2x2) and 9 (3x3).
    area_col = node.properties.index("area")
    props = out["properties"][0]
    areas = sorted(props[i, area_col].item() for i in range(2))
    assert areas == pytest.approx([4.0, 9.0])


def test_shape_morphology_padding_and_shapes() -> None:
    mask = _build_mask()
    node = ShapeMorphology(max_objects=16)
    out = node.forward(mask=mask)

    assert out["identity_mask"].shape == (1, 60, 80)
    assert out["properties"].shape == (1, 16, len(node.properties))
    assert out["valid"].shape == (1, 16)
    assert out["properties"].dtype == torch.float32
    assert out["identity_mask"].dtype == torch.int32

    # Padded slots carry zero descriptors.
    valid = out["valid"][0]
    pad = out["properties"][0][~valid]
    assert torch.all(pad == 0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_objects": 0}, "max_objects"),
        ({"connectivity": 6}, "connectivity"),
        ({"properties": ["area", "bogus"]}, "unknown properties"),
        ({"properties": []}, "at least one"),
    ],
)
def test_shape_morphology_validates_constructor_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        ShapeMorphology(**kwargs)
