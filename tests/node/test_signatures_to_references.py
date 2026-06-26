from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.spectral_angle_mapper import SpectralAngleMapper
from cuvis_ai.node.spectral_extractor import SignaturesToReferences

pytestmark = pytest.mark.unit


@torch.no_grad()
def test_reshape_preserves_values() -> None:
    sig = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # [1, 2, 3]
    out = SignaturesToReferences().forward(signatures=sig)["spectral_signature"]
    assert out.shape == (2, 1, 1, 3)
    assert torch.equal(out, sig[0].view(2, 1, 1, 3))


@torch.no_grad()
def test_unit_mean_normalization() -> None:
    sig = torch.tensor([[[2.0, 4.0, 6.0], [1.0, 1.0, 4.0]]])  # row means 4 and 2
    out = SignaturesToReferences(normalize="unit_mean").forward(signatures=sig)
    rows = out["spectral_signature"].view(2, 3)
    assert torch.allclose(rows.mean(dim=1), torch.ones(2), atol=1e-6)


@torch.no_grad()
def test_l2_normalization() -> None:
    sig = torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]]])
    out = SignaturesToReferences(normalize="l2").forward(signatures=sig)
    rows = out["spectral_signature"].view(2, 3)
    assert torch.allclose(rows.norm(dim=1), torch.ones(2), atol=1e-6)


def test_rejects_bad_normalize() -> None:
    with pytest.raises(ValueError):
        SignaturesToReferences(normalize="zscore")


@torch.no_grad()
def test_feeds_spectral_angle_mapper() -> None:
    """The [N,1,1,C] output must plug straight into SpectralAngleMapper."""
    channels = 4
    # two distinct reference spectra
    sig = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])  # [1, 2, 4]
    refs = SignaturesToReferences().forward(signatures=sig)["spectral_signature"]

    # a 1x2 cube whose two pixels match reference 0 and reference 1 respectively
    cube = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]]])  # [1, 1, 2, 4]
    sam = SpectralAngleMapper(num_channels=channels)
    out = sam.forward(cube=cube, spectral_signature=refs)

    ident = out["identity_mask"]  # 1-based nearest reference per pixel
    assert ident.shape == (1, 1, 2)
    assert int(ident[0, 0, 0]) == 1
    assert int(ident[0, 0, 1]) == 2
