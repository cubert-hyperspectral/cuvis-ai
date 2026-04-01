from __future__ import annotations

import pytest
import torch

from cuvis_ai.utils.poisson_inpaint import poisson_inpaint


def test_no_mask_returns_unchanged() -> None:
    image = torch.rand((12, 16, 3), dtype=torch.float32)
    mask = torch.zeros((12, 16), dtype=torch.bool)
    out = poisson_inpaint(image, mask)
    torch.testing.assert_close(out, image)
    assert out.data_ptr() != image.data_ptr()


def test_single_pixel_inpaint() -> None:
    image = torch.zeros((5, 5, 1), dtype=torch.float32)
    image[1, 2, 0] = 0.2
    image[3, 2, 0] = 0.6
    image[2, 1, 0] = 0.4
    image[2, 3, 0] = 0.8

    mask = torch.zeros((5, 5), dtype=torch.bool)
    mask[2, 2] = True

    out = poisson_inpaint(image, mask, max_iter=100, tol=1e-8)
    assert out[2, 2, 0].item() == pytest.approx(0.5, abs=1e-5)
    torch.testing.assert_close(out[~mask, :], image[~mask, :])


def test_gradient_continuity() -> None:
    h, w = 24, 30
    x = torch.linspace(0.0, 1.0, w, dtype=torch.float32)
    y = torch.linspace(0.0, 1.0, h, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    image = torch.stack((xx, yy, 0.25 * xx + 0.75 * yy), dim=-1)

    mask = torch.zeros((h, w), dtype=torch.bool)
    mask[5:19, 10:15] = True

    out = poisson_inpaint(image, mask, max_iter=2000, tol=1e-7)
    torch.testing.assert_close(out, image, atol=5e-4, rtol=1e-4)


@pytest.mark.parametrize("channels", [1, 3, 51])
def test_multichannel(channels: int) -> None:
    h, w = 20, 24
    base = torch.linspace(0.0, 1.0, h * w, dtype=torch.float32).reshape(h, w)
    image = torch.stack([(base + 0.01 * c) for c in range(channels)], dim=-1)

    mask = torch.zeros((h, w), dtype=torch.bool)
    mask[6:14, 8:16] = True

    out = poisson_inpaint(image, mask, max_iter=1500, tol=1e-6)
    assert out.shape == image.shape
    assert torch.isfinite(out).all()


def test_preserves_device_and_dtype() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.rand((10, 11, 3), dtype=torch.float32, device=device)
    mask = torch.zeros((10, 11), dtype=torch.bool)
    mask[3:7, 4:9] = True

    out = poisson_inpaint(image, mask, max_iter=500, tol=1e-6)
    assert out.device == image.device
    assert out.dtype == image.dtype


def test_full_mask_raises() -> None:
    image = torch.rand((8, 8, 3), dtype=torch.float32)
    mask = torch.ones((8, 8), dtype=torch.bool)
    with pytest.raises(ValueError, match="whole image"):
        _ = poisson_inpaint(image, mask)


def test_boundary_only_mask() -> None:
    image = torch.zeros((4, 4, 1), dtype=torch.float32)
    image[0, 1, 0] = 0.3
    image[1, 0, 0] = 0.9
    mask = torch.zeros((4, 4), dtype=torch.bool)
    mask[0, 0] = True

    out = poisson_inpaint(image, mask, max_iter=100, tol=1e-8)
    assert out[0, 0, 0].item() == pytest.approx(0.6, abs=1e-5)


def test_convergence_within_tolerance() -> None:
    h, w = 24, 24
    image = torch.rand((h, w, 1), dtype=torch.float32)
    mask = torch.zeros((h, w), dtype=torch.bool)
    mask[6:18, 6:18] = True

    out = poisson_inpaint(image, mask, max_iter=3000, tol=1e-7)
    u = out[..., 0]

    residual_max = 0.0
    for y, x in mask.nonzero(as_tuple=False):
        yi = int(y.item())
        xi = int(x.item())
        degree = 0
        neigh_sum = 0.0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny = yi + dy
            nx = xi + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            degree += 1
            neigh_sum += float(u[ny, nx].item())
        residual = abs(degree * float(u[yi, xi].item()) - neigh_sum)
        residual_max = max(residual_max, residual)

    assert residual_max < 2e-3
