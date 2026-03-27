"""Color space conversion utilities (PyTorch, differentiable)."""

from __future__ import annotations

import torch
from torch import Tensor


def srgb_to_linear(x: Tensor) -> Tensor:
    """Apply inverse sRGB EOTF (companding to linear).

    Parameters
    ----------
    x : Tensor
        sRGB values in [0, 1] with arbitrary batch shape.

    Returns
    -------
    Tensor
        Linear-light values, same shape as input.
    """
    a = 0.055
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + a) / (1.0 + a)).pow(2.4),
    )


def linear_rgb_to_oklab(rgb: Tensor) -> Tensor:
    """Convert linear RGB to OKLab.

    Parameters
    ----------
    rgb : Tensor
        Linear RGB tensor with shape ``[..., 3]`` in range [0, 1].
        Input should **not** have an sRGB gamma curve applied;
        use :func:`srgb_to_linear` first if working with display sRGB values.

    Returns
    -------
    Tensor
        OKLab tensor ``[..., 3]`` where channels are (L, a, b).
    """
    # Linear RGB -> LMS
    M1 = rgb.new_tensor(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    lms = rgb @ M1.T  # [..., 3]

    # Cube root with sign handling (robust if values go slightly negative)
    lms_cbrt = torch.sign(lms) * torch.abs(lms).clamp_min(1e-12).pow(1.0 / 3.0)

    # LMS -> OKLab
    M2 = rgb.new_tensor(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    return lms_cbrt @ M2.T  # [..., 3]


def rgb_to_oklab(rgb: Tensor, assume_srgb: bool = True) -> Tensor:
    """Convert RGB to OKLab, optionally handling sRGB input.

    Parameters
    ----------
    rgb : Tensor
        RGB tensor ``[..., 3]`` in [0, 1].
    assume_srgb : bool
        If ``True``, apply inverse sRGB gamma first via :func:`srgb_to_linear`.
        If ``False``, assume input is already linear RGB.

    Returns
    -------
    Tensor
        OKLab tensor ``[..., 3]``.
    """
    if assume_srgb:
        rgb = srgb_to_linear(rgb)
    return linear_rgb_to_oklab(rgb)
