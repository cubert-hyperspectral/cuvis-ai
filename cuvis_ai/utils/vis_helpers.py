"""Visualization helper utilities for converting figures and tensors to arrays."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    import matplotlib.figure


def fig_to_array(fig: matplotlib.figure.Figure, dpi: int = 150) -> np.ndarray:
    """Convert matplotlib figure to numpy array in RGB format.

    This utility handles the conversion of a matplotlib figure to a numpy array
    by saving it to a BytesIO buffer, loading it with PIL, and converting to
    a numpy array. The figure is automatically closed after conversion.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure to convert
    dpi : int, optional
        Resolution for the saved image (default: 150)

    Returns
    -------
    np.ndarray
        RGB image as numpy array with shape (H, W, 3) and dtype uint8

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> img_array = fig_to_array(fig, dpi=150)
    >>> img_array.shape[-1]
    3
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img.convert("RGB"))
    buf.close()

    # Close the figure to free memory
    import matplotlib.pyplot as plt

    plt.close(fig)

    return img_array


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array on CPU.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor (can be on any device)

    Returns
    -------
    np.ndarray
        Numpy array representation
    """
    return tensor.detach().cpu().numpy()


@torch.no_grad()
def create_mask_overlay(
    rgb: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.4,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> torch.Tensor:
    """Alpha-blend a colored tint on foreground pixels.

    Pure PyTorch, no gradients.  Works for both single images ``[H, W, 3]``
    and batched ``[B, H, W, 3]`` thanks to broadcasting.

    Parameters
    ----------
    rgb : torch.Tensor
        RGB image(s) in ``[0, 1]``.  Shape ``[H, W, 3]`` or ``[B, H, W, 3]``.
    mask : torch.Tensor
        Segmentation mask where ``> 0`` is foreground.
        Shape ``[H, W]`` or ``[B, H, W]``.
    alpha : float, optional
        Blend factor for the overlay colour (default: 0.4).
    color : tuple[float, float, float], optional
        RGB overlay colour in ``[0, 1]`` (default: red ``(1, 0, 0)``).

    Returns
    -------
    torch.Tensor
        Blended image, same shape and device as *rgb*, clamped to ``[0, 1]``.
    """
    if mask.device != rgb.device:
        mask = mask.to(rgb.device)
    fg = (mask > 0).unsqueeze(-1).to(dtype=rgb.dtype)  # [..., 1] for channel broadcast
    tint = torch.tensor(color, dtype=rgb.dtype, device=rgb.device)
    return ((1.0 - alpha * fg) * rgb + alpha * fg * tint).clamp(0.0, 1.0)


# 12 perceptually-distinct colours for multi-object overlays (RGB, 0-255).
OBJECT_PALETTE: list[tuple[int, int, int]] = [
    (230, 25, 75),  # red
    (60, 180, 75),  # green
    (255, 225, 25),  # yellow
    (0, 130, 200),  # blue
    (245, 130, 48),  # orange
    (145, 30, 180),  # purple
    (70, 240, 240),  # cyan
    (240, 50, 230),  # magenta
    (210, 245, 60),  # lime
    (250, 190, 212),  # pink
    (0, 128, 128),  # teal
    (220, 190, 255),  # lavender
]


__all__ = [
    "fig_to_array",
    "tensor_to_numpy",
    "create_mask_overlay",
    "OBJECT_PALETTE",
]
