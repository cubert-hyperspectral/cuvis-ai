from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import cycle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def to_numpy_np(x):
    """Convert a tensor or array to numpy safely."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise TypeError(f"Unsupported type for to_numpy(): {type(x)}")


def check_ndim(x: np.ndarray, expected_dims: Sequence[int], name: str = "array") -> None:
    """Check number of dimensions (generic; works for NumPy/Torch tensors already on CPU)."""
    if x.ndim not in expected_dims:
        raise ValueError(
            f"{name} must have ndim in {expected_dims}, got {x.ndim} (shape={x.shape})"
        )


def ensure_mask_shape(mask, target_shape):
    """Return a 2D mask aligned with the target spatial shape."""
    if mask is None:
        return None
    check_ndim(mask, [2, 3, 4], name="mask")

    mask = to_numpy_np(mask)
    if mask.ndim == 4:  # [B, H, W, 1]
        mask = mask[0, :, :, 0]
    elif mask.ndim == 3:  # [B, H, W]
        mask = mask[0, :, :]
    elif mask.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    # optional shape consistency check
    if mask.shape != target_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match target {target_shape}")
    return mask


def config(config_path: Path, overrides: Sequence[str] | None = None) -> DictConfig:
    logger.info(f"Loading configuration from {config_path}")
    cfg = OmegaConf.load(config_path)
    if overrides:
        logger.info(f"Applying overrides: {overrides}")
        cli_cfg = OmegaConf.from_cli(list(overrides))
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


def normalize_per_channel_vectorized(
    array: np.ndarray, range_min: float = 0.0, range_max: float = 1.0
) -> np.ndarray:
    arr = array.astype(np.float32)
    min_vals = arr.min(axis=(0, 1), keepdims=True)
    max_vals = arr.max(axis=(0, 1), keepdims=True)
    denom = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)
    norm = (arr - min_vals) / denom
    norm = norm * (range_max - range_min) + range_min
    return norm


def to_display_image(array: Any) -> np.ndarray:
    """Convert array shaped (..., H, W[, C]) into an RGB display image."""
    img = to_numpy_np(array)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.moveaxis(img, 0, -1)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.ndim == 3 and img.shape[-1] == 3:
        return img
    raise ValueError(f"Unsupported image shape {img.shape} for display conversion.")


def save_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray | None,
    out_path: Path,
    title: str | None = None,
) -> str:
    """Save an overlay of mask contours on top of an image."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    if image.ndim == 3 and image.shape[-1] == 3:
        plt.imshow(image)
    else:
        plt.imshow(image.squeeze(), cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    if mask is not None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_cycle = cycle(colors)
        unique_labels = [label for label in np.unique(mask) if label != 0]
        if not unique_labels:
            unique_labels = [1]
        for label in unique_labels:
            current_mask = (mask == label).astype(np.uint8)
            color = next(color_cycle)
            try:
                plt.contour(current_mask, levels=[0.5], colors=[color], linewidths=1.5)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"Skipping contour for label {label}: {exc}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def _ensure_channels_last(input_tensor: torch.Tensor) -> torch.Tensor:
    if input_tensor.dim() == 3:
        return input_tensor.unsqueeze(-1)
    if input_tensor.dim() != 4:
        raise ValueError(
            "Binary decider expects a 3D (B,H,W) or 4D (B,H,W,C) tensor; "
            f"received shape {tuple(input_tensor.shape)}"
        )
    return input_tensor


def _resolve_measurement_indices(
    indices: Sequence[int] | Iterable[int] | None, max_index: int = None
) -> list[int]:
    """Coerce, validate and store  indices."""
    print("indices:", indices, "max_index:", max_index)
    if indices is None and max_index is not None:
        resolved = list(range(max_index))
    elif isinstance(indices, range):
        resolved = list(indices)
    else:
        resolved = list(indices)

    if not resolved:
        if max_index is not None and max_index == 0:
            return []
        raise ValueError("At least one index is required.")

    invalid_indices = [idx for idx in resolved if idx < 0 or idx >= max_index]
    if invalid_indices:
        raise IndexError(
            f"Indices {invalid_indices} are out of bounds selection with {max_index} ."
        )

    if len(set(resolved)) != len(resolved):
        raise ValueError("Indices contain duplicates; provide unique indices.")

    return resolved


def _to_int_list(indices: Any) -> list[int]:
    if isinstance(indices, torch.Tensor):
        return [int(x) for x in indices.flatten().tolist()]
    if hasattr(indices, "numpy"):
        return [int(x) for x in indices.numpy().tolist()]
    if isinstance(indices, Iterable):
        return [int(x) for x in indices]
    return [int(indices)]
