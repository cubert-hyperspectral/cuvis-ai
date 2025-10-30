from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import dotenv
import numpy as np
import requests
import torch
import wandb
from PIL import Image

from cuvis_ai.utils.general import _to_int_list

dotenv.load_dotenv(override=True)

# -------- helpers ------------------------------------------------------------


def _to_display_image(x: Any) -> np.ndarray:
    """
    Accepts: HWC NumPy, CHW torch.Tensor, or PIL.Image.
    Returns: HWC NumPy (dtype preserved). Normalization is left to W&B.
    """
    if isinstance(x, Image.Image):
        return np.asarray(x)

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3, 4):  # CHW -> HWC
            x = x.permute(1, 2, 0)
        return x.numpy()

    return np.asarray(x)


def _as_bhwc(x: Any) -> np.ndarray:
    """
    Accepts:
      - np.ndarray (B,H,W,C)
      - torch.Tensor (B,C,H,W)
      - list/tuple of images (each PIL/np/torch CHW/HWC) -> stacked (B,H,W,C)
    Returns: NumPy (B,H,W,C)
    """
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.ndim != 4:
            raise ValueError(f"images tensor must be 4D; got {tuple(t.shape)}")
        if t.shape[1] in (1, 3, 4):  # (B,C,H,W) -> (B,H,W,C)
            t = t.permute(0, 2, 3, 1)
        return t.numpy()

    if isinstance(x, (list, tuple)):
        arrs = [_to_display_image(e) for e in x]  # each HWC
        return np.stack(arrs, axis=0)

    a = np.asarray(x)
    if a.ndim != 4:
        raise ValueError(f"images must be (B,H,W,C); got {a.shape}")
    return a


def _as_bhw1(x: Any) -> np.ndarray:
    """
    Accepts np.ndarray or torch.Tensor shaped (B,H,W) or (B,H,W,1).
    Returns: NumPy (B,H,W,1).
    """
    if isinstance(x, torch.Tensor):
        a = x.detach().cpu().numpy()
    else:
        a = np.asarray(x)

    if a.ndim == 3:
        a = a[..., np.newaxis]
    if a.ndim != 4 or a.shape[-1] != 1:
        raise ValueError(f"masks must be (B,H,W,1); got {a.shape}")
    return a


def _prepare_mask_2d(x: np.ndarray) -> np.ndarray:
    """
    Input: (H,W,1) or (H,W) NumPy.
    Output: (H,W) integer mask for W&B (class ids).
    Ensures values are in valid range [0, num_classes-1]
    """
    a = np.asarray(x)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"mask must be (H,W); got {a.shape}")
    if not np.issubdtype(a.dtype, np.integer):
        a = (a > 0).astype(np.uint8)  # binarize non-integer masks

    # Ensure mask is in valid range [0, 1] for binary masks
    a = np.clip(a, 0, 1).astype(np.uint8)

    return a


# -------- visualizer ---------------------------------------------------------


@dataclass
class WandBMaskVisualizer:
    """
    Reusable utility to log image + mask overlays and scalars to Weights & Biases.
    No images are written to disk; everything is logged in-memory.
    Uses filled red overlays for reliable visualization.
    """

    project: str = "cuvis-ai"
    entity: str | None = None
    namespace: str = "rx_anomaly"
    class_labels: dict[int, str] | None = field(
        default_factory=lambda: {0: "background", 1: "anomaly"}
    )
    tags: list[str] = field(default_factory=list)
    run: Any | None = None  # inject an external run if you have one
    image_normalize: bool | None = None  # forwarded to wandb.Image(normalize=...)
    overlay_alpha: float = 0.5  # Transparency for overlays

    # internals
    _run: wandb.sdk.wandb_run.Run | None = field(init=False, default=None)
    _step: int = field(init=False, default=0)

    def __post_init__(self):
        """Initialize color cycle for contours."""
        pass

    # context manager (optional)
    def __enter__(self) -> WandBMaskVisualizer:
        self._ensure_run()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()

    def _check_wandb_server_connectivity(self) -> bool:
        """Check if wandb server is reachable when WANDB_MODE is online."""
        wandb_mode = os.getenv("WANDB_MODE", "online")
        if wandb_mode != "online":
            return True  # No need to check in offline mode

        wandb_base_url = os.getenv("WANDB_BASE_URL")
        if not wandb_base_url:
            return True  # Let wandb use default cloud server

        try:
            # Try to reach the server with a short timeout
            response = requests.get(wandb_base_url, timeout=5)
            return response.status_code < 500
        except (requests.RequestException, Exception) as e:
            logger.warning(f"Cannot reach wandb server at {wandb_base_url}: {e}")
            return False

    def _ensure_run(self) -> wandb.sdk.wandb_run.Run:
        if self._run is not None:
            return self._run
        if self.run is not None:
            self._run = self.run
        elif wandb.run is not None:
            self._run = wandb.run
        else:
            # Check server connectivity and fallback to offline if needed
            original_mode = os.getenv("WANDB_MODE", "online")
            mode = original_mode

            if original_mode == "online":
                assert self._check_wandb_server_connectivity(), (
                    "Wandb server is not reachable. Falling back to offline mode. "
                    "Run 'wandb sync' later to upload the data."
                )

            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                mode=mode,
                tags=self.tags,
                job_type=self.namespace,
            )
        return self._run

    def _create_overlay(self, img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Create an RGB overlay of image and mask with filled regions.
        img: (H,W,C) numpy array
        mask: (H,W) numpy array with values 0 or 1
        Returns: (H,W,C) numpy array with mask overlay
        """
        # Ensure image is RGB
        if img.ndim == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:
            img_rgb = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] == 4:
            img_rgb = img[..., :3]
        else:
            img_rgb = img.copy()

        # Normalize image to [0, 1] if needed
        if img_rgb.dtype == np.uint8:
            img_rgb = img_rgb.astype(np.float32) / 255.0
        elif img_rgb.max() > 1.0:
            img_rgb = img_rgb / img_rgb.max()

        # Create red overlay for mask
        overlay = img_rgb.copy()
        overlay[mask > 0] = overlay[mask > 0] * (1 - alpha) + np.array([1.0, 0, 0]) * alpha

        # Convert back to uint8
        overlay = (overlay * 255).astype(np.uint8)

        return overlay

    def log(
        self,
        images_bhwc: Any,
        pred_masks: Any,
        measurement_indices: Sequence[int] | Iterable[int],
        *,
        class_labels: dict[int, str] | None = None,
        gt_masks: Any | None = None,
        scalar_logs: dict[str, float] | None = None,
        captions: Sequence[str] | None = None,
    ) -> None:
        """
        Log images + predicted/GT masks to W&B. Writes no files.
        - images_bhwc: np (B,H,W,C), torch (B,C,H,W), or list/PIL
        - pred_masks: np/torch (B,H,W[,(1)])
        - gt_masks: optional, same shape semantics
        - measurement_indices: per-sample integer ids
        - captions: optional per-sample strings
        - scalar_logs: merged into each step payload
        """
        images = _as_bhwc(images_bhwc)
        masks = _as_bhw1(pred_masks)
        B, H, W, C = images.shape

        mesu_indices = _to_int_list(measurement_indices)

        assert len(mesu_indices) == images.shape[0], (
            f"measurement_indices length must match batch size: {len(mesu_indices)} vs {B}"
        )

        gt = _as_bhw1(gt_masks) if gt_masks is not None else None
        if gt is not None and gt.shape[0] != B:
            raise ValueError(f"ground_truth_masks batch {gt.shape[0]} vs {B}")

        caps = list(captions) if captions is not None else [None] * B
        assert len(caps) == B, f"captions length must match batch size: {len(caps)} vs {B}"

        run = self._ensure_run()
        if class_labels is None:
            class_labels = self.class_labels

        for i in range(B):
            img = _to_display_image(images[i])  # HWC np
            pred2d = _prepare_mask_2d(masks[i])  # (H,W) int
            gt2d = _prepare_mask_2d(gt[i]) if gt is not None else None
            mesu_idx = mesu_indices[i]
            cap = caps[i] or f"Measurement {mesu_idx}"

            # Create overlays
            pred_overlay = self._create_overlay(img, pred2d, alpha=self.overlay_alpha)
            gt_overlay = (
                self._create_overlay(img, gt2d, alpha=self.overlay_alpha)
                if gt2d is not None
                else None
            )

            # Prepare payload with separate images
            wb_img_kwargs = {}
            if self.image_normalize is not None:
                wb_img_kwargs["normalize"] = self.image_normalize

            payload: dict[str, Any] = {
                # "image/original": wandb.Image(img, caption=f"{cap} (original)", **wb_img_kwargs),
                "image/pred_overlay": wandb.Image(pred_overlay, caption=f"{cap} (prediction)"),
                "image/pred_mask": wandb.Image(pred2d * 255, caption=f"{cap} (pred mask)"),
                "mesu_index": mesu_idx,
                "pred_mask_sum": int(np.sum(pred2d)),
            }

            # Add ground truth visualizations if available
            if gt2d is not None:
                payload["image/gt_overlay"] = wandb.Image(
                    gt_overlay, caption=f"{cap} (ground truth)"
                )
                payload["image/gt_mask"] = wandb.Image(gt2d * 255, caption=f"{cap} (GT mask)")
                payload["gt_mask_sum"] = int(np.sum(gt2d))

                # Add comparison metrics
                if np.any(gt2d) or np.any(pred2d):  # Only if there's something to compare
                    intersection = np.sum(np.logical_and(pred2d, gt2d))
                    union = np.sum(np.logical_or(pred2d, gt2d))
                    payload["iou"] = float(intersection / union) if union > 0 else 0.0

            if scalar_logs:
                payload.update(scalar_logs)

            run.log(payload, step=self._step)
            self._step += 1

    def finish(self) -> None:
        if self._run is not None and self._run is not self.run:
            self._run.finish()
            self._run = None
