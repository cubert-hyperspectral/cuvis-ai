"""Spectral Angle Mapper node."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec


class SpectralAngleMapper(Node):
    """Compute per-pixel spectral angle against one or more reference spectra."""

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C]",
        ),
        "spectral_signature": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Reference spectra [N, 1, 1, C]",
        ),
    }

    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Spectral angle scores [B, H, W, N] in radians",
        ),
        "best_scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Best score per pixel [B, H, W, 1]",
        ),
        "identity_mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="1-based best-matching identity [B, H, W]",
        ),
    }

    def __init__(self, num_channels: int, eps: float = 1e-12, **kwargs: Any) -> None:
        if int(num_channels) <= 0:
            raise ValueError(f"num_channels must be > 0, got {num_channels}")
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        super().__init__(num_channels=self.num_channels, eps=self.eps, **kwargs)

    @torch.no_grad()
    def forward(
        self,
        cube: torch.Tensor,
        spectral_signature: torch.Tensor,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Run spectral-angle scoring for all references."""
        ref = spectral_signature.squeeze(1).squeeze(1)  # [N, C]
        channel_count = int(ref.shape[-1])
        ref_mean = ref.mean(dim=-1, keepdim=True)
        ref_norm = ref / (ref_mean + self.eps)

        pixel_mean = cube.mean(dim=-1, keepdim=True)
        cube_norm = cube / (pixel_mean + self.eps)

        ref_expanded = ref_norm.view(1, 1, 1, ref_norm.shape[0], channel_count)
        cube_expanded = cube_norm.unsqueeze(-2)

        dot = (cube_expanded * ref_expanded).sum(dim=-1)
        norms = cube_norm.norm(dim=-1, keepdim=True) * ref_norm.norm(dim=-1).view(1, 1, 1, -1)
        cos_sim = dot / (norms + self.eps)
        scores = torch.acos(cos_sim.clamp(-1.0, 1.0))

        best_scores = scores.amin(dim=-1, keepdim=True)
        identity_mask = scores.argmin(dim=-1).to(torch.int32) + 1

        return {
            "scores": scores,
            "best_scores": best_scores,
            "identity_mask": identity_mask,
        }


__all__ = ["SpectralAngleMapper"]
