"""Spectral Angle Mapper nodes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_core.node import Node


class SpectralAngleMapper(Node):
    """Compute per-pixel spectral angle against one or more reference spectra."""

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {NodeTag.HYPERSPECTRAL, NodeTag.CLASSIFICATION, NodeTag.STATEFUL, NodeTag.NUMPY}
    )

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
        # Ensure signature is on the same device as the cube (handles gRPC weight-reload race)
        spectral_signature = spectral_signature.to(cube.device)
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


class StatefulSpectralAngleMapper(SpectralAngleMapper):
    """Stateful Spectral Angle Mapper node that can persist a learned reference signature in `.pt` weights."""

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C]",
        ),
        "spectral_signature": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Optional runtime spectra [N, 1, 1, C]; falls back to learned buffer",
            optional=True,
        ),
    }

    def __init__(self, num_channels: int, eps: float = 1e-12, **kwargs: Any) -> None:
        super().__init__(num_channels=num_channels, eps=eps, **kwargs)
        self.register_buffer(
            "learned_signature",
            torch.zeros((1, 1, 1, self.num_channels), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "_has_learned_signature",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        self._statistically_initialized = False

    def _canonicalize_signature_tensor(self, signature: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Convert input signatures to canonical shape [N, C]."""
        tensor = torch.as_tensor(
            signature, dtype=torch.float32, device=self.learned_signature.device
        )
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)  # [C] -> [1, C]
        elif tensor.ndim == 2:
            pass  # [N, C]
        elif tensor.ndim == 4:
            # [N, 1, 1, C] -> [N, C]
            tensor = tensor.squeeze(1).squeeze(1)
        else:
            raise ValueError(
                "signature must have shape [C], [N, C], or [N, 1, 1, C], "
                f"got {tuple(tensor.shape)}."
            )

        if tensor.ndim != 2:
            raise ValueError(f"signature must canonicalize to [N, C], got {tuple(tensor.shape)}.")

        if int(tensor.shape[-1]) != self.num_channels:
            raise ValueError(
                "signature channel mismatch: "
                f"expected {self.num_channels}, got {int(tensor.shape[-1])}."
            )

        if int(tensor.shape[0]) != 1:
            raise ValueError(
                f"StatefulSpectralAngleMapper expects one signature [1, C], got {tuple(tensor.shape)}."
            )

        return tensor.contiguous()

    @torch.no_grad()
    def fit_signature(self, signature: torch.Tensor | np.ndarray) -> None:
        """Set and persist one or more reference signatures with shape [N, C]."""
        tensor = self._canonicalize_signature_tensor(signature)
        self.learned_signature.copy_(tensor.unsqueeze(1).unsqueeze(1).contiguous())
        self._has_learned_signature.fill_(True)
        self._statistically_initialized = True

    @torch.no_grad()
    def statistical_initialization(self, input_stream: Iterable[dict[str, torch.Tensor]]) -> None:
        """Initialize learned signature from a port-style stream.

        Each stream item must include a ``spectral_signature`` tensor in one of:
        [C], [N, C], or [N, 1, 1, C]. This node enforces a single signature.
        """
        self._statistically_initialized = False

        collected: list[torch.Tensor] = []
        for batch_data in input_stream:
            signature = batch_data.get("spectral_signature")
            if signature is None:
                continue
            canonical = self._canonicalize_signature_tensor(signature)
            collected.append(canonical)

        if not collected:
            raise RuntimeError(
                "StatefulSpectralAngleMapper.statistical_initialization() did not receive "
                "'spectral_signature' data."
            )

        merged = torch.cat(collected, dim=0)
        if int(merged.shape[0]) != 1:
            raise RuntimeError(
                "StatefulSpectralAngleMapper expects exactly one total signature during "
                f"statistical_initialization(), got {int(merged.shape[0])}."
            )

        self.fit_signature(merged)

    @torch.no_grad()
    def forward(
        self,
        cube: torch.Tensor,
        spectral_signature: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if spectral_signature is None:
            if not bool(self._has_learned_signature.item()):
                raise ValueError(
                    "No learned_signature present. Call fit_signature(...) or pass spectral_signature."
                )
            spectral_signature = self.learned_signature
        return super().forward(cube=cube, spectral_signature=spectral_signature, **kwargs)


__all__ = ["SpectralAngleMapper", "StatefulSpectralAngleMapper"]
