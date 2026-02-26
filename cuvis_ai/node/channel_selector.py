"""Channel selector nodes for HSI to RGB conversion.

This module provides port-based nodes for selecting spectral channels from
hyperspectral cubes and composing RGB images for downstream processing
(e.g., with AdaCLIP).

**Selectors** gate/reweight individual channels independently:
``output[c] = weight[c] * input[c]`` (diagonal operation, preserves channel count).

For cross-channel linear projection (full matrix, reduces channel count),
see :mod:`cuvis_ai.node.channel_mixer`.

Normalization design
--------------------
All channel selectors share a common RGB normalization strategy in
``ChannelSelectorBase``, controlled by ``NormMode``:

- **Percentile bounds** (not absolute min/max): SpectralRadiance data contains
  outlier pixels whose absolute max can be 10x the median, compressing 99% of
  the image into the bottom of the brightness range. Using the 0.5th / 99.5th
  percentile clips these outliers and preserves visual dynamic range.

- **Per-channel [3] bounds**: Separate min/max per R/G/B channel preserves
  colour balance. A single scalar bound would distort hue if one channel has a
  wider range than the others.

- **Three modes** (``NormMode``):
  ``running`` (default) — warmup + min/max accumulation. The first N frames use
      per-frame normalization (visually good immediately) while silently
      building global percentile bounds. After warmup the accumulated bounds are
      used, giving temporal stability identical to ``statistical`` mode.
  ``statistical`` — pre-computed global percentiles via ``StatisticalTrainer``.
      Use when exact global stats matter and a full first pass is acceptable.
  ``per_frame`` — each frame normalized independently; no inter-frame state.
      Use for unrelated images or single-frame pipelines.

- **Why warmup + accumulation** (not EMA): Exponential moving averages have
  recency bias — for long videos the early-frame statistics are forgotten. The
  min/max accumulation bounds only ever *expand* (min-of-lows, max-of-highs),
  so they converge to the exact same result as a full statistical-init pass
  without requiring a separate first pass. The warmup period ensures the first
  few frames look natural before enough data has been accumulated.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import Context, InputStream
from cuvis_ai_schemas.pipeline import PortSpec
from scipy.ndimage import laplace
from sklearn.metrics import roc_auc_score
from torch import Tensor

from cuvis_ai.utils.welford import WelfordAccumulator


class NormMode(StrEnum):
    """RGB normalization mode for channel selectors."""

    PER_FRAME = "per_frame"
    RUNNING = "running"
    STATISTICAL = "statistical"


class ChannelSelectorBase(Node):
    """Base class for hyperspectral band selection strategies.

    This base class defines the common input/output ports for band selection
    nodes and provides shared percentile-based RGB normalization (see module
    docstring for design rationale).

    Subclasses should implement ``forward()`` and ``_compute_raw_rgb()`` (the
    latter is used by ``statistical_initialization`` and ``_running_normalize``).

    Parameters
    ----------
    norm_mode : str | NormMode
        RGB normalization mode.  Default ``NormMode.RUNNING``.
    apply_gamma : bool
        Apply sRGB gamma curve after normalization.  Default ``True``.
        Lifts midtones so linear [0, 1] values appear natural on standard
        displays.

    Ports
    -----
    INPUT_SPECS
        ``cube`` : float32, shape (-1, -1, -1, -1)
            Hyperspectral cube in BHWC format.
        ``wavelengths`` : float32, shape (-1,)
            Wavelength array in nanometers.
    OUTPUT_SPECS
        ``rgb_image`` : float32, shape (-1, -1, -1, 3)
            Composed RGB image in BHWC format (0-1 range).
        ``band_info`` : dict
            Metadata about selected bands.
    """

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C]",
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Wavelength array [C] in nanometers",
        ),
    }

    OUTPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Composed RGB image [B, H, W, 3] in 0-1 range",
        ),
        "band_info": PortSpec(
            dtype=dict,
            shape=(),
            description="Selected band metadata",
        ),
    }

    # Percentile bounds for normalization (fractions, not percentages).
    # Prevents outlier pixels from compressing the dynamic range.
    _NORM_QUANTILE_LOW = 0.005  # 0.5th percentile
    _NORM_QUANTILE_HIGH = 0.995  # 99.5th percentile
    _WARMUP_FRAMES = 10  # per-frame normalization during warmup

    def __init__(
        self,
        norm_mode: str | NormMode = NormMode.RUNNING,
        apply_gamma: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            norm_mode=str(norm_mode) if isinstance(norm_mode, NormMode) else norm_mode,
            apply_gamma=apply_gamma,
            **kwargs,
        )
        self.norm_mode = NormMode(norm_mode)
        self.apply_gamma = apply_gamma

        # Per-channel [3] running bounds for normalization.
        self.register_buffer("running_min", torch.full((3,), float("nan")))
        self.register_buffer("running_max", torch.full((3,), float("nan")))
        self._norm_frame_count = 0
        self._statistically_initialized = False

        if self.norm_mode == NormMode.STATISTICAL:
            self._requires_initial_fit_override = True

    @staticmethod
    def _nearest_band_index(wavelengths: np.ndarray, target_nm: float) -> int:
        """Find the index of the band nearest to the target wavelength."""
        return int(np.argmin(np.abs(wavelengths - target_nm)))

    # ------------------------------------------------------------------
    # RGB normalization
    # ------------------------------------------------------------------

    def _normalize_rgb(self, raw_rgb: torch.Tensor) -> torch.Tensor:
        """Normalize raw RGB tensor to [0, 1] and optionally apply sRGB gamma.

        Parameters
        ----------
        raw_rgb : torch.Tensor
            Unnormalized RGB image [B, H, W, 3].

        Returns
        -------
        torch.Tensor
            Normalized (and gamma-corrected if ``apply_gamma``) RGB [B, H, W, 3].
        """
        if self.norm_mode == NormMode.STATISTICAL and self._statistically_initialized:
            result = self._apply_accumulated_stats(raw_rgb)
        elif self.norm_mode == NormMode.RUNNING:
            result = self._running_normalize(raw_rgb)
        else:
            result = self._per_frame_normalize(raw_rgb)

        if self.apply_gamma:
            result = self._srgb_gamma(result)
        return result

    @staticmethod
    def _per_frame_normalize(rgb: torch.Tensor) -> torch.Tensor:
        """Per-batch, per-channel min-max normalization to [0, 1]."""
        rgb_min = rgb.amin(dim=(1, 2), keepdim=True)
        rgb_max = rgb.amax(dim=(1, 2), keepdim=True)
        denom = (rgb_max - rgb_min).clamp_min(1e-8)
        return ((rgb - rgb_min) / denom).clamp_(0.0, 1.0)

    def _per_frame_percentile_normalize(self, rgb: torch.Tensor) -> torch.Tensor:
        """Per-frame percentile normalization matching the running accumulation quantiles."""
        flat = rgb.reshape(-1, 3).float()  # quantile() requires float/double
        lo = torch.quantile(flat, self._NORM_QUANTILE_LOW, dim=0).view(1, 1, 1, 3)
        hi = torch.quantile(flat, self._NORM_QUANTILE_HIGH, dim=0).view(1, 1, 1, 3)
        denom = (hi - lo).clamp_min(1e-8)
        return ((rgb - lo) / denom).clamp_(0.0, 1.0)

    def _apply_accumulated_stats(self, rgb: torch.Tensor) -> torch.Tensor:
        """Normalize using accumulated per-channel bounds."""
        lo = self.running_min.view(1, 1, 1, 3)
        hi = self.running_max.view(1, 1, 1, 3)
        denom = (hi - lo).clamp_min(1e-8)
        return ((rgb - lo) / denom).clamp_(0.0, 1.0)

    @staticmethod
    def _srgb_gamma(linear: torch.Tensor) -> torch.Tensor:
        """Apply sRGB companding (IEC 61966-2-1).

        Converts linear [0, 1] values to sRGB gamma-encoded [0, 1].
        This lifts midtones so images appear natural on standard displays.
        """
        low = 12.92 * linear
        high = 1.055 * linear.clamp_min(1e-10).pow(1.0 / 2.4) - 0.055
        return torch.where(linear <= 0.0031308, low, high)

    @torch.no_grad()
    def _running_normalize(self, rgb: torch.Tensor) -> torch.Tensor:
        """Warmup + min/max accumulation hybrid normalization.

        Always accumulates per-frame percentile bounds (monotonic: bounds only
        expand).  During the warmup period the output uses per-frame
        normalization so the first frames look natural.  After warmup, switches
        to the accumulated bounds for temporal stability.
        """
        flat = rgb.reshape(-1, 3).float()  # quantile() requires float/double
        frame_lo = torch.quantile(flat, self._NORM_QUANTILE_LOW, dim=0)  # [3]
        frame_hi = torch.quantile(flat, self._NORM_QUANTILE_HIGH, dim=0)  # [3]

        if torch.isnan(self.running_min).any():
            self.running_min.copy_(frame_lo)
            self.running_max.copy_(frame_hi)
        else:
            torch.minimum(self.running_min, frame_lo, out=self.running_min)
            torch.maximum(self.running_max, frame_hi, out=self.running_max)

        self._norm_frame_count += 1

        if self._norm_frame_count <= self._WARMUP_FRAMES:
            return self._per_frame_percentile_normalize(rgb)
        return self._apply_accumulated_stats(rgb)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _compute_raw_rgb(self, cube: torch.Tensor, wavelengths: Any) -> torch.Tensor:
        """Compute unnormalized RGB from a hyperspectral cube.

        Subclasses must override this to support ``statistical`` and
        ``running`` normalization modes.

        Parameters
        ----------
        cube : torch.Tensor
            Hyperspectral cube [B, H, W, C].
        wavelengths : Any
            Wavelength array [C] in nanometers.

        Returns
        -------
        torch.Tensor
            Unnormalized RGB [B, H, W, 3].
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _compute_raw_rgb() "
            "to support statistical normalization."
        )

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Compute global percentile bounds across the entire dataset.

        Uses ``_compute_raw_rgb()`` to convert each batch, then accumulates
        per-channel percentile bounds (min-of-lows, max-of-highs).
        """
        for batch_data in input_stream:
            raw_rgb = self._compute_raw_rgb(batch_data["cube"], batch_data["wavelengths"])
            flat = raw_rgb.reshape(-1, 3).float()  # quantile() requires float/double
            frame_lo = torch.quantile(flat, self._NORM_QUANTILE_LOW, dim=0)
            frame_hi = torch.quantile(flat, self._NORM_QUANTILE_HIGH, dim=0)

            if torch.isnan(self.running_min).any():
                self.running_min.copy_(frame_lo)
                self.running_max.copy_(frame_hi)
            else:
                torch.minimum(self.running_min, frame_lo, out=self.running_min)
                torch.maximum(self.running_max, frame_hi, out=self.running_max)

        if not torch.isnan(self.running_min).any():
            self._statistically_initialized = True

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def _compose_rgb(self, cube: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
        """Compose and normalize RGB from selected band indices.

        Parameters
        ----------
        cube : torch.Tensor
            Hyperspectral cube [B, H, W, C].
        indices : Sequence[int]
            Indices of bands to use for R, G, B channels.

        Returns
        -------
        torch.Tensor
            RGB image [B, H, W, 3] in 0-1 range, on the same device as input.
        """
        bands = [cube[..., idx] for idx in indices]  # each [B, H, W]
        rgb = torch.stack(bands, dim=-1)  # [B, H, W, 3]
        return self._normalize_rgb(rgb)


class FixedWavelengthSelector(ChannelSelectorBase):
    """Fixed wavelength band selection (e.g., 650, 550, 450 nm).

    Selects bands nearest to the specified target wavelengths for R, G, B channels.
    This is the simplest band selection strategy that produces "true color-ish" images.

    Parameters
    ----------
    target_wavelengths : tuple[float, float, float]
        Target wavelengths for R, G, B channels in nanometers.
        Default: (650.0, 550.0, 450.0)
    """

    def __init__(
        self,
        target_wavelengths: tuple[float, float, float] = (650.0, 550.0, 450.0),
        **kwargs,
    ) -> None:
        super().__init__(target_wavelengths=target_wavelengths, **kwargs)
        self.target_wavelengths = target_wavelengths

    def _compute_raw_rgb(self, cube: torch.Tensor, wavelengths: Any) -> torch.Tensor:
        wavelengths_np = np.asarray(wavelengths, dtype=np.float32)
        indices = [self._nearest_band_index(wavelengths_np, nm) for nm in self.target_wavelengths]
        bands = [cube[..., idx] for idx in indices]
        return torch.stack(bands, dim=-1)

    def forward(
        self,
        cube: torch.Tensor,
        wavelengths: Any,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        """Select bands and compose RGB image.

        Parameters
        ----------
        cube : torch.Tensor
            Hyperspectral cube [B, H, W, C].
        wavelengths : torch.Tensor
            Wavelength array [C].

        Returns
        -------
        dict[str, Any]
            Dictionary with "rgb_image" and "band_info" keys.
        """
        wavelengths_np = np.asarray(wavelengths, dtype=np.float32)

        # Find nearest bands
        indices = [self._nearest_band_index(wavelengths_np, nm) for nm in self.target_wavelengths]

        # Compose RGB (includes normalization via _normalize_rgb)
        rgb = self._compose_rgb(cube, indices)

        band_info = {
            "strategy": "baseline_false_rgb",
            "band_indices": indices,
            "band_wavelengths_nm": [float(wavelengths_np[i]) for i in indices],
            "target_wavelengths_nm": list(self.target_wavelengths),
        }

        return {"rgb_image": rgb, "band_info": band_info}


class RangeAverageFalseRGBSelector(ChannelSelectorBase):
    """Range-based false RGB selection by averaging bands per channel.

    For each output channel (R/G/B), all spectral bands within the configured
    wavelength range are averaged per pixel. Channels with no matching bands are
    filled with zeros.

    Parameters
    ----------
    red_range : tuple[float, float]
        Inclusive wavelength range for red channel in nanometers.
    green_range : tuple[float, float]
        Inclusive wavelength range for green channel in nanometers.
    blue_range : tuple[float, float]
        Inclusive wavelength range for blue channel in nanometers.
    """

    def __init__(
        self,
        red_range: tuple[float, float] = (580.0, 650.0),
        green_range: tuple[float, float] = (500.0, 580.0),
        blue_range: tuple[float, float] = (420.0, 500.0),
        **kwargs: Any,
    ) -> None:
        for name, rng in {
            "red_range": red_range,
            "green_range": green_range,
            "blue_range": blue_range,
        }.items():
            if len(rng) != 2 or rng[0] > rng[1]:
                raise ValueError(f"{name} must be (min_nm, max_nm) with min_nm <= max_nm")

        super().__init__(
            red_range=red_range, green_range=green_range, blue_range=blue_range, **kwargs
        )
        self.red_range = red_range
        self.green_range = green_range
        self.blue_range = blue_range

        # Static channel range boundaries [3, 2]; buffer so .to(device) moves it.
        self.register_buffer(
            "_ranges",
            torch.tensor(
                [
                    [red_range[0], red_range[1]],
                    [green_range[0], green_range[1]],
                    [blue_range[0], blue_range[1]],
                ],
                dtype=torch.float32,
            ),
        )
        # Wavelength-dependent channel weights; lazily computed on first forward.
        self.register_buffer("_avg_weights", None, persistent=False)
        self.register_buffer("_avg_mask", None, persistent=False)
        self._cached_wl_key: tuple[float, ...] | None = None

    @staticmethod
    def _prepare_wavelengths_tensor(
        wavelengths: Any,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert wavelengths input to 1D float32 tensor on target device."""
        if isinstance(wavelengths, torch.Tensor):
            wavelengths_t = wavelengths
        else:
            wavelengths_t = torch.as_tensor(wavelengths)

        if wavelengths_t.ndim == 2:
            wavelengths_t = wavelengths_t[0]
        if wavelengths_t.ndim != 1:
            raise ValueError(f"Expected 1D wavelengths [C], got shape {tuple(wavelengths_t.shape)}")

        return wavelengths_t.to(device=device, dtype=torch.float32)

    def _build_channel_weights(
        self, wavelengths_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build per-channel averaging weights and inclusion mask from wavelength ranges."""
        # Use pre-registered _ranges buffer (moves with .to(device)).
        ranges_t = self._ranges.to(device=wavelengths_t.device, dtype=wavelengths_t.dtype)
        low = ranges_t[:, 0:1]  # [3,1]
        high = ranges_t[:, 1:2]  # [3,1]

        # channel_mask[k, c] == True iff wavelength c belongs to channel k range.
        channel_mask = (wavelengths_t.unsqueeze(0) >= low) & (wavelengths_t.unsqueeze(0) <= high)

        counts = channel_mask.sum(dim=1, keepdim=True)  # [3,1]
        weights = channel_mask.to(torch.float32) / counts.clamp_min(1).to(torch.float32)
        return weights, channel_mask

    def _ensure_weights(self, wavelengths: Any, device: torch.device) -> None:
        """Lazily compute & cache channel weights when wavelengths change."""
        wavelengths_t = self._prepare_wavelengths_tensor(wavelengths, device)
        wl_key = tuple(wavelengths_t.tolist())
        if self._avg_weights is None or self._cached_wl_key != wl_key:
            self._avg_weights, self._avg_mask = self._build_channel_weights(wavelengths_t)
            self._cached_wl_key = wl_key

    def _compute_raw_rgb(self, cube: torch.Tensor, wavelengths: Any) -> torch.Tensor:
        self._ensure_weights(wavelengths, cube.device)
        return torch.einsum("bhwc,kc->bhwk", cube, self._avg_weights)

    def forward(
        self,
        cube: torch.Tensor,
        wavelengths: Any,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        """Average spectral bands inside RGB ranges and compose normalized RGB."""
        self._ensure_weights(wavelengths, cube.device)
        wavelengths_t = self._prepare_wavelengths_tensor(wavelengths, cube.device)

        # Vectorized channel averaging:
        # cube [B,H,W,C] and weights [3,C] -> rgb [B,H,W,3]
        rgb = self._compute_raw_rgb(cube, wavelengths)
        rgb = self._normalize_rgb(rgb)

        channel_indices = [
            torch.where(self._avg_mask[i])[0].tolist() for i in range(self._avg_mask.shape[0])
        ]
        channel_names = ["red", "green", "blue"]
        missing_channels = [
            channel_names[i] for i, indices in enumerate(channel_indices) if len(indices) == 0
        ]

        band_info = {
            "strategy": "range_average_false_rgb",
            "band_indices": channel_indices,  # [R, G, B]
            "band_wavelengths_nm": [wavelengths_t[idxs].tolist() for idxs in channel_indices],
            "ranges_nm": {
                "red": [float(self.red_range[0]), float(self.red_range[1])],
                "green": [float(self.green_range[0]), float(self.green_range[1])],
                "blue": [float(self.blue_range[0]), float(self.blue_range[1])],
            },
            "aggregation": "mean",
            "missing_channels": missing_channels,
        }
        return {"rgb_image": rgb, "band_info": band_info}


class HighContrastSelector(ChannelSelectorBase):
    """Data-driven band selection using spatial variance + Laplacian energy.

    For each wavelength window, selects the band with the highest score based on:
    score = variance + alpha * Laplacian_energy

    This produces "high contrast" images that may work better for visual
    anomaly detection.

    Parameters
    ----------
    windows : Sequence[tuple[float, float]]
        Wavelength windows for Blue, Green, Red channels.
        Default: ((440, 500), (500, 580), (610, 700)) for visible spectrum.
    alpha : float
        Weight for Laplacian energy term. Default: 0.1
    """

    def __init__(
        self,
        windows: Sequence[tuple[float, float]] = ((440, 500), (500, 580), (610, 700)),
        alpha: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(windows=windows, alpha=alpha, **kwargs)
        self.windows = list(windows)
        self.alpha = alpha

    def forward(
        self,
        cube: torch.Tensor,
        wavelengths: Any,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        """Select high-contrast bands and compose RGB image.

        Parameters
        ----------
        cube : torch.Tensor
            Hyperspectral cube [B, H, W, C].
        wavelengths : torch.Tensor
            Wavelength array [C].

        Returns
        -------
        dict[str, Any]
            Dictionary with "rgb_image" and "band_info" keys.
        """
        wavelengths_np = np.asarray(wavelengths, dtype=np.float32)
        # Use first batch item for band selection
        cube_np = cube[0].cpu().numpy()

        selected_indices = []
        for start, end in self.windows:
            mask = (wavelengths_np >= start) & (wavelengths_np <= end)
            window_indices = np.where(mask)[0]

            if len(window_indices) == 0:
                # Fallback to nearest single wavelength
                nearest = self._nearest_band_index(wavelengths_np, (start + end) / 2.0)
                selected_indices.append(int(nearest))
                continue

            scores = []
            for idx in window_indices:
                band = cube_np[..., idx]
                variance = float(np.var(band))
                lap_energy = float(np.mean(np.abs(laplace(band))))
                scores.append(variance + self.alpha * lap_energy)

            best_idx = int(window_indices[int(np.argmax(scores))])
            selected_indices.append(best_idx)

        rgb = self._compose_rgb(cube, selected_indices)

        band_info = {
            "strategy": "high_contrast",
            "band_indices": selected_indices,
            "band_wavelengths_nm": [float(wavelengths_np[i]) for i in selected_indices],
            "windows_nm": [[float(s), float(e)] for s, e in self.windows],
            "alpha": self.alpha,
        }

        return {"rgb_image": rgb, "band_info": band_info}


class CIRSelector(ChannelSelectorBase):
    """Color Infrared (CIR) false color composition.

    Maps NIR to Red, Red to Green, Green to Blue for false-color composites.
    This is useful for highlighting vegetation and certain anomalies.

    Parameters
    ----------
    nir_nm : float
        Near-infrared wavelength in nm. Default: 860.0
    red_nm : float
        Red wavelength in nm. Default: 670.0
    green_nm : float
        Green wavelength in nm. Default: 560.0
    """

    def __init__(
        self,
        nir_nm: float = 860.0,
        red_nm: float = 670.0,
        green_nm: float = 560.0,
        **kwargs,
    ) -> None:
        super().__init__(nir_nm=nir_nm, red_nm=red_nm, green_nm=green_nm, **kwargs)
        self.nir_nm = nir_nm
        self.red_nm = red_nm
        self.green_nm = green_nm

    def forward(
        self,
        cube: torch.Tensor,
        wavelengths: Any,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        """Select CIR bands and compose false-color image.

        Parameters
        ----------
        cube : torch.Tensor
            Hyperspectral cube [B, H, W, C].
        wavelengths : torch.Tensor
            Wavelength array [C].

        Returns
        -------
        dict[str, Any]
            Dictionary with "rgb_image" and "band_info" keys.
        """
        wavelengths_np = np.asarray(wavelengths, dtype=np.float32)

        # CIR mapping: NIR -> R, Red -> G, Green -> B
        nir_idx = self._nearest_band_index(wavelengths_np, self.nir_nm)
        red_idx = self._nearest_band_index(wavelengths_np, self.red_nm)
        green_idx = self._nearest_band_index(wavelengths_np, self.green_nm)

        indices = [nir_idx, red_idx, green_idx]
        rgb = self._compose_rgb(cube, indices)

        band_info = {
            "strategy": "cir_false_color",
            "band_indices": indices,
            "band_wavelengths_nm": [float(wavelengths_np[i]) for i in indices],
            "target_wavelengths_nm": [self.nir_nm, self.red_nm, self.green_nm],
            "channel_mapping": {"R": "NIR", "G": "Red", "B": "Green"},
        }

        return {"rgb_image": rgb, "band_info": band_info}


class CIETristimulusFalseRGBSelector(ChannelSelectorBase):
    """CIE 1931 tristimulus-based false RGB rendering.

    Converts a hyperspectral cube to sRGB by integrating each pixel's spectrum
    with the CIE 1931 2-degree standard observer color matching functions
    (x_bar, y_bar, z_bar), applying a D65 white point normalization, and
    converting from CIE XYZ to linear sRGB.

    Normalization and sRGB gamma are handled by ``ChannelSelectorBase`` (see
    ``apply_gamma`` parameter inherited from the base class).

    This produces the most physically grounded false RGB and lands closest to
    the distribution SAM3's Perception Encoder expects.

    For wavelengths outside the visible range (approx. >780 nm), the CMFs are
    zero, so NIR bands do not contribute to the output.
    """

    # CIE 1931 2-degree observer CMFs at 5 nm intervals, 380-780 nm.
    # Source: CIE 015:2004, Table 1 (standard tabulation).
    # fmt: off
    _CMF_WAVELENGTHS = np.array([
        380, 385, 390, 395, 400, 405, 410, 415, 420, 425,
        430, 435, 440, 445, 450, 455, 460, 465, 470, 475,
        480, 485, 490, 495, 500, 505, 510, 515, 520, 525,
        530, 535, 540, 545, 550, 555, 560, 565, 570, 575,
        580, 585, 590, 595, 600, 605, 610, 615, 620, 625,
        630, 635, 640, 645, 650, 655, 660, 665, 670, 675,
        680, 685, 690, 695, 700, 705, 710, 715, 720, 725,
        730, 735, 740, 745, 750, 755, 760, 765, 770, 775, 780,
    ], dtype=np.float64)

    _X_BAR = np.array([
        0.0014, 0.0022, 0.0042, 0.0076, 0.0143, 0.0232, 0.0435, 0.0776,
        0.1344, 0.2148, 0.2839, 0.3285, 0.3483, 0.3481, 0.3362, 0.3187,
        0.2908, 0.2511, 0.1954, 0.1421, 0.0956, 0.0580, 0.0320, 0.0147,
        0.0049, 0.0024, 0.0093, 0.0291, 0.0633, 0.1096, 0.1655, 0.2257,
        0.2904, 0.3597, 0.4334, 0.5121, 0.5945, 0.6784, 0.7621, 0.8425,
        0.9163, 0.9786, 1.0263, 1.0567, 1.0622, 1.0456, 1.0026, 0.9384,
        0.8544, 0.7514, 0.6424, 0.5419, 0.4479, 0.3608, 0.2835, 0.2187,
        0.1649, 0.1212, 0.0874, 0.0636, 0.0468, 0.0329, 0.0227, 0.0158,
        0.0114, 0.0081, 0.0058, 0.0041, 0.0029, 0.0020, 0.0014, 0.0010,
        0.0007, 0.0005, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001,
        0.0000,
    ], dtype=np.float64)

    _Y_BAR = np.array([
        0.0000, 0.0001, 0.0001, 0.0002, 0.0004, 0.0006, 0.0012, 0.0022,
        0.0040, 0.0073, 0.0116, 0.0168, 0.0230, 0.0298, 0.0380, 0.0480,
        0.0600, 0.0739, 0.0910, 0.1126, 0.1390, 0.1693, 0.2080, 0.2586,
        0.3230, 0.4073, 0.5030, 0.6082, 0.7100, 0.7932, 0.8620, 0.9149,
        0.9540, 0.9803, 0.9950, 1.0000, 0.9950, 0.9786, 0.9520, 0.9154,
        0.8700, 0.8163, 0.7570, 0.6949, 0.6310, 0.5668, 0.5030, 0.4412,
        0.3810, 0.3210, 0.2650, 0.2170, 0.1750, 0.1382, 0.1070, 0.0816,
        0.0610, 0.0446, 0.0320, 0.0232, 0.0170, 0.0119, 0.0082, 0.0057,
        0.0041, 0.0029, 0.0021, 0.0015, 0.0010, 0.0007, 0.0005, 0.0004,
        0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000,
        0.0000,
    ], dtype=np.float64)

    _Z_BAR = np.array([
        0.0065, 0.0105, 0.0201, 0.0362, 0.0679, 0.1102, 0.2074, 0.3713,
        0.6456, 1.0391, 1.3856, 1.6230, 1.7471, 1.7826, 1.7721, 1.7441,
        1.6692, 1.5281, 1.2876, 1.0419, 0.8130, 0.6162, 0.4652, 0.3533,
        0.2720, 0.2123, 0.1582, 0.1117, 0.0782, 0.0573, 0.0422, 0.0298,
        0.0203, 0.0134, 0.0087, 0.0057, 0.0039, 0.0027, 0.0021, 0.0018,
        0.0017, 0.0014, 0.0011, 0.0010, 0.0008, 0.0006, 0.0003, 0.0002,
        0.0002, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000,
    ], dtype=np.float64)
    # fmt: on

    # XYZ -> linear sRGB matrix (IEC 61966-2-1 / Rec. 709 primaries, D65).
    _XYZ_TO_SRGB = np.array(
        [
            [3.2406255, -1.5372080, -0.4986286],
            [-0.9689307, 1.8757561, 0.0415175],
            [0.0557101, -0.2040211, 1.0569959],
        ],
        dtype=np.float64,
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Static XYZ -> linear sRGB matrix; buffer so .to(device) moves it.
        self.register_buffer(
            "_xyz_to_srgb_matrix",
            torch.from_numpy(self._XYZ_TO_SRGB.astype(np.float32)),
        )
        # Wavelength-dependent CMF integration weights; lazily computed on first forward.
        self.register_buffer("_cmf_weights", None, persistent=False)
        self._cached_wl_key: tuple[float, ...] | None = None
        self._cached_n_visible: int = 0

    def _interpolate_cmfs(
        self,
        wavelengths_nm: np.ndarray,
    ) -> np.ndarray:
        """Interpolate CIE CMFs to the sensor's wavelength grid.

        Parameters
        ----------
        wavelengths_nm : np.ndarray
            Sensor wavelengths in nm, shape (C,).

        Returns
        -------
        np.ndarray
            CMF weights, shape (3, C) — rows are x_bar, y_bar, z_bar.
        """
        x_interp = np.interp(
            wavelengths_nm, self._CMF_WAVELENGTHS, self._X_BAR, left=0.0, right=0.0
        )
        y_interp = np.interp(
            wavelengths_nm, self._CMF_WAVELENGTHS, self._Y_BAR, left=0.0, right=0.0
        )
        z_interp = np.interp(
            wavelengths_nm, self._CMF_WAVELENGTHS, self._Z_BAR, left=0.0, right=0.0
        )
        return np.stack([x_interp, y_interp, z_interp], axis=0)  # (3, C)

    def _ensure_cmf_weights(self, wavelengths: Any, device: torch.device) -> np.ndarray | None:
        """Lazily compute & cache CMF integration weights when wavelengths change.

        Returns the raw CMFs array (for n_visible count) if weights were
        recomputed, or None if the cache was valid.
        """
        wavelengths_np = np.asarray(wavelengths, dtype=np.float64).ravel()
        wl_key = tuple(wavelengths_np.tolist())
        if self._cmf_weights is None or self._cached_wl_key != wl_key:
            cmfs = self._interpolate_cmfs(wavelengths_np)
            spacing = np.gradient(wavelengths_np)
            iw = (cmfs * spacing[np.newaxis, :]).astype(np.float32)
            self._cmf_weights = torch.from_numpy(iw).to(device=device)
            self._cached_wl_key = wl_key
            self._cached_n_visible = int((cmfs.sum(axis=0) > 1e-6).sum())
            return cmfs
        return None

    def _compute_raw_rgb(self, cube: torch.Tensor, wavelengths: Any) -> torch.Tensor:
        self._ensure_cmf_weights(wavelengths, cube.device)
        xyz = torch.einsum("bhwc,kc->bhwk", cube, self._cmf_weights)
        rgb_linear = torch.einsum("bhwj,ij->bhwi", xyz, self._xyz_to_srgb_matrix)
        return rgb_linear.clamp_min(0.0)

    def forward(
        self,
        cube: torch.Tensor,
        wavelengths: Any,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        """Convert HSI cube to sRGB via CIE 1931 tristimulus integration.

        Parameters
        ----------
        cube : torch.Tensor
            Hyperspectral cube [B, H, W, C].
        wavelengths : torch.Tensor | np.ndarray
            Wavelength array [C] in nanometers.

        Returns
        -------
        dict[str, Any]
            Dictionary with "rgb_image" [B, H, W, 3] and "band_info".
        """
        wavelengths_np = np.asarray(wavelengths, dtype=np.float64).ravel()
        if wavelengths_np.ndim == 0:
            raise ValueError("wavelengths must be a 1-D array")

        # Compute unnormalized linear sRGB, then normalize + gamma via base class.
        rgb = self._normalize_rgb(self._compute_raw_rgb(cube, wavelengths))

        band_info = {
            "strategy": "cie_tristimulus",
            "illuminant": "D65",
            "apply_gamma": self.apply_gamma,
            "sensor_bands_total": len(wavelengths_np),
            "sensor_bands_visible": self._cached_n_visible,
            "wavelength_range_nm": [float(wavelengths_np[0]), float(wavelengths_np[-1])],
        }

        return {"rgb_image": rgb, "band_info": band_info}


class CameraEmulationFalseRGBSelector(ChannelSelectorBase):
    """Camera-emulation false RGB using smooth Gaussian sensitivity curves.

    Defines three broad, smooth Gaussian weighting curves over the spectral
    bands that mimic R/G/B camera sensitivity (peaks at configurable
    wavelengths). The weight matrix W is [3, num_bands], applied as
    ``rgb = W @ spectrum``. Non-negativity is enforced by construction.

    This is simple, stable, and requires no training. Good middle ground
    between single-band selection and learned mapping.

    Parameters
    ----------
    r_peak : float
        Red channel peak wavelength in nm. Default: 610.0
    g_peak : float
        Green channel peak wavelength in nm. Default: 540.0
    b_peak : float
        Blue channel peak wavelength in nm. Default: 460.0
    r_sigma : float
        Red channel Gaussian sigma in nm. Default: 40.0
    g_sigma : float
        Green channel Gaussian sigma in nm. Default: 35.0
    b_sigma : float
        Blue channel Gaussian sigma in nm. Default: 30.0
    """

    def __init__(
        self,
        r_peak: float = 610.0,
        g_peak: float = 540.0,
        b_peak: float = 460.0,
        r_sigma: float = 40.0,
        g_sigma: float = 35.0,
        b_sigma: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            r_peak=r_peak,
            g_peak=g_peak,
            b_peak=b_peak,
            r_sigma=r_sigma,
            g_sigma=g_sigma,
            b_sigma=b_sigma,
            **kwargs,
        )
        self.peaks = (r_peak, g_peak, b_peak)
        self.sigmas = (r_sigma, g_sigma, b_sigma)

        # Wavelength-dependent Gaussian weights; lazily computed on first forward.
        self.register_buffer("_channel_weights", None, persistent=False)
        self._cached_wl_key: tuple[float, ...] | None = None

    def _build_weights(self, wavelengths_np: np.ndarray) -> np.ndarray:
        """Build [3, C] Gaussian weight matrix, each row sums to 1."""
        w = np.zeros((3, len(wavelengths_np)), dtype=np.float64)
        for i, (peak, sigma) in enumerate(zip(self.peaks, self.sigmas, strict=False)):
            w[i] = np.exp(-0.5 * ((wavelengths_np - peak) / sigma) ** 2)
        # Normalize each row so weights sum to 1
        row_sums = w.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        w /= row_sums
        return w

    def _ensure_channel_weights(self, wavelengths: Any, device: torch.device) -> None:
        """Lazily compute & cache Gaussian weights when wavelengths change."""
        wavelengths_np = np.asarray(wavelengths, dtype=np.float64).ravel()
        wl_key = tuple(wavelengths_np.tolist())
        if self._channel_weights is None or self._cached_wl_key != wl_key:
            weights = self._build_weights(wavelengths_np)
            self._channel_weights = torch.from_numpy(weights.astype(np.float32)).to(device=device)
            self._cached_wl_key = wl_key

    def _compute_raw_rgb(self, cube: torch.Tensor, wavelengths: Any) -> torch.Tensor:
        self._ensure_channel_weights(wavelengths, cube.device)
        return torch.einsum("bhwc,kc->bhwk", cube, self._channel_weights)

    def forward(
        self,
        cube: torch.Tensor,
        wavelengths: Any,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        """Convert HSI cube to false RGB using Gaussian camera sensitivity.

        Parameters
        ----------
        cube : torch.Tensor
            Hyperspectral cube [B, H, W, C].
        wavelengths : torch.Tensor | np.ndarray
            Wavelength array [C] in nanometers.

        Returns
        -------
        dict[str, Any]
            Dictionary with "rgb_image" [B, H, W, 3] and "band_info".
        """
        wavelengths_np = np.asarray(wavelengths, dtype=np.float64).ravel()

        rgb = self._compute_raw_rgb(cube, wavelengths)
        rgb = self._normalize_rgb(rgb)

        band_info = {
            "strategy": "camera_emulation",
            "peaks_nm": {"R": self.peaks[0], "G": self.peaks[1], "B": self.peaks[2]},
            "sigmas_nm": {"R": self.sigmas[0], "G": self.sigmas[1], "B": self.sigmas[2]},
            "sensor_bands_total": len(wavelengths_np),
        }

        return {"rgb_image": rgb, "band_info": band_info}


# ---------------------------------------------------------------------------
# Supervised band selection helpers
# ---------------------------------------------------------------------------


def _compute_fisher_score(class0_intensities: np.ndarray, class1_intensities: np.ndarray) -> float:
    """Compute Fisher score for class separation."""
    mu0 = float(np.mean(class0_intensities))
    mu1 = float(np.mean(class1_intensities))
    sigma0_sq = float(np.var(class0_intensities))
    sigma1_sq = float(np.var(class1_intensities))
    denominator = sigma0_sq + sigma1_sq
    if denominator < 1e-10:
        return 0.0
    return float((mu1 - mu0) ** 2 / denominator)


def _compute_band_auc(band_intensities: np.ndarray, binary_labels: np.ndarray) -> float:
    """Compute ROC AUC using band intensity as score for binary classification."""
    if len(np.unique(binary_labels)) < 2:
        return 0.0
    try:
        return float(roc_auc_score(binary_labels, band_intensities))
    except ValueError:
        return 0.0


def _compute_mutual_information(
    band_intensities: np.ndarray,
    binary_labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute mutual information between discretized band intensities and labels."""
    if len(np.unique(binary_labels)) < 2:
        return 0.0
    try:
        band_min = float(np.min(band_intensities))
        band_max = float(np.max(band_intensities))
        if not np.isfinite(band_min) or not np.isfinite(band_max) or band_max <= band_min:
            return 0.0
        band_discrete = np.digitize(
            band_intensities,
            bins=np.linspace(band_min, band_max, n_bins, endpoint=True),
        )
        # Joint and marginal distributions
        joint = np.histogram2d(band_discrete, binary_labels, bins=(n_bins, 2))[0] + 1e-10
        joint = joint / np.sum(joint)
        marginal_band = np.sum(joint, axis=1)
        marginal_label = np.sum(joint, axis=0)
        mi = 0.0
        for i in range(n_bins):
            for j in range(2):
                if joint[i, j] > 0:
                    mi += joint[i, j] * np.log2(
                        joint[i, j] / (marginal_band[i] * marginal_label[j])
                    )
        return float(mi)
    except Exception:
        return 0.0


def _compute_band_scores_supervised(
    training_cubes: list[np.ndarray],
    training_masks: list[np.ndarray],
    wavelengths: np.ndarray,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute discriminative scores for each band using supervised metrics.

    We assume binary masks y ∈ {0,1} where 1 = positive (stone) and 0 = negative (lentil/background).
    Returns: (combined_scores, fisher_scores, auc_scores, mi_scores)
    """
    n_bands = len(wavelengths)
    fisher_scores = np.zeros(n_bands, dtype=np.float64)
    auc_scores = np.zeros(n_bands, dtype=np.float64)
    mi_scores = np.zeros(n_bands, dtype=np.float64)

    # Combine all training data
    all_intensities_per_band: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for cube, mask in zip(training_cubes, training_masks, strict=True):
        h, w, c = cube.shape
        cube_flat = cube.reshape(-1, c)  # [H*W, C]
        mask_flat = mask.reshape(-1)  # [H*W]

        # Only use pixels that are positive (1) or negative (0)
        valid_mask = (mask_flat == 0) | (mask_flat == 1)
        if np.sum(valid_mask) == 0:
            continue

        cube_valid = cube_flat[valid_mask]  # [N, C]
        labels_valid = mask_flat[valid_mask]  # [N]

        all_intensities_per_band.append(cube_valid)
        all_labels.append(labels_valid)

    if len(all_intensities_per_band) == 0:
        return (
            np.zeros(n_bands, dtype=np.float64),
            np.zeros(n_bands, dtype=np.float64),
            np.zeros(n_bands, dtype=np.float64),
            np.zeros(n_bands, dtype=np.float64),
        )

    # Stack all training data
    all_intensities = np.vstack(all_intensities_per_band)  # [total_pixels, C]
    all_labels_combined = np.hstack(all_labels)  # [total_pixels]

    # Compute scores for each band
    for band_idx in range(n_bands):
        band_intensities = all_intensities[:, band_idx]

        # Split by class
        class0_mask = all_labels_combined == 0
        class1_mask = all_labels_combined == 1
        if np.sum(class0_mask) == 0 or np.sum(class1_mask) == 0:
            continue

        class0_intensities = band_intensities[class0_mask]
        class1_intensities = band_intensities[class1_mask]

        fisher_scores[band_idx] = _compute_fisher_score(class0_intensities, class1_intensities)
        auc_scores[band_idx] = _compute_band_auc(band_intensities, all_labels_combined)
        mi_scores[band_idx] = _compute_mutual_information(band_intensities, all_labels_combined)

    # Normalize scores (z-score) to comparable scales
    def normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Z-score normalize scores to zero mean and unit variance.

        Parameters
        ----------
        scores : np.ndarray
            Input scores to normalize.

        Returns
        -------
        np.ndarray
            Normalized scores. If std is near zero, only centers the scores.
        """
        mean = np.mean(scores)
        std = np.std(scores)
        if std < 1e-10:
            return scores - mean
        return (scores - mean) / std

    fisher_norm = normalize_scores(fisher_scores)
    auc_norm = normalize_scores(auc_scores)
    mi_norm = normalize_scores(mi_scores)

    w_fisher, w_auc, w_mi = weights
    combined_scores = w_fisher * fisher_norm + w_auc * auc_norm + w_mi * mi_norm
    return combined_scores, fisher_scores, auc_scores, mi_scores


def _compute_band_correlation_matrix(
    training_cubes: list[np.ndarray], num_bands: int
) -> np.ndarray:
    """Compute absolute correlation matrix across bands from training cubes.

    Uses a single-pass ``WelfordAccumulator`` with covariance tracking to
    compute mean, variance, and covariance in one streaming pass over all
    cubes, then derives the absolute correlation matrix.
    """
    if len(training_cubes) == 0:
        return np.eye(num_bands, dtype=np.float64)

    acc = WelfordAccumulator(num_bands, track_covariance=True)
    for cube in training_cubes:
        h, w, c = cube.shape
        assert c == num_bands, "num_bands must match cube.shape[2]"
        flat = torch.from_numpy(cube.reshape(-1, c))  # (N, C)
        acc.update(flat)

    if acc.count <= 1:
        return np.eye(num_bands, dtype=np.float64)

    return acc.corr.numpy()


def _mrmr_band_selection(
    scores: np.ndarray,
    wavelengths: np.ndarray,
    windows: Sequence[tuple[float, float]],
    corr_matrix: np.ndarray,
    lambda_penalty: float = 0.5,
) -> list[int]:
    """Select bands using mRMR (max relevance, min redundancy) within windows."""
    selected_indices: list[int] = []

    for window_start, window_end in windows:
        window_mask = (wavelengths >= window_start) & (wavelengths <= window_end)
        window_indices = np.where(window_mask)[0]

        if len(window_indices) == 0:
            center = (window_start + window_end) / 2.0
            nearest = int(np.argmin(np.abs(wavelengths - center)))
            selected_indices.append(nearest)
            continue

        if len(selected_indices) == 0:
            best_in_window = window_indices[int(np.argmax(scores[window_indices]))]
            selected_indices.append(int(best_in_window))
        else:
            best_idx = None
            best_adjusted_score = -np.inf
            for candidate_idx in window_indices:
                max_corr = max(corr_matrix[candidate_idx, sel] for sel in selected_indices)
                adjusted_score = scores[candidate_idx] - lambda_penalty * max_corr
                if adjusted_score > best_adjusted_score:
                    best_adjusted_score = adjusted_score
                    best_idx = candidate_idx
            if best_idx is not None:
                selected_indices.append(int(best_idx))

    return selected_indices


def _select_top_k_bands(
    scores: np.ndarray,
    corr_matrix: np.ndarray,
    k: int = 3,
    lambda_penalty: float = 0.5,
) -> list[int]:
    """Select top-k bands globally with redundancy penalty."""
    selected: list[int] = []
    available = set(range(len(scores)))

    while len(selected) < k and available:
        best_idx = None
        best_adjusted = -np.inf
        for idx in available:
            if len(selected) == 0:
                adjusted = scores[idx]
            else:
                max_corr = max(corr_matrix[idx, sel] for sel in selected)
                adjusted = scores[idx] - lambda_penalty * max_corr
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_idx = idx
        if best_idx is None:
            break
        selected.append(int(best_idx))
        available.remove(best_idx)

    return selected


class SupervisedSelectorBase(ChannelSelectorBase):
    """Base class for supervised band selection strategies.

    This class adds an optional ``mask`` input port and implements common
    logic for statistical initialization via :meth:`fit`.

    The mask is assumed to be binary (0/1), where 1 denotes the positive
    class (e.g. stone) and 0 denotes the negative class (e.g. lentil/background).
    """

    INPUT_SPECS = {
        **ChannelSelectorBase.INPUT_SPECS,
        "mask": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary mask [B, H, W, 1] with 1=positive, 0=negative",
            optional=True,
        ),
    }

    def __init__(
        self,
        num_spectral_bands: int,
        score_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        lambda_penalty: float = 0.5,
        **kwargs: Any,
    ) -> None:
        # Call super().__init__ FIRST so Serializable captures hparams correctly
        super().__init__(
            num_spectral_bands=num_spectral_bands,
            score_weights=score_weights,
            lambda_penalty=lambda_penalty,
            **kwargs,
        )
        # Then set instance attributes
        self.num_spectral_bands = num_spectral_bands
        self.score_weights = score_weights
        self.lambda_penalty = lambda_penalty
        # Initialize buffers with correct shapes (not empty)
        # selected_indices: always 3 for RGB
        # score buffers: num_spectral_bands
        self.register_buffer("selected_indices", torch.zeros(3, dtype=torch.long), persistent=True)
        self.register_buffer(
            "band_scores", torch.zeros(num_spectral_bands, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "fisher_scores", torch.zeros(num_spectral_bands, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "auc_scores", torch.zeros(num_spectral_bands, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "mi_scores", torch.zeros(num_spectral_bands, dtype=torch.float32), persistent=True
        )
        # Use standard instance attribute for initialization tracking
        self._statistically_initialized = False

    @property
    def requires_initial_fit(self) -> bool:
        """Whether this node requires statistical initialization from training data.

        Returns
        -------
        bool
            Always True for supervised band selectors.
        """
        return True

    def _collect_training_data(
        self,
        input_stream: InputStream,
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """Collect training cubes, masks, and wavelengths from an InputStream."""
        training_cubes: list[np.ndarray] = []
        training_masks: list[np.ndarray] = []
        wavelengths_np: np.ndarray | None = None

        for batch in input_stream:
            cube = batch.get("cube")
            mask = batch.get("mask")
            wavelengths = batch.get("wavelengths")
            if cube is None or mask is None:
                continue

            cube_np = cube.detach().cpu().numpy()  # [B, H, W, C]
            mask_np = mask.detach().cpu().numpy()  # [B, H, W, 1] or [B, H, W]
            if mask_np.ndim == 4:
                mask_np = mask_np[..., 0]

            B = cube_np.shape[0]
            for b in range(B):
                training_cubes.append(cube_np[b])
                training_masks.append(mask_np[b])

            if wavelengths is not None and wavelengths_np is None:
                if isinstance(wavelengths, torch.Tensor):
                    wavelengths_np = wavelengths.detach().cpu().numpy()
                else:
                    wavelengths_np = np.asarray(wavelengths)
                wavelengths_np = wavelengths_np.astype(np.float32, copy=False)
                # DataLoader stacks per-sample [C] wavelengths into [B, C];
                # we only need the 1-D channel vector.
                if wavelengths_np.ndim > 1:
                    wavelengths_np = wavelengths_np[0]

        if wavelengths_np is None:
            raise RuntimeError("SupervisedBandSelector requires wavelengths in the input stream.")

        return training_cubes, training_masks, wavelengths_np

    def _store_scores_and_indices(
        self,
        band_scores: np.ndarray,
        fisher_scores: np.ndarray,
        auc_scores: np.ndarray,
        mi_scores: np.ndarray,
        selected_indices: list[int],
    ) -> None:
        """Store scores and selected indices into buffers.

        Buffers are already initialized with correct shapes in __init__,
        so we just update their values using copy_.
        """
        # Verify shapes match
        if len(band_scores) != self.num_spectral_bands:
            raise ValueError(
                f"band_scores length {len(band_scores)} != num_spectral_bands {self.num_spectral_bands}"
            )
        if len(selected_indices) != 3:
            raise ValueError(f"selected_indices must have 3 elements, got {len(selected_indices)}")

        # Update buffer values (buffers already exist with correct shapes)
        # .copy_() handles cross-device transfer automatically
        self.band_scores.copy_(torch.from_numpy(band_scores.astype(np.float32)))
        self.fisher_scores.copy_(torch.from_numpy(fisher_scores.astype(np.float32)))
        self.auc_scores.copy_(torch.from_numpy(auc_scores.astype(np.float32)))
        self.mi_scores.copy_(torch.from_numpy(mi_scores.astype(np.float32)))
        self.selected_indices.copy_(torch.as_tensor(selected_indices, dtype=torch.long))

        self._statistically_initialized = True

    # -- Hooks for subclasses ------------------------------------------------

    _strategy_name: str = ""
    """Strategy identifier included in ``band_info``. Subclasses must set this."""

    def _select_bands(
        self,
        band_scores: np.ndarray,
        wavelengths: np.ndarray,
        corr_matrix: np.ndarray,
    ) -> list[int]:
        """Select 3 band indices from scored bands. Subclasses must implement."""
        raise NotImplementedError

    def _extra_band_info(self, wavelengths_np: np.ndarray) -> dict[str, Any]:
        """Return additional ``band_info`` entries (e.g., windows). Override as needed."""
        return {}

    # -- Common implementations ----------------------------------------------

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Initialize band selection using supervised scoring.

        Computes Fisher, AUC, and MI scores for each band, delegates to
        :meth:`_select_bands` for strategy-specific selection, and stores
        the 3 selected bands.

        Parameters
        ----------
        input_stream : InputStream
            Training data stream with cube, mask, and wavelengths.

        Raises
        ------
        ValueError
            If band selection doesn't return exactly 3 bands.
        """
        cubes, masks, wavelengths = self._collect_training_data(input_stream)
        band_scores, fisher_scores, auc_scores, mi_scores = _compute_band_scores_supervised(
            cubes,
            masks,
            wavelengths,
            self.score_weights,
        )
        corr_matrix = _compute_band_correlation_matrix(cubes, len(wavelengths))
        selected_indices = self._select_bands(band_scores, wavelengths, corr_matrix)
        if len(selected_indices) != 3:
            raise ValueError(f"{type(self).__name__} expected 3 bands, got {len(selected_indices)}")
        self._store_scores_and_indices(
            band_scores, fisher_scores, auc_scores, mi_scores, selected_indices
        )

    def forward(
        self,
        cube: torch.Tensor,
        wavelengths: np.ndarray,
        mask: torch.Tensor | None = None,  # noqa: ARG002
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, Any]:
        """Generate false-color RGB from selected bands.

        Parameters
        ----------
        cube : torch.Tensor
            Hyperspectral cube [B, H, W, C].
        wavelengths : np.ndarray
            Wavelengths for each channel [C].
        mask : torch.Tensor, optional
            Ground truth mask (unused in forward, required for initialization).
        context : Context, optional
            Pipeline execution context (unused).
        **_ : Any
            Additional unused keyword arguments.

        Returns
        -------
        dict[str, Any]
            Dictionary with "rgb_image" [B, H, W, 3] and "band_info" metadata.

        Raises
        ------
        RuntimeError
            If the node has not been statistically initialized.
        """
        if not self._statistically_initialized or self.selected_indices.numel() != 3:
            raise RuntimeError(f"{type(self).__name__} not fitted")

        wavelengths_np = np.asarray(wavelengths, dtype=np.float32)
        indices = self.selected_indices.tolist()
        rgb = self._compose_rgb(cube, indices)

        band_info = {
            "strategy": self._strategy_name,
            "band_indices": indices,
            "band_wavelengths_nm": [float(wavelengths_np[i]) for i in indices],
            "score_weights": list(self.score_weights),
            "lambda_penalty": float(self.lambda_penalty),
            **self._extra_band_info(wavelengths_np),
        }
        return {"rgb_image": rgb, "band_info": band_info}


class SupervisedCIRSelector(SupervisedSelectorBase):
    """Supervised CIR/NIR band selection with window constraints.

    Windows are typically set to:
        - NIR: 840-910 nm
        - Red: 650-720 nm
        - Green: 500-570 nm

    The selector chooses one band per window using a supervised score
    (Fisher + AUC + MI) with an mRMR-style redundancy penalty.
    """

    _strategy_name = "supervised_cir"

    def __init__(
        self,
        windows: Sequence[tuple[float, float]] = ((840.0, 910.0), (650.0, 720.0), (500.0, 570.0)),
        score_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        lambda_penalty: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            score_weights=score_weights,
            lambda_penalty=lambda_penalty,
            windows=list(windows),
            **kwargs,
        )
        self.windows = list(windows)

    def _select_bands(
        self,
        band_scores: np.ndarray,
        wavelengths: np.ndarray,
        corr_matrix: np.ndarray,
    ) -> list[int]:
        return _mrmr_band_selection(
            band_scores, wavelengths, self.windows, corr_matrix, self.lambda_penalty
        )

    def _extra_band_info(self, wavelengths_np: np.ndarray) -> dict[str, Any]:
        return {"windows_nm": [[float(s), float(e)] for s, e in self.windows]}


class SupervisedWindowedSelector(SupervisedSelectorBase):
    """Supervised band selection constrained to visible RGB windows.

    Similar to :class:`HighContrastSelector`, but uses label-driven scores.
    Default windows:
        - Blue: 440-500 nm
        - Green: 500-580 nm
        - Red: 610-700 nm
    """

    _strategy_name = "supervised_windowed_false_rgb"

    def __init__(
        self,
        windows: Sequence[tuple[float, float]] = ((440.0, 500.0), (500.0, 580.0), (610.0, 700.0)),
        score_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        lambda_penalty: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            score_weights=score_weights,
            lambda_penalty=lambda_penalty,
            windows=list(windows),
            **kwargs,
        )
        self.windows = list(windows)

    def _select_bands(
        self,
        band_scores: np.ndarray,
        wavelengths: np.ndarray,
        corr_matrix: np.ndarray,
    ) -> list[int]:
        return _mrmr_band_selection(
            band_scores, wavelengths, self.windows, corr_matrix, self.lambda_penalty
        )

    def _extra_band_info(self, wavelengths_np: np.ndarray) -> dict[str, Any]:
        return {"windows_nm": [[float(s), float(e)] for s, e in self.windows]}


class SupervisedFullSpectrumSelector(SupervisedSelectorBase):
    """Supervised selection without window constraints.

    Picks the top-3 discriminative bands globally with an mRMR-style
    redundancy penalty applied over the full spectrum.
    """

    _strategy_name = "supervised_full_spectrum"

    def __init__(
        self,
        score_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        lambda_penalty: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(score_weights=score_weights, lambda_penalty=lambda_penalty, **kwargs)

    def _select_bands(
        self,
        band_scores: np.ndarray,
        wavelengths: np.ndarray,  # noqa: ARG002
        corr_matrix: np.ndarray,
    ) -> list[int]:
        return _select_top_k_bands(
            band_scores, corr_matrix, k=3, lambda_penalty=self.lambda_penalty
        )


class SoftChannelSelector(Node):
    """Soft channel selector with temperature-based Gumbel-Softmax selection.

    This is a **selector** node — it gates/reweights individual channels independently:
    ``output[c] = weight[c] * input[c]`` (diagonal operation, preserves channel count).

    For cross-channel linear projection that reduces channel count, see
    :class:`cuvis_ai.node.channel_mixer.ConcreteChannelMixer` or
    :class:`cuvis_ai.node.channel_mixer.LearnableChannelMixer`.

    This node learns to select a subset of input channels using differentiable
    channel selection with temperature annealing. Supports:
    - Statistical initialization (uniform or importance-based)
    - Gradient-based optimization with temperature scheduling
    - Entropy and diversity regularization
    - Hard selection at inference time

    Parameters
    ----------
    n_select : int
        Number of channels to select
    input_channels : int
        Number of input channels
    init_method : {"uniform", "variance"}, optional
        Initialization method for channel weights (default: "uniform")
    temperature_init : float, optional
        Initial temperature for Gumbel-Softmax (default: 5.0)
    temperature_min : float, optional
        Minimum temperature (default: 0.1)
    temperature_decay : float, optional
        Temperature decay factor per epoch (default: 0.9)
    hard : bool, optional
        If True, use hard selection at inference (default: False)
    eps : float, optional
        Small constant for numerical stability (default: 1e-6)

    Attributes
    ----------
    channel_logits : nn.Parameter or Tensor
        Unnormalized channel importance scores [n_channels]
    temperature : float
        Current temperature for Gumbel-Softmax
    """

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube (BHWC format)",
        )
    }

    OUTPUT_SPECS = {
        "selected": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Channel-weighted output (same shape as input)",
        ),
        "weights": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Current channel selection weights",
        ),
    }

    TRAINABLE_BUFFERS = ("channel_logits",)

    def __init__(
        self,
        n_select: int,
        input_channels: int,
        init_method: Literal["uniform", "variance"] = "uniform",
        temperature_init: float = 5.0,
        temperature_min: float = 0.1,
        temperature_decay: float = 0.9,
        hard: bool = False,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        self.n_select = n_select
        self.input_channels = input_channels
        self.init_method = init_method
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.hard = hard
        self.eps = eps

        super().__init__(
            n_select=n_select,
            input_channels=input_channels,
            init_method=init_method,
            temperature_init=temperature_init,
            temperature_min=temperature_min,
            temperature_decay=temperature_decay,
            hard=hard,
            eps=eps,
            **kwargs,
        )

        # Temperature tracking (not a parameter, managed externally)
        self.temperature = temperature_init
        self._n_channels = input_channels

        # Validate selection size
        if self.n_select > self._n_channels:
            raise ValueError(
                f"Cannot select {self.n_select} channels from {self._n_channels} available channels"  # nosec B608
            )

        # Initialize channel logits based on method - always as buffer
        if self.init_method == "uniform":
            # Uniform initialization
            logits = torch.zeros(self._n_channels)
        elif self.init_method == "variance":
            # Random initialization - will be refined with fit if called
            logits = torch.randn(self._n_channels) * 0.01
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

        # Store as buffer initially
        self.register_buffer("channel_logits", logits)

        self._statistically_initialized = False

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Initialize channel selection weights from data.

        Parameters
        ----------
        input_stream : InputStream
            Iterator yielding dicts matching INPUT_SPECS (port-based format)
            Expected format: {"data": tensor} where tensor is BHWC
        """
        # Collect statistics from first batch to determine n_channels
        first_batch = next(iter(input_stream))
        x = first_batch["data"]

        if x is None:
            raise ValueError("No data provided for selector initialization")

        self._n_channels = x.shape[-1]

        if self.n_select > self._n_channels:
            raise ValueError(
                f"Cannot select {self.n_select} channels from {self._n_channels} available channels"  # nosec B608
            )

        # Initialize channel logits based on method
        if self.init_method == "uniform":
            # Uniform initialization
            logits = torch.zeros(self._n_channels)
        elif self.init_method == "variance":
            # Importance-based initialization using channel variance
            acc = WelfordAccumulator(self._n_channels)
            acc.update(x.reshape(-1, x.shape[-1]))
            for batch_data in input_stream:
                x_batch = batch_data["data"]
                if x_batch is not None:
                    acc.update(x_batch.reshape(-1, x_batch.shape[-1]))

            variance = acc.var  # [C]

            # Use log variance as initial logits (high variance = high importance)
            logits = torch.log(variance + self.eps)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

        # Store as buffer
        self.channel_logits.data[:] = logits.clone()
        self._statistically_initialized = True

    def update_temperature(self, epoch: int | None = None, step: int | None = None) -> None:
        """Update temperature with decay schedule.

        Parameters
        ----------
        epoch : int, optional
            Current epoch number (used for per-epoch decay)
        step : int, optional
            Current training step (for more granular control)
        """
        if epoch is not None:
            # Exponential decay per epoch
            self.temperature = max(
                self.temperature_min, self.temperature_init * (self.temperature_decay**epoch)
            )

    def get_selection_weights(self, hard: bool | None = None) -> Tensor:
        """Get current channel selection weights.

        Parameters
        ----------
        hard : bool, optional
            If True, use hard selection (top-k). If None, uses self.hard.

        Returns
        -------
        Tensor
            Selection weights [n_channels] summing to n_select
        """
        if hard is None:
            hard = self.hard and not self.training

        if hard:
            # Hard selection: top-k channels
            _, top_indices = torch.topk(self.channel_logits, self.n_select)
            weights = torch.zeros_like(self.channel_logits)
            weights[top_indices] = 1.0
        else:
            # Soft selection with Gumbel-Softmax
            # First, compute selection probabilities
            probs = F.softmax(self.channel_logits / self.temperature, dim=-1)

            # Scale to sum to n_select instead of 1
            weights = probs * self.n_select

        return weights

    def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
        """Apply soft channel selection to input.

        Parameters
        ----------
        data : Tensor
            Input tensor [B, H, W, C]

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "selected" key containing reweighted channels
            and optional "weights" key containing selection weights
        """
        # Get selection weights
        weights = self.get_selection_weights()

        # Apply channel-wise weighting: [B, H, W, C] * [C]
        selected = data * weights.view(1, 1, 1, -1)

        # Prepare output dictionary - weights always exposed for loss/metric nodes
        outputs = {"selected": selected, "weights": weights}

        return outputs


class TopKIndices(Node):
    """Utility node that surfaces the top-k channel indices from selector weights.

    This node extracts the indices of the top-k weighted channels from a selector's
    weight vector. Useful for introspection and reporting which channels were selected.

    Parameters
    ----------
    k : int
        Number of top indices to return

    Attributes
    ----------
    k : int
        Number of top indices to return
    """

    INPUT_SPECS = {
        "weights": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Channel selection weights",
        )
    }
    OUTPUT_SPECS = {
        "indices": PortSpec(
            dtype=torch.int64,
            shape=(-1,),
            description="Top-k channel indices",
        )
    }

    def __init__(self, k: int, **kwargs: Any) -> None:
        self.k = int(k)

        # Extract Node base parameters from kwargs to avoid duplication
        name = kwargs.pop("name", None)
        execution_stages = kwargs.pop("execution_stages", None)

        super().__init__(
            name=name,
            execution_stages=execution_stages,
            k=self.k,
            **kwargs,
        )

    def forward(self, weights: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Return the indices of the top-k weighted channels.

        Parameters
        ----------
        weights : torch.Tensor
            Channel selection weights [n_channels]

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with "indices" key containing top-k indices
        """
        top_k = min(self.k, weights.shape[-1]) if weights.numel() else 0
        if top_k == 0:
            return {"indices": torch.zeros(0, dtype=torch.int64, device=weights.device)}

        _, indices = torch.topk(weights, top_k)
        return {"indices": indices}


__all__ = [
    "CameraEmulationFalseRGBSelector",
    "ChannelSelectorBase",
    "CIETristimulusFalseRGBSelector",
    "NormMode",
    "CIRSelector",
    "FixedWavelengthSelector",
    "HighContrastSelector",
    "RangeAverageFalseRGBSelector",
    "SoftChannelSelector",
    "SupervisedCIRSelector",
    "SupervisedFullSpectrumSelector",
    "SupervisedSelectorBase",
    "SupervisedWindowedSelector",
    "TopKIndices",
]
