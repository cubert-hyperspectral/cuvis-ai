"""Band selection nodes for HSI to RGB conversion.

This module provides port-based nodes for selecting spectral bands from
hyperspectral cubes and composing RGB images for downstream processing
(e.g., with AdaCLIP).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.ports import PortSpec
from cuvis_ai_core.utils.types import Context, InputStream
from scipy.ndimage import laplace
from sklearn.metrics import roc_auc_score


class BandSelectorBase(Node):
    """Base class for hyperspectral band selection strategies.

    This base class defines the common input/output ports for band selection nodes.
    Subclasses should implement the `forward()` method to perform specific
    band selection strategies.

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

    @staticmethod
    def _nearest_band_index(wavelengths: np.ndarray, target_nm: float) -> int:
        """Find the index of the band nearest to the target wavelength."""
        return int(np.argmin(np.abs(wavelengths - target_nm)))

    def _compose_rgb(self, cube: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
        """Compose RGB from selected band indices.

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
        # Gather selected bands on the SAME device as cube
        # cube: [B, H, W, C] → [B, H, W, 3]
        bands = [cube[..., idx] for idx in indices]  # each [B, H, W]
        rgb = torch.stack(bands, dim=-1)  # [B, H, W, 3]

        # Per-batch, per-channel min/max normalization to [0, 1]
        # Keep dims so broadcasting works: [B, 1, 1, 3]
        rgb_min = rgb.amin(dim=(1, 2), keepdim=True)
        rgb_max = rgb.amax(dim=(1, 2), keepdim=True)
        denom = (rgb_max - rgb_min).clamp_min(1e-8)

        rgb = (rgb - rgb_min) / denom
        return rgb.clamp_(0.0, 1.0)


class BaselineFalseRGBSelector(BandSelectorBase):
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

        # Compose RGB
        rgb = self._compose_rgb(cube, indices)

        band_info = {
            "strategy": "baseline_false_rgb",
            "band_indices": indices,
            "band_wavelengths_nm": [float(wavelengths_np[i]) for i in indices],
            "target_wavelengths_nm": list(self.target_wavelengths),
        }

        return {"rgb_image": rgb, "band_info": band_info}


class HighContrastBandSelector(BandSelectorBase):
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


class CIRFalseColorSelector(BandSelectorBase):
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

    This implementation is streaming, it avoids stacking
    all pixels from all cubes into a single huge matrix. Instead it performs
    two passes:

    1) Accumulate per-band mean and (uncentered) second moment using Welford's
       method to obtain global means and variances.
    2) Accumulate cross terms E[x_i * x_j] across batches to derive the
       covariance and correlation.
    """
    if len(training_cubes) == 0:
        return np.eye(num_bands, dtype=np.float64)

    # ------------------------------------------------------------------
    # Pass 1: compute per-band mean and variance via Welford's algorithm
    # ------------------------------------------------------------------
    count = 0
    mean = np.zeros(num_bands, dtype=np.float64)
    m2 = np.zeros(num_bands, dtype=np.float64)  # sum of squared diffs

    for cube in training_cubes:
        # cube: [H, W, C]
        h, w, c = cube.shape
        assert c == num_bands, "num_bands must match cube.shape[2]"
        flat = cube.reshape(-1, c).astype(np.float64, copy=False)  # [N, C]
        # Iterate over rows to keep memory bounded per update
        for x in flat:
            count += 1
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            m2 += delta2 * delta2

    if count <= 1:
        return np.eye(num_bands, dtype=np.float64)

    var = m2 / (count - 1)
    # Avoid division by zero later
    var = np.clip(var, 1e-12, None)
    std = np.sqrt(var)

    # ------------------------------------------------------------------
    # Pass 2: accumulate E[x_i * x_j] to compute covariance / correlation
    # ------------------------------------------------------------------
    exx = np.zeros((num_bands, num_bands), dtype=np.float64)
    total_count = 0

    for cube in training_cubes:
        h, w, c = cube.shape
        flat = cube.reshape(-1, c).astype(np.float64, copy=False)  # [N, C]
        # Sum x^T x over all rows: flat.T @ flat
        exx += flat.T @ flat
        total_count += flat.shape[0]

    if total_count == 0:
        return np.eye(num_bands, dtype=np.float64)

    exx /= float(total_count)  # E[x_i * x_j]

    # Cov[i,j] = E[x_i * x_j] - mu_i * mu_j
    mu_outer = np.outer(mean, mean)
    cov = exx - mu_outer

    # Convert covariance to correlation: corr_ij = cov_ij / (std_i * std_j)
    denom = np.outer(std, std)
    denom = np.clip(denom, 1e-12, None)
    corr = cov / denom

    # Replace NaNs (from zero variance) with 0 and return absolute values
    corr = np.nan_to_num(corr, nan=0.0)
    return np.abs(corr)


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


class SupervisedBandSelectorBase(BandSelectorBase):
    """Base class for supervised band selection strategies.

    This class adds an optional ``mask`` input port and implements common
    logic for statistical initialization via :meth:`fit`.

    The mask is assumed to be binary (0/1), where 1 denotes the positive
    class (e.g. stone) and 0 denotes the negative class (e.g. lentil/background).
    """

    INPUT_SPECS = {
        **BandSelectorBase.INPUT_SPECS,
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

        # Get device from existing buffers
        device = self.band_scores.device

        # Update buffer values (buffers already exist with correct shapes)
        self.band_scores.copy_(torch.from_numpy(band_scores.astype(np.float32)).to(device))
        self.fisher_scores.copy_(torch.from_numpy(fisher_scores.astype(np.float32)).to(device))
        self.auc_scores.copy_(torch.from_numpy(auc_scores.astype(np.float32)).to(device))
        self.mi_scores.copy_(torch.from_numpy(mi_scores.astype(np.float32)).to(device))
        self.selected_indices.copy_(
            torch.as_tensor(selected_indices, dtype=torch.long, device=device)
        )

        self._statistically_initialized = True


class SupervisedCIRBandSelector(SupervisedBandSelectorBase):
    """Supervised CIR/NIR band selection with window constraints.

    Windows are typically set to:
        - NIR: 840-910 nm
        - Red: 650-720 nm
        - Green: 500-570 nm

    The selector chooses one band per window using a supervised score
    (Fisher + AUC + MI) with an mRMR-style redundancy penalty.
    """

    def __init__(
        self,
        windows: Sequence[tuple[float, float]] = ((840.0, 910.0), (650.0, 720.0), (500.0, 570.0)),
        score_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        lambda_penalty: float = 0.5,
        **kwargs: Any,
    ) -> None:
        # Call super().__init__ FIRST so Serializable captures hparams correctly
        super().__init__(
            score_weights=score_weights,
            lambda_penalty=lambda_penalty,
            windows=list(windows),
            **kwargs,
        )
        # Then set instance attributes
        self.windows = list(windows)

    def statistical_initialization(self, input_stream: InputStream) -> None:
        cubes, masks, wavelengths = self._collect_training_data(input_stream)
        band_scores, fisher_scores, auc_scores, mi_scores = _compute_band_scores_supervised(
            cubes,
            masks,
            wavelengths,
            self.score_weights,
        )
        corr_matrix = _compute_band_correlation_matrix(cubes, len(wavelengths))
        selected_indices = _mrmr_band_selection(
            band_scores,
            wavelengths,
            self.windows,
            corr_matrix,
            self.lambda_penalty,
        )
        if len(selected_indices) != 3:
            raise ValueError(
                f"SupervisedCIRBandSelector expected 3 bands, got {len(selected_indices)}"
            )
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
        if not self._statistically_initialized or self.selected_indices.numel() != 3:
            raise RuntimeError("SupervisedCIRBandSelector not fitted")

        wavelengths_np = np.asarray(wavelengths, dtype=np.float32)
        indices = self.selected_indices.tolist()
        rgb = self._compose_rgb(cube, indices)

        band_info = {
            "strategy": "supervised_cir",
            "band_indices": indices,
            "band_wavelengths_nm": [float(wavelengths_np[i]) for i in indices],
            "windows_nm": [[float(s), float(e)] for s, e in self.windows],
            "score_weights": list(self.score_weights),
            "lambda_penalty": float(self.lambda_penalty),
        }
        return {"rgb_image": rgb, "band_info": band_info}


class SupervisedWindowedFalseRGBSelector(SupervisedBandSelectorBase):
    """Supervised band selection constrained to visible RGB windows.

    Similar to :class:`HighContrastBandSelector`, but uses label-driven scores.
    Default windows:
        - Blue: 440–500 nm
        - Green: 500–580 nm
        - Red: 610–700 nm
    """

    def __init__(
        self,
        windows: Sequence[tuple[float, float]] = ((440.0, 500.0), (500.0, 580.0), (610.0, 700.0)),
        score_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        lambda_penalty: float = 0.5,
        **kwargs: Any,
    ) -> None:
        # Call super().__init__ FIRST so Serializable captures hparams correctly
        super().__init__(
            score_weights=score_weights,
            lambda_penalty=lambda_penalty,
            windows=list(windows),
            **kwargs,
        )
        # Then set instance attributes
        self.windows = list(windows)

    def statistical_initialization(self, input_stream: InputStream) -> None:
        cubes, masks, wavelengths = self._collect_training_data(input_stream)
        band_scores, fisher_scores, auc_scores, mi_scores = _compute_band_scores_supervised(
            cubes,
            masks,
            wavelengths,
            self.score_weights,
        )
        corr_matrix = _compute_band_correlation_matrix(cubes, len(wavelengths))
        selected_indices = _mrmr_band_selection(
            band_scores,
            wavelengths,
            self.windows,
            corr_matrix,
            self.lambda_penalty,
        )
        if len(selected_indices) != 3:
            raise ValueError(
                f"SupervisedWindowedFalseRGBSelector expected 3 bands, got {len(selected_indices)}",
            )
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
        if not self._statistically_initialized or self.selected_indices.numel() != 3:
            raise RuntimeError("SupervisedWindowedFalseRGBSelector not fitted")

        wavelengths_np = np.asarray(wavelengths, dtype=np.float32)
        indices = self.selected_indices.tolist()
        rgb = self._compose_rgb(cube, indices)

        band_info = {
            "strategy": "supervised_windowed_false_rgb",
            "band_indices": indices,
            "band_wavelengths_nm": [float(wavelengths_np[i]) for i in indices],
            "windows_nm": [[float(s), float(e)] for s, e in self.windows],
            "score_weights": list(self.score_weights),
            "lambda_penalty": float(self.lambda_penalty),
        }
        return {"rgb_image": rgb, "band_info": band_info}


class SupervisedFullSpectrumBandSelector(SupervisedBandSelectorBase):
    """Supervised selection without window constraints.

    Picks the top-3 discriminative bands globally with an mRMR-style
    redundancy penalty applied over the full spectrum.
    """

    def __init__(
        self,
        score_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        lambda_penalty: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(score_weights=score_weights, lambda_penalty=lambda_penalty, **kwargs)

    def statistical_initialization(self, input_stream: InputStream) -> None:
        cubes, masks, wavelengths = self._collect_training_data(input_stream)
        band_scores, fisher_scores, auc_scores, mi_scores = _compute_band_scores_supervised(
            cubes,
            masks,
            wavelengths,
            self.score_weights,
        )
        corr_matrix = _compute_band_correlation_matrix(cubes, len(wavelengths))
        selected_indices = _select_top_k_bands(
            band_scores,
            corr_matrix,
            k=3,
            lambda_penalty=self.lambda_penalty,
        )
        if len(selected_indices) != 3:
            raise ValueError(
                f"SupervisedFullSpectrumBandSelector expected 3 bands, got {len(selected_indices)}",
            )
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
        if not self._statistically_initialized or self.selected_indices.numel() != 3:
            raise RuntimeError("SupervisedFullSpectrumBandSelector not fitted")

        wavelengths_np = np.asarray(wavelengths, dtype=np.float32)
        indices = self.selected_indices.tolist()
        rgb = self._compose_rgb(cube, indices)

        band_info = {
            "strategy": "supervised_full_spectrum",
            "band_indices": indices,
            "band_wavelengths_nm": [float(wavelengths_np[i]) for i in indices],
            "score_weights": list(self.score_weights),
            "lambda_penalty": float(self.lambda_penalty),
        }
        return {"rgb_image": rgb, "band_info": band_info}


__all__ = [
    "BandSelectorBase",
    "BaselineFalseRGBSelector",
    "CIRFalseColorSelector",
    "HighContrastBandSelector",
    "SupervisedBandSelectorBase",
    "SupervisedCIRBandSelector",
    "SupervisedFullSpectrumBandSelector",
    "SupervisedWindowedFalseRGBSelector",
]
