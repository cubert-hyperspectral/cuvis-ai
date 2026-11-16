"""Independent numpy and scipy reference for DD-SIMCA, plus the synthetic data its tests share.

The formulas follow Pomerantsev and Rodionova, "Concept and role of extreme objects in
PCA/SIMCA" (J. Chemometrics 28, 2014), with the conventions of R ``mdatools`` (``R/ldecomp.R``):
the score distance is ``sum(t_a^2 / eigenvalue_a)``, the orthogonal distance is the squared
residual, each is a scaled chi-square with ``u0 = mean(u)`` and ``N = 2 (u0 / sd(u))^2`` (``sd``
on ``n - 1``, a non-finite ``N`` set to 1, then rounded and clamped to 1..250), and the limit is
``qchisq(1 - alpha, N_h + N_q)``.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import chi2


def low_rank_cubes(
    seed: int = 0,
    n_batches: int = 3,
    shape: tuple[int, int, int] = (2, 6, 5),
    n_channels: int = 6,
    scales: tuple[float, ...] = (9.0, 4.0, 2.0, 0.3, 0.2, 0.1),
) -> list[torch.Tensor]:
    """Seeded float32 cubes [B, H, W, C]: strong latent directions, small noise, non-zero mean."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(n_channels, n_channels)))[0]
    mean = rng.normal(size=n_channels) * 2.0
    batch, height, width = shape
    cubes = []
    for _ in range(n_batches):
        latent = rng.normal(size=(batch * height * width, n_channels)) * np.asarray(scales)
        cube = (latent @ basis.T + mean).reshape(batch, height, width, n_channels)
        cubes.append(torch.from_numpy(cube.astype(np.float32)))
    return cubes


def pixels_of(cubes: list[torch.Tensor]) -> np.ndarray:
    """All pixels of the cubes as one float64 [N, C] matrix."""
    return np.concatenate([c.cpu().numpy().reshape(-1, c.shape[-1]) for c in cubes]).astype(
        np.float64
    )


def pca_model(pixels: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA by SVD of the centred pixels: mean [C], components [K, C], eigenvalues [K] (n - 1)."""
    mean = pixels.mean(axis=0)
    _, singular, vt = np.linalg.svd(pixels - mean, full_matrices=False)
    eigenvalues = singular[:n_components] ** 2 / (pixels.shape[0] - 1)
    return mean, vt[:n_components], eigenvalues


def distances(
    pixels: np.ndarray, mean: np.ndarray, components: np.ndarray, eigenvalues: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Score distance T2 and orthogonal distance Q of every pixel, each [N]."""
    scores = (pixels - mean) @ components.T
    residual = pixels - (mean + scores @ components)
    return (scores**2 / eigenvalues).sum(axis=1), (residual**2).sum(axis=1)


def moments(u: np.ndarray) -> tuple[float, float]:
    """Scaled chi-square parameters of a distance by the method of moments: ``u0`` and ``N``."""
    u0 = float(u.mean())
    sd = float(u.std(ddof=1))
    dof = 2.0 * (u0 / sd) ** 2 if sd > 0 else float("inf")
    if not np.isfinite(dof):
        dof = 1.0
    return u0, float(np.clip(np.round(dof), 1, 250))


def limit(alpha: float, n_h: float, n_q: float) -> float:
    """Critical value of the combined distance at significance ``alpha``."""
    return float(chi2.ppf(1.0 - alpha, n_h + n_q))


def combined(
    h: np.ndarray, q: np.ndarray, h0: float, n_h: float, q0: float, n_q: float
) -> np.ndarray:
    """Combined distance ``f = N_h h / h0 + N_q q / q0``."""
    return n_h * h / h0 + n_q * q / q0
