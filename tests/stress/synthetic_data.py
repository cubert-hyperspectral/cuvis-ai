"""Synthetic hyperspectral data generator for stress testing.

This module provides utilities for generating synthetic hyperspectral cubes
with controllable characteristics (size, channels, anomalies) for stress testing.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticHyperspectralDataset(Dataset):
    """Generate synthetic hyperspectral data for stress testing.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset
    height : int
        Height of each hyperspectral cube (pixels)
    width : int
        Width of each hyperspectral cube (pixels)
    n_channels : int
        Number of spectral channels
    anomaly_ratio : float
        Percentage of anomalous pixels (0.0 to 1.0)
    seed : int
        Random seed for reproducibility
    add_noise : bool
        Whether to add Gaussian noise to the data
    noise_std : float
        Standard deviation of added Gaussian noise
    anomaly_magnitude : float
        Magnitude multiplier for anomalies (higher = more distinct)

    Examples
    --------
    >>> # Small dataset for quick tests
    >>> small_data = SyntheticHyperspectralDataset(
    ...     n_samples=10, height=64, width=64, n_channels=10
    ... )
    >>>
    >>> # Large dataset for stress testing
    >>> large_data = SyntheticHyperspectralDataset(
    ...     n_samples=1000, height=256, width=256, n_channels=100
    ... )
    """

    def __init__(
        self,
        n_samples: int = 10,
        height: int = 64,
        width: int = 64,
        n_channels: int = 10,
        anomaly_ratio: float = 0.05,
        seed: int = 42,
        add_noise: bool = True,
        noise_std: float = 0.1,
        anomaly_magnitude: float = 3.0,
    ):
        self.n_samples = n_samples
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.anomaly_ratio = anomaly_ratio
        self.seed = seed
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.anomaly_magnitude = anomaly_magnitude

        # Set random seed for reproducibility
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        # Pre-compute data characteristics
        self._setup_data_characteristics()

    def _setup_data_characteristics(self):
        """Set up the base spectral characteristics for the dataset."""
        # Create base spectral signature (normal class)
        # Use a smooth spectral curve resembling real hyperspectral data
        wavelengths = np.linspace(0, 1, self.n_channels)

        # Normal class: smooth gaussian-like curve
        self.normal_signature = np.exp(-((wavelengths - 0.5) ** 2) / 0.1)

        # Anomaly class: different spectral signature with peaks
        self.anomaly_signature = 0.5 * np.exp(-((wavelengths - 0.3) ** 2) / 0.05) + 0.5 * np.exp(
            -((wavelengths - 0.7) ** 2) / 0.05
        )

        # Scale signatures to reasonable range
        self.normal_signature = self.normal_signature * 100.0
        self.anomaly_signature = self.anomaly_signature * 100.0 * self.anomaly_magnitude

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Generate a single synthetic hyperspectral sample.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        dict
            Dictionary containing:
            - 'cube': Hyperspectral cube tensor [H, W, C]
            - 'labels': Binary anomaly labels [H, W] (1 for anomaly, 0 for normal)
            - 'mask': Valid pixel mask [H, W] (all 1s for synthetic data)
        """
        # Use index-dependent seed for variety
        sample_rng = np.random.RandomState(self.seed + idx)

        # Create base cube with normal spectral signature
        cube = np.zeros((self.height, self.width, self.n_channels), dtype=np.float32)

        # Fill with normal signature
        for i in range(self.height):
            for j in range(self.width):
                # Add spatial variation (smooth gradients)
                spatial_factor = 1.0 + 0.2 * np.sin(i / self.height * 2 * np.pi)
                spatial_factor *= 1.0 + 0.2 * np.cos(j / self.width * 2 * np.pi)
                cube[i, j, :] = self.normal_signature * spatial_factor

        # Add Gaussian noise if requested
        if self.add_noise:
            noise = sample_rng.normal(0, self.noise_std * 100.0, cube.shape)
            cube += noise

        # Generate anomaly mask
        n_anomaly_pixels = int(self.height * self.width * self.anomaly_ratio)
        labels = np.zeros((self.height, self.width), dtype=np.float32)

        if n_anomaly_pixels > 0:
            # Create spatially clustered anomalies (more realistic)
            n_clusters = max(1, n_anomaly_pixels // 20)  # Roughly 20 pixels per cluster

            for _ in range(n_clusters):
                # Random cluster center
                center_i = sample_rng.randint(0, self.height)
                center_j = sample_rng.randint(0, self.width)

                # Random cluster radius (2-5 pixels)
                radius = sample_rng.randint(2, 6)

                # Fill cluster
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        i = center_i + di
                        j = center_j + dj

                        # Check bounds
                        if 0 <= i < self.height and 0 <= j < self.width:
                            # Check if within circular radius
                            if di**2 + dj**2 <= radius**2:
                                labels[i, j] = 1.0
                                # Replace with anomaly signature
                                cube[i, j, :] = self.anomaly_signature

        # Create valid pixel mask (all valid for synthetic data)
        mask = np.ones((self.height, self.width), dtype=np.float32)

        # Convert to tensors
        cube_tensor = torch.from_numpy(cube)
        labels_tensor = torch.from_numpy(labels)
        mask_tensor = torch.from_numpy(mask)

        return {
            "cube": cube_tensor,
            "labels": labels_tensor,
            "mask": mask_tensor,
        }

    def get_statistics(self) -> dict[str, float]:
        """Get dataset statistics.

        Returns
        -------
        dict
            Statistics including:
            - total_pixels: Total number of pixels across all samples
            - anomaly_pixels: Expected number of anomaly pixels
            - memory_mb: Approximate memory usage in megabytes
            - cube_shape: Shape of each cube (H, W, C)
        """
        total_pixels = self.n_samples * self.height * self.width
        anomaly_pixels = int(total_pixels * self.anomaly_ratio)

        # Memory estimate: 4 bytes per float32 value
        # cube: H * W * C floats
        # labels: H * W floats
        # mask: H * W floats
        bytes_per_sample = (
            self.height * self.width * self.n_channels * 4  # cube
            + self.height * self.width * 4  # labels
            + self.height * self.width * 4  # mask
        )
        memory_mb = (bytes_per_sample * self.n_samples) / (1024 * 1024)

        return {
            "total_pixels": total_pixels,
            "anomaly_pixels": anomaly_pixels,
            "memory_mb": memory_mb,
            "cube_shape": (self.height, self.width, self.n_channels),
            "n_samples": self.n_samples,
        }


def create_small_scale_dataset(**kwargs) -> SyntheticHyperspectralDataset:
    """Create a small-scale dataset for quick tests.

    Default: 10 samples × 64×64 × 10 channels (~0.3 MB)

    Parameters
    ----------
    **kwargs
        Override default parameters

    Returns
    -------
    SyntheticHyperspectralDataset
        Small-scale synthetic dataset
    """
    defaults = {
        "n_samples": 10,
        "height": 64,
        "width": 64,
        "n_channels": 10,
        "anomaly_ratio": 0.05,
    }
    defaults.update(kwargs)
    return SyntheticHyperspectralDataset(**defaults)


def create_medium_scale_dataset(**kwargs) -> SyntheticHyperspectralDataset:
    """Create a medium-scale dataset for realistic tests.

    Default: 100 samples × 128×128 × 50 channels (~80 MB)

    Parameters
    ----------
    **kwargs
        Override default parameters

    Returns
    -------
    SyntheticHyperspectralDataset
        Medium-scale synthetic dataset
    """
    defaults = {
        "n_samples": 100,
        "height": 128,
        "width": 128,
        "n_channels": 50,
        "anomaly_ratio": 0.05,
    }
    defaults.update(kwargs)
    return SyntheticHyperspectralDataset(**defaults)


def create_large_scale_dataset(**kwargs) -> SyntheticHyperspectralDataset:
    """Create a large-scale dataset for stress testing.

    Default: 1000 samples × 256×256 × 100 channels (~6.5 GB)

    Parameters
    ----------
    **kwargs
        Override default parameters

    Returns
    -------
    SyntheticHyperspectralDataset
        Large-scale synthetic dataset
    """
    defaults = {
        "n_samples": 1000,
        "height": 256,
        "width": 256,
        "n_channels": 100,
        "anomaly_ratio": 0.05,
    }
    defaults.update(kwargs)
    return SyntheticHyperspectralDataset(**defaults)


def create_extra_large_scale_dataset(**kwargs) -> SyntheticHyperspectralDataset:
    """Create an extra-large dataset for extreme stress testing.

    Default: 10000 samples × 512×512 × 200 channels (~524 GB)

    WARNING: This dataset requires significant memory and disk space.
    Consider using streaming or chunked processing.

    Parameters
    ----------
    **kwargs
        Override default parameters

    Returns
    -------
    SyntheticHyperspectralDataset
        Extra-large-scale synthetic dataset
    """
    defaults = {
        "n_samples": 10000,
        "height": 512,
        "width": 512,
        "n_channels": 200,
        "anomaly_ratio": 0.05,
    }
    defaults.update(kwargs)
    return SyntheticHyperspectralDataset(**defaults)
