"""Test data factory fixtures for creating hyperspectral cubes."""

import functools
from collections.abc import Generator
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from cuvis_ai.grpc.v1 import cuvis_ai_pb2
from cuvis_ai.training.config import DataConfig, OptimizerConfig, TrainerConfig, TrainingConfig
from cuvis_ai.training.datamodule import CuvisDataModule

# Session-scoped cache for test data files to avoid repeated file system operations
_test_data_cache = {}


@pytest.fixture(scope="session")
def test_data_files_cached(test_data_path: Path) -> Generator[tuple[Path, Path], None, None]:
    """Session-scoped cached version of test data files.

    Caches the test data file paths to avoid repeated file existence checks
    and path resolution across multiple tests.

    Args:
        test_data_path: Base path for test data

    Yields:
        tuple[Path, Path]: (cu3s_file, json_file) paths

    Raises:
        pytest.skip: If test data files not found
    """
    cache_key = str(test_data_path)
    if cache_key not in _test_data_cache:
        cu3s_file = test_data_path / "Lentils" / "Lentils_000.cu3s"
        json_file = test_data_path / "Lentils" / "Lentils_000.json"

        if not cu3s_file.exists() or not json_file.exists():
            pytest.skip(f"Test data not found under {test_data_path}")

        _test_data_cache[cache_key] = (cu3s_file, json_file)

    yield _test_data_cache[cache_key]


# Memoize data config creation to avoid redundant proto serialization
@functools.lru_cache(maxsize=32)
def _create_cached_data_config(
    cu3s_file_path: str,
    json_file_path: str,
    batch_size: int,
    processing_mode: str,
    train_ids: tuple[int, ...],
    val_ids: tuple[int, ...],
    test_ids: tuple[int, ...],
) -> cuvis_ai_pb2.DataConfig:
    """Cached version of DataConfig creation."""
    return DataConfig(
        cu3s_file_path=cu3s_file_path,
        annotation_json_path=json_file_path,
        train_ids=list(train_ids),
        val_ids=list(val_ids),
        test_ids=list(test_ids),
        batch_size=batch_size,
        processing_mode=processing_mode,
    ).to_proto()


@pytest.fixture
def data_config_factory(test_data_files_cached: tuple[Path, Path]):
    """Factory for creating DataConfig proto objects using cached files.

    Provides convenient creation of DataConfig with sensible defaults
    for test data files, using cached file paths and memoized configuration.

    Returns:
        Callable: Function that builds and returns a DataConfig proto
    """
    cu3s_file, json_file = test_data_files_cached

    def _create_config(
        batch_size: int = 2,
        processing_mode: cuvis_ai_pb2.ProcessingMode = cuvis_ai_pb2.PROCESSING_MODE_RAW,
        train_ids: list[int] | None = None,
        val_ids: list[int] | None = None,
        test_ids: list[int] | None = None,
        cu3s_override: Path | None = None,
        json_override: Path | None = None,
    ) -> cuvis_ai_pb2.DataConfig:
        """Create DataConfig with defaults using cached files.

        Args:
            batch_size: Batch size (default: 2)
            processing_mode: Processing mode (default: RAW)
            train_ids: Training IDs (default: [0, 1, 2])
            val_ids: Validation IDs (default: [3, 4])
            test_ids: Test IDs (default: [5, 6])
            cu3s_override: Override default cu3s file path
            json_override: Override default json file path
        """
        cu3s_file_path = cu3s_override or cu3s_file
        json_path = json_override or json_file

        if not cu3s_file_path.exists() or not json_path.exists():
            pytest.skip(f"Test data not found under {cu3s_file_path.parent}")

        processing_mode_str = (
            "Raw" if processing_mode == cuvis_ai_pb2.PROCESSING_MODE_RAW else "Reflectance"
        )

        # Use memoized function for caching
        return _create_cached_data_config(
            str(cu3s_file_path),
            str(json_path),
            batch_size,
            processing_mode_str,
            tuple(train_ids or [0, 1, 2]),
            tuple(val_ids or [3, 4]),
            tuple(test_ids or [5, 6]),
        )

    return _create_config


@pytest.fixture
def create_test_cube():
    """Factory fixture for creating test hyperspectral cubes with various patterns.

    This fixture provides a flexible way to generate test cubes for different scenarios:
    - wavelength_dependent: Values equal to wavelengths (for bandpass/filtering tests)
    - random: Random values (for normalization/statistical tests)
    - synthetic: Structured synthetic data with trends (for anomaly detection tests)

    Returns:
        Factory function that creates (cube, wavelengths) tuple

    Examples:
        >>> create_cube = create_test_cube  # in a test function
        >>> cube, waves = create_cube(batch_size=2, height=10, width=10, num_channels=100)
        >>> cube, waves = create_cube(mode="random", seed=42)
    """

    def _create(
        batch_size: int = 2,
        height: int = 10,
        width: int = 10,
        num_channels: int = 61,
        mode: Literal["wavelength_dependent", "random", "synthetic"] = "wavelength_dependent",
        wavelength_range: tuple[float, float] = (430.0, 910.0),
        seed: int = 123,
        dtype: torch.dtype = torch.uint16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a test cube and corresponding wavelengths.

        Parameters
        ----------
        batch_size : int
            Batch dimension size
        height : int
            Height dimension size
        width : int
            Width dimension size
        num_channels : int
            Number of spectral channels
        mode : str
            Generation mode:
            - "wavelength_dependent": Each channel has values equal to its wavelength
            - "random": Random values from normal distribution
            - "synthetic": Random with spectral trend (for anomaly detection)
        wavelength_range : tuple[float, float]
            Min and max wavelengths in nanometers
        seed : int
            Random seed for reproducibility

        Returns
        -------
        cube : torch.Tensor
            Test cube tensor [B, H, W, C]
        wavelengths : torch.Tensor
            Wavelength tensor [B, C] in nanometers with dtype int32
        """
        # Create wavelengths from min to max in equal steps as torch tensor with int32 dtype
        wavelengths_1d = torch.linspace(
            wavelength_range[0], wavelength_range[1], num_channels, dtype=torch.float32
        ).int()  # Convert to int32

        # Expand to 2D [B, C] - same wavelengths for all batch samples
        wavelengths = wavelengths_1d.unsqueeze(0).expand(batch_size, -1).clone()

        if mode == "wavelength_dependent":
            # Each channel has values equal to its wavelength
            cube = (
                wavelengths_1d.float()
                .view(1, 1, 1, num_channels)
                .expand(batch_size, height, width, num_channels)
                .clone()
            )
        elif mode == "random":
            # Random values from normal distribution
            rng = torch.Generator().manual_seed(seed)
            cube = torch.randn(
                (batch_size, height, width, num_channels), generator=rng, dtype=torch.float32
            )
        elif mode == "synthetic":
            # Random with spectral trend (for anomaly detection)
            rng = np.random.default_rng(seed)
            base = rng.normal(0, 1, size=(batch_size, height, width, num_channels)).astype(
                np.float32
            )
            # Add mild trend across channels
            trend = np.linspace(-0.2, 0.2, num_channels, dtype=np.float32)
            base += trend.reshape(1, 1, 1, num_channels)
            cube = torch.from_numpy(base)
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'wavelength_dependent', 'random', or 'synthetic'"
            )

        # Convert to specified dtype
        cube = cube.to(dtype)
        wavelengths = wavelengths.to(torch.int32).reshape(batch_size, -1)
        return cube, wavelengths

    return _create


# Rest of the file remains unchanged (SyntheticAnomalyDataModule, etc.)
class _SyntheticDictDataset(Dataset):
    """Dataset that returns batch dicts with cube, mask, and wavelengths.

    Compatible with LentilsAnomalyDataNode which requires wavelengths as an input port.
    """

    def __init__(
        self, cubes: torch.Tensor, masks: torch.Tensor | None, wavelengths: np.ndarray
    ) -> None:
        self._cubes = cubes
        self._masks = masks
        self._wavelengths = torch.from_numpy(wavelengths).int()

    def __len__(self) -> int:
        return self._cubes.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {
            "cube": self._cubes[idx],
            "wavelengths": self._wavelengths,  # Same wavelengths for all samples in batch
        }
        if self._masks is not None:
            sample["mask"] = self._masks[idx]
        return sample


class SyntheticAnomalyDataModule(CuvisDataModule):
    """Lightweight datamodule that generates deterministic synthetic anomaly data.

    This datamodule reuses the create_test_cube logic to ensure consistency
    across tests and properly includes wavelengths for compatibility with
    LentilsAnomalyDataNode.
    """

    def __init__(
        self,
        *,
        batch_size: int = 4,
        num_samples: int = 24,
        height: int = 8,
        width: int = 8,
        channels: int = 20,
        seed: int = 0,
        include_labels: bool = True,
        mode: Literal["wavelength_dependent", "random", "synthetic"] = "random",
        wavelength_range: tuple[float, float] = (430.0, 910.0),
        dtype: torch.dtype = torch.uint16,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size

        # Generate cubes using create_test_cube logic (wavelengths as int32 to match LentilsAnomalyDataNode)
        wavelengths = np.linspace(
            wavelength_range[0],
            wavelength_range[1],
            channels,
            dtype=np.int32,
        )

        if mode == "wavelength_dependent":
            wavelengths_tensor = torch.from_numpy(wavelengths).float()
            cubes = (
                wavelengths_tensor.view(1, 1, 1, channels)
                .expand(num_samples, height, width, channels)
                .clone()
            )
        elif mode == "random":
            generator = torch.Generator().manual_seed(seed)
            cubes = torch.randn(
                (num_samples, height, width, channels),
                generator=generator,
                dtype=torch.float32,
            )
        elif mode == "synthetic":
            rng = np.random.default_rng(seed)
            base = rng.normal(0, 1, size=(num_samples, height, width, channels)).astype(np.float32)
            trend = np.linspace(-0.2, 0.2, channels, dtype=np.float32)
            base += trend.reshape(1, 1, 1, channels)
            cubes = torch.from_numpy(base)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Convert to specified dtype (defaults to uint16 to match real sensor data)
        cubes = cubes.to(dtype)

        # Generate masks if needed
        masks = (
            torch.randint(
                0,
                2,
                (num_samples, height, width),
                generator=torch.Generator().manual_seed(seed),
                dtype=torch.int32,
            )
            if include_labels
            else None
        )

        self._train_dataset = _SyntheticDictDataset(cubes, masks, wavelengths)

        val_count = max(1, num_samples // 4)
        self._val_dataset = _SyntheticDictDataset(
            cubes[:val_count],
            masks[:val_count] if masks is not None else None,
            wavelengths,
        )

    def prepare_data(self) -> None:  # pragma: no cover - no external downloads
        pass

    def setup(self, stage=None) -> None:  # pragma: no cover - nothing to stage
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_dataset, batch_size=self.batch_size, shuffle=False)


@pytest.fixture
def synthetic_anomaly_datamodule():
    """Factory fixture for creating synthetic anomaly datamodules.

    Generates deterministic synthetic data with wavelengths properly included
    for compatibility with LentilsAnomalyDataNode.

    Returns:
        Callable that creates SyntheticAnomalyDataModule instances

    Examples:
        >>> datamodule = synthetic_anomaly_datamodule(
        ...     batch_size=4,
        ...     num_samples=24,
        ...     channels=20,
        ...     include_labels=True
        ... )
    """

    def _create(**kwargs) -> SyntheticAnomalyDataModule:
        return SyntheticAnomalyDataModule(**kwargs)

    return _create


@pytest.fixture
def create_batch_with_wavelengths():
    """Helper fixture for creating batches with properly formatted wavelengths.

    Wavelengths must be 2D [Batch, Channels] for LentilsAnomalyDataNode.
    When using DataLoader, the loader automatically stacks 1D wavelengths from
    individual samples into 2D batches. But for manual batch creation in tests, we need to create 2D wavelengths explicitly.

    Returns:
        Callable that adds wavelengths to a batch dict

    Examples:
        >>> batch = {"cube": torch.randn(2, 10, 10, 100)}
        >>> batch = create_batch_with_wavelengths(batch, num_channels=100)
        >>> # batch now has wavelengths with shape [2, 100]
    """

    def _add_wavelengths(
        batch: dict[str, torch.Tensor],
        num_channels: int | None = None,
        wavelength_range: tuple[float, float] = (430.0, 910.0),
    ) -> dict[str, torch.Tensor]:
        """Add properly formatted wavelengths to a batch dict.

        Parameters
        ----------
        batch : dict
            Batch dictionary containing at least 'cube' tensor with shape [B, H, W, C]
        num_channels : int, optional
            Number of channels. If None, inferred from cube shape
        wavelength_range : tuple[float, float]
            Min and max wavelengths in nanometers

        Returns
        -------
        dict
            Updated batch dict with 'wavelengths' key containing [B, C] tensor
        """
        if "cube" not in batch:
            raise ValueError("Batch must contain 'cube' key")

        cube = batch["cube"]
        batch_size = cube.shape[0]
        channels = num_channels if num_channels is not None else cube.shape[-1]

        # Create 1D wavelengths
        wavelengths_1d = torch.linspace(
            wavelength_range[0],
            wavelength_range[1],
            channels,
            dtype=torch.float32,
        ).int()

        # Expand to 2D [B, C] - same wavelengths for all batch samples
        wavelengths_2d = wavelengths_1d.unsqueeze(0).expand(batch_size, -1)

        batch["wavelengths"] = wavelengths_2d
        return batch

    return _add_wavelengths


@pytest.fixture
def training_config_factory():
    """Factory for creating TrainingConfig with test-friendly defaults.

    Provides convenient creation of TrainingConfig objects optimized for
    fast CPU-based unit tests.

    Returns:
        Callable that creates TrainingConfig instances

    Examples:
        >>> config = training_config_factory(max_epochs=2, lr=1e-2)
    """

    def _create(max_epochs: int = 2, lr: float = 1e-2) -> TrainingConfig:
        """Create a TrainingConfig with CPU defaults for fast unit tests."""
        trainer = TrainerConfig(
            max_epochs=max_epochs,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_checkpointing=False,
            log_every_n_steps=1,
        )
        optimizer = OptimizerConfig(
            name="adam",
            lr=lr,
            weight_decay=0.0,
            betas=None,
        )
        return TrainingConfig(
            seed=123,
            trainer=trainer,
            optimizer=optimizer,
        )

    return _create
