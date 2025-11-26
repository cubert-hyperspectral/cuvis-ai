"""Unit tests for BandpassByWavelength node.

This test suite verifies that the BandpassByWavelength node correctly:
- Selects channels based on wavelength ranges
- Matches results with classic numpy array slicing
- Handles both min-only and min-max bounds
- Preserves correct shapes and wavelength lists
- Works with wavelengths from input port, constructor, or caching
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from cuvis_ai.node.preprocessors import BandpassByWavelength


def _create_test_cube(
    batch_size: int = 2,
    height: int = 10,
    width: int = 10,
    num_channels: int = 100,
) -> tuple[Tensor, np.ndarray]:
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

    Returns
    -------
    cube : Tensor
        Test cube tensor [B, H, W, C]
    wavelengths : np.ndarray
        Wavelength array [C] in nanometers
    """
    # Create wavelengths from 400nm to 1000nm in equal steps
    wavelengths = np.linspace(400.0, 1000.0, num_channels, dtype=np.float32)

    # Create cube with channel-dependent values for easy verification
    # Each channel has a unique value based on its wavelength
    # Use broadcasting: create [1, 1, 1, C] and expand to [B, H, W, C]
    wavelengths_tensor = torch.from_numpy(wavelengths).float()
    cube = (
        wavelengths_tensor.view(1, 1, 1, num_channels)
        .expand(batch_size, height, width, num_channels)
        .clone()
    )

    return cube, wavelengths


def _classic_bandpass_numpy(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    min_wavelength_nm: float,
    max_wavelength_nm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Classic numpy-based bandpass implementation for verification.

    Parameters
    ----------
    cube : np.ndarray
        Input cube [B, H, W, C]
    wavelengths : np.ndarray
        Wavelength array [C]
    min_wavelength_nm : float
        Minimum wavelength (inclusive)
    max_wavelength_nm : float | None
        Maximum wavelength (inclusive), or None for no upper bound

    Returns
    -------
    filtered_cube : np.ndarray
        Bandpassed cube
    filtered_wavelengths : np.ndarray
        Wavelengths included in bandpass
    """
    if max_wavelength_nm is None:
        mask = wavelengths >= min_wavelength_nm
    else:
        mask = (wavelengths >= min_wavelength_nm) & (wavelengths <= max_wavelength_nm)

    filtered_cube = cube[..., mask]
    filtered_wavelengths = wavelengths[mask]

    return filtered_cube, filtered_wavelengths


@torch.no_grad()
def test_bandpass_with_min_and_max() -> None:
    """Test bandpass with both min and max wavelength bounds."""
    cube, wavelengths = _create_test_cube(batch_size=2, height=10, width=10, num_channels=100)
    min_nm = 500.0
    max_nm = 700.0

    # Create node with cached wavelengths
    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
        wavelengths=wavelengths,
    )

    # Apply bandpass
    filtered_cube = bandpass.forward(data=cube)["filtered"]

    # Verify shape
    expected_channels = np.sum((wavelengths >= min_nm) & (wavelengths <= max_nm))
    assert filtered_cube.shape == (2, 10, 10, expected_channels), (
        f"Expected shape (2, 10, 10, {expected_channels}), got {filtered_cube.shape}"
    )

    # Verify with classic method
    cube_np = cube.numpy()
    classic_cube, classic_wavelengths = _classic_bandpass_numpy(
        cube_np, wavelengths, min_nm, max_nm
    )

    # Check shapes match
    assert filtered_cube.shape == classic_cube.shape, (
        f"Shape mismatch: {filtered_cube.shape} vs {classic_cube.shape}"
    )

    # Check wavelengths match
    filtered_wavelengths = wavelengths[(wavelengths >= min_nm) & (wavelengths <= max_nm)]
    np.testing.assert_array_equal(filtered_wavelengths, classic_wavelengths)

    # Check cube values match (within tolerance)
    np.testing.assert_allclose(
        filtered_cube.numpy(),
        classic_cube,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Filtered cube values don't match classic implementation",
    )


@torch.no_grad()
def test_bandpass_with_min_only() -> None:
    """Test bandpass with only min wavelength bound (max=None)."""
    cube, wavelengths = _create_test_cube(batch_size=1, height=5, width=5, num_channels=50)
    min_nm = 600.0
    max_nm = None

    # Create node
    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
    )

    # Apply bandpass with wavelengths from input port
    wavelengths_tensor = torch.from_numpy(wavelengths).float()
    filtered_cube = bandpass.forward(data=cube, wavelengths=wavelengths_tensor)["filtered"]

    # Verify shape
    expected_channels = np.sum(wavelengths >= min_nm)
    assert filtered_cube.shape == (1, 5, 5, expected_channels), (
        f"Expected shape (1, 5, 5, {expected_channels}), got {filtered_cube.shape}"
    )

    # Verify with classic method
    cube_np = cube.numpy()
    classic_cube, classic_wavelengths = _classic_bandpass_numpy(
        cube_np, wavelengths, min_nm, max_nm
    )

    # Check shapes match
    assert filtered_cube.shape == classic_cube.shape, (
        f"Shape mismatch: {filtered_cube.shape} vs {classic_cube.shape}"
    )

    # Check wavelengths match
    filtered_wavelengths = wavelengths[wavelengths >= min_nm]
    np.testing.assert_array_equal(filtered_wavelengths, classic_wavelengths)

    # Check cube values match
    np.testing.assert_allclose(
        filtered_cube.numpy(),
        classic_cube,
        rtol=1e-5,
        atol=1e-6,
    )


@torch.no_grad()
def test_bandpass_wavelength_list_matches() -> None:
    """Test that wavelengths in bandpass match expected list."""
    cube, wavelengths = _create_test_cube(num_channels=100)
    min_nm = 450.0
    max_nm = 550.0

    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
        wavelengths=wavelengths,
    )

    filtered_cube = bandpass.forward(data=cube)["filtered"]

    # Compute expected wavelengths
    expected_mask = (wavelengths >= min_nm) & (wavelengths <= max_nm)
    expected_wavelengths = wavelengths[expected_mask]

    # Verify number of channels matches number of wavelengths
    assert filtered_cube.shape[-1] == len(expected_wavelengths), (
        f"Number of channels ({filtered_cube.shape[-1]}) doesn't match "
        f"number of wavelengths ({len(expected_wavelengths)})"
    )

    # Verify wavelengths are correct (using cube values as proxy)
    # Since we created cube with channel values = wavelengths
    filtered_values = filtered_cube[0, 0, 0, :].numpy()
    np.testing.assert_allclose(
        filtered_values,
        expected_wavelengths,
        rtol=1e-5,
        err_msg="Wavelengths in bandpass don't match expected values",
    )


@torch.no_grad()
def test_bandpass_same_result_one_or_two_bounds() -> None:
    """Test that using one bound or two bounds gives same result when max equals dataset max."""
    cube, wavelengths = _create_test_cube(num_channels=100)
    min_nm = 500.0
    max_nm = float(wavelengths.max())  # Set max to dataset maximum

    # Test with two bounds
    bandpass_two = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
        wavelengths=wavelengths,
    )

    # Test with one bound (max=None)
    bandpass_one = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=None,
        wavelengths=wavelengths,
    )

    filtered_two = bandpass_two.forward(data=cube)["filtered"]
    filtered_one = bandpass_one.forward(data=cube)["filtered"]

    # Should have same shape
    assert filtered_one.shape == filtered_two.shape, (
        f"Shapes should match: {filtered_one.shape} vs {filtered_two.shape}"
    )

    # Should have same values
    np.testing.assert_allclose(
        filtered_one.numpy(),
        filtered_two.numpy(),
        rtol=1e-5,
        atol=1e-6,
        err_msg="Results should be identical when max equals dataset max",
    )

    # Wavelengths should match
    expected_wavelengths = wavelengths[wavelengths >= min_nm]
    # Verify using cube values
    values_one = filtered_one[0, 0, 0, :].numpy()
    values_two = filtered_two[0, 0, 0, :].numpy()
    np.testing.assert_allclose(values_one, values_two, rtol=1e-5)
    np.testing.assert_allclose(values_one, expected_wavelengths, rtol=1e-5)


@torch.no_grad()
def test_bandpass_wavelengths_caching() -> None:
    """Test that wavelengths are cached after first forward pass."""
    cube, wavelengths = _create_test_cube()
    min_nm = 550.0
    max_nm = 650.0

    # Create node without cached wavelengths
    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
    )

    # First call with wavelengths from input port
    wavelengths_tensor = torch.from_numpy(wavelengths).float()
    filtered_cube = bandpass.forward(data=cube, wavelengths=wavelengths_tensor)["filtered"]

    # Verify shape
    expected_channels = np.sum((wavelengths >= min_nm) & (wavelengths <= max_nm))
    assert filtered_cube.shape[-1] == expected_channels
    assert bandpass._cached_wavelengths is not None
    np.testing.assert_allclose(bandpass._cached_wavelengths, wavelengths, rtol=1e-6)

    # Subsequent call without wavelengths should use cached wavelengths
    cached_cube = bandpass.forward(data=cube)["filtered"]
    np.testing.assert_allclose(cached_cube.numpy(), filtered_cube.numpy(), rtol=1e-5)


@torch.no_grad()
def test_bandpass_wavelengths_via_kwargs() -> None:
    """Test that wavelengths provided via kwargs are used (backward compatibility)."""
    cube, wavelengths = _create_test_cube(batch_size=1, height=6, width=7, num_channels=32)
    min_nm = 450.0
    max_nm = 650.0

    # Create node without cached wavelengths
    bandpass = BandpassByWavelength(min_wavelength_nm=min_nm, max_wavelength_nm=max_nm)

    # Call forward providing wavelengths via kwargs (backward compatibility)
    filtered_cube = bandpass.forward(data=cube, wavelengths=wavelengths)["filtered"]

    expected_mask = (wavelengths >= min_nm) & (wavelengths <= max_nm)
    assert filtered_cube.shape[-1] == int(expected_mask.sum())

    # Cache should now be populated; call again without kwargs
    filtered_again = bandpass.forward(data=cube)["filtered"]
    np.testing.assert_allclose(filtered_again.numpy(), filtered_cube.numpy(), rtol=1e-5)


@torch.no_grad()
def test_bandpass_3d_input() -> None:
    """Test that callers can use HWC by adding an explicit batch dimension."""
    cube, wavelengths = _create_test_cube(batch_size=1, height=5, width=5, num_channels=50)
    min_nm = 500.0
    max_nm = 600.0

    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
        wavelengths=wavelengths,
    )

    # Remove batch dimension for HWC representation, then add it back as 1HWC
    cube_3d = cube[0]  # [H, W, C]
    filtered_cube = bandpass.forward(data=cube_3d.unsqueeze(0))["filtered"]

    # Should have batch dimension added back
    assert filtered_cube.dim() == 4
    assert filtered_cube.shape[0] == 1  # Batch dimension added

    # Compare with classic method using full BHWC cube
    cube_np = cube.numpy()
    classic_cube, _ = _classic_bandpass_numpy(cube_np, wavelengths, min_nm, max_nm)
    np.testing.assert_allclose(filtered_cube.numpy(), classic_cube, rtol=1e-5)


@torch.no_grad()
def test_bandpass_empty_range_error() -> None:
    """Test that empty wavelength range raises error."""
    cube, wavelengths = _create_test_cube()
    min_nm = 2000.0  # Way beyond dataset range
    max_nm = 3000.0

    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
        wavelengths=wavelengths,
    )

    with pytest.raises(ValueError, match="No channels selected"):
        bandpass.forward(data=cube)


@torch.no_grad()
def test_bandpass_no_wavelengths_error() -> None:
    """Test that missing wavelengths raises error."""
    cube, _ = _create_test_cube()
    min_nm = 500.0
    max_nm = 600.0

    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
    )

    with pytest.raises(RuntimeError, match="requires 'wavelengths'"):
        bandpass.forward(data=cube)  # No wavelengths provided


@torch.no_grad()
def test_bandpass_serialization(tmp_path) -> None:
    """Test serialization and loading of bandpass node."""
    _, wavelengths = _create_test_cube()
    min_nm = 550.0
    max_nm = 750.0

    # Create original node
    original = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
        wavelengths=wavelengths,
    )

    # Serialize
    serial_dir = str(tmp_path / "test_serial")
    serialized = original.serialize(serial_dir=serial_dir)

    # Create new node and load
    loaded = BandpassByWavelength(
        min_wavelength_nm=0.0,  # Will be overwritten
        max_wavelength_nm=None,
    )

    # Load expects params dict with "params" key
    params_dict = {"params": serialized} if "params" not in serialized else serialized
    loaded.load(params_dict, serial_dir=serial_dir)

    # Verify loaded parameters
    assert loaded.min_wavelength_nm == min_nm
    assert loaded.max_wavelength_nm == max_nm
    assert loaded._cached_wavelengths is not None
    np.testing.assert_allclose(loaded._cached_wavelengths, wavelengths, rtol=1e-6)

    # Test that loaded node produces same results
    cube, _ = _create_test_cube()
    original_output = original.forward(data=cube)["filtered"]
    loaded_output = loaded.forward(data=cube)["filtered"]

    np.testing.assert_allclose(
        original_output.numpy(),
        loaded_output.numpy(),
        rtol=1e-5,
        err_msg="Loaded node should produce same output as original",
    )


@torch.no_grad()
def test_bandpass_spectrum_matches_classic() -> None:
    """Test that average spectrum matches classic numpy implementation."""
    cube, wavelengths = _create_test_cube(batch_size=1, height=20, width=20, num_channels=100)
    min_nm = 500.0
    max_nm = 700.0

    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
        wavelengths=wavelengths,
    )

    filtered_cube = bandpass.forward(data=cube)["filtered"]

    # Compute average spectrum (over spatial dimensions)
    avg_spectrum = filtered_cube[0].mean(dim=(0, 1)).numpy()

    # Compare with classic method
    cube_np = cube.numpy()
    classic_cube, classic_wavelengths = _classic_bandpass_numpy(
        cube_np, wavelengths, min_nm, max_nm
    )
    classic_avg_spectrum = classic_cube[0].mean(axis=(0, 1))

    # Should match
    np.testing.assert_allclose(
        avg_spectrum,
        classic_avg_spectrum,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Average spectrum should match classic implementation",
    )

    # Verify wavelengths match
    expected_wavelengths = wavelengths[(wavelengths >= min_nm) & (wavelengths <= max_nm)]
    np.testing.assert_array_equal(expected_wavelengths, classic_wavelengths)
    assert len(avg_spectrum) == len(expected_wavelengths)
