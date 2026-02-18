"""Unit tests for BandpassByWavelength node.

This test suite verifies that the BandpassByWavelength node correctly:
- Selects channels based on wavelength ranges
- Matches results with classic numpy array slicing
- Handles both min-only and min-max bounds
- Preserves correct shapes and wavelength lists
- Works with wavelengths from input port only
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.preprocessors import BandpassByWavelength


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
def test_bandpass_with_min_and_max(create_test_cube) -> None:
    """Test bandpass with both min and max wavelength bounds."""
    cube, wavelengths = create_test_cube(
        batch_size=2, height=10, width=10, num_channels=100, dtype=torch.float32
    )
    min_nm = 500.0
    max_nm = 700.0

    # Create node
    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
    )

    # Apply bandpass with wavelengths from input port
    # wavelengths is now a 2D tensor [B, C], extract first batch for testing
    wavelengths_tensor = wavelengths[0].float()
    filtered_cube = bandpass.forward(data=cube, wavelengths=wavelengths_tensor)["filtered"]

    # Extract wavelengths as numpy for comparison
    wavelengths_np = wavelengths[0].cpu().numpy()

    # Verify shape
    expected_channels = np.sum((wavelengths_np >= min_nm) & (wavelengths_np <= max_nm))
    assert filtered_cube.shape == (2, 10, 10, expected_channels), (
        f"Expected shape (2, 10, 10, {expected_channels}), got {filtered_cube.shape}"
    )

    # Verify with classic method
    cube_np = cube.numpy()
    classic_cube, classic_wavelengths = _classic_bandpass_numpy(
        cube_np, wavelengths_np, min_nm, max_nm
    )

    # Check shapes match
    assert filtered_cube.shape == classic_cube.shape, (
        f"Shape mismatch: {filtered_cube.shape} vs {classic_cube.shape}"
    )

    # Check wavelengths match
    filtered_wavelengths = wavelengths_np[(wavelengths_np >= min_nm) & (wavelengths_np <= max_nm)]
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
def test_bandpass_with_min_only(create_test_cube) -> None:
    """Test bandpass with only min wavelength bound (max=None)."""
    cube, wavelengths = create_test_cube(
        batch_size=1, height=5, width=5, num_channels=50, dtype=torch.float32
    )
    min_nm = 600.0
    max_nm = None

    # Create node
    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
    )

    # Apply bandpass with wavelengths from input port
    wavelengths_tensor = wavelengths[0].float()

    # Extract wavelengths as numpy for comparison
    wavelengths_np = wavelengths[0].cpu().numpy()

    filtered_cube = bandpass.forward(data=cube, wavelengths=wavelengths_tensor)["filtered"]

    # Verify shape
    expected_channels = np.sum(wavelengths_np >= min_nm)
    assert filtered_cube.shape == (1, 5, 5, expected_channels), (
        f"Expected shape (1, 5, 5, {expected_channels}), got {filtered_cube.shape}"
    )

    # Verify with classic method
    cube_np = cube.numpy()
    classic_cube, classic_wavelengths = _classic_bandpass_numpy(
        cube_np, wavelengths_np, min_nm, max_nm
    )

    # Check shapes match
    assert filtered_cube.shape == classic_cube.shape, (
        f"Shape mismatch: {filtered_cube.shape} vs {classic_cube.shape}"
    )

    # Check wavelengths match
    filtered_wavelengths = wavelengths_np[wavelengths_np >= min_nm]
    np.testing.assert_array_equal(filtered_wavelengths, classic_wavelengths)

    # Check cube values match
    np.testing.assert_allclose(
        filtered_cube.numpy(),
        classic_cube,
        rtol=1e-5,
        atol=1e-6,
    )


@torch.no_grad()
def test_bandpass_wavelength_list_matches(create_test_cube) -> None:
    """Test that wavelengths in bandpass match expected list."""
    cube, wavelengths = create_test_cube(num_channels=100, dtype=torch.float32)
    min_nm = 450.0
    max_nm = 550.0

    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
    )

    wavelengths_tensor = wavelengths[0].float()
    wavelengths_np = wavelengths[0].cpu().numpy()

    filtered_cube = bandpass.forward(data=cube, wavelengths=wavelengths_tensor)["filtered"]

    # Compute expected wavelengths
    expected_mask = (wavelengths_np >= min_nm) & (wavelengths_np <= max_nm)
    expected_wavelengths = wavelengths_np[expected_mask]

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
def test_bandpass_same_result_one_or_two_bounds(create_test_cube) -> None:
    """Test that using one bound or two bounds gives same result when max equals dataset max."""
    cube, wavelengths = create_test_cube(num_channels=100, dtype=torch.float32)
    wavelengths_np = wavelengths[0].cpu().numpy()
    min_nm = 500.0
    max_nm = float(wavelengths_np.max())  # Set max to dataset maximum

    # Test with two bounds
    bandpass_two = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
    )

    # Test with one bound (max=None)
    bandpass_one = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=None,
    )

    wavelengths_tensor = wavelengths[0].float()
    filtered_two = bandpass_two.forward(data=cube, wavelengths=wavelengths_tensor)["filtered"]
    filtered_one = bandpass_one.forward(data=cube, wavelengths=wavelengths_tensor)["filtered"]

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
    expected_wavelengths = wavelengths_np[wavelengths_np >= min_nm]
    # Verify using cube values
    values_one = filtered_one[0, 0, 0, :].numpy()
    values_two = filtered_two[0, 0, 0, :].numpy()
    np.testing.assert_allclose(values_one, values_two, rtol=1e-5)
    np.testing.assert_allclose(values_one, expected_wavelengths, rtol=1e-5)


@torch.no_grad()
def test_bandpass_3d_input(create_test_cube) -> None:
    """Test that callers can use HWC by adding an explicit batch dimension."""
    cube, wavelengths = create_test_cube(
        batch_size=1, height=5, width=5, num_channels=50, dtype=torch.float32
    )
    min_nm = 500.0
    max_nm = 600.0

    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
    )

    # Remove batch dimension for HWC representation, then add it back as 1HWC
    cube_3d = cube[0]  # [H, W, C]
    wavelengths_tensor = wavelengths[0].float()
    wavelengths_np = wavelengths[0].cpu().numpy()

    filtered_cube = bandpass.forward(data=cube_3d.unsqueeze(0), wavelengths=wavelengths_tensor)[
        "filtered"
    ]

    # Should have batch dimension added back
    assert filtered_cube.dim() == 4
    assert filtered_cube.shape[0] == 1  # Batch dimension added

    # Compare with classic method using full BHWC cube
    cube_np = cube.numpy()
    classic_cube, _ = _classic_bandpass_numpy(cube_np, wavelengths_np, min_nm, max_nm)
    np.testing.assert_allclose(filtered_cube.numpy(), classic_cube, rtol=1e-5)


@torch.no_grad()
def test_bandpass_empty_range_error(create_test_cube) -> None:
    """Test that empty wavelength range raises error."""
    cube, wavelengths = create_test_cube(dtype=torch.float32)
    min_nm = 2000.0  # Way beyond dataset range
    max_nm = 3000.0

    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
    )

    wavelengths_tensor = wavelengths[0].float()
    with pytest.raises(ValueError, match="No channels selected"):
        bandpass.forward(data=cube, wavelengths=wavelengths_tensor)


@torch.no_grad()
def test_bandpass_serialization(create_test_cube, tmp_path) -> None:
    """Test serialization and loading of bandpass node."""
    cube, wavelengths = create_test_cube(dtype=torch.float32)
    min_nm = 550.0
    max_nm = 750.0

    # Create original node
    original = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
    )

    # Serialize
    state = original.state_dict()

    # Create new node and load
    loaded = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
    )

    loaded.load_state_dict(state)

    # Verify loaded parameters
    assert loaded.min_wavelength_nm == min_nm
    assert loaded.max_wavelength_nm == max_nm

    # Test that loaded node produces same results
    wavelengths_tensor = wavelengths[0].float()
    original_output = original.forward(data=cube, wavelengths=wavelengths_tensor)["filtered"]
    loaded_output = loaded.forward(data=cube, wavelengths=wavelengths_tensor)["filtered"]

    np.testing.assert_allclose(
        original_output.numpy(),
        loaded_output.numpy(),
        rtol=1e-5,
        err_msg="Loaded node should produce same output as original",
    )


@torch.no_grad()
def test_bandpass_spectrum_matches_classic(create_test_cube) -> None:
    """Test that average spectrum matches classic numpy implementation."""
    cube, wavelengths = create_test_cube(
        batch_size=1, height=20, width=20, num_channels=100, dtype=torch.float32
    )
    min_nm = 500.0
    max_nm = 700.0

    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
    )

    wavelengths_tensor = wavelengths[0].float()
    wavelengths_np = wavelengths[0].cpu().numpy()

    filtered_cube = bandpass.forward(data=cube, wavelengths=wavelengths_tensor)["filtered"]

    # Compute average spectrum (over spatial dimensions)
    avg_spectrum = filtered_cube[0].mean(dim=(0, 1)).numpy()

    # Compare with classic method
    cube_np = cube.numpy()
    classic_cube, classic_wavelengths = _classic_bandpass_numpy(
        cube_np, wavelengths_np, min_nm, max_nm
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
    expected_wavelengths = wavelengths_np[(wavelengths_np >= min_nm) & (wavelengths_np <= max_nm)]
    np.testing.assert_array_equal(expected_wavelengths, classic_wavelengths)
    assert len(avg_spectrum) == len(expected_wavelengths)


@torch.no_grad()
def test_bandpass_unsorted_wavelengths(create_test_cube) -> None:
    """Bandpass handles wavelengths that are not in ascending order."""
    cube, wavelengths = create_test_cube(
        batch_size=1, height=5, width=5, num_channels=20, dtype=torch.float32
    )

    # Shuffle wavelength order
    wavelengths_np = wavelengths[0].cpu().numpy()
    perm = np.random.default_rng(42).permutation(len(wavelengths_np))
    shuffled_wavelengths = torch.tensor(wavelengths_np[perm], dtype=torch.float32)
    shuffled_cube = cube[..., perm]

    min_nm = float(wavelengths_np.min()) + 50
    max_nm = float(wavelengths_np.max()) - 50

    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
    )

    filtered = bandpass.forward(data=shuffled_cube, wavelengths=shuffled_wavelengths)["filtered"]

    # Compute expected count from the unsorted wavelengths
    mask = (shuffled_wavelengths.numpy() >= min_nm) & (shuffled_wavelengths.numpy() <= max_nm)
    expected_channels = int(mask.sum())

    assert filtered.shape[-1] == expected_channels
    assert torch.isfinite(filtered).all()


@torch.no_grad()
def test_bandpass_single_band_selection(create_test_cube) -> None:
    """Bandpass correctly selects exactly one band when range is very narrow."""
    cube, wavelengths = create_test_cube(
        batch_size=2, height=4, width=4, num_channels=50, dtype=torch.float32
    )

    wavelengths_np = wavelengths[0].cpu().numpy()
    # Pick a narrow range around an actual wavelength (not median, which may fall between values)
    mid_idx = len(wavelengths_np) // 2
    target_wl = float(wavelengths_np[mid_idx])
    step = float(np.diff(wavelengths_np).min()) if len(wavelengths_np) > 1 else 1.0
    min_nm = target_wl - step * 0.4
    max_nm = target_wl + step * 0.4

    bandpass = BandpassByWavelength(
        min_wavelength_nm=min_nm,
        max_wavelength_nm=max_nm,
    )

    wavelengths_tensor = wavelengths[0].float()
    filtered = bandpass.forward(data=cube, wavelengths=wavelengths_tensor)["filtered"]

    assert filtered.shape[-1] == 1
    assert filtered.shape[:3] == cube.shape[:3]
