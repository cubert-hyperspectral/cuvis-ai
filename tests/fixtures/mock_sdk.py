"""Mock CUVIS SDK fixtures for testing."""

from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_cuvis_sdk(create_test_cube):
    """Mock CUVIS SDK to avoid thread-safety issues in tests.

    Uses the unified create_test_cube fixture to generate realistic
    wavelength-dependent hyperspectral data for mock measurements.
    """
    # Generate realistic test cube data using wavelength_dependent mode
    cube, wavelengths = create_test_cube(
        batch_size=1,
        height=64,
        width=64,
        num_channels=61,
        mode="wavelength_dependent",
        wavelength_range=(430.0, 910.0),
        seed=42,
    )

    # Create mock measurement object
    mock_measurement = Mock()
    mock_measurement.cube = Mock()
    mock_measurement.cube.array = cube[0].numpy()  # Remove batch dimension for mock
    mock_measurement.cube.channels = 61
    mock_measurement.cube.wavelength = wavelengths
    mock_measurement.data = {"cube": True}  # Pretend cube is already loaded

    # Create mock session file
    mock_session = Mock()
    mock_session.get_measurement = Mock(return_value=mock_measurement)
    mock_session.__len__ = Mock(return_value=7)  # 7 measurements total

    # Create mock processing context
    mock_pc = Mock()
    mock_pc.apply = Mock(return_value=mock_measurement)
    mock_pc.processing_mode = Mock()

    # Mock COCO annotations
    mock_coco = Mock()
    mock_coco.category_id_to_name = {0: "background", 1: "anomaly"}
    mock_coco.image_ids = [0, 1, 2, 3, 4, 5, 6]  # Match measurement count
    mock_coco.annotations = Mock()
    mock_coco.annotations.where = Mock(return_value=[])  # No annotations for simplicity

    # Patch cuvis module imports
    with (
        patch("cuvis_ai.data.datasets.cuvis.SessionFile", return_value=mock_session),
        patch("cuvis_ai.data.datasets.cuvis.ProcessingContext", return_value=mock_pc),
        patch("cuvis_ai.data.datasets.cuvis.ProcessingMode") as mock_pm,
        patch("cuvis_ai.data.datasets.cuvis.ReferenceType") as mock_rt,
        patch("cuvis_ai.data.datasets.COCOData.from_path", return_value=mock_coco),
    ):
        # Setup ProcessingMode enum
        mock_pm.Raw = "Raw"
        mock_pm.Reflectance = "Reflectance"
        mock_rt.White = "White"
        mock_rt.Dark = "Dark"

        white_ref = Mock(name="white_reference")
        dark_ref = Mock(name="dark_reference")

        def _get_reference(idx, ref_type):
            if ref_type == mock_rt.White:
                return white_ref
            if ref_type == mock_rt.Dark:
                return dark_ref
            return None

        mock_session.get_reference = Mock(side_effect=_get_reference)

        yield {
            "session": mock_session,
            "processing_context": mock_pc,
            "measurement": mock_measurement,
            "coco": mock_coco,
        }
