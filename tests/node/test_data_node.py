"""Unit tests for LentilsAnomalyDataNode."""

import numpy as np
import pytest
import torch

from cuvis_ai.node.data import LentilsAnomalyDataNode

pytestmark = pytest.mark.unit


class TestLentilsAnomalyDataNode:
    """Tests for LentilsAnomalyDataNode input/output transformations."""

    def test_dtype_conversion_uint16_to_float32(self):
        """Cube should be converted from uint16 to float32."""
        node = LentilsAnomalyDataNode(normal_class_ids=[0])
        cube = torch.randint(0, 65535, (1, 4, 4, 5), dtype=torch.uint16)
        wavelengths = torch.arange(5, dtype=torch.int32).unsqueeze(0)
        result = node.forward(cube=cube, wavelengths=wavelengths)
        assert result["cube"].dtype == torch.float32

    def test_cube_values_preserved(self):
        """Float32 values should equal the original uint16 values."""
        node = LentilsAnomalyDataNode(normal_class_ids=[0])
        cube = torch.tensor([[[[100, 200, 300]]]], dtype=torch.uint16)
        result = node.forward(cube=cube)
        expected = torch.tensor([[[[100.0, 200.0, 300.0]]]], dtype=torch.float32)
        assert torch.equal(result["cube"], expected)

    def test_cube_shape_preserved(self):
        """Output cube shape should match input cube shape."""
        node = LentilsAnomalyDataNode(normal_class_ids=[0])
        cube = torch.randint(0, 65535, (2, 8, 8, 10), dtype=torch.uint16)
        result = node.forward(cube=cube)
        assert result["cube"].shape == (2, 8, 8, 10)

    def test_mask_shape_3d_to_4d(self):
        """Mask should be transformed from [B,H,W] to [B,H,W,1]."""
        node = LentilsAnomalyDataNode(normal_class_ids=[0])
        cube = torch.randint(0, 65535, (2, 4, 4, 5), dtype=torch.uint16)
        mask = torch.zeros(2, 4, 4, dtype=torch.int32)
        wavelengths = torch.arange(5, dtype=torch.int32).unsqueeze(0).expand(2, -1)
        result = node.forward(cube=cube, mask=mask, wavelengths=wavelengths)
        assert result["mask"].shape == (2, 4, 4, 1)
        assert result["mask"].dtype == torch.bool

    def test_wavelength_extraction_2d_to_1d_numpy(self):
        """Wavelengths should be extracted from 2D tensor to 1D numpy array."""
        node = LentilsAnomalyDataNode(normal_class_ids=[0])
        cube = torch.randint(0, 65535, (2, 4, 4, 5), dtype=torch.uint16)
        wavelengths = torch.tensor(
            [[430, 550, 670, 790, 910], [430, 550, 670, 790, 910]], dtype=torch.int32
        )
        result = node.forward(cube=cube, wavelengths=wavelengths)
        assert isinstance(result["wavelengths"], np.ndarray)
        assert result["wavelengths"].shape == (5,)

    def test_no_mask_case(self):
        """When mask is None, result should not contain 'mask' key."""
        node = LentilsAnomalyDataNode(normal_class_ids=[0])
        cube = torch.randint(0, 65535, (1, 4, 4, 5), dtype=torch.uint16)
        wavelengths = torch.arange(5, dtype=torch.int32).unsqueeze(0)
        result = node.forward(cube=cube, wavelengths=wavelengths)
        assert "mask" not in result

    def test_no_wavelengths_case(self):
        """When wavelengths is None, result should not contain 'wavelengths' key."""
        node = LentilsAnomalyDataNode(normal_class_ids=[0])
        cube = torch.randint(0, 65535, (1, 4, 4, 5), dtype=torch.uint16)
        result = node.forward(cube=cube)
        assert "wavelengths" not in result
