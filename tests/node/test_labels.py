from __future__ import annotations

import warnings

import pytest
import torch

from cuvis_ai.node.labels import BinaryAnomalyLabelMapper


class TestBinaryAnomalyLabelMapper:
    """Test suite for BinaryAnomalyLabelMapper node."""

    def test_normal_behavior_no_anomaly_ids(self):
        """Test normal behavior when anomaly_class_ids is None."""
        mapper = BinaryAnomalyLabelMapper(normal_class_ids=[0, 2])

        # Create test data with classes 0, 1, 2, 3
        cube = torch.ones(1, 4, 4, 3)
        mask = torch.tensor([[[[0], [1], [2], [3]]]], dtype=torch.int32)

        result = mapper.forward(cube, mask)

        # Classes 0 and 2 should be normal (False), classes 1 and 3 should be anomaly (True)
        expected = torch.tensor([[[[False], [True], [False], [True]]]], dtype=torch.bool)
        assert torch.equal(result["mask"], expected)

    def test_explicit_anomaly_ids_no_gaps(self):
        """Test behavior with explicit anomaly_class_ids and no gaps."""
        mapper = BinaryAnomalyLabelMapper(normal_class_ids=[0, 1], anomaly_class_ids=[2, 3])

        cube = torch.ones(1, 4, 4, 3)
        mask = torch.tensor([[[[0], [1], [2], [3]]]], dtype=torch.int32)

        result = mapper.forward(cube, mask)

        # Classes 0,1 should be normal (False), classes 2,3 should be anomaly (True)
        expected = torch.tensor([[[[False], [False], [True], [True]]]], dtype=torch.bool)
        assert torch.equal(result["mask"], expected)

    def test_gap_handling_with_warning(self):
        """Test that gaps are handled correctly with warning."""
        # This should trigger a warning about class ID 2 being missing
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapper = BinaryAnomalyLabelMapper(normal_class_ids=[0, 1], anomaly_class_ids=[3])

            # Check that warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Gap detected" in str(w[0].message)
            assert "2" in str(w[0].message)  # Class ID 2 should be mentioned

        # Class 2 should now be included in normal_class_ids
        assert 2 in mapper.normal_class_ids

        # Test the mapping behavior
        cube = torch.ones(1, 4, 4, 3)
        mask = torch.tensor([[[[0], [1], [2], [3]]]], dtype=torch.int32)

        result = mapper.forward(cube, mask)

        # Classes 0,1,2 should be normal (False), class 3 should be anomaly (True)
        expected = torch.tensor([[[[False], [False], [False], [True]]]], dtype=torch.bool)
        assert torch.equal(result["mask"], expected)

    def test_overlap_error(self):
        """Test that overlaps between normal and anomaly class IDs raise an error."""
        with pytest.raises(ValueError, match="Overlap detected"):
            BinaryAnomalyLabelMapper(normal_class_ids=[0, 1, 2], anomaly_class_ids=[2, 3])

    def test_empty_anomaly_ids(self):
        """Test behavior with empty anomaly_class_ids."""
        mapper = BinaryAnomalyLabelMapper(normal_class_ids=[0, 1], anomaly_class_ids=[])

        cube = torch.ones(1, 4, 4, 3)
        mask = torch.tensor([[[[0], [1], [2], [3]]]], dtype=torch.int32)

        result = mapper.forward(cube, mask)

        # All classes should be normal since anomaly_class_ids is empty
        expected = torch.tensor([[[[False], [False], [False], [False]]]], dtype=torch.bool)
        assert torch.equal(result["mask"], expected)

    def test_single_class_ids(self):
        """Test with single class IDs."""
        mapper = BinaryAnomalyLabelMapper(normal_class_ids=[0], anomaly_class_ids=[1])

        cube = torch.ones(1, 2, 2, 3)
        mask = torch.tensor([[[[0], [1]]]], dtype=torch.int32)

        result = mapper.forward(cube, mask)

        expected = torch.tensor([[[[False], [True]]]], dtype=torch.bool)
        assert torch.equal(result["mask"], expected)

    def test_large_gap_range(self):
        """Test with larger gap range."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapper = BinaryAnomalyLabelMapper(normal_class_ids=[0], anomaly_class_ids=[5])

            # Should warn about classes 1, 2, 3, 4 being missing
            assert len(w) == 1
            assert "1, 2, 3, 4" in str(w[0].message)

        # All missing classes should be added to normal_class_ids
        assert set(mapper.normal_class_ids) == {0, 1, 2, 3, 4}

    def test_no_gap_no_warning(self):
        """Test that no warning is issued when there are no gaps."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapper = BinaryAnomalyLabelMapper(normal_class_ids=[0, 1], anomaly_class_ids=[2, 3])

            # No warning should be issued
            assert len(w) == 0

        assert mapper.normal_class_ids == (0, 1)
        assert mapper.anomaly_class_ids == (2, 3)
