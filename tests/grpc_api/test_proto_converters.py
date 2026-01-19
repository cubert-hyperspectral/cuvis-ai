import numpy as np
import pytest
import torch

from cuvis_ai_core.grpc import cuvis_ai_pb2, helpers


class TestProtoToNumpy:
    """Test proto → numpy conversion"""

    def test_proto_to_numpy_float32(self):
        """Test converting float32 tensor proto to numpy"""
        # Arrange
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[2, 2], dtype=cuvis_ai_pb2.D_TYPE_FLOAT32, raw_data=arr.tobytes()
        )

        # Act
        result = helpers.proto_to_numpy(tensor_proto)

        # Assert
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.float32

    def test_proto_to_numpy_int32(self):
        """Test converting int32 tensor proto to numpy"""
        arr = np.array([1, 2, 3, 4], dtype=np.int32)
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[4], dtype=cuvis_ai_pb2.D_TYPE_INT32, raw_data=arr.tobytes()
        )

        result = helpers.proto_to_numpy(tensor_proto)

        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.int32

    def test_proto_to_numpy_invalid_dtype(self):
        """Test error handling for unsupported dtype"""
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[2, 2],
            dtype=999,  # Invalid
            raw_data=b"invalid",
        )

        with pytest.raises(ValueError, match="Unsupported dtype"):
            helpers.proto_to_numpy(tensor_proto)

    def test_proto_to_numpy_empty_tensor(self):
        """Test handling empty tensors"""
        arr = np.array([], dtype=np.float32)
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[0], dtype=cuvis_ai_pb2.D_TYPE_FLOAT32, raw_data=arr.tobytes()
        )

        result = helpers.proto_to_numpy(tensor_proto)

        assert result.shape == (0,)
        assert result.dtype == np.float32

    def test_proto_to_numpy_writable_by_default(self):
        """Test that proto_to_numpy returns writable arrays by default"""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[2, 2], dtype=cuvis_ai_pb2.D_TYPE_FLOAT32, raw_data=arr.tobytes()
        )

        result = helpers.proto_to_numpy(tensor_proto)

        # Should be writable
        assert result.flags.writeable
        # Verify we can modify it
        result[0, 0] = 999.0
        assert result[0, 0] == 999.0

    def test_proto_to_numpy_copy_true(self):
        """Test proto_to_numpy with copy=True returns writable array"""
        arr = np.array([1, 2, 3, 4], dtype=np.int32)
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[4], dtype=cuvis_ai_pb2.D_TYPE_INT32, raw_data=arr.tobytes()
        )

        result = helpers.proto_to_numpy(tensor_proto, copy=True)

        # Should be writable
        assert result.flags.writeable
        result[0] = 999
        assert result[0] == 999

    def test_proto_to_numpy_copy_false(self):
        """Test proto_to_numpy with copy=False returns read-only view"""
        arr = np.array([1, 2, 3, 4], dtype=np.int32)
        tensor_proto = cuvis_ai_pb2.Tensor(
            shape=[4], dtype=cuvis_ai_pb2.D_TYPE_INT32, raw_data=arr.tobytes()
        )

        result = helpers.proto_to_numpy(tensor_proto, copy=False)

        # Should be read-only
        assert not result.flags.writeable
        # Verify modification raises error
        with pytest.raises(ValueError, match="assignment destination is read-only"):
            result[0] = 999


class TestNumpyToProto:
    """Test numpy → proto conversion"""

    def test_numpy_to_proto_float32(self):
        """Test converting numpy array to proto"""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        tensor_proto = helpers.numpy_to_proto(arr)

        assert list(tensor_proto.shape) == [2, 2]
        assert tensor_proto.dtype == cuvis_ai_pb2.D_TYPE_FLOAT32
        assert tensor_proto.raw_data == arr.tobytes()

    def test_numpy_to_proto_roundtrip(self):
        """Test numpy → proto → numpy preserves data"""
        arr = np.random.randn(3, 4, 5).astype(np.float32)

        proto = helpers.numpy_to_proto(arr)
        result = helpers.proto_to_numpy(proto)

        np.testing.assert_array_almost_equal(result, arr)
        assert result.shape == arr.shape
        assert result.dtype == arr.dtype


class TestTorchConversion:
    """Test torch tensor conversion"""

    def test_proto_to_tensor(self):
        """Test proto → torch tensor conversion"""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor_proto = helpers.numpy_to_proto(arr)

        tensor = helpers.proto_to_tensor(tensor_proto)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == torch.Size([2, 2])
        assert tensor.dtype == torch.float32
        torch.testing.assert_close(tensor, torch.tensor(arr))

    def test_tensor_to_proto(self):
        """Test torch tensor → proto conversion"""
        tensor = torch.randn(2, 3, 4)

        proto = helpers.tensor_to_proto(tensor)

        assert list(proto.shape) == [2, 3, 4]
        assert proto.dtype == cuvis_ai_pb2.D_TYPE_FLOAT32

    def test_tensor_to_proto_roundtrip(self):
        """Test tensor → proto → tensor preserves data"""
        tensor = torch.randn(3, 4, 5)

        proto = helpers.tensor_to_proto(tensor)
        result = helpers.proto_to_tensor(proto)

        torch.testing.assert_close(result, tensor)


class TestProcessingModeMapping:
    """Test ProcessingMode enum mapping"""

    def test_processing_mode_to_cuvis(self):
        """Test proto ProcessingMode → cuvis ProcessingMode"""
        import cuvis

        proto_raw = cuvis_ai_pb2.PROCESSING_MODE_RAW
        proto_refl = cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE

        assert helpers.proto_to_processing_mode(proto_raw) == cuvis.ProcessingMode.Raw
        assert helpers.proto_to_processing_mode(proto_refl) == cuvis.ProcessingMode.Reflectance

    def test_processing_mode_invalid(self):
        """Test error handling for invalid processing mode"""
        with pytest.raises(ValueError, match="Unsupported ProcessingMode"):
            helpers.proto_to_processing_mode(999)

    def test_processing_mode_new_values(self):
        """Test new ProcessingMode enum values (DarkSubtract, SpectralRadiance)"""
        import cuvis

        # Test DarkSubtract mapping
        proto_darksubtract = cuvis_ai_pb2.PROCESSING_MODE_DARKSUBTRACT
        assert (
            helpers.proto_to_processing_mode(proto_darksubtract)
            == cuvis.ProcessingMode.DarkSubtract
        )

        # Test SpectralRadiance mapping
        proto_spectral_radiance = cuvis_ai_pb2.PROCESSING_MODE_SPECTRAL_RADIANCE
        assert (
            helpers.proto_to_processing_mode(proto_spectral_radiance)
            == cuvis.ProcessingMode.SpectralRadiance
        )

    def test_processing_mode_enum_values(self):
        """Test that all ProcessingMode enum values are correctly defined"""
        # Verify the enum values match expected numbers
        assert cuvis_ai_pb2.PROCESSING_MODE_UNSPECIFIED == 0
        assert cuvis_ai_pb2.PROCESSING_MODE_RAW == 1
        assert cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE == 2
        assert cuvis_ai_pb2.PROCESSING_MODE_DARKSUBTRACT == 3
        assert cuvis_ai_pb2.PROCESSING_MODE_SPECTRAL_RADIANCE == 4
