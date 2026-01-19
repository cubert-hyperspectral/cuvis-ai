from cuvis_ai_core.grpc import cuvis_ai_pb2


class TestProtoDefinitions:
    """Verify proto definitions are generated correctly"""

    def test_tensor_message_exists(self):
        """Verify Tensor message has required fields"""
        tensor = cuvis_ai_pb2.Tensor()
        assert hasattr(tensor, "shape")
        assert hasattr(tensor, "dtype")
        assert hasattr(tensor, "raw_data")

    def test_data_config_message_exists(self):
        """Verify DataConfig message structure"""
        config = cuvis_ai_pb2.DataConfig()
        assert hasattr(config, "config_bytes")

    def test_context_message_exists(self):
        """Verify Context message structure"""
        context = cuvis_ai_pb2.Context()
        assert hasattr(context, "stage")
        assert hasattr(context, "epoch")
        assert hasattr(context, "batch_idx")
        assert hasattr(context, "global_step")

    def test_execution_stage_enum(self):
        """Verify ExecutionStage enum values"""
        assert cuvis_ai_pb2.EXECUTION_STAGE_TRAIN == 1
        assert cuvis_ai_pb2.EXECUTION_STAGE_VAL == 2
        assert cuvis_ai_pb2.EXECUTION_STAGE_TEST == 3
        assert cuvis_ai_pb2.EXECUTION_STAGE_INFERENCE == 4

    def test_dtype_enum(self):
        """Verify DType enum values"""
        assert cuvis_ai_pb2.D_TYPE_FLOAT32 == 1
        assert cuvis_ai_pb2.D_TYPE_FLOAT64 == 2
        assert cuvis_ai_pb2.D_TYPE_INT32 == 3
        assert cuvis_ai_pb2.D_TYPE_INT64 == 4
        assert cuvis_ai_pb2.D_TYPE_UINT8 == 5
        assert cuvis_ai_pb2.D_TYPE_BOOL == 6

    def test_processing_mode_enum(self):
        """Verify ProcessingMode enum values"""
        assert cuvis_ai_pb2.PROCESSING_MODE_RAW == 1
        assert cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE == 2

    def test_train_status_enum(self):
        """Verify TrainStatus enum values"""
        assert cuvis_ai_pb2.TRAIN_STATUS_UNSPECIFIED == 0
        assert cuvis_ai_pb2.TRAIN_STATUS_RUNNING == 1
        assert cuvis_ai_pb2.TRAIN_STATUS_COMPLETE == 2
        assert cuvis_ai_pb2.TRAIN_STATUS_ERROR == 3

    def test_point_type_enum(self):
        """Verify PointType enum values"""
        assert cuvis_ai_pb2.POINT_TYPE_UNSPECIFIED == 0
        assert cuvis_ai_pb2.POINT_TYPE_POSITIVE == 1
        assert cuvis_ai_pb2.POINT_TYPE_NEGATIVE == 2
        assert cuvis_ai_pb2.POINT_TYPE_NEUTRAL == 3
