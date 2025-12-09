import time

import numpy as np
import pytest

from cuvis_ai.grpc import cuvis_ai_pb2, helpers
from tests.fixtures import create_pipeline_config_proto


@pytest.mark.slow
class TestPerformance:
    """Lightweight performance smoke tests."""

    def test_session_creation_performance(self, grpc_stub, mock_cuvis_sdk):
        """Ensure session creation stays reasonably fast without relying on pytest-benchmark."""
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            resp = grpc_stub.CreateSession(
                cuvis_ai_pb2.CreateSessionRequest(
                    pipeline=create_pipeline_config_proto("rx_statistical"),
                )
            )
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=resp.session_id))
            timings.append(time.perf_counter() - start)

        avg_duration = sum(timings) / len(timings)
        assert avg_duration < 1.0

    def test_inference_latency(
        self, grpc_stub, data_config_factory, mock_cuvis_sdk, create_test_cube
    ):
        data_config = data_config_factory(
            batch_size=2, processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE
        )

        session_resp = grpc_stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(
                pipeline=create_pipeline_config_proto("rx_statistical"),
            )
        )
        session_id = session_resp.session_id

        for _ in grpc_stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
                data=data_config,
            )
        ):
            pass

        cube, wavelengths = create_test_cube(batch_size=1, height=16, width=16, num_channels=61)
        # Convert to numpy arrays for proto
        cube = cube.numpy()
        wavelengths = wavelengths.cpu().numpy().astype(np.int32).reshape(1, -1)

        request = cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.numpy_to_proto(cube),
                wavelengths=helpers.numpy_to_proto(wavelengths),
            ),
        )

        # Warm up once to avoid first-call overhead
        grpc_stub.Inference(request)

        iterations = 5
        start = time.perf_counter()
        for _ in range(iterations):
            grpc_stub.Inference(request)
        avg_latency = (time.perf_counter() - start) / iterations

        assert avg_latency < 0.5

        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_throughput(self, grpc_stub, data_config_factory, mock_cuvis_sdk, create_test_cube):
        data_config = data_config_factory(
            batch_size=2, processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE
        )

        session_resp = grpc_stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(
                pipeline=create_pipeline_config_proto("rx_statistical"),
            )
        )
        session_id = session_resp.session_id

        for _ in grpc_stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
                data=data_config,
            )
        ):
            pass

        cube, wavelengths = create_test_cube(batch_size=1, height=16, width=16, num_channels=61)
        # Convert to numpy arrays for proto
        cube = cube.numpy()
        wavelengths = wavelengths.cpu().numpy().astype(np.int32).reshape(1, -1)
        request = cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.numpy_to_proto(cube),
                wavelengths=helpers.numpy_to_proto(wavelengths),
            ),
        )

        num_requests = 50
        start = time.perf_counter()
        for _ in range(num_requests):
            grpc_stub.Inference(request)
        elapsed = time.perf_counter() - start
        throughput = num_requests / elapsed

        print(f"\nThroughput: {throughput:.2f} req/s")
        # Lightweight guardrail to catch major regressions while staying stable on CI
        assert throughput > 2

        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
