import time

import pytest

from cuvis_ai_core.grpc import cuvis_ai_pb2, helpers


def _load_pipeline(grpc_stub, session_id: str, pipeline_name: str = "rx_statistical"):
    """Helper to load pipeline using bytes-based RPC."""
    config_response = grpc_stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="pipeline",
            path=f"pipeline/{pipeline_name}",
        )
    )
    response = grpc_stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=config_response.config_bytes),
        )
    )
    assert response.success
    return response


@pytest.mark.slow
class TestPerformance:
    """Lightweight performance smoke tests."""

    def test_session_creation_performance(self, grpc_stub):
        """Ensure session creation stays reasonably fast without relying on pytest-benchmark."""
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            # Step 1: Create empty session
            resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
            # Step 2: Load pipeline using ResolveConfig + LoadPipeline
            _load_pipeline(grpc_stub, resp.session_id)
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=resp.session_id))
            timings.append(time.perf_counter() - start)

        avg_duration = sum(timings) / len(timings)
        assert avg_duration < 1.0

    def test_inference_latency(self, grpc_stub, data_config_factory, create_test_cube):
        data_config = data_config_factory(
            batch_size=2, processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE
        )

        # Step 1: Create empty session
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Step 2: Load pipeline using four-step workflow
        _load_pipeline(grpc_stub, session_id)

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

        request = cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.tensor_to_proto(cube),
                wavelengths=helpers.tensor_to_proto(wavelengths),
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

    def test_throughput(self, grpc_stub, data_config_factory, create_test_cube):
        data_config = data_config_factory(
            batch_size=2, processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE
        )

        # Step 1: Create empty session
        session_resp = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = session_resp.session_id

        # Step 2: Load pipeline using four-step workflow
        _load_pipeline(grpc_stub, session_id)

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

        request = cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.tensor_to_proto(cube),
                wavelengths=helpers.tensor_to_proto(wavelengths),
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
