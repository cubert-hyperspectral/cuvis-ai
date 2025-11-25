import time
from pathlib import Path

import numpy as np
import pytest

from cuvis_ai.grpc import cuvis_ai_pb2, helpers


def _data_files(test_data_path: Path) -> tuple[Path, Path]:
    cu3s_file = test_data_path / "Lentils" / "Lentils_000.cu3s"
    json_file = test_data_path / "Lentils" / "Lentils_000.json"
    if not cu3s_file.exists() or not json_file.exists():
        pytest.skip(f"Test data not found under {test_data_path}")
    return cu3s_file, json_file


@pytest.mark.slow
class TestPerformance:
    """Lightweight performance smoke tests."""

    def test_session_creation_performance(
        self, grpc_stub, test_data_path, mock_cuvis_sdk, benchmark
    ):
        cu3s_file, _ = _data_files(test_data_path)

        def create_session() -> None:
            resp = grpc_stub.CreateSession(
                cuvis_ai_pb2.CreateSessionRequest(
                    pipeline_type="statistical",
                    data_config=cuvis_ai_pb2.DataConfig(
                        cu3s_file_path=str(cu3s_file),
                        batch_size=2,
                        processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
                    ),
                )
            )
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=resp.session_id))

        benchmark(create_session)
        assert benchmark.stats.stats["mean"] < 1.0

    def test_inference_latency(self, grpc_stub, test_data_path, mock_cuvis_sdk, benchmark):
        cu3s_file, _ = _data_files(test_data_path)

        session_resp = grpc_stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(
                pipeline_type="statistical",
                data_config=cuvis_ai_pb2.DataConfig(
                    cu3s_file_path=str(cu3s_file),
                    batch_size=2,
                    processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
                ),
            )
        )
        session_id = session_resp.session_id

        for _ in grpc_stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            )
        ):
            pass

        cube = np.random.rand(1, 16, 16, 61).astype(np.float32)
        request = cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube)),
        )

        benchmark(lambda: grpc_stub.Inference(request))
        assert benchmark.stats.stats["mean"] < 0.1

        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    def test_throughput(self, grpc_stub, test_data_path, mock_cuvis_sdk):
        cu3s_file, _ = _data_files(test_data_path)

        session_resp = grpc_stub.CreateSession(
            cuvis_ai_pb2.CreateSessionRequest(
                pipeline_type="statistical",
                data_config=cuvis_ai_pb2.DataConfig(
                    cu3s_file_path=str(cu3s_file),
                    batch_size=2,
                    processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
                ),
            )
        )
        session_id = session_resp.session_id

        for _ in grpc_stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            )
        ):
            pass

        cube = np.random.rand(1, 16, 16, 61).astype(np.float32)
        request = cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube)),
        )

        num_requests = 50
        start = time.perf_counter()
        for _ in range(num_requests):
            grpc_stub.Inference(request)
        elapsed = time.perf_counter() - start
        throughput = num_requests / elapsed

        print(f"\nThroughput: {throughput:.2f} req/s")
        assert throughput > 10

        grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
