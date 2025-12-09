"""Example client demonstrating statistical training workflow."""

import grpc
import numpy as np

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc, helpers


def main() -> None:
    channel = grpc.insecure_channel("localhost:50051")
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    # Use RestoreExperiment to load full config (pipeline + data)
    session_resp = stub.RestoreExperiment(
        cuvis_ai_pb2.RestoreExperimentRequest(
            experiment_path="configs/experiment/rx_statistical.yaml"
        )
    )
    session_id = session_resp.session_id
    print(f"Session: {session_id}")

    # Statistical training (config already loaded from experiment)
    train_req = cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
    )

    for progress in stub.Train(train_req):
        stage = cuvis_ai_pb2.ExecutionStage.Name(progress.context.stage)
        status = cuvis_ai_pb2.TrainStatus.Name(progress.status)
        print(f"[{stage}] {status}")

    cube = np.random.rand(1, 32, 32, 61).astype(np.uint16)
    inference_resp = stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube)),
        )
    )
    print(f"Outputs: {list(inference_resp.outputs.keys())}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


if __name__ == "__main__":
    main()
