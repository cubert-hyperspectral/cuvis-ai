"""Example client demonstrating gradient training with streamed progress."""

from __future__ import annotations

import grpc

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc


def main(server_address: str = "localhost:50051") -> None:
    channel = grpc.insecure_channel(server_address)
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    # Use RestoreExperiment to load full config (pipeline + data + training + loss/metric nodes)
    session_id = stub.RestoreExperiment(
        cuvis_ai_pb2.RestoreExperimentRequest(experiment_path="configs/experiment/deep_svdd.yaml")
    ).session_id
    print(f"Session: {session_id}")

    # Statistical training
    stat_req = cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
    )
    for msg in stub.Train(stat_req):
        print(f"[statistical] {cuvis_ai_pb2.TrainStatus.Name(msg.status)}")

    # Gradient training (config already loaded from experiment)
    grad_req = cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
    )

    for progress in stub.Train(grad_req):
        stage_name = cuvis_ai_pb2.ExecutionStage.Name(progress.context.stage)
        status_name = cuvis_ai_pb2.TrainStatus.Name(progress.status)
        loss_str = f"losses={progress.losses}" if progress.losses else ""
        metric_str = f"metrics={progress.metrics}" if progress.metrics else ""
        print(
            f"[{stage_name}] epoch={progress.context.epoch} {status_name} {loss_str} {metric_str}"
        )

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


if __name__ == "__main__":
    main()
