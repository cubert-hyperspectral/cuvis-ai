"""Example client demonstrating gradient training with streamed progress."""

from __future__ import annotations

import grpc

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig


def main(server_address: str = "localhost:50051") -> None:
    channel = grpc.insecure_channel(server_address)
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    print("Creating session...")
    create_req = cuvis_ai_pb2.CreateSessionRequest(
        pipeline_type="gradient",
        data_config=cuvis_ai_pb2.DataConfig(
            cu3s_file_path="data/Lentils/Lentils_000.cu3s",
            annotation_json_path="data/Lentils/Lentils_000.json",
            train_ids=[0, 1, 2],
            val_ids=[3, 4],
            test_ids=[5, 6],
            batch_size=2,
            processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
        ),
    )
    session_id = stub.CreateSession(create_req).session_id
    print(f"Session created: {session_id}")

    print("Running statistical training...")
    stat_req = cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
    )
    for msg in stub.Train(stat_req):
        print(f"  status={cuvis_ai_pb2.TrainStatus.Name(msg.status)}")

    print("Starting gradient training...")
    config = TrainingConfig(
        trainer=TrainerConfig(max_epochs=3, accelerator="gpu", log_every_n_steps=1),
        optimizer=OptimizerConfig(name="adam", lr=0.001),
    )
    grad_req = cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        config=cuvis_ai_pb2.TrainingConfig(config_json=config.to_json().encode()),
    )

    for progress in stub.Train(grad_req):
        stage_name = cuvis_ai_pb2.ExecutionStage.Name(progress.context.stage)
        status_name = cuvis_ai_pb2.TrainStatus.Name(progress.status)
        loss_str = f"losses={progress.losses}" if progress.losses else ""
        metric_str = f"metrics={progress.metrics}" if progress.metrics else ""
        print(f"[{stage_name}] epoch={progress.context.epoch} {status_name} {loss_str} {metric_str}")

    print("Gradient training complete, closing session.")
    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))


if __name__ == "__main__":
    main()
