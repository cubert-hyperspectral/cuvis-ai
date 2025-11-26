"""Example client demonstrating checkpoint save/load over gRPC."""

import grpc

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig


def main() -> None:
    channel = grpc.insecure_channel("localhost:50051")
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    print("Creating and training session...")
    session_resp = stub.CreateSession(
        cuvis_ai_pb2.CreateSessionRequest(
            pipeline_type="gradient",
            data_config=cuvis_ai_pb2.DataConfig(
                cu3s_file_path="/path/to/data.cu3s",
                annotation_json_path="/path/to/data.json",
                train_ids=[0, 1, 2],
                val_ids=[3, 4],
                batch_size=2,
                processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
            ),
        )
    )
    session_id = session_resp.session_id

    train_cfg = TrainingConfig(
        trainer=TrainerConfig(max_epochs=2, accelerator="cpu"),
        optimizer=OptimizerConfig(name="adam", lr=0.001),
    )
    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
            config=cuvis_ai_pb2.TrainingConfig(config_json=train_cfg.to_json().encode()),
        )
    ):
        if progress.status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE:
            print("Training complete.")

    checkpoint_path = "./model_checkpoint.ckpt"
    print(f"Saving checkpoint to {checkpoint_path}...")
    save_resp = stub.SaveCheckpoint(
        cuvis_ai_pb2.SaveCheckpointRequest(
            session_id=session_id,
            checkpoint_path=checkpoint_path,
        )
    )
    print(f"Checkpoint saved: {save_resp.success}")

    print("Creating new session and loading checkpoint...")
    new_session = stub.CreateSession(
        cuvis_ai_pb2.CreateSessionRequest(
            pipeline_type="gradient",
            data_config=cuvis_ai_pb2.DataConfig(
                cu3s_file_path="/path/to/data.cu3s",
                batch_size=2,
            ),
        )
    )

    load_resp = stub.LoadCheckpoint(
        cuvis_ai_pb2.LoadCheckpointRequest(
            session_id=new_session.session_id,
            checkpoint_path=checkpoint_path,
        )
    )
    print(f"Checkpoint loaded: {load_resp.success}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=new_session.session_id))


if __name__ == "__main__":
    main()
