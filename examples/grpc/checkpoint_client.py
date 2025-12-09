"""Example client demonstrating pipeline save/load over gRPC.

Note: SaveCheckpoint/LoadCheckpoint RPCs were removed in Phase 5.
Use SavePipeline/LoadPipeline or SaveExperiment/RestoreExperiment instead.
"""

import grpc

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig


def main() -> None:
    channel = grpc.insecure_channel("localhost:50051")
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    print("Creating and training session...")
    session_resp = stub.CreateSession(
        cuvis_ai_pb2.CreateSessionRequest(
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=b"channel_selector")
        )
    )
    session_id = session_resp.session_id

    data_cfg = cuvis_ai_pb2.DataConfig(
        cu3s_file_path="data/Lentils/Lentils_000.cu3s",
        annotation_json_path="data/Lentils/Lentils_000.json",
        train_ids=[0, 1, 2],
        val_ids=[3, 4],
        batch_size=2,
        processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
    )

    train_cfg = TrainingConfig(
        trainer=TrainerConfig(max_epochs=2, accelerator="cuda"),
        optimizer=OptimizerConfig(name="adam", lr=0.001),
    )

    print("Training...")
    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
            data=data_cfg,
            training=cuvis_ai_pb2.TrainingConfig(config_bytes=train_cfg.to_json().encode()),
        )
    ):
        stage_name = cuvis_ai_pb2.ExecutionStage.Name(progress.context.stage)
        status_name = cuvis_ai_pb2.TrainStatus.Name(progress.status)
        loss_str = f"losses={dict(progress.losses)}" if progress.losses else ""
        metric_str = f"metrics={dict(progress.metrics)}" if progress.metrics else ""

        print(
            f"[{stage_name}] epoch={progress.context.epoch} {status_name} {loss_str} {metric_str}"
        )

        if progress.status == cuvis_ai_pb2.TRAIN_STATUS_COMPLETE:
            print("Training complete.")

    pipeline_path = "models/trained_pipeline.yaml"
    print(f"Saving pipeline to {pipeline_path}...")
    save_resp = stub.SavePipeline(
        cuvis_ai_pb2.SavePipelineRequest(
            session_id=session_id,
            pipeline_path=pipeline_path,
            metadata=cuvis_ai_pb2.PipelineMetadata(
                name="Trained Pipeline",
                description="Pipeline with trained weights",
                created="2024-12-08",
            ),
        )
    )
    print(f"Pipeline saved: {save_resp.pipeline_path}")
    print(f"Weights saved: {save_resp.weights_path}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    print("\nCreating new session from saved pipeline...")
    new_session = stub.CreateSession(
        cuvis_ai_pb2.CreateSessionRequest(
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=pipeline_path.encode("utf-8"))
        )
    )
    print(f"New session created: {new_session.session_id}")
    print("Pipeline and weights automatically loaded!")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=new_session.session_id))


if __name__ == "__main__":
    main()
