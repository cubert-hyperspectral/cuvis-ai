"""Train a pipeline from scratch using explicit config composition."""

from __future__ import annotations

import json
from pathlib import Path

from cuvis_ai_core.grpc import cuvis_ai_pb2
from workflow_utils import (
    apply_trainrun_config,
    build_stub,
    config_search_paths,
    create_session_with_search_paths,
    format_progress,
    resolve_trainrun_config,
)


def main() -> None:
    stub = build_stub()
    session_id = create_session_with_search_paths(stub, config_search_paths())
    print(f"Session created: {session_id}")

    # Resolve the channel_selector trainrun and tweak it client-side
    resolved, config_dict = resolve_trainrun_config(
        stub,
        session_id,
        "channel_selector",
        overrides=[],
    )

    # (Optional) Build the pipeline explicitly before applying the trainrun
    stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=session_id,
            pipeline=cuvis_ai_pb2.PipelineConfig(
                config_bytes=json.dumps(config_dict["pipeline"]).encode("utf-8")
            ),
        )
    )

    # Customize a few training hyperparameters before applying
    config_dict["training"]["trainer"]["max_epochs"] = 2
    config_dict["training"]["optimizer"]["lr"] = 5e-4

    trainrun_bytes = json.dumps(config_dict).encode("utf-8")
    apply_trainrun_config(stub, session_id, trainrun_bytes)
    print(f"Applied custom trainrun '{config_dict['name']}' with manual tweaks")

    # Statistical training (server streams progress)
    print("Starting statistical training...")
    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
        )
    ):
        print(format_progress(progress))

    # Gradient training
    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        )
    ):
        print(format_progress(progress))

    # Save artifacts
    pipeline_path = str(Path("outputs/train_from_scratch.yaml").resolve())
    trainrun_path = str(Path("outputs/train_from_scratch_trainrun.yaml").resolve())

    save_pipeline = stub.SavePipeline(
        cuvis_ai_pb2.SavePipelineRequest(
            session_id=session_id,
            pipeline_path=pipeline_path,
            metadata=cuvis_ai_pb2.PipelineMetadata(
                name="Trained Channel Selector",
                description="Channel selector trained from scratch",
            ),
        )
    )
    print(f"Pipeline saved: {save_pipeline.pipeline_path}")
    print(f"Weights saved: {save_pipeline.weights_path}")

    save_trainrun = stub.SaveTrainRun(
        cuvis_ai_pb2.SaveTrainRunRequest(session_id=session_id, trainrun_path=trainrun_path)
    )
    print(f"Trainrun saved: {save_trainrun.trainrun_path}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
    print("Session closed.")


if __name__ == "__main__":
    main()
