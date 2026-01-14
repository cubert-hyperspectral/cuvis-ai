"""Demonstrate SavePipeline/LoadPipeline with the Phase 5 workflow."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from workflow_utils import (
    build_stub,
    config_search_paths,
    create_session_with_search_paths,
    format_progress,
)

from cuvis_ai.grpc import cuvis_ai_pb2


def main() -> None:
    stub = build_stub()
    search_paths = config_search_paths()

    # Train a pipeline (quick pass) using resolved trainrun
    session_id = create_session_with_search_paths(stub, search_paths)

    # Direct gRPC call to resolve trainrun config (alternative: workflow_utils.resolve_trainrun_config)
    config_path = (
        "channel_selector"
        if "channel_selector".startswith("trainrun/")
        else "trainrun/channel_selector"
    )
    resolve_response = stub.ResolveConfig(
        cuvis_ai_pb2.ResolveConfigRequest(
            session_id=session_id,
            config_type="trainrun",
            path=config_path,
            overrides=["training.trainer.max_epochs=1"],
        )
    )

    # Direct gRPC call to apply trainrun config (alternative: workflow_utils.apply_trainrun_config)
    stub.SetTrainRunConfig(
        cuvis_ai_pb2.SetTrainRunConfigRequest(
            session_id=session_id,
            config=cuvis_ai_pb2.TrainRunConfig(config_bytes=resolve_response.config_bytes),
        )
    )

    # Run statistical initialization first (required for RXGlobal)
    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
        )
    ):
        print(f"[statistical init] {format_progress(progress)}")

    # Now run gradient training
    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        )
    ):
        print(format_progress(progress))

    # Save the trained pipeline
    pipeline_path = str(Path("outputs/checkpoint_pipeline.yaml").resolve())
    save_resp = stub.SavePipeline(
        cuvis_ai_pb2.SavePipelineRequest(
            session_id=session_id,
            pipeline_path=pipeline_path,
            metadata=cuvis_ai_pb2.PipelineMetadata(
                name="Checkpoint Demo",
                description="Example trained pipeline saved via gRPC",
            ),
        )
    )
    print(f"Pipeline saved: {save_resp.pipeline_path}")
    print(f"Weights saved: {save_resp.weights_path}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))

    # Load pipeline into a fresh session
    new_session = create_session_with_search_paths(stub, search_paths)
    pipeline_bytes = json.dumps(yaml.safe_load(Path(save_resp.pipeline_path).read_text())).encode(
        "utf-8"
    )
    stub.LoadPipeline(
        cuvis_ai_pb2.LoadPipelineRequest(
            session_id=new_session,
            pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=pipeline_bytes),
        )
    )
    stub.LoadPipelineWeights(
        cuvis_ai_pb2.LoadPipelineWeightsRequest(
            session_id=new_session,
            weights_path=save_resp.weights_path,
            strict=True,
        )
    )
    print(f"Reloaded pipeline into session {new_session}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=new_session))


if __name__ == "__main__":
    main()
