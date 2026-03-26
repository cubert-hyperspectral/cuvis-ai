"""End-to-end gRPC workflow using the explicit ResolveConfig pipeline."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
from cuvis_ai_core.grpc import helpers
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
from workflow_utils import (
    apply_trainrun_config,
    build_stub,
    config_search_paths,
    create_session_with_search_paths,
    format_progress,
    resolve_trainrun_config,
)


def main(
    *,
    target: str,
    trainrun: str,
    pipeline_out: Path,
    trainrun_out: Path,
) -> None:
    stub = build_stub(target)

    # 1) Create session + register config search paths
    session_id = create_session_with_search_paths(stub, config_search_paths())
    print(f"Session created: {session_id}")

    # 2) Resolve trainrun config via ConfigService (Hydra composition + overrides)
    resolved, config_dict = resolve_trainrun_config(
        stub,
        session_id,
        trainrun,
        overrides=["training.trainer.max_epochs=2"],
    )
    print(
        f"Resolved trainrun '{config_dict['name']}' "
        f"with pipeline '{config_dict['pipeline']['metadata']['name']}'"
    )

    # 3) Apply trainrun config (builds pipeline + persists data/training configs)
    apply_trainrun_config(stub, session_id, resolved.config_bytes)

    # 4) Train: statistical pass followed by gradient fine-tuning
    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
        )
    ):
        print(f"[statistical] {format_progress(progress)}")

    for progress in stub.Train(
        cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
        )
    ):
        print(f"[gradient] {format_progress(progress)}")

    # 5) Save pipeline + composed trainrun for reproducibility
    pipeline_path = str(pipeline_out.resolve())
    trainrun_path = str(trainrun_out.resolve())

    save_pipeline = stub.SavePipeline(
        cuvis_ai_pb2.SavePipelineRequest(
            session_id=session_id,
            pipeline_path=pipeline_path,
        )
    )
    print(f"Pipeline saved to {save_pipeline.pipeline_path}")
    print(f"Weights saved to {save_pipeline.weights_path}")

    save_trainrun = stub.SaveTrainRun(
        cuvis_ai_pb2.SaveTrainRunRequest(session_id=session_id, trainrun_path=trainrun_path)
    )
    print(f"Trainrun saved to {save_trainrun.trainrun_path}")

    # 6) Run inference using the freshly trained pipeline

    cube = np.random.randn(1, 16, 16, 61).astype(np.uint16)
    wavelengths = np.linspace(430, 910, 61).reshape(1, -1).astype(np.int32)
    inference = stub.Inference(
        cuvis_ai_pb2.InferenceRequest(
            session_id=session_id,
            inputs=cuvis_ai_pb2.InputBatch(
                cube=helpers.numpy_to_proto(cube), wavelengths=helpers.numpy_to_proto(wavelengths)
            ),
        )
    )
    print(f"Inference outputs: {list(inference.outputs.keys())}")

    stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
    print("Session closed.")


@click.command()
@click.option(
    "--target",
    type=str,
    default="localhost:50051",
    show_default=True,
    help="gRPC target host:port.",
)
@click.option(
    "--trainrun",
    type=str,
    default="deep_svdd",
    show_default=True,
    help="Trainrun name (under configs/trainrun) to resolve with Hydra.",
)
@click.option(
    "--pipeline-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("outputs") / "demo_pipeline.yaml",
    show_default=True,
    help="Path to save the trained pipeline (YAML).",
)
@click.option(
    "--trainrun-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("outputs") / "demo_trainrun.yaml",
    show_default=True,
    help="Path to save the composed trainrun config (for RestoreTrainRun).",
)
def cli(
    target: str,
    trainrun: str,
    pipeline_out: Path,
    trainrun_out: Path,
) -> None:
    main(
        target=target,
        trainrun=trainrun,
        pipeline_out=pipeline_out,
        trainrun_out=trainrun_out,
    )


if __name__ == "__main__":
    cli()
