"""End-to-end gRPC workflow: session setup, training, inference, checkpointing."""

import argparse
from pathlib import Path

import grpc
import numpy as np

from cuvis_ai.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc, helpers
from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig


def parse_args() -> argparse.Namespace:
    default_cu3s = Path(__file__).resolve().parents[2] / "data" / "Lentils" / "Lentils_000.cu3s"
    default_annotations = default_cu3s.with_suffix(".json")
    parser = argparse.ArgumentParser(description="cuvis.ai gRPC end-to-end client")
    parser.add_argument("--target", default="localhost:50051", help="gRPC target host:port")
    parser.add_argument("--cu3s", default=str(default_cu3s), help="Path to .cu3s file")
    parser.add_argument("--annotations", default=str(default_annotations), help="Path to COCO JSON")
    parser.add_argument("--checkpoint", default="model.ckpt", help="Checkpoint output path")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Gradient training epochs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    channel = grpc.insecure_channel(args.target)
    stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

    data_config = cuvis_ai_pb2.DataConfig(
        cu3s_file_path=str(args.cu3s),
        annotation_json_path=str(args.annotations),
        train_ids=[0, 1, 2],
        val_ids=[3, 4],
        test_ids=[5, 6],
        batch_size=args.batch_size,
        processing_mode=cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE,
    )

    session_id = stub.CreateSession(
        cuvis_ai_pb2.CreateSessionRequest(
            pipeline_type="gradient",
            data_config=data_config,
        )
    ).session_id

    try:
        inputs = stub.GetPipelineInputs(
            cuvis_ai_pb2.GetPipelineInputsRequest(session_id=session_id)
        )
        outputs = stub.GetPipelineOutputs(
            cuvis_ai_pb2.GetPipelineOutputsRequest(session_id=session_id)
        )
        print(f"Inputs: {inputs.input_names}")
        print(f"Outputs: {outputs.output_names}")

        for update in stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            )
        ):
            print(f"[statistical] status={update.status}")

        training_cfg = TrainingConfig(
            trainer=TrainerConfig(max_epochs=args.epochs, accelerator="cpu"),
            optimizer=OptimizerConfig(name="adam", lr=1e-3),
        )
        cfg_bytes = training_cfg.to_json().encode()

        validation = stub.ValidateTrainingConfig(
            cuvis_ai_pb2.ValidateTrainingConfigRequest(
                config=cuvis_ai_pb2.TrainingConfig(config_json=cfg_bytes)
            )
        )
        if not validation.valid:
            raise RuntimeError(f"Invalid training config: {validation.errors}")

        for progress in stub.Train(
            cuvis_ai_pb2.TrainRequest(
                session_id=session_id,
                trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT,
                config=cuvis_ai_pb2.TrainingConfig(config_json=cfg_bytes),
            )
        ):
            print(
                f"[gradient] epoch={progress.context.epoch} "
                f"status={progress.status} losses={dict(progress.losses)} "
                f"metrics={dict(progress.metrics)}"
            )

        checkpoint_path = Path(args.checkpoint)
        save_resp = stub.SaveCheckpoint(
            cuvis_ai_pb2.SaveCheckpointRequest(
                session_id=session_id,
                checkpoint_path=str(checkpoint_path),
            )
        )
        print(f"Checkpoint saved: {save_resp.success} -> {checkpoint_path}")

        cube = np.random.rand(1, 32, 32, 61).astype(np.float32)
        inference = stub.Inference(
            cuvis_ai_pb2.InferenceRequest(
                session_id=session_id,
                inputs=cuvis_ai_pb2.InputBatch(cube=helpers.numpy_to_proto(cube)),
                output_specs=["selector.selected", "anomaly_metrics"],
            )
        )
        print(f"Outputs: {list(inference.outputs.keys())}")
        print(f"Metrics: {dict(inference.metrics)}")
    finally:
        stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
        channel.close()


if __name__ == "__main__":
    main()
