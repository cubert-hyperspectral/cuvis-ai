"""Experiment/trainrun management service component."""

from __future__ import annotations

from pathlib import Path

import grpc
import torch
import yaml

from cuvis_ai.training.config import DataConfig, TrainingConfig, TrainRunConfig

from . import helpers
from .session_manager import SessionManager
from .v1 import cuvis_ai_pb2


class TrainRunService:
    """Save and restore train run configuration."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    def save_train_run(
        self,
        request: cuvis_ai_pb2.SaveTrainRunRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.SaveTrainRunResponse:
        """Persist the current session TrainRunConfig to disk, optionally with weights."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.SaveTrainRunResponse(success=False)

        if not request.trainrun_path:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("trainrun_path is required")
            return cuvis_ai_pb2.SaveTrainRunResponse(success=False)

        try:
            trainrun_path = Path(request.trainrun_path)
            trainrun_path.parent.mkdir(parents=True, exist_ok=True)

            if session.trainrun_config is not None:
                trainrun_config = session.trainrun_config
            else:
                if session.pipeline is None and session._pipeline_config is None:
                    context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                    context.set_details(
                        "Pipeline configuration is required before saving a train run."
                    )
                    return cuvis_ai_pb2.SaveTrainRunResponse(success=False)

                try:
                    pipeline_config = session.pipeline_config
                except ValueError as exc:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(str(exc))
                    return cuvis_ai_pb2.SaveTrainRunResponse(success=False)

                trainrun_config = TrainRunConfig(
                    name=getattr(session.pipeline, "name", "trainrun"),
                    pipeline=pipeline_config,
                    data=session.data_config or DataConfig(cu3s_file_path=""),
                    training=session.training_config or TrainingConfig(),
                    loss_nodes=[],
                    metric_nodes=[],
                    freeze_nodes=[],
                    unfreeze_nodes=[],
                    output_dir="./outputs",
                    tags={},
                )

            with trainrun_path.open("w", encoding="utf-8") as f:
                yaml.dump(
                    trainrun_config.model_dump(mode="json"),
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )

            # Simplified: Only save weights if explicitly requested
            weights_path = None
            if request.save_weights and session.pipeline is not None:
                try:
                    # Save weights with same name as trainrun but .pt extension
                    weights_path_obj = trainrun_path.with_suffix(".pt")
                    weights_path = str(weights_path_obj)

                    # Save only the state_dict (weights) without pipeline config
                    state_dict = {}
                    for node in session.pipeline.nodes():
                        if hasattr(node, "state_dict"):
                            state_dict[node.name] = node.state_dict()

                    checkpoint = {
                        "state_dict": state_dict,
                        "metadata": {
                            "name": trainrun_config.name,
                            "created": trainrun_config.pipeline.metadata.created
                            if trainrun_config.pipeline and trainrun_config.pipeline.metadata
                            else "",
                        },
                    }

                    import torch

                    torch.save(checkpoint, weights_path_obj)

                except Exception as exc:
                    # Log weights save failure but don't fail the entire operation
                    context.set_details(
                        f"Train run config saved successfully, but weights save failed: {exc}"
                    )

            return cuvis_ai_pb2.SaveTrainRunResponse(
                success=True,
                trainrun_path=str(trainrun_path),
                weights_path=weights_path,
            )
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to save train run: {exc}")
            return cuvis_ai_pb2.SaveTrainRunResponse(success=False)

    def restore_train_run(
        self,
        request: cuvis_ai_pb2.RestoreTrainRunRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.RestoreTrainRunResponse:
        """Restore a TrainRunConfig from disk and create a session, optionally with weights."""
        trainrun_path = Path(request.trainrun_path)
        if not trainrun_path.exists():
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Train run file not found: {trainrun_path}")
            return cuvis_ai_pb2.RestoreTrainRunResponse()

        try:
            from cuvis_ai.pipeline.factory import PipelineBuilder
            from cuvis_ai.pipeline.pipeline import CuvisPipeline

            with trainrun_path.open("r", encoding="utf-8") as f:
                trainrun_raw = f.read()
            trainrun_dict = yaml.safe_load(trainrun_raw)
            if isinstance(trainrun_dict, dict) and "defaults" in trainrun_dict:
                # By design: the server does not run Hydra composition. Clients must compose/resolve
                # configs before sending (e.g., pass resolved YAML/JSON via config_bytes).
                raise ValueError(
                    "Train run config contains Hydra defaults; please compose/resolve it first "
                    "and pass the resolved YAML/JSON via config_bytes."
                )

            # If pipeline is provided as a reference to saved files, inline the config
            # and keep track of the weights for later loading.
            pipeline_config_path: Path | None = None
            pipeline_weights_path: str | None = None
            pipeline_section = trainrun_dict.get("pipeline")
            if isinstance(pipeline_section, dict) and "config_path" in pipeline_section:
                if "nodes" not in pipeline_section or "connections" not in pipeline_section:
                    pipeline_config_path = helpers.resolve_pipeline_path(
                        pipeline_section["config_path"]
                    )
                    with pipeline_config_path.open("r", encoding="utf-8") as pipeline_file:
                        pipeline_config_dict = yaml.safe_load(pipeline_file)

                    if not isinstance(pipeline_config_dict, dict):
                        raise ValueError(
                            f"Pipeline config file {pipeline_config_path} did not contain a mapping"
                        )

                    pipeline_weights_path = pipeline_section.get("weights_path")
                    if pipeline_weights_path:
                        weights_path = Path(pipeline_weights_path)
                        if not weights_path.exists():
                            raise FileNotFoundError(
                                f"Pipeline weights file not found: {weights_path}"
                            )

                    # Replace the reference with the actual pipeline config so validation passes
                    trainrun_dict["pipeline"] = pipeline_config_dict

            trainrun_config = TrainRunConfig.from_dict(trainrun_dict)

            if trainrun_config.pipeline is None:
                raise ValueError("Train run config missing pipeline section")

            # Enhanced: Check for associated pipeline files using the naming convention
            # If no explicit pipeline config path was found, try the _pipeline.yaml pattern
            if pipeline_config_path is None:
                pipeline_config_pattern = trainrun_path.with_stem(f"{trainrun_path.stem}_pipeline")
                if pipeline_config_pattern.with_suffix(".yaml").exists():
                    pipeline_config_path = pipeline_config_pattern.with_suffix(".yaml")

            # Enhanced: Load weights if weights_path is provided in request
            weights_path_from_request = None
            if request.HasField("weights_path") and request.weights_path:
                weights_path_from_request = Path(request.weights_path)
                if not weights_path_from_request.exists():
                    raise FileNotFoundError(f"Weights file not found: {weights_path_from_request}")

            # Build or load pipeline
            if pipeline_config_path is not None:
                # Load pipeline from file, using weights from request if provided
                pipeline = CuvisPipeline.load_pipeline(
                    config_path=pipeline_config_path,
                    weights_path=str(weights_path_from_request)
                    if weights_path_from_request
                    else None,
                    strict_weight_loading=request.strict if request.HasField("strict") else True,
                    device="cuda" if torch.cuda.is_available() else None,
                )
            else:
                # Build pipeline from trainrun config (no weights)
                builder = PipelineBuilder()
                pipeline_dict = trainrun_config.pipeline.to_dict()
                pipeline = builder.build_from_config(pipeline_dict)

                # Enhanced: Load weights into the built pipeline if requested
                if weights_path_from_request:
                    pipeline._restore_weights_from_checkpoint(
                        weights_path=str(weights_path_from_request),
                        strict_weight_loading=request.strict
                        if request.HasField("strict")
                        else True,
                    )

            # Move pipeline to GPU if available (for PipelineBuilder path)
            if torch.cuda.is_available():
                pipeline = pipeline.to("cuda")

            session_id = self.session_manager.create_session(
                pipeline=pipeline,
                data_config=trainrun_config.data,
                training_config=trainrun_config.training,
                trainrun_config=trainrun_config,
            )

            return cuvis_ai_pb2.RestoreTrainRunResponse(
                session_id=session_id,
                trainrun=trainrun_config.to_proto(),
            )

        except FileNotFoundError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.RestoreTrainRunResponse()
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return cuvis_ai_pb2.RestoreTrainRunResponse()
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to restore train run: {exc}")
            return cuvis_ai_pb2.RestoreTrainRunResponse()


__all__ = ["TrainRunService"]
