"""gRPC service implementation for basic inference, introspection, and sessions."""

from __future__ import annotations

import json
import queue
import tempfile
import threading
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import grpc
import numpy as np
import torch
import yaml

from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule
from cuvis_ai.grpc.callbacks import ProgressStreamCallback
from cuvis_ai.training.config import (
    DataConfig,
    ExperimentConfig,
    PipelineConfig,
    TrainingConfig,
    create_callbacks_from_config,
)
from cuvis_ai.training.trainers import GradientTrainer, StatisticalTrainer
from cuvis_ai.utils.types import Context, ExecutionStage

from . import helpers
from .session_manager import SessionManager, SessionState
from .v1 import cuvis_ai_pb2, cuvis_ai_pb2_grpc


class CuvisAIService(cuvis_ai_pb2_grpc.CuvisAIServiceServicer):
    """Cuvis.ai gRPC surface for session management and inference."""

    def __init__(self, session_manager: SessionManager | None = None) -> None:
        self.session_manager = session_manager or SessionManager()

    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------
    def CreateSession(
        self,
        request: cuvis_ai_pb2.CreateSessionRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.CreateSessionResponse:
        """Create a new session with pipeline configuration."""
        try:
            from cuvis_ai.pipeline.pipeline import CuvisPipeline
            from cuvis_ai.pipeline.pipeline_builder import PipelineBuilder

            # Decode config_bytes - could be either a file path or full config JSON
            config_bytes_str = request.pipeline.config_bytes.decode("utf-8")

            # Try to determine if it's a file path or JSON config
            try:
                # Try parsing as JSON first
                config_dict = json.loads(config_bytes_str)

                # If it's a dict with the expected PipelineConfig structure, build from it
                if (
                    isinstance(config_dict, dict)
                    and "nodes" in config_dict
                    and "connections" in config_dict
                ):
                    # Full pipeline config provided - build pipeline from config
                    pipeline_config = PipelineConfig.from_dict(config_dict)
                    builder = PipelineBuilder()

                    # Build pipeline from config structure
                    pipeline_dict = pipeline_config.to_dict()
                    pipeline = builder.build_from_config(pipeline_dict)

                    # Create session without file paths (config is self-contained)
                    session_id = self.session_manager.create_session(pipeline=pipeline)

                    return cuvis_ai_pb2.CreateSessionResponse(session_id=session_id)
                else:
                    # Not a valid pipeline config structure
                    raise ValueError("Invalid pipeline config structure")

            except (json.JSONDecodeError, ValueError):
                # Not JSON, treat as file path
                pipeline_path = helpers.resolve_pipeline_path(config_bytes_str)
                pt_path = pipeline_path.with_suffix(".pt")
                weights_path = str(pt_path) if pt_path.exists() else None

                try:
                    pipeline = CuvisPipeline.load_from_file(
                        str(pipeline_path),
                        weights_path=weights_path,
                        strict_weight_loading=False,
                    )
                except Exception:
                    # Fallback to structure-only load if weights cannot be applied
                    pipeline = CuvisPipeline.load_from_file(str(pipeline_path))

                # Create session with pipeline
                session_id = self.session_manager.create_session(pipeline=pipeline)

                return cuvis_ai_pb2.CreateSessionResponse(session_id=session_id)

        except FileNotFoundError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.CreateSessionResponse()
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return cuvis_ai_pb2.CreateSessionResponse()
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to create session: {exc}")
            return cuvis_ai_pb2.CreateSessionResponse()

    def CloseSession(
        self,
        request: cuvis_ai_pb2.CloseSessionRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.CloseSessionResponse:
        """Close and clean up a session."""
        try:
            self.session_manager.close_session(request.session_id)
            return cuvis_ai_pb2.CloseSessionResponse(success=True)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.CloseSessionResponse(success=False)
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to close session: {exc}")
            return cuvis_ai_pb2.CloseSessionResponse(success=False)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def Inference(
        self,
        request: cuvis_ai_pb2.InferenceRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.InferenceResponse:
        """Run a forward pass for the requested session."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.InferenceResponse()

        try:
            batch = self._parse_input_batch(request.inputs)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return cuvis_ai_pb2.InferenceResponse()

        try:
            outputs = session.pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)
        except Exception as exc:  # pragma: no cover - exercise in tests via validation
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Inference failed: {exc}")
            return cuvis_ai_pb2.InferenceResponse()

        output_specs = set(request.output_specs)
        tensor_outputs: dict[str, cuvis_ai_pb2.Tensor] = {}
        metrics: dict[str, float] = {}

        for raw_key, value in outputs.items():
            output_name = self._format_output_key(raw_key)
            if not self._should_return(output_name, output_specs):
                continue

            # Metrics: plain scalars
            if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                metrics[output_name] = float(value)
                continue

            try:
                tensor = self._to_tensor(value)
            except Exception:
                # Skip non-tensorizable outputs (e.g., artifacts or custom objects)
                continue
            tensor_outputs[output_name] = helpers.tensor_to_proto(tensor)

        return cuvis_ai_pb2.InferenceResponse(outputs=tensor_outputs, metrics=metrics)

    # ------------------------------------------------------------------
    # Pipeline Introspection
    # ------------------------------------------------------------------
    def GetPipelineInputs(
        self,
        request: cuvis_ai_pb2.GetPipelineInputsRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetPipelineInputsResponse:
        """Return pipeline entrypoint specifications for the session."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.GetPipelineInputsResponse()

        try:
            input_specs_dict = session.pipeline.get_input_specs()
            input_specs = {
                name: cuvis_ai_pb2.TensorSpec(
                    name=spec.get("name", name),
                    shape=list(spec.get("shape", [])),
                    dtype=self._dtype_str_to_proto(spec.get("dtype")),
                    required=bool(spec.get("required", False)),
                )
                for name, spec in input_specs_dict.items()
            }

            return cuvis_ai_pb2.GetPipelineInputsResponse(
                input_names=list(input_specs.keys()),
                input_specs=input_specs,
            )
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get inputs: {exc}")
            return cuvis_ai_pb2.GetPipelineInputsResponse()

    def GetPipelineOutputs(
        self,
        request: cuvis_ai_pb2.GetPipelineOutputsRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetPipelineOutputsResponse:
        """Return pipeline exit specifications for the session."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.GetPipelineOutputsResponse()

        try:
            output_specs_dict = session.pipeline.get_output_specs()
            output_specs = {
                name: cuvis_ai_pb2.TensorSpec(
                    name=spec.get("name", name),
                    shape=list(spec.get("shape", [])),
                    dtype=self._dtype_str_to_proto(spec.get("dtype")),
                    required=bool(spec.get("required", False)),
                )
                for name, spec in output_specs_dict.items()
            }

            return cuvis_ai_pb2.GetPipelineOutputsResponse(
                output_names=list(output_specs.keys()),
                output_specs=output_specs,
            )
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get outputs: {exc}")
            return cuvis_ai_pb2.GetPipelineOutputsResponse()

    def GetPipelineVisualization(
        self,
        request: cuvis_ai_pb2.GetPipelineVisualizationRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetPipelineVisualizationResponse:
        """Return a visualization of the session pipeline."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.GetPipelineVisualizationResponse()

        from cuvis_ai.pipeline.visualizer import PipelineVisualizer

        format_type = (request.format or "png").lower()
        visualizer = PipelineVisualizer(session.pipeline)

        try:
            if format_type in {"png", "svg"}:
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = Path(tmpdir) / f"pipeline.{format_type}"
                    rendered = visualizer.render_graphviz(
                        output_path=output_path, format=format_type
                    )
                    image_data = Path(rendered).read_bytes()
            elif format_type in {"dot", "graphviz"}:
                dot_source = visualizer.to_graphviz()
                image_data = dot_source.encode("utf-8")
            elif format_type in {"mermaid"}:
                mermaid_source = visualizer.to_mermaid()
                image_data = mermaid_source.encode("utf-8")
            else:
                raise ValueError(f"Unsupported visualization format: {format_type}")
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return cuvis_ai_pb2.GetPipelineVisualizationResponse()
        except Exception:
            # Fallback to a DOT string if rendering dependencies are unavailable
            dot_source = visualizer.to_graphviz()
            image_data = dot_source.encode("utf-8")

        return cuvis_ai_pb2.GetPipelineVisualizationResponse(
            image_data=image_data,
            format=format_type,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def Train(
        self,
        request: cuvis_ai_pb2.TrainRequest,
        context: grpc.ServicerContext,
    ) -> Iterator[cuvis_ai_pb2.TrainResponse]:
        """Train the pipeline with statistical or gradient methods.

        Args:
            request: TrainRequest with session ID, data config, and training config
            context: gRPC context

        Yields:
            TrainResponse messages with status and metrics
        """
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return

        # Get data config - either from request or from session's experiment config
        data_config_proto = None
        data_config_py = None

        if request.HasField("data"):
            # Use data config from request
            data_config_proto = request.data
            data_config_py = DataConfig.from_proto(request.data)
        elif session.data_config is not None:
            # Use data config from session (loaded via RestoreExperiment)
            data_config_py = session.data_config
            data_config_proto = data_config_py.to_proto()
        else:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                "Training requires data config to be provided either in request or loaded via RestoreExperiment"
            )
            return

        try:
            # Create datamodule from data config
            datamodule = self._create_single_cu3s_data_module(data_config_proto)
            training_config_py: TrainingConfig | None = None

            if request.trainer_type == cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL:
                # Statistical training - single pass, no streaming
                # Preserve existing training_config from session (if loaded via RestoreExperiment)
                # or create default if not present
                training_config_py = session.training_config or TrainingConfig()
                self._capture_experiment_context(session, data_config_py, training_config_py)
                yield from self._train_statistical(session, datamodule)

            elif request.trainer_type == cuvis_ai_pb2.TRAINER_TYPE_GRADIENT:
                # Gradient training - streaming progress
                # Get training config - either from request or from session
                if request.HasField("training"):
                    training_config_py = self._deserialize_training_config(request.training)
                elif session.training_config is not None:
                    training_config_py = session.training_config
                else:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        "Gradient training requires training config either in request or loaded via RestoreExperiment"
                    )
                    return

                self._capture_experiment_context(session, data_config_py, training_config_py)
                yield from self._train_gradient(
                    session,
                    datamodule,
                    data_config_proto,
                    training_config_py,
                )

            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Unknown trainer type: {request.trainer_type}")
                return

        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Training failed: {str(exc)}")
            return

    def GetTrainStatus(
        self,
        request: cuvis_ai_pb2.GetTrainStatusRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetTrainStatusResponse:
        """Get current training status.

        Args:
            request: GetTrainStatusRequest with session ID
            context: gRPC context

        Returns:
            GetTrainStatusResponse with current status
        """
        try:
            session = self.session_manager.get_session(request.session_id)

            # Simple status tracking (can be enhanced with async training in future)
            if session.trainer is None:
                status = cuvis_ai_pb2.TRAIN_STATUS_UNSPECIFIED
            else:
                status = cuvis_ai_pb2.TRAIN_STATUS_COMPLETE  # Simplified for Phase 5

            # Create a TrainResponse with the status
            latest_progress = cuvis_ai_pb2.TrainResponse(
                context=cuvis_ai_pb2.Context(
                    stage=cuvis_ai_pb2.EXECUTION_STAGE_TRAIN,
                    epoch=0,
                    batch_idx=0,
                    global_step=0,
                ),
                status=status,
                message="Training status query",
            )

            return cuvis_ai_pb2.GetTrainStatusResponse(latest_progress=latest_progress)

        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.GetTrainStatusResponse()
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get status: {str(exc)}")
            return cuvis_ai_pb2.GetTrainStatusResponse()

    # ------------------------------------------------------------------
    # Pipeline Management (Model Deployment)
    # ------------------------------------------------------------------
    def SavePipeline(
        self,
        request: cuvis_ai_pb2.SavePipelineRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.SavePipelineResponse:
        """Save trained pipeline (structure + weights) to disk."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.SavePipelineResponse(success=False)

        if not request.pipeline_path:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("pipeline_path is required")
            return cuvis_ai_pb2.SavePipelineResponse(success=False)

        try:
            from datetime import datetime

            from cuvis_ai.training.config import PipelineMetadata

            # Use resolve_pipeline_save_path for consistent path resolution
            pipeline_path = helpers.resolve_pipeline_save_path(request.pipeline_path)
            pipeline_path.parent.mkdir(parents=True, exist_ok=True)

            # Build metadata from proto or defaults
            metadata = PipelineMetadata(
                name=request.metadata.name if request.metadata.name else pipeline_path.stem,
                description=request.metadata.description if request.metadata.description else "",
                created=request.metadata.created
                if request.metadata.created
                else datetime.now().isoformat(),
                cuvis_ai_version=request.metadata.cuvis_ai_version
                if request.metadata.cuvis_ai_version
                else "0.1.5",
                tags=list(request.metadata.tags) if request.metadata.tags else [],
                author=request.metadata.author if request.metadata.author else "",
            )

            # Save pipeline using CuvisPipeline.save_to_file
            session.pipeline.save_to_file(
                str(pipeline_path),
                metadata=metadata,
            )

            # Compute weights path (save_to_file creates it as pipeline_path.with_suffix('.pt'))
            weights_path = pipeline_path.with_suffix(".pt")

            return cuvis_ai_pb2.SavePipelineResponse(
                success=True,
                pipeline_path=str(pipeline_path),
                weights_path=str(weights_path),
            )
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to save pipeline: {exc}")
            return cuvis_ai_pb2.SavePipelineResponse(success=False)

    def LoadPipeline(
        self,
        request: cuvis_ai_pb2.LoadPipelineRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.LoadPipelineResponse:
        """Load pipeline (structure + optionally weights) into existing session."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)

        # Use resolve_pipeline_path for consistent path resolution
        try:
            pipeline_path = helpers.resolve_pipeline_path(request.pipeline_path)
        except FileNotFoundError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)

        weights_path = None
        if request.HasField("weights_path") and request.weights_path:
            weights_path = request.weights_path

        try:
            from cuvis_ai.pipeline.pipeline import CuvisPipeline

            # Default strict to True if not specified (proto3 optional field)
            strict = request.strict if request.HasField("strict") else True

            pipeline = CuvisPipeline.load_from_file(
                str(pipeline_path),
                weights_path=weights_path,
                strict_weight_loading=strict,
            )

            # Update session pipeline
            session.pipeline = pipeline

            # Use pipeline metadata directly (loaded from YAML during load_from_file)
            metadata_proto = pipeline.metadata.to_proto()

            return cuvis_ai_pb2.LoadPipelineResponse(
                success=True,
                metadata=metadata_proto,
            )
        except FileNotFoundError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to load pipeline: {exc}")
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)

    # ------------------------------------------------------------------
    # Experiment Management (Reproducibility)
    # ------------------------------------------------------------------
    def SaveExperiment(
        self,
        request: cuvis_ai_pb2.SaveExperimentRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.SaveExperimentResponse:
        """Save complete experiment specification (pipeline + data + training configs)."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.SaveExperimentResponse(success=False)

        if not request.experiment_path:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("experiment_path is required")
            return cuvis_ai_pb2.SaveExperimentResponse(success=False)

        try:
            experiment_path = Path(request.experiment_path)
            experiment_path.parent.mkdir(parents=True, exist_ok=True)

            experiment_config = session.experiment_config
            if experiment_config is None:
                pipeline_config = session.pipeline_config

                experiment_dict: dict[str, Any] = {
                    "name": getattr(
                        session.pipeline, "name", pipeline_config.metadata.name or "experiment"
                    ),
                    "pipeline": pipeline_config.to_dict(),
                }
                if session.data_config is not None:
                    experiment_dict["data"] = session.data_config.to_dict()
                if session.training_config is not None:
                    experiment_dict["training"] = asdict(session.training_config)

                with experiment_path.open("w", encoding="utf-8") as f:
                    yaml.dump(experiment_dict, f, default_flow_style=False, sort_keys=False)
            else:
                # Save experiment config to YAML
                experiment_config.save_to_file(str(experiment_path))

            return cuvis_ai_pb2.SaveExperimentResponse(
                success=True,
                experiment_path=str(experiment_path),
            )
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to save experiment: {exc}")
            return cuvis_ai_pb2.SaveExperimentResponse(success=False)

    def RestoreExperiment(
        self,
        request: cuvis_ai_pb2.RestoreExperimentRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.RestoreExperimentResponse:
        """Restore experiment from YAML file and create new session."""
        experiment_path = Path(request.experiment_path)
        if not experiment_path.exists():
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Experiment file not found: {experiment_path}")
            return cuvis_ai_pb2.RestoreExperimentResponse()

        try:
            from cuvis_ai.pipeline.pipeline_builder import PipelineBuilder
            from cuvis_ai.training.config import ExperimentConfig

            # Use the same tested code path as the CLI examples
            experiment_config = ExperimentConfig.load_from_file(str(experiment_path))

            # Build pipeline from experiment config
            builder = PipelineBuilder()
            pipeline_dict = experiment_config.pipeline.to_dict()
            pipeline = builder.build_from_config(pipeline_dict)

            # Create session with components from loaded experiment config
            session_id = self.session_manager.create_session(
                pipeline=pipeline,
                data_config=experiment_config.data,
                training_config=experiment_config.training,
                experiment_config=experiment_config,
            )

            # Build response
            return cuvis_ai_pb2.RestoreExperimentResponse(
                session_id=session_id,
                experiment=experiment_config.to_proto(),
            )

        except FileNotFoundError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.RestoreExperimentResponse()
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to restore experiment: {exc}")
            return cuvis_ai_pb2.RestoreExperimentResponse()

    # ------------------------------------------------------------------
    # Pipeline Discovery
    # ------------------------------------------------------------------
    def ListAvailablePipelinees(
        self,
        request: cuvis_ai_pb2.ListAvailablePipelineesRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.ListAvailablePipelineesResponse:
        """List all available pipeline configurations."""
        try:
            filter_tag = request.filter_tag if request.HasField("filter_tag") else None

            pipelinees_list = helpers.list_available_pipelinees(filter_tag=filter_tag)

            pipeline_infos = []
            for pipeline_dict in pipelinees_list:
                metadata = pipeline_dict["metadata"]
                pipeline_info = cuvis_ai_pb2.PipelineInfo(
                    name=pipeline_dict["name"],
                    path=pipeline_dict["path"],
                    metadata=cuvis_ai_pb2.PipelineMetadata(
                        name=metadata["name"],
                        description=metadata["description"],
                        created=metadata["created"],
                        cuvis_ai_version=metadata["cuvis_ai_version"],
                        tags=metadata["tags"],
                        author=metadata["author"],
                    ),
                    tags=pipeline_dict["tags"],
                    has_weights=pipeline_dict["has_weights"],
                    weights_path=pipeline_dict["weights_path"],
                )
                pipeline_infos.append(pipeline_info)

            return cuvis_ai_pb2.ListAvailablePipelineesResponse(pipelinees=pipeline_infos)
        except FileNotFoundError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.ListAvailablePipelineesResponse()
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to list pipelinees: {exc}")
            return cuvis_ai_pb2.ListAvailablePipelineesResponse()

    def GetPipelineInfo(
        self,
        request: cuvis_ai_pb2.GetPipelineInfoRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetPipelineInfoResponse:
        """Get detailed information about a specific pipeline."""
        try:
            pipeline_dict = helpers.get_pipeline_info(
                pipeline_name=request.pipeline_name,
                include_yaml_content=True,
            )

            metadata = pipeline_dict["metadata"]
            pipeline_info = cuvis_ai_pb2.PipelineInfo(
                name=pipeline_dict["name"],
                path=pipeline_dict["path"],
                metadata=cuvis_ai_pb2.PipelineMetadata(
                    name=metadata["name"],
                    description=metadata["description"],
                    created=metadata["created"],
                    cuvis_ai_version=metadata["cuvis_ai_version"],
                    tags=metadata["tags"],
                    author=metadata["author"],
                ),
                tags=pipeline_dict["tags"],
                has_weights=pipeline_dict["has_weights"],
                weights_path=pipeline_dict["weights_path"],
                yaml_content=pipeline_dict.get("yaml_content", ""),
            )

            return cuvis_ai_pb2.GetPipelineInfoResponse(pipeline_info=pipeline_info)
        except FileNotFoundError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.GetPipelineInfoResponse()
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get pipeline info: {exc}")
            return cuvis_ai_pb2.GetPipelineInfoResponse()

    # ------------------------------------------------------------------
    # Training Capabilities & Validation
    # ------------------------------------------------------------------
    def GetTrainingCapabilities(
        self,
        request: cuvis_ai_pb2.GetTrainingCapabilitiesRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetTrainingCapabilitiesResponse:
        """Return supported optimizers, schedulers, and callbacks."""
        try:
            supported_optimizers = ["adam", "adamw", "sgd"]
            supported_schedulers = ["reduce_on_plateau"]

            callbacks = [
                cuvis_ai_pb2.CallbackTypeInfo(
                    type="EarlyStopping",
                    description="Stop training when a monitored metric stops improving.",
                    parameters=[
                        cuvis_ai_pb2.ParamSpec(
                            name="monitor",
                            type="string",
                            required=True,
                            description="Metric to monitor (e.g., 'val_loss').",
                        ),
                        cuvis_ai_pb2.ParamSpec(
                            name="patience",
                            type="int",
                            required=False,
                            default_value="10",
                            description="Number of epochs with no improvement.",
                        ),
                        cuvis_ai_pb2.ParamSpec(
                            name="mode",
                            type="string",
                            required=False,
                            default_value="min",
                            validation="in ['min', 'max']",
                            description="Optimization direction for monitored metric.",
                        ),
                    ],
                ),
                cuvis_ai_pb2.CallbackTypeInfo(
                    type="ModelCheckpoint",
                    description="Persist checkpoints during training.",
                    parameters=[
                        cuvis_ai_pb2.ParamSpec(
                            name="dirpath",
                            type="string",
                            required=True,
                            description="Directory to store checkpoints.",
                        ),
                        cuvis_ai_pb2.ParamSpec(
                            name="monitor",
                            type="string",
                            required=True,
                            description="Metric to monitor for best checkpoint.",
                        ),
                        cuvis_ai_pb2.ParamSpec(
                            name="mode",
                            type="string",
                            required=False,
                            default_value="max",
                            validation="in ['min', 'max']",
                            description="Optimization direction for checkpoint metric.",
                        ),
                        cuvis_ai_pb2.ParamSpec(
                            name="save_top_k",
                            type="int",
                            required=False,
                            default_value="1",
                            description="Number of best checkpoints to keep.",
                        ),
                    ],
                ),
                cuvis_ai_pb2.CallbackTypeInfo(
                    type="LearningRateMonitor",
                    description="Log learning rate during training.",
                    parameters=[
                        cuvis_ai_pb2.ParamSpec(
                            name="logging_interval",
                            type="string",
                            required=False,
                            default_value="epoch",
                            validation="in ['step', 'epoch']",
                            description="Frequency to log learning rate.",
                        ),
                        cuvis_ai_pb2.ParamSpec(
                            name="log_momentum",
                            type="bool",
                            required=False,
                            default_value="False",
                            description="Whether to log optimizer momentum.",
                        ),
                    ],
                ),
            ]

            optimizer_params = cuvis_ai_pb2.OptimizerParamsSchema(
                parameters=[
                    cuvis_ai_pb2.ParamSpec(
                        name="lr",
                        type="float",
                        required=True,
                        description="Learning rate.",
                    ),
                    cuvis_ai_pb2.ParamSpec(
                        name="weight_decay",
                        type="float",
                        required=False,
                        default_value="0.0",
                        description="Weight decay (L2 regularization).",
                    ),
                    cuvis_ai_pb2.ParamSpec(
                        name="betas",
                        type="tuple",
                        required=False,
                        description="Adam/AdamW betas (beta1, beta2).",
                    ),
                ]
            )

            scheduler_params = cuvis_ai_pb2.SchedulerParamsSchema(
                parameters=[
                    cuvis_ai_pb2.ParamSpec(
                        name="monitor",
                        type="string",
                        required=False,
                        description="Metric to monitor for scheduler decisions.",
                    ),
                    cuvis_ai_pb2.ParamSpec(
                        name="factor",
                        type="float",
                        required=False,
                        default_value="0.1",
                        description="LR reduction factor for ReduceLROnPlateau.",
                    ),
                    cuvis_ai_pb2.ParamSpec(
                        name="patience",
                        type="int",
                        required=False,
                        default_value="10",
                        description="Epochs to wait before reducing LR.",
                    ),
                ]
            )

            return cuvis_ai_pb2.GetTrainingCapabilitiesResponse(
                supported_optimizers=supported_optimizers,
                supported_schedulers=supported_schedulers,
                supported_callbacks=callbacks,
                optimizer_params=optimizer_params,
                scheduler_params=scheduler_params,
            )
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get capabilities: {exc}")
            return cuvis_ai_pb2.GetTrainingCapabilitiesResponse()

    def ValidateTrainingConfig(
        self,
        request: cuvis_ai_pb2.ValidateTrainingConfigRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.ValidateTrainingConfigResponse:
        """Validate training config payloads before submission."""
        try:
            config_json = request.config.config_bytes.decode("utf-8")
            training_config = TrainingConfig.from_json(config_json)
        except Exception as exc:
            return cuvis_ai_pb2.ValidateTrainingConfigResponse(
                valid=False, errors=[f"Failed to parse config: {exc}"]
            )

        errors: list[str] = []
        warnings: list[str] = []

        supported_optimizers = {"adam", "adamw", "sgd"}
        optimizer_name = training_config.optimizer.name.lower()
        if optimizer_name not in supported_optimizers:
            errors.append(
                f"Unsupported optimizer '{training_config.optimizer.name}'. "
                f"Supported: {', '.join(sorted(supported_optimizers))}"
            )

        if training_config.optimizer.lr <= 0:
            errors.append(f"Learning rate must be > 0, got {training_config.optimizer.lr}")
        if training_config.optimizer.lr > 1.0:
            warnings.append(f"Learning rate {training_config.optimizer.lr} is unusually high")

        if training_config.trainer.max_epochs <= 0:
            errors.append(f"max_epochs must be > 0, got {training_config.trainer.max_epochs}")

        scheduler = training_config.optimizer.scheduler
        if scheduler is not None and scheduler.name.lower() != "reduce_on_plateau":
            errors.append(f"Unsupported scheduler '{scheduler.name}'. Supported: reduce_on_plateau")

        return cuvis_ai_pb2.ValidateTrainingConfigResponse(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _capture_experiment_context(
        self,
        session: SessionState,
        data_config: Any,
        training_config: TrainingConfig,
    ) -> None:
        """Persist experiment context on the session for SaveExperiment."""
        pipeline_config = session.pipeline_config

        pipeline_name = getattr(
            session.pipeline, "name", pipeline_config.metadata.name or "experiment"
        )

        session.data_config = data_config
        session.training_config = training_config

        # Preserve loss_nodes, metric_nodes, and unfreeze_nodes from existing experiment_config if available
        loss_nodes = []
        metric_nodes = []
        unfreeze_nodes = []

        if session.experiment_config is not None:
            loss_nodes = session.experiment_config.loss_nodes
            metric_nodes = session.experiment_config.metric_nodes
            unfreeze_nodes = session.experiment_config.unfreeze_nodes

        session.experiment_config = ExperimentConfig(
            name=str(pipeline_name),
            pipeline=pipeline_config,
            data=data_config,
            training=training_config,
            loss_nodes=loss_nodes,
            metric_nodes=metric_nodes,
            unfreeze_nodes=unfreeze_nodes,
        )

    def _create_single_cu3s_data_module(
        self,
        data_config: cuvis_ai_pb2.DataConfig,
    ) -> SingleCu3sDataModule:
        """Create SingleCu3sDataModule from proto config.

        Args:
            data_config: Proto DataConfig message

        Returns:
            SingleCu3sDataModule instance
        """
        # Map proto ProcessingMode to string
        processing_mode_map = {
            cuvis_ai_pb2.PROCESSING_MODE_RAW: "Raw",
            cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE: "Reflectance",
        }

        processing_mode = processing_mode_map.get(data_config.processing_mode, "Reflectance")

        # Handle optional annotation_json_path (may be empty string or not set)
        annotation_json_path = (
            data_config.annotation_json_path if data_config.annotation_json_path else None
        )

        # Use explicit paths from proto config (takes precedence)
        return SingleCu3sDataModule(
            cu3s_file_path=data_config.cu3s_file_path,
            annotation_json_path=annotation_json_path,
            train_ids=list(data_config.train_ids),
            val_ids=list(data_config.val_ids),
            test_ids=list(data_config.test_ids),
            batch_size=data_config.batch_size,
            processing_mode=processing_mode,
        )

    def _train_statistical(
        self,
        session: SessionState,
        datamodule: SingleCu3sDataModule,
    ) -> Iterator[cuvis_ai_pb2.TrainResponse]:
        """Train with statistical method.

        Args:
            session: Session state
            datamodule: Data module for training

        Yields:
            TrainResponse messages
        """
        # Yield initial status
        yield cuvis_ai_pb2.TrainResponse(
            context=cuvis_ai_pb2.Context(
                stage=cuvis_ai_pb2.EXECUTION_STAGE_TRAIN,
                epoch=0,
                batch_idx=0,
                global_step=0,
            ),
            status=cuvis_ai_pb2.TRAIN_STATUS_RUNNING,
            message="Starting statistical training",
        )

        # Create and run statistical trainer
        trainer = StatisticalTrainer(
            pipeline=session.pipeline,
            datamodule=datamodule,
        )

        # Fit the pipeline (initializes normalizers, selectors, PCA, RX)
        trainer.fit()

        # Store trainer in session for potential later use
        session.trainer = trainer

        # Yield completion
        yield cuvis_ai_pb2.TrainResponse(
            context=cuvis_ai_pb2.Context(
                stage=cuvis_ai_pb2.EXECUTION_STAGE_TRAIN,
                epoch=1,
                batch_idx=0,
                global_step=1,
            ),
            status=cuvis_ai_pb2.TRAIN_STATUS_COMPLETE,
            message="Statistical training complete",
        )

    def _train_gradient(
        self,
        session: SessionState,
        datamodule: SingleCu3sDataModule,
        data_config: cuvis_ai_pb2.DataConfig,
        training_config: TrainingConfig,
    ) -> Iterator[cuvis_ai_pb2.TrainResponse]:
        """Train with gradient-based method and stream progress."""
        loss_nodes, metric_nodes = self._configure_gradient_components(
            session, data_config, training_config
        )

        progress_queue: queue.Queue[cuvis_ai_pb2.TrainResponse] = queue.Queue()

        def progress_handler(
            context_obj: Context, losses: dict, metrics: dict, status: str
        ) -> None:
            progress_queue.put(
                self._create_progress_response(
                    context_obj,
                    losses,
                    metrics,
                    status=status,
                    message="Gradient training",
                )
            )

        callback_list = [ProgressStreamCallback(progress_handler)]
        callback_list.extend(create_callbacks_from_config(training_config.trainer.callbacks))

        trainer = GradientTrainer(
            pipeline=session.pipeline,
            datamodule=datamodule,
            trainer_config=training_config.trainer,
            optimizer_config=training_config.optimizer,
            loss_nodes=loss_nodes,
            metric_nodes=metric_nodes,
            callbacks=callback_list,
        )
        session.trainer = trainer

        training_complete = threading.Event()
        training_error: Exception | None = None

        def _run_training() -> None:
            nonlocal training_error
            try:
                trainer.fit()
            except Exception as exc:  # pragma: no cover - surfaced via progress stream
                training_error = exc
            finally:
                training_complete.set()

        thread = threading.Thread(target=_run_training, daemon=True)
        thread.start()

        while not training_complete.is_set() or not progress_queue.empty():
            try:
                progress = progress_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            else:
                yield progress

        if training_error is not None:
            raise training_error

        final_context = Context(
            stage=ExecutionStage.TRAIN,
            epoch=training_config.trainer.max_epochs,
            batch_idx=0,
            global_step=getattr(getattr(trainer, "trainer", None), "global_step", 0),
        )
        yield self._create_progress_response(
            final_context,
            losses={},
            metrics={},
            status="complete",
            message="Gradient training complete",
        )

    def _parse_pipeline_config(self, raw_config: str) -> dict:
        if not raw_config:
            return {}
        try:
            parsed = json.loads(raw_config)
            if not isinstance(parsed, dict):
                raise ValueError("pipeline_config must decode to a JSON object")
            return parsed
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid pipeline_config JSON: {exc.msg}") from exc

    def _parse_input_batch(self, inputs: cuvis_ai_pb2.InputBatch) -> dict[str, Any]:
        """Convert InputBatch proto to dict for CuvisPipeline.

        Converts proto messages to Python types. The pipeline determines
        which inputs are required and validates shapes/types.
        """
        batch: dict[str, Any] = {}

        # Parse tensor inputs (if provided)
        if inputs.HasField("cube"):
            batch["cube"] = helpers.proto_to_tensor(inputs.cube)

        if inputs.HasField("wavelengths"):
            batch["wavelengths"] = helpers.proto_to_tensor(inputs.wavelengths)

        if inputs.HasField("mask"):
            batch["mask"] = helpers.proto_to_tensor(inputs.mask)

        # Parse structured inputs (if provided)
        if inputs.HasField("bboxes"):
            batch["bboxes"] = self._parse_bounding_boxes(inputs.bboxes)

        if inputs.HasField("points"):
            batch["points"] = self._parse_points(inputs.points)

        if inputs.text_prompt:
            batch["text_prompt"] = inputs.text_prompt

        return batch

    def _parse_bounding_boxes(self, bboxes_proto: cuvis_ai_pb2.BoundingBoxes) -> list[dict]:
        """Parse bounding boxes from proto into dictionaries."""
        return [
            {
                "element_id": box.element_id,
                "x_min": box.x_min,
                "y_min": box.y_min,
                "x_max": box.x_max,
                "y_max": box.y_max,
            }
            for box in bboxes_proto.boxes
        ]

    def _parse_points(self, points_proto: cuvis_ai_pb2.Points) -> list[dict]:
        """Parse points from proto into dictionaries."""
        return [
            {
                "element_id": point.element_id,
                "x": point.x,
                "y": point.y,
                "type": (
                    "neutral"
                    if point.type == cuvis_ai_pb2.POINT_TYPE_UNSPECIFIED
                    else helpers.point_type_to_string(point.type).lower()
                ),
            }
            for point in points_proto.points
        ]

    def _format_output_key(self, key: Any) -> str:
        """Normalize pipeline output keys (tuple -> 'node.port')."""
        if isinstance(key, tuple) and len(key) == 2:
            node_name, port = key
            return f"{node_name}.{port}"
        return str(key)

    def _should_return(self, output_name: str, specs: set[str]) -> bool:
        if not specs:
            return True
        port_name = output_name.split(".", maxsplit=1)[-1]
        return output_name in specs or port_name in specs

    def _to_tensor(self, value: Any) -> torch.Tensor:
        """Coerce supported outputs to torch.Tensor."""
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        if isinstance(value, Iterable):
            return torch.tensor(list(value))
        # Last resort for scalars (avoid bool/metric routing)
        return torch.tensor(value)

    def _dtype_str_to_proto(self, dtype_str: str | None) -> int:
        """Convert dtype strings produced by pipeline introspection to proto enums."""
        dtype_map = {
            "float32": cuvis_ai_pb2.D_TYPE_FLOAT32,
            "float64": cuvis_ai_pb2.D_TYPE_FLOAT64,
            "int32": cuvis_ai_pb2.D_TYPE_INT32,
            "int64": cuvis_ai_pb2.D_TYPE_INT64,
            "uint8": cuvis_ai_pb2.D_TYPE_UINT8,
            "bool": cuvis_ai_pb2.D_TYPE_BOOL,
            "float16": cuvis_ai_pb2.D_TYPE_FLOAT16,
        }
        return dtype_map.get((dtype_str or "").lower(), cuvis_ai_pb2.D_TYPE_UNSPECIFIED)

    # ------------------------------------------------------------------
    # Gradient training helpers
    # ------------------------------------------------------------------
    def _deserialize_training_config(
        self, config_proto: cuvis_ai_pb2.TrainingConfig
    ) -> TrainingConfig:
        """Decode TrainingConfig from proto bytes."""
        try:
            config_json = config_proto.config_bytes.decode("utf-8")
        except Exception as exc:
            raise ValueError("Failed to decode training config") from exc

        if not config_json:
            raise ValueError("Training config cannot be empty")

        try:
            return TrainingConfig.from_json(config_json)
        except Exception as exc:
            raise ValueError(f"Invalid training config: {exc}") from exc

    def _configure_gradient_components(
        self,
        session: SessionState,
        data_config: cuvis_ai_pb2.DataConfig,
        training_config: TrainingConfig,
    ) -> tuple[list, list]:
        """Configure loss and metric nodes from experiment config.

        Uses explicit node names from experiment_config.loss_nodes and
        experiment_config.metric_nodes. No automatic discovery or hardcoded types.

        Args:
            session: Session state containing pipeline and experiment config
            data_config: Data configuration (unused, kept for API compatibility)
            training_config: Training configuration (unused, kept for API compatibility)

        Returns:
            Tuple of (loss_nodes, metric_nodes) lists

        Raises:
            ValueError: If experiment config is missing or invalid
        """
        pipeline = session.pipeline

        # Require explicit experiment config
        if session.experiment_config is None:
            raise ValueError(
                "Gradient training requires explicit ExperimentConfig with loss_nodes and metric_nodes. "
                "Please provide experiment configuration via RestoreExperiment or session creation."
            )

        experiment_config = session.experiment_config

        # Validate required fields
        if not experiment_config.loss_nodes:
            raise ValueError(
                "experiment_config.loss_nodes must specify at least one loss node for gradient training. "
                "Add loss node names to your experiment config YAML."
            )

        # Build node lookup map
        node_map = {node.name: node for node in pipeline.nodes()}

        # Look up loss nodes by name
        loss_nodes = []
        for loss_name in experiment_config.loss_nodes:
            if loss_name not in node_map:
                raise ValueError(
                    f"Loss node '{loss_name}' not found in pipeline. "
                    f"Available nodes: {', '.join(sorted(node_map.keys()))}"
                )
            loss_nodes.append(node_map[loss_name])

        # Look up metric nodes by name
        metric_nodes = []
        for metric_name in experiment_config.metric_nodes:
            if metric_name not in node_map:
                raise ValueError(
                    f"Metric node '{metric_name}' not found in pipeline. "
                    f"Available nodes: {', '.join(sorted(node_map.keys()))}"
                )
            metric_nodes.append(node_map[metric_name])

        # Handle unfreeze_nodes from experiment config
        if experiment_config.unfreeze_nodes:
            pipeline.unfreeze_nodes_by_name(list(experiment_config.unfreeze_nodes))

        # Validate we have trainable parameters
        has_trainable = any(p.requires_grad for p in pipeline.parameters())
        if not has_trainable:
            raise ValueError(
                "No trainable parameters found after unfreezing. "
                "Configure unfreeze_nodes in experiment config to include trainable nodes."
            )

        return loss_nodes, metric_nodes

    def _create_progress_response(
        self,
        context_obj: Context,
        losses: dict,
        metrics: dict,
        status: str,
        message: str = "",
    ) -> cuvis_ai_pb2.TrainResponse:
        """Map internal progress to proto TrainResponse."""
        stage_map = {
            ExecutionStage.TRAIN: cuvis_ai_pb2.EXECUTION_STAGE_TRAIN,
            ExecutionStage.VAL: cuvis_ai_pb2.EXECUTION_STAGE_VAL,
            ExecutionStage.TEST: cuvis_ai_pb2.EXECUTION_STAGE_TEST,
            ExecutionStage.INFERENCE: cuvis_ai_pb2.EXECUTION_STAGE_INFERENCE,
        }
        status_map = {
            "running": cuvis_ai_pb2.TRAIN_STATUS_RUNNING,
            "complete": cuvis_ai_pb2.TRAIN_STATUS_COMPLETE,
            "error": cuvis_ai_pb2.TRAIN_STATUS_ERROR,
        }

        return cuvis_ai_pb2.TrainResponse(
            context=cuvis_ai_pb2.Context(
                stage=stage_map.get(context_obj.stage, cuvis_ai_pb2.EXECUTION_STAGE_TRAIN),
                epoch=context_obj.epoch,
                batch_idx=context_obj.batch_idx,
                global_step=context_obj.global_step,
            ),
            losses={k: float(v) for k, v in (losses or {}).items()},
            metrics={k: float(v) for k, v in (metrics or {}).items()},
            status=status_map.get(status, cuvis_ai_pb2.TRAIN_STATUS_RUNNING),
            message=message,
        )


__all__ = ["CuvisAIService"]
