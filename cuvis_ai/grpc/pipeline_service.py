"""Pipeline construction and management service component."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import grpc
import torch
from pydantic import ValidationError

from cuvis_ai.training.config import TrainRunConfig

from . import helpers
from .session_manager import SessionManager
from .v1 import cuvis_ai_pb2


class PipelineService:
    """Pipeline construction, weight loading, and persistence."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    def load_pipeline_weights(
        self,
        request: cuvis_ai_pb2.LoadPipelineWeightsRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.LoadPipelineWeightsResponse:
        """Load weights into an existing pipeline (path or raw bytes)."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.LoadPipelineWeightsResponse(success=False)

        if session.pipeline is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("No pipeline is available for this session. Build pipeline first.")
            return cuvis_ai_pb2.LoadPipelineWeightsResponse(success=False)

        strict = request.strict if request.HasField("strict") else True

        try:
            resolved_path = ""

            if request.HasField("weights_path") and request.weights_path:
                resolved = helpers.find_weights_file(request.weights_path, session.search_paths)
                session.pipeline._restore_weights_from_checkpoint(
                    weights_path=str(resolved),
                    strict_weight_loading=strict,
                )
                resolved_path = str(resolved)
            elif request.HasField("weights_bytes") and request.weights_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                    tmp.write(request.weights_bytes)
                    tmp_path = Path(tmp.name)

                try:
                    session.pipeline._restore_weights_from_checkpoint(
                        weights_path=str(tmp_path),
                        strict_weight_loading=strict,
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)
                resolved_path = "(client-bytes)"
            else:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Provide either weights_path or weights_bytes")
                return cuvis_ai_pb2.LoadPipelineWeightsResponse(success=False)

            return cuvis_ai_pb2.LoadPipelineWeightsResponse(
                success=True, resolved_path=resolved_path
            )
        except FileNotFoundError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.LoadPipelineWeightsResponse(success=False)
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to load weights: {exc}")
            return cuvis_ai_pb2.LoadPipelineWeightsResponse(success=False)

    def set_train_run_config(
        self,
        request: cuvis_ai_pb2.SetTrainRunConfigRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.SetTrainRunConfigResponse:
        """Persist trainrun configuration and apply pipeline precedence logic."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.SetTrainRunConfigResponse(success=False)

        if not request.config.config_bytes:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("trainrun config_bytes is required")
            return cuvis_ai_pb2.SetTrainRunConfigResponse(success=False)

        try:
            trainrun_config = TrainRunConfig.from_proto(request.config)

            if session.pipeline is not None:
                # If a pipeline already exists, only error when the trainrun also supplies one.
                if trainrun_config.pipeline is not None:
                    context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                    context.set_details(
                        "Pipeline already exists. Either remove the pipeline from the session "
                        "manually or don't pass a pipeline in the trainrun config."
                    )
                    return cuvis_ai_pb2.SetTrainRunConfigResponse(success=False)
            else:
                # No pipeline in session; require trainrun to supply it and build.
                if trainrun_config.pipeline is None:
                    context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                    context.set_details(
                        "No pipeline exists and no pipeline configuration provided in trainrun "
                        "config."
                    )
                    return cuvis_ai_pb2.SetTrainRunConfigResponse(success=False)

                from cuvis_ai.pipeline.factory import PipelineBuilder

                pipeline = PipelineBuilder().build_from_config(
                    trainrun_config.pipeline.model_dump()
                )
                # Move pipeline to GPU if available
                if torch.cuda.is_available():
                    pipeline = pipeline.to("cuda")
                session.pipeline = pipeline
                session.pipeline_config = trainrun_config.pipeline

            # Set session configurations
            session.data_config = trainrun_config.data
            session.training_config = trainrun_config.training
            session.trainrun_config = trainrun_config

            return cuvis_ai_pb2.SetTrainRunConfigResponse(success=True, pipeline_from_config=True)
        except (ValidationError, ValueError) as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return cuvis_ai_pb2.SetTrainRunConfigResponse(success=False)
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to set train run config: {exc}")
            return cuvis_ai_pb2.SetTrainRunConfigResponse(success=False)

    def save_pipeline(
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

    def load_pipeline(
        self,
        request: cuvis_ai_pb2.LoadPipelineRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.LoadPipelineResponse:
        """Load pipeline structure from resolved config bytes.

        Expected workflow:
        1) ResolveConfig to produce JSON config bytes
        2) LoadPipeline to build pipeline structure
        3) LoadPipelineWeights to load weights (optional, explicit)
        """
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)

        if not request.pipeline or not request.pipeline.config_bytes:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("pipeline.config_bytes is required")
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)

        try:
            from cuvis_ai.pipeline.factory import PipelineBuilder
            from cuvis_ai.training.config import PipelineConfig

            config_dict = json.loads(request.pipeline.config_bytes)
            # Some YAML sources include a top-level version field that is not part of the schema.
            config_dict.pop("version", None)
            pipeline_config = PipelineConfig(**config_dict)

            pipeline = PipelineBuilder().build_from_config(pipeline_config.model_dump())
            # Move pipeline to GPU if available
            if torch.cuda.is_available():
                pipeline = pipeline.to("cuda")

            session.pipeline = pipeline
            session.pipeline_config = pipeline_config

            metadata_proto = (
                pipeline_config.metadata.to_proto()
                if getattr(pipeline_config, "metadata", None)
                else cuvis_ai_pb2.PipelineMetadata()
            )

            return cuvis_ai_pb2.LoadPipelineResponse(success=True, metadata=metadata_proto)
        except json.JSONDecodeError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Invalid JSON in pipeline.config_bytes: {exc}")
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)
        except ValidationError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Invalid pipeline config: {exc}")
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to load pipeline: {exc}")
            return cuvis_ai_pb2.LoadPipelineResponse(success=False)


__all__ = ["PipelineService"]
