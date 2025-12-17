"""Pipeline introspection and visualization service component."""

from __future__ import annotations

import tempfile
from pathlib import Path

import grpc

from .session_manager import SessionManager
from .v1 import cuvis_ai_pb2


class IntrospectionService:
    """Pipeline introspection helpers."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    def get_pipeline_inputs(
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

    def get_pipeline_outputs(
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

    def get_pipeline_visualization(
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


__all__ = ["IntrospectionService"]
