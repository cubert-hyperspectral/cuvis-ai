"""Pipeline discovery service component."""

from __future__ import annotations

import grpc

from . import helpers
from .v1 import cuvis_ai_pb2


class DiscoveryService:
    """Discover available pipelines."""

    def list_available_pipelinees(
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

    def get_pipeline_info(
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


__all__ = ["DiscoveryService"]
