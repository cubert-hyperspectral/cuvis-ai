"""Inference service component."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import grpc
import numpy as np
import torch

from cuvis_ai.utils.types import ExecutionStage

from . import helpers
from .session_manager import SessionManager
from .v1 import cuvis_ai_pb2


class InferenceService:
    """Inference operations for Cuvis AI."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    def inference(
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

        # Check if pipeline exists
        if session.pipeline is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("No pipeline is available for this session. Build pipeline first.")
            return cuvis_ai_pb2.InferenceResponse()

        try:
            batch = self._parse_input_batch(request.inputs)
            # Ensure all tensor inputs are on the same device as the pipeline
            batch = self._move_batch_to_pipeline_device(batch, session.pipeline)
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

    def _move_batch_to_pipeline_device(
        self,
        batch: dict[str, Any],
        pipeline,
    ) -> dict[str, Any]:
        """Move tensor inputs to the same device as the pipeline.

        This mirrors the Python API pattern where the pipeline is moved to a device
        (e.g., via ``pipeline.to('cuda')``) and dataloader batches are produced on
        that device. Here, gRPC deserialization always yields CPU tensors, so we
        align them with the pipeline device before forwarding.

        Uses the same robust device detection pattern as StatisticalTrainer,
        iterating through all nodes to find one with parameters or buffers.
        """
        if pipeline is None:
            return batch

        device = self._get_pipeline_device(pipeline)

        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value

        return moved

    def _get_pipeline_device(self, pipeline) -> torch.device:
        """Get the device of the pipeline from its parameters/buffers.

        Iterates through all nodes in the pipeline to find one with
        parameters or buffers, returning its device. Falls back to CPU
        if no device information is found.

        This follows the same pattern as StatisticalTrainer._get_pipeline_device
        to ensure consistent device detection across the codebase.
        """
        # Iterate through all torch-backed layers (not just the first one)
        for layer in pipeline.torch_layers:
            # Check parameters first (preferred for device detection)
            for param in layer.parameters():
                return param.device
            # Fall back to buffers if no parameters
            for buf in layer.buffers():
                return buf.device
        # Explicit fallback to CPU if no device information found
        return torch.device("cpu")

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


__all__ = ["InferenceService"]
