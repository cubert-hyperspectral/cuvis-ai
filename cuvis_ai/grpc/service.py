"""gRPC service implementation for basic inference, introspection, and sessions."""

from __future__ import annotations

import json
import queue
import tempfile
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

import grpc
from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule
from cuvis_ai.grpc.callbacks import ProgressStreamCallback
from cuvis_ai.training.config import TrainingConfig, create_callbacks_from_config
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
        """Create a new pipeline session."""
        try:
            pipeline_config = self._parse_pipeline_config(request.pipeline_config)
            # data_config is optional - only pass if provided
            data_config = request.data_config if request.HasField("data_config") else None
            session_id = self.session_manager.create_session(
                pipeline_type=request.pipeline_type,
                pipeline_config=pipeline_config,
                data_config=data_config,
            )
            return cuvis_ai_pb2.CreateSessionResponse(session_id=session_id)
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
            outputs = session.canvas.forward(batch=batch, stage=ExecutionStage.INFERENCE)
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

            tensor = self._to_tensor(value)
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
            input_specs_dict = session.canvas.get_input_specs()
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
            output_specs_dict = session.canvas.get_output_specs()
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

        from cuvis_ai.pipeline.visualizer import CanvasVisualizer

        format_type = (request.format or "png").lower()
        visualizer = CanvasVisualizer(session.canvas)

        try:
            if format_type in {"png", "svg"}:
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = Path(tmpdir) / f"canvas.{format_type}"
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
            request: TrainRequest with session ID and trainer type
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

        # Validate that data_config is present for training
        if session.data_config is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Training requires data_config to be set during session creation")
            return

        try:
            # Create datamodule from session's data config
            datamodule = self._create_single_cu3s_data_module(session.data_config)

            if request.trainer_type == cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL:
                # Statistical training - single pass, no streaming
                yield from self._train_statistical(session, datamodule)

            elif request.trainer_type == cuvis_ai_pb2.TRAINER_TYPE_GRADIENT:
                # Gradient training - streaming progress (Phase 6)
                if not request.HasField("config"):
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("Gradient training requires config")
                    return

                yield from self._train_gradient(session, datamodule, request.config)

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

            return cuvis_ai_pb2.GetTrainStatusResponse(status=status)

        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.GetTrainStatusResponse()
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get status: {str(exc)}")
            return cuvis_ai_pb2.GetTrainStatusResponse()

    # ------------------------------------------------------------------
    # Checkpoint Management
    # ------------------------------------------------------------------
    def SaveCheckpoint(
        self,
        request: cuvis_ai_pb2.SaveCheckpointRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.SaveCheckpointResponse:
        """Persist the current trainer or canvas state to disk."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.SaveCheckpointResponse(success=False)

        if not request.checkpoint_path:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("checkpoint_path is required")
            return cuvis_ai_pb2.SaveCheckpointResponse(success=False)

        if session.trainer is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("No trainer available. Train the model first.")
            return cuvis_ai_pb2.SaveCheckpointResponse(success=False)

        try:
            checkpoint_path = Path(request.checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            if hasattr(session.trainer, "save_checkpoint"):
                session.trainer.save_checkpoint(str(checkpoint_path))
            elif hasattr(session.trainer, "trainer") and hasattr(
                session.trainer.trainer, "save_checkpoint"
            ):
                session.trainer.trainer.save_checkpoint(str(checkpoint_path))
            else:
                canvas_state_fn = getattr(session.canvas, "state_dict", None)
                canvas_state = canvas_state_fn() if callable(canvas_state_fn) else {}
                torch.save(
                    {
                        "pipeline_type": session.pipeline_type,
                        "pipeline_config": session.pipeline_config,
                        "canvas_state": canvas_state,
                    },
                    checkpoint_path,
                )

            return cuvis_ai_pb2.SaveCheckpointResponse(success=True)
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to save checkpoint: {exc}")
            return cuvis_ai_pb2.SaveCheckpointResponse(success=False)

    def LoadCheckpoint(
        self,
        request: cuvis_ai_pb2.LoadCheckpointRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.LoadCheckpointResponse:
        """Load a checkpoint into an existing session."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.LoadCheckpointResponse(success=False)

        checkpoint_path = Path(request.checkpoint_path)
        if not checkpoint_path.exists():
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Checkpoint not found: {checkpoint_path}")
            return cuvis_ai_pb2.LoadCheckpointResponse(success=False)

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            if hasattr(session.trainer, "load_state_dict") and isinstance(checkpoint, dict):
                state_dict = checkpoint.get("state_dict") or checkpoint
                session.trainer.load_state_dict(state_dict, strict=False)  # type: ignore[arg-type]
            elif hasattr(session.trainer, "trainer") and hasattr(
                session.trainer.trainer, "strategy"
            ):
                # Lightning checkpoints can be restored via strategy helper
                session.trainer.trainer.strategy.load_checkpoint(checkpoint_path)  # type: ignore[call-arg]
            elif isinstance(checkpoint, dict) and "canvas_state" in checkpoint:
                load_fn = getattr(session.canvas, "load_state_dict", None)
                if callable(load_fn):
                    load_fn(checkpoint["canvas_state"])

            return cuvis_ai_pb2.LoadCheckpointResponse(success=True)
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to load checkpoint: {exc}")
            return cuvis_ai_pb2.LoadCheckpointResponse(success=False)

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
            config_json = request.config.config_json.decode("utf-8")
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

        # Use explicit paths from proto config (takes precedence)
        return SingleCu3sDataModule(
            cu3s_path=data_config.cu3s_file_path,
            label_path=data_config.annotation_json_path,
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
            canvas=session.canvas,
            datamodule=datamodule,
        )

        # Fit the canvas (initializes normalizers, selectors, PCA, RX)
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
        config_proto: cuvis_ai_pb2.TrainingConfig,
    ) -> Iterator[cuvis_ai_pb2.TrainResponse]:
        """Train with gradient-based method and stream progress."""
        training_config = self._deserialize_training_config(config_proto)

        # data_config should be validated by caller, but add safety check
        if session.data_config is None:
            raise ValueError("data_config is required for gradient training")
        
        loss_nodes, metric_nodes = self._configure_gradient_components(session, session.data_config)

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
            canvas=session.canvas,
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
        """Convert InputBatch proto to a dict consumable by CuvisCanvas."""
        if not inputs.HasField("cube"):
            raise ValueError("inputs.cube is required for inference")

        batch: dict[str, Any] = {"cube": helpers.proto_to_tensor(inputs.cube)}

        if inputs.HasField("wavelengths"):
            batch["wavelengths"] = helpers.proto_to_tensor(inputs.wavelengths)
        if inputs.HasField("mask"):
            batch["mask"] = helpers.proto_to_tensor(inputs.mask)

        if inputs.HasField("bboxes"):
            batch["bboxes"] = self._parse_bounding_boxes(inputs.bboxes)
        if inputs.HasField("points"):
            batch["points"] = self._parse_points(inputs.points)
        if inputs.text_prompt:
            batch["text_prompt"] = inputs.text_prompt

        for key, tensor_proto in inputs.extra_inputs.items():
            batch[key] = helpers.proto_to_tensor(tensor_proto)

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
        """Normalize canvas output keys (tuple -> 'node.port')."""
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
        """Convert dtype strings produced by canvas introspection to proto enums."""
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
            config_json = config_proto.config_json.decode("utf-8")
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
    ) -> tuple[list, list]:
        """Attach default loss/metric nodes for gradient training."""
        from cuvis_ai.anomaly.rx_logit_head import RXLogitHead
        from cuvis_ai.deciders.binary_decider import BinaryDecider
        from cuvis_ai.node.data import LentilsAnomalyDataNode
        from cuvis_ai.node.losses import AnomalyBCEWithLogits, SelectorEntropyRegularizer
        from cuvis_ai.node.metrics import AnomalyDetectionMetrics
        from cuvis_ai.node.selector import SoftChannelSelector

        canvas = session.canvas
        nodes = list(canvas.nodes())  # type: ignore[arg-type]

        def _first_of(cls) -> Any | None:
            for node in nodes:
                if isinstance(node, cls):
                    return node
            return None

        selector = _first_of(SoftChannelSelector)
        logit_head = _first_of(RXLogitHead)
        decider = _first_of(BinaryDecider)
        data_node = _first_of(LentilsAnomalyDataNode)

        if selector is not None:
            selector.unfreeze()
        if logit_head is not None:
            logit_head.unfreeze()

        loss_nodes: list = []
        metric_nodes: list = []

        # Selector regularization
        if selector is not None:
            entropy_loss = _first_of(SelectorEntropyRegularizer)
            if entropy_loss is None:
                entropy_loss = SelectorEntropyRegularizer(
                    name="selector_entropy",
                    execution_stages={ExecutionStage.TRAIN},
                )
                canvas.connect(selector.outputs.weights, entropy_loss.inputs.weights)
                nodes.append(entropy_loss)
            loss_nodes.append(entropy_loss)

        # BCE loss if labels are available
        has_labels = bool(getattr(data_config, "annotation_json_path", ""))
        if logit_head is not None and data_node is not None and has_labels:
            bce_loss = _first_of(AnomalyBCEWithLogits)
            if bce_loss is None:
                bce_loss = AnomalyBCEWithLogits(
                    name="bce_loss",
                    execution_stages={ExecutionStage.TRAIN, ExecutionStage.VAL},
                )
                canvas.connect(
                    (logit_head.outputs.logits, bce_loss.inputs.predictions),
                    (data_node.outputs.mask, bce_loss.inputs.targets),
                )
                nodes.append(bce_loss)
            loss_nodes.append(bce_loss)

        # Anomaly detection metrics for val/test
        if decider is not None and data_node is not None and has_labels:
            metrics_node = _first_of(AnomalyDetectionMetrics)
            if metrics_node is None:
                metrics_node = AnomalyDetectionMetrics(
                    name="anomaly_metrics",
                    execution_stages={ExecutionStage.VAL, ExecutionStage.TEST},
                )
                canvas.connect(
                    (decider.outputs.decisions, metrics_node.inputs.decisions),
                    (data_node.outputs.mask, metrics_node.inputs.targets),
                )
                nodes.append(metrics_node)
            metric_nodes.append(metrics_node)

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
