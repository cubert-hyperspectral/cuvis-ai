"""Training operations service component."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import grpc

from cuvis_ai.grpc.callbacks import ProgressStreamCallback
from cuvis_ai.training.config import (
    DataConfig,
    TrainingConfig,
    TrainRunConfig,
    create_callbacks_from_config,
)
from cuvis_ai.training.trainers import GradientTrainer, StatisticalTrainer
from cuvis_ai.utils.types import Context, ExecutionStage

from .session_manager import SessionManager, SessionState
from .v1 import cuvis_ai_pb2

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule


class TrainingService:
    """Handles statistical and gradient training operations."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    def train(
        self,
        request: cuvis_ai_pb2.TrainRequest,
        context: grpc.ServicerContext,
    ) -> Iterator[cuvis_ai_pb2.TrainResponse]:
        """Train the pipeline with statistical or gradient methods."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return

        if session.pipeline is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(
                "No pipeline to train. Build pipeline or set trainrun config first."
            )
            return

        # Get data config - either from request or from session's experiment config
        data_config_py: DataConfig | None = None

        if request.HasField("data"):
            # Use data config from request
            data_config_py = DataConfig.from_proto(request.data)
        elif session.data_config is not None:
            # Use data config from session (loaded via RestoreTrainRun/SetTrainRunConfig)
            data_config_py = session.data_config
        else:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                "Training requires data config to be provided either in request or loaded via RestoreTrainRun"
            )
            return

        try:
            # Create datamodule from data config
            datamodule = self._create_single_cu3s_data_module(data_config_py)
            training_config_py: TrainingConfig | None = None

            if request.trainer_type == cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL:
                # Statistical training - single pass, no streaming
                # Preserve existing training_config from session (if loaded via RestoreTrainRun)
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
                        "Gradient training requires training config either in request or loaded via RestoreTrainRun"
                    )
                    return

                self._capture_experiment_context(session, data_config_py, training_config_py)
                yield from self._train_gradient(
                    session,
                    datamodule,
                    data_config_py,
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

    def get_train_status(
        self,
        request: cuvis_ai_pb2.GetTrainStatusRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetTrainStatusResponse:
        """Get current training status."""
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

    def get_training_capabilities(
        self,
        request: cuvis_ai_pb2.GetTrainingCapabilitiesRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetTrainingCapabilitiesResponse:
        """Return supported optimizers, schedulers, and callbacks."""
        try:
            from cuvis_ai.training.optimizer_registry import (
                get_supported_optimizers,
                get_supported_schedulers,
            )

            supported_optimizers = get_supported_optimizers()
            supported_schedulers = get_supported_schedulers()

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

    def _capture_experiment_context(
        self,
        session: SessionState,
        data_config: Any,
        training_config: TrainingConfig,
    ) -> None:
        """Persist experiment context on the session for SaveTrainRun."""
        pipeline_config = session.pipeline_config

        metadata_name = pipeline_config.metadata.name if pipeline_config.metadata else ""
        pipeline_name = getattr(session.pipeline, "name", metadata_name or "experiment")

        session.data_config = data_config
        session.training_config = training_config

        # Preserve loss_nodes, metric_nodes, and unfreeze_nodes from existing trainrun_config if available
        loss_nodes = []
        metric_nodes = []
        unfreeze_nodes = []
        freeze_nodes = []

        if session.trainrun_config is not None:
            loss_nodes = session.trainrun_config.loss_nodes
            metric_nodes = session.trainrun_config.metric_nodes
            unfreeze_nodes = session.trainrun_config.unfreeze_nodes
            freeze_nodes = session.trainrun_config.freeze_nodes

        session.trainrun_config = TrainRunConfig(
            name=str(pipeline_name),
            pipeline=pipeline_config,
            data=data_config,
            training=training_config,
            loss_nodes=loss_nodes,
            metric_nodes=metric_nodes,
            freeze_nodes=freeze_nodes,
            unfreeze_nodes=unfreeze_nodes,
        )

    def _create_single_cu3s_data_module(
        self,
        data_config: DataConfig,
    ) -> SingleCu3sDataModule:
        """Create SingleCu3sDataModule from parsed DataConfig."""
        try:
            from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule
        except ImportError as exc:  # pragma: no cover - environment dependency
            raise ValueError(
                "SingleCu3sDataModule dependencies are missing. "
                "Install scikit-image and related optional deps to enable training."
            ) from exc

        annotation_json_path = data_config.annotation_json_path or None

        return SingleCu3sDataModule(
            cu3s_file_path=data_config.cu3s_file_path,
            annotation_json_path=annotation_json_path,
            train_ids=list(data_config.train_ids),
            val_ids=list(data_config.val_ids),
            test_ids=list(data_config.test_ids),
            batch_size=data_config.batch_size,
            processing_mode=data_config.processing_mode,
        )

    def _train_statistical(
        self,
        session: SessionState,
        datamodule: SingleCu3sDataModule,
    ) -> Iterator[cuvis_ai_pb2.TrainResponse]:
        """Train with statistical method."""
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
        data_config: DataConfig,
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
            scheduler_config=training_config.scheduler,
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

    def _deserialize_training_config(
        self, config_proto: cuvis_ai_pb2.TrainingConfig
    ) -> TrainingConfig:
        """Decode TrainingConfig from proto bytes."""
        if not config_proto.config_bytes:
            raise ValueError("Training config cannot be empty")

        try:
            return TrainingConfig.from_proto(config_proto)
        except Exception as exc:
            raise ValueError(f"Invalid training config: {exc}") from exc

    def _configure_gradient_components(
        self,
        session: SessionState,
        data_config: DataConfig,
        training_config: TrainingConfig,
    ) -> tuple[list, list]:
        """Configure loss and metric nodes from trainrun config."""
        pipeline = session.pipeline

        # Require explicit trainrun config
        if session.trainrun_config is None:
            raise ValueError(
                "Gradient training requires explicit TrainRunConfig with loss_nodes and metric_nodes. "
                "Please provide train run configuration via RestoreTrainRun or session creation."
            )

        trainrun_config = session.trainrun_config

        # Validate required fields
        if not trainrun_config.loss_nodes:
            raise ValueError(
                "trainrun_config.loss_nodes must specify at least one loss node for gradient training. "
                "Add loss node names to your trainrun config YAML."
            )

        # Build node lookup map
        node_map = {node.name: node for node in pipeline.nodes()}

        # Look up loss nodes by name
        loss_nodes = []
        for loss_name in trainrun_config.loss_nodes:
            if loss_name not in node_map:
                raise ValueError(
                    f"Loss node '{loss_name}' not found in pipeline. "
                    f"Available nodes: {', '.join(sorted(node_map.keys()))}"
                )
            loss_nodes.append(node_map[loss_name])

        # Look up metric nodes by name
        metric_nodes = []
        for metric_name in trainrun_config.metric_nodes:
            if metric_name not in node_map:
                raise ValueError(
                    f"Metric node '{metric_name}' not found in pipeline. "
                    f"Available nodes: {', '.join(sorted(node_map.keys()))}"
                )
            metric_nodes.append(node_map[metric_name])

        # Handle unfreeze_nodes from trainrun config
        if trainrun_config.unfreeze_nodes:
            pipeline.unfreeze_nodes_by_name(list(trainrun_config.unfreeze_nodes))

        # Validate we have trainable parameters
        has_trainable = any(p.requires_grad for p in pipeline.parameters())
        if not has_trainable:
            raise ValueError(
                "No trainable parameters found after unfreezing. "
                "Configure unfreeze_nodes in trainrun config to include trainable nodes."
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


__all__ = ["TrainingService"]
