"""Training configuration models using Pydantic with proto helpers."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cuvis_ai import __version__

if TYPE_CHECKING:
    from cuvis_ai.grpc.v1 import cuvis_ai_pb2


class _BaseConfig(BaseModel):
    """Base model with strict validation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)


class EarlyStoppingConfig(_BaseConfig):
    """Early stopping callback configuration."""

    monitor: str = Field(description="Metric to monitor")
    patience: int = Field(default=10, ge=1, description="Number of epochs to wait")
    mode: str = Field(default="min", description="min or max")
    min_delta: float = Field(default=0.0, ge=0.0, description="Minimum change to qualify")
    stopping_threshold: float | None = Field(
        default=None, description="Stop once monitored metric reaches this threshold"
    )
    verbose: bool = Field(default=True, description="Whether to log state changes")
    strict: bool = Field(default=True, description="Whether to crash if monitor is not found")
    check_finite: bool = Field(
        default=True, description="Stop when monitor becomes NaN or infinite"
    )
    divergence_threshold: float | None = Field(
        default=None, description="Stop training when monitor becomes worse than this threshold"
    )
    check_on_train_epoch_end: bool | None = Field(
        default=None, description="Whether to run early stopping at end of training epoch"
    )
    log_rank_zero_only: bool = Field(
        default=False, description="Log status only for rank 0 process"
    )


class ModelCheckpointConfig(_BaseConfig):
    """Model checkpoint callback configuration."""

    dirpath: str = Field(default="checkpoints", description="Directory to save checkpoints")
    filename: str | None = Field(default=None, description="Checkpoint filename pattern")
    monitor: str = Field(default="val_loss", description="Metric to monitor")
    mode: str = Field(default="min", description="min or max")
    save_top_k: int = Field(default=3, ge=-1, description="Save top k checkpoints (-1 for all)")
    every_n_epochs: int = Field(default=1, ge=1, description="Save every n epochs")
    save_last: bool | Literal["link"] | None = Field(
        default=False, description="Also save last checkpoint (or 'link' for symlink)"
    )
    auto_insert_metric_name: bool = Field(
        default=True, description="Automatically insert metric name into filename"
    )
    verbose: bool = Field(default=False, description="Verbosity mode")
    save_on_exception: bool = Field(
        default=False, description="Whether to save checkpoint when exception is raised"
    )
    save_weights_only: bool = Field(
        default=False, description="If True, only save model weights, not optimizer states"
    )
    every_n_train_steps: int | None = Field(
        default=None, description="How many training steps to wait before saving checkpoint"
    )
    train_time_interval: timedelta | None = Field(
        default=None, description="Checkpoints monitored at specified time interval"
    )
    save_on_train_epoch_end: bool | None = Field(
        default=None, description="Whether to run checkpointing at end of training epoch"
    )
    enable_version_counter: bool = Field(
        default=True, description="Whether to append version to existing file name"
    )


class LearningRateMonitorConfig(_BaseConfig):
    """Learning rate monitor callback configuration."""

    logging_interval: Literal["step", "epoch"] | None = Field(
        default="epoch", description="Log lr at 'epoch' or 'step'"
    )
    log_momentum: bool = Field(default=False, description="Log momentum values as well")
    log_weight_decay: bool = Field(default=False, description="Log weight decay values as well")


class CallbacksConfig(_BaseConfig):
    """Callbacks configuration."""

    checkpoint: ModelCheckpointConfig | None = Field(
        default=None, description="Model checkpoint configuration", alias="model_checkpoint"
    )
    early_stopping: list[EarlyStoppingConfig] = Field(
        default_factory=list, description="Early stopping configuration(s)"
    )
    learning_rate_monitor: LearningRateMonitorConfig | None = Field(
        default=None, description="Learning rate monitor configuration"
    )

    def to_proto(self) -> cuvis_ai_pb2.CallbacksConfig:
        from cuvis_ai.grpc.v1 import cuvis_ai_pb2

        return cuvis_ai_pb2.CallbacksConfig(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config):
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))


class TrainerConfig(_BaseConfig):
    """Lightning Trainer configuration."""

    max_epochs: int = Field(default=100, ge=1, description="Maximum number of epochs")
    accelerator: str = Field(default="auto", description="Accelerator type")
    devices: int | str | None = Field(default=None, description="Number of devices or IDs")
    default_root_dir: str | None = Field(default=None, description="Root directory for outputs")
    precision: str | int = Field(default="32-true", description="Precision mode")
    accumulate_grad_batches: int = Field(default=1, ge=1, description="Accumulate gradients")
    enable_progress_bar: bool = Field(default=True, description="Show progress bar")
    enable_checkpointing: bool = Field(default=False, description="Enable checkpointing")
    log_every_n_steps: int = Field(default=50, ge=1, description="Log frequency in steps")
    val_check_interval: float | int | None = Field(
        default=1.0, ge=0.0, description="Validation interval"
    )
    check_val_every_n_epoch: int | None = Field(
        default=1, ge=1, description="Validate every n epochs"
    )
    gradient_clip_val: float | None = Field(
        default=None, ge=0.0, description="Gradient clipping value"
    )
    deterministic: bool = Field(default=False, description="Deterministic training")
    benchmark: bool = Field(default=False, description="Enable cudnn benchmark")
    callbacks: CallbacksConfig | None = Field(
        default=None, description="Callback configurations for trainer"
    )


class SchedulerConfig(_BaseConfig):
    """Learning rate scheduler configuration."""

    name: str | None = Field(
        default=None, description="Scheduler type: cosine, step, exponential, plateau"
    )
    warmup_epochs: int = Field(default=0, ge=0, description="Number of warmup epochs")
    min_lr: float = Field(default=1e-6, ge=0.0, description="Minimum learning rate")
    t_max: int | None = Field(
        default=None, ge=1, description="Maximum iterations (for cosine annealing)"
    )
    step_size: int | None = Field(
        default=None, ge=1, description="Period of LR decay (for step scheduler)"
    )
    gamma: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Multiplicative factor of LR decay"
    )
    monitor: str | None = Field(
        default=None, description="Metric to monitor (for plateau/reduce_on_plateau)"
    )
    mode: str = Field(default="min", description="min or max for monitored metrics")
    factor: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="LR reduction factor for ReduceLROnPlateau",
    )
    patience: int = Field(default=10, ge=0, description="Patience for plateau scheduler")
    threshold: float = Field(default=1e-4, ge=0.0, description="Plateau threshold")
    threshold_mode: str = Field(default="rel", description="Plateau threshold mode")
    cooldown: int = Field(default=0, ge=0, description="Cooldown epochs for plateau")
    eps: float = Field(default=1e-8, ge=0.0, description="Minimum change in LR for plateau")
    verbose: bool = Field(default=False, description="Verbose scheduler logging")

    def to_proto(self) -> cuvis_ai_pb2.SchedulerConfig:
        from cuvis_ai.grpc.v1 import cuvis_ai_pb2

        return cuvis_ai_pb2.SchedulerConfig(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config):
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))


class OptimizerConfig(_BaseConfig):
    """Optimizer configuration with constraints and documentation."""

    name: str = Field(
        default="adamw",
        description="Optimizer type: adamw, sgd, adam",
    )
    lr: float = Field(
        default=1e-3,
        gt=0.0,
        le=1.0,
        description="Learning rate",
        json_schema_extra={"minimum": 1e-6},
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="L2 regularization coefficient",
    )
    momentum: float | None = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Momentum factor (for SGD)",
    )
    betas: tuple[float, float] | None = Field(default=None, description="Adam betas (beta1, beta2)")

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "name": "adamw",
                    "lr": 0.001,
                    "weight_decay": 0.01,
                }
            ]
        },
    )

    @field_validator("betas")
    @classmethod
    def _validate_betas(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is None:
            return value
        if len(value) != 2:
            raise ValueError("betas must be a tuple of length 2")
        return value

    @field_validator("lr")
    @classmethod
    def _validate_lr(cls, value: float) -> float:
        if value == 0:
            raise ValueError("Learning rate must be non-zero")
        return value

    def to_proto(self) -> cuvis_ai_pb2.OptimizerConfig:
        from cuvis_ai.grpc.v1 import cuvis_ai_pb2

        return cuvis_ai_pb2.OptimizerConfig(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config):
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))


class TrainingConfig(_BaseConfig):
    """Complete training configuration."""

    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig, description="Optimizer configuration"
    )
    scheduler: SchedulerConfig | None = Field(
        default=None, description="Learning rate scheduler (optional)"
    )
    callbacks: CallbacksConfig | None = Field(
        default=None, description="Training callbacks (optional)"
    )
    trainer: TrainerConfig = Field(
        default_factory=TrainerConfig, description="Lightning Trainer configuration"
    )
    max_epochs: int = Field(default=100, ge=1, le=10000, description="Maximum training epochs")
    batch_size: int = Field(default=32, ge=1, description="Batch size")
    num_workers: int = Field(default=4, ge=0, description="Number of data loading workers")
    gradient_clip_val: float | None = Field(
        default=None, ge=0.0, description="Gradient clipping value (optional)"
    )
    accumulate_grad_batches: int = Field(
        default=1, ge=1, description="Accumulate gradients over n batches"
    )

    @model_validator(mode="after")
    def _sync_trainer_fields(self) -> TrainingConfig:
        """Keep top-level hyperparameters in sync with trainer config."""
        fields_set = getattr(self, "model_fields_set", set())

        # max_epochs: prefer explicit trainer value when top-level not provided
        if "max_epochs" not in fields_set and self.trainer.max_epochs is not None:
            self.max_epochs = self.trainer.max_epochs
        else:
            self.trainer.max_epochs = self.max_epochs

        # gradient_clip_val
        if "gradient_clip_val" not in fields_set and self.trainer.gradient_clip_val is not None:
            self.gradient_clip_val = self.trainer.gradient_clip_val
        elif self.gradient_clip_val is not None:
            self.trainer.gradient_clip_val = self.gradient_clip_val

        # accumulate_grad_batches
        if (
            "accumulate_grad_batches" not in fields_set
            and self.trainer.accumulate_grad_batches is not None
        ):
            self.accumulate_grad_batches = self.trainer.accumulate_grad_batches
        else:
            self.trainer.accumulate_grad_batches = self.accumulate_grad_batches

        # callbacks
        if self.callbacks is not None:
            self.trainer.callbacks = self.callbacks
        return self

    def to_proto(self) -> cuvis_ai_pb2.TrainingConfig:
        from cuvis_ai.grpc.v1 import cuvis_ai_pb2

        return cuvis_ai_pb2.TrainingConfig(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config):
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))

    def to_json(self) -> str:
        """JSON serialization helper for legacy callers."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> TrainingConfig:
        return cls.model_validate_json(payload)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_dict_config(self) -> dict[str, Any]:
        """Compatibility shim for legacy OmegaConf usage."""
        try:
            from omegaconf import OmegaConf
        except Exception:
            return self.model_dump()

        return OmegaConf.create(self.model_dump())  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        return cls.model_validate(data)

    @classmethod
    def from_dict_config(cls, config: dict[str, Any]) -> TrainingConfig:
        if config.__class__.__name__ == "DictConfig":  # Avoid hard dependency in type hints
            from omegaconf import OmegaConf

            config = OmegaConf.to_container(config, resolve=True)  # type: ignore[assignment]
        elif not isinstance(config, dict):
            config = dict(config)
        return cls.model_validate(config)


class PipelineMetadata(_BaseConfig):
    """Pipeline metadata for documentation and discovery."""

    name: str
    description: str = ""
    created: str = ""
    tags: list[str] = Field(default_factory=list)
    author: str = ""
    cuvis_ai_version: str = Field(default_factory=lambda: __version__)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineMetadata:
        return cls.model_validate(data)

    def to_proto(self) -> cuvis_ai_pb2.PipelineMetadata:
        from cuvis_ai.grpc.v1 import cuvis_ai_pb2

        return cuvis_ai_pb2.PipelineMetadata(
            name=self.name,
            description=self.description,
            created=self.created,
            tags=list(self.tags),
            author=self.author,
            cuvis_ai_version=self.cuvis_ai_version,
        )


class PipelineConfig(_BaseConfig):
    """Pipeline structure configuration."""

    name: str = Field(default="", description="Pipeline name")
    nodes: list[dict[str, Any]] = Field(description="Node definitions")
    connections: list[dict[str, Any]] = Field(description="Node connections")
    frozen_nodes: list[str] = Field(
        default_factory=list, description="Node names to keep frozen during training"
    )
    metadata: PipelineMetadata | None = Field(
        default=None, description="Optional pipeline metadata"
    )

    def to_proto(self) -> cuvis_ai_pb2.PipelineConfig:
        from cuvis_ai.grpc.v1 import cuvis_ai_pb2

        return cuvis_ai_pb2.PipelineConfig(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config):
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> PipelineConfig:
        return cls.model_validate_json(payload)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        return cls.model_validate(data)


class DataConfig(_BaseConfig):
    """Data loading configuration."""

    cu3s_file_path: str = Field(description="Path to .cu3s file")
    annotation_json_path: str | None = Field(
        default=None, description="Path to annotation JSON (optional)"
    )
    train_ids: list[int] = Field(default_factory=list, description="Training sample IDs")
    val_ids: list[int] = Field(default_factory=list, description="Validation sample IDs")
    test_ids: list[int] = Field(default_factory=list, description="Test sample IDs")
    train_split: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Training split ratio"
    )
    val_split: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Validation split ratio"
    )
    shuffle: bool = Field(default=True, description="Shuffle dataset")
    batch_size: int = Field(default=1, ge=1, description="Batch size")
    processing_mode: str = Field(default="Reflectance", description="Raw or Reflectance mode")

    def to_proto(self) -> cuvis_ai_pb2.DataConfig:
        from cuvis_ai.grpc.v1 import cuvis_ai_pb2

        return cuvis_ai_pb2.DataConfig(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config):
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> DataConfig:
        return cls.model_validate_json(payload)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataConfig:
        return cls.model_validate(data)


class TrainRunConfig(_BaseConfig):
    """Complete reproducible training configuration."""

    name: str = Field(description="Train run identifier")
    pipeline: PipelineConfig | None = Field(
        default=None, description="Pipeline configuration (optional if already built)"
    )
    data: DataConfig = Field(description="Data configuration")
    training: TrainingConfig = Field(description="Training configuration")
    loss_nodes: list[str] = Field(
        default_factory=list, description="Loss node names for gradient training"
    )
    metric_nodes: list[str] = Field(
        default_factory=list, description="Metric node names for monitoring"
    )
    freeze_nodes: list[str] = Field(
        default_factory=list, description="Node names to keep frozen during training"
    )
    unfreeze_nodes: list[str] = Field(
        default_factory=list, description="Node names to unfreeze during training"
    )
    output_dir: str = Field(default="./outputs", description="Output directory for artifacts")
    tags: dict[str, str] = Field(default_factory=dict, description="Metadata tags for tracking")

    def to_proto(self) -> cuvis_ai_pb2.TrainRunConfig:
        from cuvis_ai.grpc.v1 import cuvis_ai_pb2

        return cuvis_ai_pb2.TrainRunConfig(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config):
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> TrainRunConfig:
        return cls.model_validate_json(payload)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainRunConfig:
        return cls.model_validate(data)

    def save_to_file(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)

    @classmethod
    def load_from_file(cls, path: str | Path) -> TrainRunConfig:
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


def create_callbacks_from_config(config: CallbacksConfig | None) -> list:
    """Create PyTorch Lightning callback instances from configuration."""
    if config is None:
        return []

    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

    callbacks = []

    if config.early_stopping:
        for es_config in config.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=es_config.monitor,
                    patience=es_config.patience,
                    mode=es_config.mode,
                    min_delta=es_config.min_delta,
                    stopping_threshold=es_config.stopping_threshold,
                    verbose=es_config.verbose,
                    strict=es_config.strict,
                    check_finite=es_config.check_finite,
                    divergence_threshold=es_config.divergence_threshold,
                    check_on_train_epoch_end=es_config.check_on_train_epoch_end,
                    log_rank_zero_only=es_config.log_rank_zero_only,
                )
            )

    if config.checkpoint is not None:
        mc_config = config.checkpoint
        callbacks.append(
            ModelCheckpoint(
                dirpath=mc_config.dirpath,
                filename=mc_config.filename,
                monitor=mc_config.monitor,
                mode=mc_config.mode,
                save_top_k=mc_config.save_top_k,
                save_last=mc_config.save_last,
                verbose=mc_config.verbose,
                auto_insert_metric_name=mc_config.auto_insert_metric_name,
                every_n_epochs=mc_config.every_n_epochs,
                save_on_exception=mc_config.save_on_exception,
                save_weights_only=mc_config.save_weights_only,
                every_n_train_steps=mc_config.every_n_train_steps,
                train_time_interval=mc_config.train_time_interval,
                save_on_train_epoch_end=mc_config.save_on_train_epoch_end,
                enable_version_counter=mc_config.enable_version_counter,
            )
        )

    if config.learning_rate_monitor is not None:
        lr_config = config.learning_rate_monitor
        callbacks.append(
            LearningRateMonitor(
                logging_interval=lr_config.logging_interval,
                log_momentum=lr_config.log_momentum,
                log_weight_decay=lr_config.log_weight_decay,
            )
        )

    return callbacks


__all__ = [
    "EarlyStoppingConfig",
    "ModelCheckpointConfig",
    "LearningRateMonitorConfig",
    "CallbacksConfig",
    "TrainerConfig",
    "SchedulerConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "PipelineMetadata",
    "PipelineConfig",
    "DataConfig",
    "TrainRunConfig",
    "create_callbacks_from_config",
]
