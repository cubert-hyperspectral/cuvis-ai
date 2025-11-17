"""Training configuration infrastructure with Hydra support for full reproducibility."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from cuvis_ai.utils.serializer import Serializable


@dataclass
class EarlyStoppingConfig:
    """Configuration for PyTorch Lightning EarlyStopping callback.

    Parameters
    ----------
    monitor : str
        Metric name to monitor (e.g., 'train/bce', 'metrics_anomaly/iou')
    patience : int
        Number of epochs with no improvement after which training will be stopped
    mode : str
        One of 'min' or 'max'. In 'min' mode, training stops when monitored metric stops decreasing.
        In 'max' mode, training stops when monitored metric stops increasing.
    min_delta : float
        Minimum change in the monitored quantity to qualify as an improvement
    stopping_threshold : float | None
        Stop training immediately once the monitored quantity reaches this threshold
    verbose : bool
        Whether to print messages when stopping
    """

    monitor: str
    patience: int = 10
    mode: str = "min"
    min_delta: float = 1e-7
    stopping_threshold: float | None = None
    verbose: bool = True


@dataclass
class ModelCheckpointConfig:
    """Configuration for PyTorch Lightning ModelCheckpoint callback.

    Parameters
    ----------
    dirpath : str
        Directory to save checkpoints
    filename : str | None
        Checkpoint filename pattern (e.g., 'best-{epoch:02d}-{val_loss:.4f}')
    monitor : str
        Metric name to monitor for best checkpoint
    mode : str
        One of 'min' or 'max'
    save_top_k : int
        Save top k models. -1 saves all, 0 disables saving
    save_last : bool
        Save a checkpoint with name 'last.ckpt' in addition to top k
    verbose : bool
        Whether to print messages when saving checkpoints
    auto_insert_metric_name : bool
        Whether to automatically insert metric name in filename
    """

    dirpath: str
    monitor: str
    filename: str | None = None
    mode: str = "max"
    save_top_k: int = 1
    save_last: bool = False
    verbose: bool = True
    auto_insert_metric_name: bool = True


@dataclass
class LearningRateMonitorConfig:
    """Configuration for PyTorch Lightning LearningRateMonitor callback.

    Parameters
    ----------
    logging_interval : str
        Set to 'epoch' or 'step' to log lr at epoch or batch level
    log_momentum : bool
        Whether to log momentum alongside learning rate
    """

    logging_interval: str = (
        "epoch"  # Note: Could be Literal['step', 'epoch'] but keeping as str for flexibility
    )
    log_momentum: bool = False


@dataclass
class CallbacksConfig:
    """Container for all callback configurations.

    Parameters
    ----------
    early_stopping : list[EarlyStoppingConfig]
        List of early stopping configurations. Multiple early stopping callbacks
        can be used to monitor different metrics simultaneously.
    model_checkpoint : ModelCheckpointConfig | None
        Model checkpoint configuration
    learning_rate_monitor : LearningRateMonitorConfig | None
        Learning rate monitor configuration
    """

    early_stopping: list[EarlyStoppingConfig] = field(default_factory=list)
    model_checkpoint: ModelCheckpointConfig | None = None
    learning_rate_monitor: LearningRateMonitorConfig | None = None


def create_callbacks_from_config(config: CallbacksConfig | None) -> list:
    """Create PyTorch Lightning callback instances from configuration.

    Parameters
    ----------
    config : CallbacksConfig | None
        Callback configuration. If None, returns empty list.

    Returns
    -------
    list
        List of instantiated PyTorch Lightning callback objects

    Examples
    --------
    >>> config = CallbacksConfig(
    ...     early_stopping=[
    ...         EarlyStoppingConfig(monitor="train/bce", patience=10, mode="min"),
    ...         EarlyStoppingConfig(monitor="val/iou", patience=10, mode="max"),
    ...     ],
    ...     model_checkpoint=ModelCheckpointConfig(
    ...         dirpath="./checkpoints",
    ...         monitor="val/iou",
    ...         mode="max",
    ...     ),
    ... )
    >>> callbacks = create_callbacks_from_config(config)
    """
    if config is None:
        return []

    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )

    callbacks = []

    # Create EarlyStopping callbacks
    for es_config in config.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=es_config.monitor,
                patience=es_config.patience,
                mode=es_config.mode,
                min_delta=es_config.min_delta,
                stopping_threshold=es_config.stopping_threshold,
                verbose=es_config.verbose,
            )
        )

    # Create ModelCheckpoint callback
    if config.model_checkpoint is not None:
        mc_config = config.model_checkpoint
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
            )
        )

    # Create LearningRateMonitor callback
    if config.learning_rate_monitor is not None:
        lr_config = config.learning_rate_monitor
        callbacks.append(
            LearningRateMonitor(
                logging_interval=lr_config.logging_interval,
                log_momentum=lr_config.log_momentum,
            )
        )

    return callbacks


@dataclass
class TrainerConfig:
    """Hydra-serializable configuration for the Lightning Trainer.

    Parameters
    ----------
    max_epochs : int
        Maximum number of training epochs
    accelerator : str
        Device accelerator ('cpu', 'gpu', 'tpu', 'auto')
    devices : int | str | None
        Number of devices or device IDs to use
    precision : str | int
        Precision mode ('32-true', '16-mixed', 'bf16-mixed', etc.)
    accumulate_grad_batches : int
        Accumulate gradients over N batches
    enable_progress_bar : bool
        Show training progress bar
    enable_checkpointing : bool
        Enable model checkpointing
    log_every_n_steps : int
        Log metrics every N steps
    val_check_interval : float | int | None
        How often to check validation set (fraction of epoch or num batches)
    check_val_every_n_epoch : int | None
        Check validation every N epochs
    gradient_clip_val : float | None
        Gradient clipping value (None to disable)
    deterministic : bool
        Use deterministic algorithms for reproducibility
    benchmark : bool
        Enable cudnn.benchmark for performance
    callbacks : CallbacksConfig | None
        Callback configurations (early stopping, checkpointing, LR monitoring, etc.)
    """

    max_epochs: int = 100
    accelerator: str = "auto"
    devices: int | str | None = None
    precision: str | int = "32-true"
    accumulate_grad_batches: int = 1
    enable_progress_bar: bool = True
    enable_checkpointing: bool = False
    log_every_n_steps: int = 50
    val_check_interval: float | int | None = 1.0
    check_val_every_n_epoch: int | None = 1
    gradient_clip_val: float | None = None
    deterministic: bool = False
    benchmark: bool = False
    callbacks: CallbacksConfig | None = None


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration.

    Parameters
    ----------
    name : str
        Scheduler name ('reduce_on_plateau', 'step', 'cosine', etc.)
    monitor : str | None
        Metric name to monitor for ReduceLROnPlateau (e.g., 'metrics_anomaly/iou')
    mode : str
        One of 'min' or 'max'. In 'min' mode, lr will be reduced when the quantity
        monitored has stopped decreasing; in 'max' mode it will be reduced when the
        quantity monitored has stopped increasing.
    factor : float
        Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience : int
        Number of epochs with no improvement after which learning rate will be reduced
    threshold : float
        Threshold for measuring the new optimum, to only focus on significant changes
    threshold_mode : str
        One of 'rel', 'abs'. In 'rel' mode, dynamic_threshold = best * (1 + threshold)
        in 'max' mode or best * (1 - threshold) in 'min' mode
    cooldown : int
        Number of epochs to wait before resuming normal operation after lr has been reduced
    min_lr : float
        A lower bound on the learning rate
    eps : float
        Minimal decay applied to lr. If the difference between new and old lr is smaller
        than eps, the update is ignored
    verbose : bool
        If True, prints a message to stdout for each update
    """

    name: str = "reduce_on_plateau"
    monitor: str | None = None
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 1e-6
    eps: float = 1e-8
    verbose: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer settings consumed by the Trainer.

    Parameters
    ----------
    name : str
        Optimizer name ('adam', 'sgd', 'adamw')
    lr : float
        Learning rate
    weight_decay : float
        Weight decay (L2 regularization)
    betas : tuple[float, float] | None
        Adam beta parameters (beta1, beta2)
    scheduler : SchedulerConfig | None
        Learning rate scheduler configuration
    """

    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] | None = None
    scheduler: SchedulerConfig | None = None


@dataclass
class TrainingConfig:
    """Top-level training configuration bundle.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    trainer : TrainerConfig
        Lightning Trainer configuration
    optimizer : OptimizerConfig
        Optimizer configuration
    monitor_plugins : list[str]
        List of monitoring plugin names to enable
    """

    seed: int = 42
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    monitor_plugins: list[str] = field(default_factory=lambda: ["loguru"])


def _config_store() -> ConfigStore:
    """Fetch the singleton ConfigStore with late import guarding."""
    return ConfigStore.instance()


def register_training_config(
    *,
    name: str = "base_training",
    group: str = "training",
    node: TrainingConfig | None = None,
    package: str | None = "training",
) -> None:
    """Register the training config with Hydra's ConfigStore.

    Parameters
    ----------
    name : str
        Schema name that Hydra will reference inside the chosen group
    group : str
        Config group under which the configuration is stored
    node : TrainingConfig | None
        Optional override node. When omitted a fresh TrainingConfig is stored
    package : str | None
        Optional Hydra package target. Defaults to 'training' so downstream
        configs can reference training.optimizer.lr
    """
    config_store = _config_store()
    config_store.store(name=name, group=group, node=node or TrainingConfig(), package=package)


def as_dict(config: TrainingConfig) -> dict[str, Any]:
    """Convert a TrainingConfig into a plain dictionary suitable for logging.

    Parameters
    ----------
    config : TrainingConfig
        Configuration to convert

    Returns
    -------
    dict[str, Any]
        Plain dictionary representation
    """
    return {
        "seed": config.seed,
        "trainer": asdict(config.trainer),
        "optimizer": asdict(config.optimizer),
        "monitor_plugins": list(config.monitor_plugins),
    }


def override_from_iterable(
    config: TrainingConfig, overrides: Iterable[tuple[str, Any]]
) -> TrainingConfig:
    """Apply key/value overrides and return the mutated config.

    Parameters
    ----------
    config : TrainingConfig
        Configuration to modify
    overrides : Iterable[tuple[str, Any]]
        Key-value pairs to override (e.g., [('trainer.max_epochs', 10)])

    Returns
    -------
    TrainingConfig
        Modified configuration

    Raises
    ------
    KeyError
        If an unsupported override key is provided
    """
    for key, value in overrides:
        if key.startswith("trainer."):
            setattr(config.trainer, key.split(".", 1)[1], value)
        elif key.startswith("optimizer."):
            setattr(config.optimizer, key.split(".", 1)[1], value)
        elif key == "seed":
            config.seed = value
        elif key == "monitor_plugins":
            config.monitor_plugins = list(value) if isinstance(value, (list, tuple)) else [value]
        else:
            raise KeyError(f"Unsupported override key: {key}")
    return config


def to_dict_config(config: TrainingConfig) -> DictConfig:
    """Convert a TrainingConfig into a structured DictConfig instance.

    Parameters
    ----------
    config : TrainingConfig
        Configuration to convert

    Returns
    -------
    DictConfig
        OmegaConf DictConfig representation
    """
    base = OmegaConf.structured(TrainingConfig)
    payload = OmegaConf.create(asdict(config))
    return OmegaConf.merge(base, payload)


def from_dict_config(config: DictConfig | Mapping[str, Any]) -> TrainingConfig:
    """Create a TrainingConfig from a DictConfig or plain mapping.

    Parameters
    ----------
    config : DictConfig | Mapping[str, Any]
        Configuration dict to parse

    Returns
    -------
    TrainingConfig
        Parsed configuration object

    Raises
    ------
    TypeError
        If the result is not a TrainingConfig instance
    """
    cfg = config if isinstance(config, DictConfig) else OmegaConf.create(config)
    merged = OmegaConf.merge(OmegaConf.structured(TrainingConfig), cfg)
    obj = OmegaConf.to_object(merged)
    if not isinstance(obj, TrainingConfig):
        raise TypeError(f"Expected TrainingConfig, received {type(obj)}")
    return obj


class TrainingConfigSerializable(Serializable):
    """Serializable wrapper for persisting TrainingConfig structures.

    This class bridges TrainingConfig with the existing Serializable protocol
    used throughout cuvis.ai for graph persistence.

    Parameters
    ----------
    config : TrainingConfig | None
        Configuration to wrap. If None, creates default TrainingConfig
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        super().__init__()

    def serialize(self, serial_dir: str) -> dict[str, Any]:
        """Return the Hydra-compatible dictionary representing the training config.

        Parameters
        ----------
        serial_dir : str
            Directory for serialization (unused, kept for protocol compatibility)

        Returns
        -------
        dict[str, Any]
            Serialized configuration
        """
        dict_config = to_dict_config(self.config)
        return OmegaConf.to_container(dict_config, resolve=True)

    def load(self, params: dict[str, Any], serial_dir: str) -> None:
        """Populate the wrapped TrainingConfig from serialized parameters.

        Parameters
        ----------
        params : dict[str, Any]
            Serialized configuration parameters
        serial_dir : str
            Directory for deserialization (unused, kept for protocol compatibility)
        """
        self.config = from_dict_config(params)


__all__ = [
    "EarlyStoppingConfig",
    "ModelCheckpointConfig",
    "LearningRateMonitorConfig",
    "CallbacksConfig",
    "create_callbacks_from_config",
    "TrainerConfig",
    "SchedulerConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "register_training_config",
    "as_dict",
    "override_from_iterable",
    "to_dict_config",
    "from_dict_config",
    "TrainingConfigSerializable",
]
