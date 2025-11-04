"""Training configuration infrastructure with Hydra support for full reproducibility."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from cuvis_ai.utils.serializer import Serializable


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
    """

    max_epochs: int = 1
    accelerator: str = "cpu"
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


@dataclass
class OptimizerConfig:
    """Optimizer settings consumed by CuvisLightningModule.
    
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
    """

    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] | None = None


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
    config_store.store(
        name=name,
        group=group,
        node=node or TrainingConfig(),
        package=package
    )


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
    config: TrainingConfig,
    overrides: Iterable[tuple[str, Any]]
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
    "TrainerConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "register_training_config",
    "as_dict",
    "override_from_iterable",
    "to_dict_config",
    "from_dict_config",
    "TrainingConfigSerializable",
]
