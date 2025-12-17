"""Centralized optimizer and scheduler registry with factory helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer

from cuvis_ai.training.config import OptimizerConfig, SchedulerConfig

try:  # PyTorch < 2.0 compatibility
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # pragma: no cover - older PyTorch fallback
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # type: ignore


# -----------------------------------------------------------------------------
# Registry definitions
# -----------------------------------------------------------------------------

SUPPORTED_OPTIMIZERS: dict[str, dict[str, Any]] = {
    "adam": {
        "class": torch.optim.Adam,
        "description": "Adam optimizer with adaptive learning rates",
        "params": {
            "lr": {"required": True, "type": "float", "description": "Learning rate"},
            "weight_decay": {"required": False, "default": 0.0, "type": "float"},
            "betas": {"required": False, "default": (0.9, 0.999), "type": "tuple"},
        },
    },
    "adamw": {
        "class": torch.optim.AdamW,
        "description": "AdamW optimizer with decoupled weight decay",
        "params": {
            "lr": {"required": True, "type": "float"},
            "weight_decay": {"required": False, "default": 0.0, "type": "float"},
            "betas": {"required": False, "default": (0.9, 0.999), "type": "tuple"},
        },
    },
    "sgd": {
        "class": torch.optim.SGD,
        "description": "Stochastic Gradient Descent optimizer",
        "params": {
            "lr": {"required": True, "type": "float"},
            "weight_decay": {"required": False, "default": 0.0, "type": "float"},
            "momentum": {"required": False, "default": 0.0, "type": "float"},
        },
    },
}

SUPPORTED_SCHEDULERS: dict[str, dict[str, Any]] = {
    "reduce_on_plateau": {
        "class": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "aliases": ["plateau"],
        "description": "Reduce learning rate when metric plateaus",
        "requires_monitor": True,
        "params": {
            "mode": {"required": False, "default": "min", "type": "str"},
            "factor": {"required": False, "default": 0.1, "type": "float"},
            "patience": {"required": False, "default": 10, "type": "int"},
            "threshold": {"required": False, "default": 1e-4, "type": "float"},
            "threshold_mode": {"required": False, "default": "rel", "type": "str"},
            "cooldown": {"required": False, "default": 0, "type": "int"},
            "min_lr": {"required": False, "default": 0.0, "type": "float"},
            "eps": {"required": False, "default": 1e-8, "type": "float"},
            "verbose": {"required": False, "default": False, "type": "bool"},
        },
    },
    "cosine": {
        "class": torch.optim.lr_scheduler.CosineAnnealingLR,
        "description": "Cosine annealing learning rate schedule",
        "param_mapping": {"t_max": "T_max", "min_lr": "eta_min"},
        "params": {
            "T_max": {"required": True, "type": "int"},
            "eta_min": {"required": False, "default": 0.0, "type": "float"},
        },
    },
    "step": {
        "class": torch.optim.lr_scheduler.StepLR,
        "description": "Step-based learning rate decay",
        "params": {
            "step_size": {"required": False, "default": 1, "type": "int"},
            "gamma": {"required": False, "default": 0.1, "type": "float"},
        },
    },
    "exponential": {
        "class": torch.optim.lr_scheduler.ExponentialLR,
        "description": "Exponential learning rate decay",
        "params": {
            "gamma": {"required": False, "default": 0.99, "type": "float"},
        },
    },
}


# -----------------------------------------------------------------------------
# Factories
# -----------------------------------------------------------------------------


def create_optimizer(config: OptimizerConfig, params) -> Optimizer:
    """Instantiate optimizer from config using the registry."""
    optimizer_name = config.name.lower()
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        supported = ", ".join(sorted(SUPPORTED_OPTIMIZERS.keys()))
        raise ValueError(f"Unsupported optimizer: {config.name}. Supported optimizers: {supported}")

    optimizer_spec = SUPPORTED_OPTIMIZERS[optimizer_name]
    optimizer_cls = optimizer_spec["class"]

    kwargs = {"lr": config.lr, "weight_decay": config.weight_decay}
    if optimizer_name in {"adam", "adamw"}:
        kwargs["betas"] = config.betas or (0.9, 0.999)
    elif optimizer_name == "sgd":
        kwargs["momentum"] = config.momentum or 0.0

    return optimizer_cls(params, **kwargs)


def create_scheduler(
    config: SchedulerConfig | None,
    optimizer: Optimizer,
    max_epochs: int,
) -> LRScheduler | None:
    """Instantiate scheduler from config using the registry."""
    if config is None or not config.name:
        return None

    scheduler_name = config.name.lower()
    if scheduler_name in {"none", ""}:
        return None

    if scheduler_name not in SUPPORTED_SCHEDULERS:
        supported = ", ".join(sorted(SUPPORTED_SCHEDULERS.keys()))
        raise ValueError(f"Unsupported scheduler: {config.name}. Supported schedulers: {supported}")

    scheduler_spec = SUPPORTED_SCHEDULERS[scheduler_name]
    scheduler_cls = scheduler_spec["class"]

    if scheduler_name == "reduce_on_plateau":
        kwargs = {
            "mode": config.mode,
            "factor": config.factor,
            "patience": config.patience,
            "threshold": config.threshold,
            "threshold_mode": config.threshold_mode,
            "cooldown": config.cooldown,
            "min_lr": config.min_lr,
            "eps": config.eps,
        }
        # Older PyTorch versions gate verbose in the signature
        import inspect

        if "verbose" in inspect.signature(scheduler_cls).parameters:
            kwargs["verbose"] = config.verbose
    elif scheduler_name == "cosine":
        kwargs = {
            "T_max": config.t_max or max_epochs,
            "eta_min": config.min_lr,
        }
    elif scheduler_name == "step":
        kwargs = {
            "step_size": config.step_size or 1,
            "gamma": config.gamma or 0.1,
        }
    elif scheduler_name == "exponential":
        kwargs = {
            "gamma": config.gamma or 0.99,
        }
    else:  # pragma: no cover - guarded by registry
        kwargs = {}

    return scheduler_cls(optimizer, **kwargs)


def wrap_scheduler_for_lightning(
    scheduler: LRScheduler, monitor: str | None = None
) -> dict | LRScheduler:
    """Wrap ReduceLROnPlateau in Lightning's expected dict format."""
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        return {
            "scheduler": scheduler,
            "monitor": monitor or "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
    return scheduler


# -----------------------------------------------------------------------------
# Introspection helpers
# -----------------------------------------------------------------------------


def get_supported_optimizers() -> list[str]:
    """Return list of supported optimizer names."""
    return list(SUPPORTED_OPTIMIZERS.keys())


def get_supported_schedulers() -> list[str]:
    """Return list of supported scheduler names including aliases."""
    names = list(SUPPORTED_SCHEDULERS.keys())
    for spec in SUPPORTED_SCHEDULERS.values():
        names.extend(spec.get("aliases", []))
    return names


def get_optimizer_info(name: str) -> dict[str, Any]:
    """Return optimizer specification by name."""
    normalized = name.lower()
    if normalized not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}")
    return SUPPORTED_OPTIMIZERS[normalized]


def get_scheduler_info(name: str) -> dict[str, Any]:
    """Return scheduler specification by name, resolving aliases."""
    normalized = name.lower()
    for scheduler_name, spec in SUPPORTED_SCHEDULERS.items():
        if normalized == scheduler_name or normalized in spec.get("aliases", []):
            return spec
    raise ValueError(f"Unknown scheduler: {name}")


__all__ = [
    "SUPPORTED_OPTIMIZERS",
    "SUPPORTED_SCHEDULERS",
    "create_optimizer",
    "create_scheduler",
    "wrap_scheduler_for_lightning",
    "get_supported_optimizers",
    "get_supported_schedulers",
    "get_optimizer_info",
    "get_scheduler_info",
]
