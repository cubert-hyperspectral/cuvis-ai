"""Configuration helper functions for the cuvis.ai framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError

from cuvis_ai.training.config import (
    CallbacksConfig,
    DataConfig,
    OptimizerConfig,
    PipelineConfig,
    SchedulerConfig,
    TrainingConfig,
    TrainRunConfig,
)

# ------------------------------------------------------------------
# Config registry and validation helpers
# ------------------------------------------------------------------

CONFIG_TYPE_REGISTRY: dict[str, type[BaseModel]] = {
    "training": TrainingConfig,
    "optimizer": OptimizerConfig,
    "scheduler": SchedulerConfig,
    "callbacks": CallbacksConfig,
    "data": DataConfig,
    "pipeline": PipelineConfig,
    "trainrun": TrainRunConfig,
}


def get_config_class(config_type: str) -> type[BaseModel]:
    """Get Pydantic model class for config type."""
    if config_type not in CONFIG_TYPE_REGISTRY:
        raise ValueError(
            f"Unknown config type: {config_type}. "
            f"Available types: {list(CONFIG_TYPE_REGISTRY.keys())}"
        )
    return CONFIG_TYPE_REGISTRY[config_type]


def generate_json_schema(config_type: str) -> dict:
    """Generate JSON Schema for config type."""
    config_class = get_config_class(config_type)
    return config_class.model_json_schema()


def validate_config_dict(config_type: str, config_dict: dict) -> tuple[bool, list[str]]:
    """Validate configuration dictionary."""
    config_class = get_config_class(config_type)

    try:
        config_class.model_validate(config_dict)
        return True, []
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            field_path = ".".join(str(x) for x in err["loc"])
            errors.append(f"{field_path}: {err['msg']}")
        return False, errors


# ------------------------------------------------------------------
# Hydra-based config resolution and file helpers
# ------------------------------------------------------------------


def apply_config_overrides(
    config: dict[str, Any] | DictConfig,
    overrides: list[str] | dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply overrides (list or dict) to an OmegaConf config and return a plain dict."""

    def _format_override_key(key: str) -> str:
        """Normalize dot notation with numeric parts into OmegaConf bracket notation."""
        if "[" in key:
            return key

        formatted_parts: list[str] = []
        for part in key.split("."):
            if part.isdigit() and formatted_parts:
                formatted_parts[-1] = f"{formatted_parts[-1]}[{part}]"
            elif part.isdigit():
                formatted_parts.append(f"[{part}]")
            else:
                formatted_parts.append(part)
        return ".".join(formatted_parts)

    if overrides in (None, [], {}):
        return OmegaConf.to_container(OmegaConf.create(config), resolve=True)  # type: ignore[return-value]

    if isinstance(config, DictConfig):
        config_omega = config.copy()
    else:
        config_omega = OmegaConf.create(config)

    if isinstance(overrides, list):
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Invalid override format: {override}. Expected format: key=value")
            key, value = override.split("=", 1)
            formatted_key = _format_override_key(key)
            parsed_value = yaml.safe_load(value)
            OmegaConf.update(config_omega, formatted_key, parsed_value, merge=True)
    elif isinstance(overrides, dict):
        override_omega = OmegaConf.create(overrides)
        config_omega = OmegaConf.merge(config_omega, override_omega)
    else:
        raise TypeError(
            f"overrides must be a list of strings, a dict, or None. Got {type(overrides)}"
        )

    return OmegaConf.to_container(config_omega, resolve=True)  # type: ignore[return-value]


def resolve_config_with_hydra(
    config_type: str,
    config_path: str,
    search_paths: list[str],
    overrides: list[str] | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve configuration using Hydra composition and validate with Pydantic."""
    try:
        from hydra import compose, initialize_config_dir
    except ImportError as exc:  # pragma: no cover - environment safeguard
        raise ImportError(
            "Hydra is required for config resolution. Install hydra-core to use ResolveConfig."
        ) from exc
    logger.info(f"Resolving {config_type} config: {config_path}")
    logger.debug(f"Search paths: {search_paths}")
    logger.debug(f"Overrides: {overrides}")

    config_file = _find_config_file(config_path, search_paths).resolve()
    logger.debug(f"Found config file: {config_file}")

    # Determine the Hydra config root and config name relative to that root.
    config_root = config_file.parent
    config_name = config_file.stem

    for search_dir in search_paths:
        resolved_dir = Path(search_dir).resolve()
        try:
            relative = config_file.relative_to(resolved_dir)
        except ValueError:
            continue

        config_root = resolved_dir
        config_name = relative.with_suffix("").as_posix()
        break

    hydra_overrides = overrides if isinstance(overrides, list) else []

    with initialize_config_dir(config_dir=str(config_root), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=hydra_overrides or [])
        config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Handle Hydra packaging for all config types
    # When resolving configs, Hydra may wrap them in a key matching the config_type
    # due to @package directives or config structure. Unwrap if needed.
    if isinstance(config_dict, dict) and config_type in config_dict:
        # Check if the wrapped value looks like the actual config (has expected fields)
        # For pipeline: should have "nodes" or "name"
        # For training: should have "seed" or "optimizer"
        # For data: should have "cu3s_file_path" or "batch_size"
        wrapped_value = config_dict[config_type]
        if isinstance(wrapped_value, dict):
            # Unwrap if the wrapper only contains the config_type key, or if the top level
            # doesn't have the expected structure
            config_class = get_config_class(config_type)
            expected_fields = set(config_class.model_fields.keys())
            top_level_fields = set(config_dict.keys())
            wrapped_fields = set(wrapped_value.keys())

            # If wrapped_value has more expected fields than top level, unwrap it
            if len(expected_fields & wrapped_fields) > len(expected_fields & top_level_fields):
                logger.debug(f"Unwrapping {config_type} config from Hydra packaging")
                config_dict = wrapped_value

    # Apply overrides using shared helper to keep behavior consistent across entrypoints
    if isinstance(config_dict, dict):
        config_dict = apply_config_overrides(config_dict, overrides)  # type: ignore[arg-type]
    else:
        # Handle case where config_dict might not be a dict (shouldn't happen in practice)
        logger.warning(f"Config dict is not a dict type, got {type(config_dict)}")
        config_dict = {}

    config_class = get_config_class(config_type)
    config_model = config_class.model_validate(config_dict)

    logger.info(f"Successfully resolved {config_type} config")
    return config_model.model_dump()


def _find_config_file(path: str, search_paths: list[str]) -> Path:
    """Find config file in search paths, supporting relative and absolute paths."""
    path_obj = Path(path)

    if not path_obj.suffix:
        path_obj = path_obj.with_suffix(".yaml")

    if path_obj.is_absolute():
        if path_obj.exists():
            return path_obj
        raise FileNotFoundError(f"Config file not found: {path_obj}")

    for search_dir in search_paths:
        candidate = Path(search_dir) / path_obj
        if candidate.exists():
            logger.debug(f"Found config in {search_dir}: {path_obj}")
            return candidate.resolve()

    raise FileNotFoundError(f"Config file '{path}' not found in search paths: {search_paths}")


__all__ = [
    "apply_config_overrides",
    "resolve_config_with_hydra",
    "validate_config_dict",
    "generate_json_schema",
    "get_config_class",
]
