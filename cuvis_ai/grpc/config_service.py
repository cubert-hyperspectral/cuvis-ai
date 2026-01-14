"""Config resolution and validation service component."""

from __future__ import annotations

import json

import grpc

from cuvis_ai.utils.config_helpers import (
    generate_json_schema,
    resolve_config_with_hydra,
    validate_config_dict,
)

from .session_manager import SessionManager
from .v1 import cuvis_ai_pb2


class ConfigService:
    """Config resolution and validation."""

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager

    def resolve_config(
        self,
        request: cuvis_ai_pb2.ResolveConfigRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.ResolveConfigResponse:
        """Resolve configuration using Hydra composition."""
        try:
            session = self.session_manager.get_session(request.session_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.ResolveConfigResponse()

        try:
            config_dict = resolve_config_with_hydra(
                config_type=request.config_type,
                config_path=request.path,
                search_paths=session.search_paths,
                overrides=list(request.overrides),
            )
            config_json = json.dumps(config_dict, indent=2)
            return cuvis_ai_pb2.ResolveConfigResponse(config_bytes=config_json.encode("utf-8"))
        except FileNotFoundError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.ResolveConfigResponse()
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Config resolution failed: {exc}")
            return cuvis_ai_pb2.ResolveConfigResponse()

    def get_parameter_schema(
        self,
        request: cuvis_ai_pb2.GetParameterSchemaRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.GetParameterSchemaResponse:
        """Return JSON Schema for requested config type."""
        try:
            schema = generate_json_schema(request.config_type)
            schema_json = json.dumps(schema, indent=2)
            return cuvis_ai_pb2.GetParameterSchemaResponse(json_schema=schema_json)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return cuvis_ai_pb2.GetParameterSchemaResponse()
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to generate schema: {exc}")
            return cuvis_ai_pb2.GetParameterSchemaResponse()

    def validate_config(
        self,
        request: cuvis_ai_pb2.ValidateConfigRequest,
        context: grpc.ServicerContext,
    ) -> cuvis_ai_pb2.ValidateConfigResponse:
        """Validate configuration payloads using Pydantic models."""
        if not request.config_bytes:
            return cuvis_ai_pb2.ValidateConfigResponse(
                valid=False, errors=["config_bytes cannot be empty"]
            )

        try:
            config_json = request.config_bytes.decode("utf-8")
            config_dict = json.loads(config_json)
        except json.JSONDecodeError as exc:
            return cuvis_ai_pb2.ValidateConfigResponse(
                valid=False, errors=[f"Invalid JSON: {exc.msg}"]
            )

        try:
            valid, errors = validate_config_dict(request.config_type, config_dict)
            warnings = []

            # Add training-specific validation for training config type
            if request.config_type == "training" and valid:
                from cuvis_ai.training.config import TrainingConfig
                from cuvis_ai.training.optimizer_registry import (
                    get_supported_optimizers,
                    get_supported_schedulers,
                )

                training_config = TrainingConfig.model_validate(config_dict)

                # Validate optimizer support
                supported_optimizers = set(get_supported_optimizers())
                optimizer_name = training_config.optimizer.name.lower()
                if optimizer_name not in supported_optimizers:
                    errors.append(
                        f"Unsupported optimizer '{training_config.optimizer.name}'. "
                        f"Supported: {', '.join(sorted(supported_optimizers))}"
                    )
                    valid = False

                # Validate scheduler support and configuration
                scheduler = training_config.scheduler
                if scheduler is not None and scheduler.name:
                    supported_schedulers = set(get_supported_schedulers())
                    scheduler_name = scheduler.name.lower()

                    # Skip validation for empty/none scheduler names
                    if scheduler_name not in {"", "none"}:
                        if scheduler_name not in supported_schedulers:
                            errors.append(
                                f"Unsupported scheduler '{scheduler.name}'. Supported: {', '.join(sorted(supported_schedulers))}"
                            )
                            valid = False

                        # Provide helpful warning for plateau schedulers without monitor
                        if (
                            scheduler_name in {"plateau", "reduce_on_plateau"}
                            and not scheduler.monitor
                        ):
                            warnings.append(
                                "Plateau scheduler configured without 'monitor'; defaulting to val_loss"
                            )

            return cuvis_ai_pb2.ValidateConfigResponse(
                valid=valid,
                errors=errors,
                warnings=warnings,
            )
        except Exception as exc:  # pragma: no cover - safety net
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to validate config: {exc}")
            return cuvis_ai_pb2.ValidateConfigResponse(valid=False, errors=[str(exc)])


__all__ = ["ConfigService"]
