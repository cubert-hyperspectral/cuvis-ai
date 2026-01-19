"""Centralized session management fixtures for gRPC tests."""

import logging
import tempfile
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import grpc
import pytest
import yaml

from cuvis_ai_core.grpc import cuvis_ai_pb2
from cuvis_ai_core.training.config import DataConfig, TrainRunConfig

# Configure logging for session management
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries, giving precedence to override."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def materialize_trainrun_config(trainrun_path: str) -> str:
    """Resolve Hydra-style defaults into a concrete trainrun YAML for testing."""
    trainrun_file = Path(trainrun_path)
    raw = yaml.safe_load(trainrun_file.read_text())
    if not isinstance(raw, dict) or "defaults" not in raw:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w", encoding="utf-8")
        yaml.safe_dump(raw, tmp, sort_keys=False)
        return tmp.name

    base_dir = trainrun_file.parent.parent
    resolved: dict = {k: v for k, v in raw.items() if k != "defaults"}

    for entry in raw.get("defaults", []):
        if not isinstance(entry, dict):
            continue
        for raw_key, name in entry.items():
            key = raw_key.split("@")[-1].lstrip("/") if isinstance(raw_key, str) else str(raw_key)
            subdir = base_dir / key
            candidate = subdir / f"{name}.yaml"
            if candidate.exists():
                default_cfg = yaml.safe_load(candidate.read_text())
                existing_value = resolved.get(key)

                if isinstance(default_cfg, dict) and isinstance(existing_value, dict):
                    # Merge defaults with explicit overrides (overrides win)
                    resolved[key] = _deep_merge(default_cfg, existing_value)
                elif key not in resolved:
                    resolved[key] = default_cfg
                else:
                    resolved[key] = existing_value

    # Hoist training-scoped fields (Hydra-style) to top-level TrainRunConfig fields
    training_cfg = resolved.get("training")
    if isinstance(training_cfg, dict):
        for field in ("loss_nodes", "metric_nodes", "unfreeze_nodes", "freeze_nodes", "output_dir"):
            if field in training_cfg and field not in resolved:
                resolved[field] = training_cfg.pop(field)

        # Normalize optimizer scheduler to TrainingConfig.scheduler
        optimizer_cfg = training_cfg.get("optimizer")
        if isinstance(optimizer_cfg, dict) and "scheduler" in optimizer_cfg:
            scheduler_cfg = optimizer_cfg.pop("scheduler")
            if "scheduler" in training_cfg and isinstance(training_cfg["scheduler"], dict):
                training_cfg["scheduler"] = _deep_merge(training_cfg["scheduler"], scheduler_cfg)
            else:
                training_cfg["scheduler"] = scheduler_cfg

        # Align callback naming with pydantic schema
        trainer_cfg = training_cfg.get("trainer")
        if isinstance(trainer_cfg, dict):
            callbacks_cfg = trainer_cfg.get("callbacks")
            if isinstance(callbacks_cfg, dict) and "model_checkpoint" in callbacks_cfg:
                checkpoint_cfg = callbacks_cfg.pop("model_checkpoint")
                if isinstance(checkpoint_cfg, dict):
                    allowed_checkpoint_keys = {
                        "dirpath",
                        "filename",
                        "monitor",
                        "mode",
                        "save_top_k",
                        "every_n_epochs",
                        "save_last",
                        "auto_insert_metric_name",
                    }
                    callbacks_cfg["checkpoint"] = {
                        k: v for k, v in checkpoint_cfg.items() if k in allowed_checkpoint_keys
                    }
                else:
                    callbacks_cfg["checkpoint"] = checkpoint_cfg

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w", encoding="utf-8")
    yaml.safe_dump(resolved, tmp, sort_keys=False)
    return tmp.name


def _safe_close_session(grpc_stub: Any, session_id: str, max_retries: int = 3) -> bool:
    """Safe session closure with retries and error handling.

    Args:
        grpc_stub: gRPC stub for making CloseSession calls
        session_id: Session ID to close
        max_retries: Maximum number of retry attempts

    Returns:
        bool: True if session was successfully closed, False otherwise
    """
    for attempt in range(max_retries):
        try:
            grpc_stub.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session_id))
            logger.debug(f"Successfully closed session {session_id}")
            return True
        except grpc.RpcError as e:
            if attempt == max_retries - 1:
                logger.warning(
                    f"Failed to close session {session_id} after {max_retries} attempts: {e}"
                )
                return False
            # Exponential backoff
            sleep_time = 0.1 * (attempt + 1)
            logger.debug(
                f"Retry {attempt + 1}/{max_retries} closing session {session_id} in {sleep_time}s"
            )
            time.sleep(sleep_time)
    return False


@pytest.fixture
def trained_pipeline_session(
    grpc_stub: Any, test_data_files_cached: tuple[Path, Path]
) -> Generator[Callable[[str], str], None, None]:
    """Factory for creating sessions with pipeline and statistical training.

    This is a simpler alternative to trained_session that creates a session
    from a pipeline (not experiment) and runs statistical training. Useful for
    basic inference tests that don't need full experiment config.

    Args:
        grpc_stub: In-process gRPC stub fixture
        test_data_files_cached: Fixture providing validated (cu3s, json) paths

    Yields:
        Callable[[str], str]: Function returning session_id
    """
    cu3s_file, json_file = test_data_files_cached
    created_sessions: list[str] = []

    def _create_trained_pipeline_session(pipeline_path: str = "channel_selector") -> str:
        # Step 1: Create empty session
        response = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = response.session_id
        created_sessions.append(session_id)

        # Step 2: Resolve + load pipeline using new API
        config_response = grpc_stub.ResolveConfig(
            cuvis_ai_pb2.ResolveConfigRequest(
                session_id=session_id,
                config_type="pipeline",
                path=f"pipeline/{pipeline_path}",
            )
        )
        load_response = grpc_stub.LoadPipeline(
            cuvis_ai_pb2.LoadPipelineRequest(
                session_id=session_id,
                pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=config_response.config_bytes),
            )
        )
        assert load_response.success

        # Create data config for statistical training
        data_config = DataConfig(
            cu3s_file_path=str(cu3s_file),
            annotation_json_path=str(json_file),
            train_ids=[0, 1, 2],
            val_ids=[3, 4],
            test_ids=[5, 6],
            batch_size=2,
            processing_mode="Reflectance",
        ).to_proto()

        # Run statistical training
        stat_req = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
            data=data_config,
        )
        for _ in grpc_stub.Train(stat_req):
            pass

        return session_id

    yield _create_trained_pipeline_session

    # Cleanup with improved error handling
    for session_id in created_sessions:
        _safe_close_session(grpc_stub, session_id)


@pytest.fixture
def session(grpc_stub: Any) -> Generator[Callable[[str], str], None, None]:
    """Factory for creating sessions with auto-cleanup.

    Creates a basic session with the specified pipeline type and tracks all
    created session IDs for cleanup after the test finishes.

    Args:
        grpc_stub: In-process gRPC stub fixture

    Yields:
        Callable[[str], str]: Function that creates a session and returns its ID
    """
    created_sessions: list[str] = []

    def _create_session(pipeline_type: str = "channel_selector") -> str:
        # Step 1: Create empty session
        response = grpc_stub.CreateSession(cuvis_ai_pb2.CreateSessionRequest())
        session_id = response.session_id
        created_sessions.append(session_id)

        # Step 2: Resolve + load pipeline using new API
        config_response = grpc_stub.ResolveConfig(
            cuvis_ai_pb2.ResolveConfigRequest(
                session_id=session_id,
                config_type="pipeline",
                path=f"pipeline/{pipeline_type}",
            )
        )
        load_response = grpc_stub.LoadPipeline(
            cuvis_ai_pb2.LoadPipelineRequest(
                session_id=session_id,
                pipeline=cuvis_ai_pb2.PipelineConfig(config_bytes=config_response.config_bytes),
            )
        )
        assert load_response.success
        return session_id

    yield _create_session

    # Improved cleanup with error handling
    for session_id in created_sessions:
        _safe_close_session(grpc_stub, session_id)


@pytest.fixture
def trained_session(
    grpc_stub: Any, test_data_files_cached: tuple[Path, Path]
) -> Generator[Callable[[str], tuple[str, cuvis_ai_pb2.DataConfig]], None, None]:
    """Factory for creating sessions with statistical training completed.

    Creates a session using RestoreTrainRun to load full config
    (including loss_nodes and metric_nodes), then runs statistical training.
    This ensures gradient training tests have access to all required config.

    Args:
        grpc_stub: In-process gRPC stub fixture
        test_data_files_cached: Fixture providing cached (cu3s, json) paths

    Yields:
        Callable[[str], tuple[str, cuvis_ai_pb2.DataConfig]]:
            Function returning (session_id, data_config)
    """
    cu3s_file, json_file = test_data_files_cached
    created_sessions: list[str] = []

    def _create_trained_session(
        trainrun_path: str = "configs/trainrun/deep_svdd.yaml",
    ) -> tuple[str, cuvis_ai_pb2.DataConfig]:
        # Resolve Hydra defaults for tests, then RestoreTrainRun to load full config
        resolved_path = materialize_trainrun_config(trainrun_path)
        restore_req = cuvis_ai_pb2.RestoreTrainRunRequest(trainrun_path=resolved_path)
        response = grpc_stub.RestoreTrainRun(restore_req)
        session_id = response.session_id
        created_sessions.append(session_id)

        # Get data config from the restored trainrun
        trainrun_config = TrainRunConfig.from_proto(response.trainrun)
        data_config = trainrun_config.data.to_proto()

        # Run statistical training
        stat_req = cuvis_ai_pb2.TrainRequest(
            session_id=session_id,
            trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL,
        )
        for _ in grpc_stub.Train(stat_req):
            pass

        return session_id, data_config

    yield _create_trained_session

    # Cleanup with improved error handling
    for session_id in created_sessions:
        _safe_close_session(grpc_stub, session_id)
