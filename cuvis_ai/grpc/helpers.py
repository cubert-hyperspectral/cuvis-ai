"""Helper functions for proto ↔ Python type conversion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cuvis
import numpy as np
import torch
import yaml

from .v1 import cuvis_ai_pb2

# Dtype mappings
DTYPE_PROTO_TO_NUMPY = {
    cuvis_ai_pb2.D_TYPE_FLOAT32: np.float32,
    cuvis_ai_pb2.D_TYPE_FLOAT64: np.float64,
    cuvis_ai_pb2.D_TYPE_INT32: np.int32,
    cuvis_ai_pb2.D_TYPE_INT64: np.int64,
    cuvis_ai_pb2.D_TYPE_UINT8: np.uint8,
    cuvis_ai_pb2.D_TYPE_BOOL: bool,
    cuvis_ai_pb2.D_TYPE_FLOAT16: np.float16,
    cuvis_ai_pb2.D_TYPE_UINT16: np.uint16,
}

DTYPE_NUMPY_TO_PROTO = {
    np.dtype(np.float32): cuvis_ai_pb2.D_TYPE_FLOAT32,
    np.dtype(np.float64): cuvis_ai_pb2.D_TYPE_FLOAT64,
    np.dtype(np.int32): cuvis_ai_pb2.D_TYPE_INT32,
    np.dtype(np.int64): cuvis_ai_pb2.D_TYPE_INT64,
    np.dtype(np.uint8): cuvis_ai_pb2.D_TYPE_UINT8,
    np.dtype(bool): cuvis_ai_pb2.D_TYPE_BOOL,
    np.dtype(np.float16): cuvis_ai_pb2.D_TYPE_FLOAT16,
    np.dtype(np.uint16): cuvis_ai_pb2.D_TYPE_UINT16,
}

DTYPE_TORCH_TO_PROTO = {
    torch.float32: cuvis_ai_pb2.D_TYPE_FLOAT32,
    torch.float64: cuvis_ai_pb2.D_TYPE_FLOAT64,
    torch.int32: cuvis_ai_pb2.D_TYPE_INT32,
    torch.int64: cuvis_ai_pb2.D_TYPE_INT64,
    torch.uint8: cuvis_ai_pb2.D_TYPE_UINT8,
    torch.bool: cuvis_ai_pb2.D_TYPE_BOOL,
    torch.float16: cuvis_ai_pb2.D_TYPE_FLOAT16,
    torch.uint16: cuvis_ai_pb2.D_TYPE_UINT16,
}

PROCESSING_MODE_MAP = {
    cuvis_ai_pb2.PROCESSING_MODE_RAW: cuvis.ProcessingMode.Raw,
    cuvis_ai_pb2.PROCESSING_MODE_REFLECTANCE: cuvis.ProcessingMode.Reflectance,
}

TRAIN_STATUS_TO_STRING = {
    cuvis_ai_pb2.TRAIN_STATUS_UNSPECIFIED: "unspecified",
    cuvis_ai_pb2.TRAIN_STATUS_RUNNING: "running",
    cuvis_ai_pb2.TRAIN_STATUS_COMPLETE: "complete",
    cuvis_ai_pb2.TRAIN_STATUS_ERROR: "error",
}

STRING_TO_TRAIN_STATUS = {
    "unspecified": cuvis_ai_pb2.TRAIN_STATUS_UNSPECIFIED,
    "running": cuvis_ai_pb2.TRAIN_STATUS_RUNNING,
    "complete": cuvis_ai_pb2.TRAIN_STATUS_COMPLETE,
    "error": cuvis_ai_pb2.TRAIN_STATUS_ERROR,
}

POINT_TYPE_TO_STRING = {
    cuvis_ai_pb2.POINT_TYPE_UNSPECIFIED: "unspecified",
    cuvis_ai_pb2.POINT_TYPE_POSITIVE: "positive",
    cuvis_ai_pb2.POINT_TYPE_NEGATIVE: "negative",
    cuvis_ai_pb2.POINT_TYPE_NEUTRAL: "neutral",
}

STRING_TO_POINT_TYPE = {
    "unspecified": cuvis_ai_pb2.POINT_TYPE_UNSPECIFIED,
    "positive": cuvis_ai_pb2.POINT_TYPE_POSITIVE,
    "negative": cuvis_ai_pb2.POINT_TYPE_NEGATIVE,
    "neutral": cuvis_ai_pb2.POINT_TYPE_NEUTRAL,
}


def proto_to_numpy(tensor_proto: cuvis_ai_pb2.Tensor, copy: bool = True) -> np.ndarray:
    """Convert proto Tensor to numpy array.

    Args:
        tensor_proto: Proto Tensor message
        copy: If True, return a writable copy. If False, return a read-only view
              of the buffer (zero-copy, but not writable). Default: True

    Returns:
        numpy array with correct shape and dtype

    Raises:
        ValueError: If dtype is not supported
    """
    if tensor_proto.dtype not in DTYPE_PROTO_TO_NUMPY:
        raise ValueError(f"Unsupported dtype: {tensor_proto.dtype}")

    dtype = DTYPE_PROTO_TO_NUMPY[tensor_proto.dtype]
    shape = tuple(tensor_proto.shape)

    # Convert raw bytes to numpy array
    arr = np.frombuffer(tensor_proto.raw_data, dtype=dtype)

    # Reshape if needed
    if shape:
        arr = arr.reshape(shape)

    # Return writable copy if requested (default), otherwise read-only view
    return arr.copy() if copy else arr


def numpy_to_proto(arr: np.ndarray) -> cuvis_ai_pb2.Tensor:
    """Convert numpy array to proto Tensor.

    Args:
        arr: numpy array

    Returns:
        Proto Tensor message

    Raises:
        ValueError: If dtype is not supported
    """
    if arr.dtype not in DTYPE_NUMPY_TO_PROTO:
        raise ValueError(f"Unsupported numpy dtype: {arr.dtype}")

    return cuvis_ai_pb2.Tensor(
        shape=list(arr.shape), dtype=DTYPE_NUMPY_TO_PROTO[arr.dtype], raw_data=arr.tobytes()
    )


def proto_to_tensor(tensor_proto: cuvis_ai_pb2.Tensor) -> torch.Tensor:
    """Convert proto Tensor to PyTorch tensor.

    Args:
        tensor_proto: Proto Tensor message

    Returns:
        PyTorch tensor
    """
    # proto_to_numpy returns a writable copy by default
    arr = proto_to_numpy(tensor_proto)
    return torch.from_numpy(arr)


def tensor_to_proto(tensor: torch.Tensor) -> cuvis_ai_pb2.Tensor:
    """Convert PyTorch tensor to proto Tensor.

    Args:
        tensor: PyTorch tensor

    Returns:
        Proto Tensor message

    Raises:
        ValueError: If dtype is not supported
    """
    if tensor.dtype not in DTYPE_TORCH_TO_PROTO:
        raise ValueError(f"Unsupported torch dtype: {tensor.dtype}")

    # Convert to numpy first to get raw bytes
    arr = tensor.detach().cpu().numpy()

    return cuvis_ai_pb2.Tensor(
        shape=list(tensor.shape), dtype=DTYPE_TORCH_TO_PROTO[tensor.dtype], raw_data=arr.tobytes()
    )


def proto_to_processing_mode(mode: int) -> cuvis.ProcessingMode:
    """Convert proto ProcessingMode to cuvis ProcessingMode.

    Args:
        mode: Proto ProcessingMode enum value

    Returns:
        cuvis.ProcessingMode enum

    Raises:
        ValueError: If mode is not supported
    """
    if mode not in PROCESSING_MODE_MAP:
        raise ValueError(f"Unsupported ProcessingMode: {mode}")

    return PROCESSING_MODE_MAP[mode]


def train_status_to_string(status: int) -> str:
    """Convert proto TrainStatus enum to string.

    Args:
        status: Proto TrainStatus enum value

    Returns:
        String representation of the status

    Raises:
        ValueError: If status is not supported
    """
    if status not in TRAIN_STATUS_TO_STRING:
        raise ValueError(f"Unsupported TrainStatus: {status}")

    return TRAIN_STATUS_TO_STRING[status]


def string_to_train_status(status: str) -> int:
    """Convert string to proto TrainStatus enum.

    Args:
        status: String representation of status

    Returns:
        Proto TrainStatus enum value

    Raises:
        ValueError: If status string is not supported
    """
    if status not in STRING_TO_TRAIN_STATUS:
        raise ValueError(f"Unsupported status string: {status}")

    return STRING_TO_TRAIN_STATUS[status]


def point_type_to_string(point_type: int) -> str:
    """Convert proto PointType enum to string.

    Args:
        point_type: Proto PointType enum value

    Returns:
        String representation of the point type

    Raises:
        ValueError: If point type is not supported
    """
    if point_type not in POINT_TYPE_TO_STRING:
        raise ValueError(f"Unsupported PointType: {point_type}")

    return POINT_TYPE_TO_STRING[point_type]


def string_to_point_type(point_type: str) -> int:
    """Convert string to proto PointType enum.

    Args:
        point_type: String representation of point type

    Returns:
        Proto PointType enum value

    Raises:
        ValueError: If point type string is not supported
    """
    if point_type not in STRING_TO_POINT_TYPE:
        raise ValueError(f"Unsupported point type string: {point_type}")

    return STRING_TO_POINT_TYPE[point_type]


# ------------------------------------------------------------------
# Pipeline Path Resolution
# ------------------------------------------------------------------


def get_server_base_dir() -> Path:
    """Get the base directory for server configurations.

    Returns:
        Path to the configs directory

    Note:
        This defaults to ./configs relative to the current working directory.
        Can be overridden via CUVIS_CONFIGS_DIR environment variable.
    """
    env_dir = os.environ.get("CUVIS_CONFIGS_DIR")
    if env_dir:
        return Path(env_dir)

    return Path.cwd() / "configs"


def resolve_pipeline_path(config_path: str, base_dir: Path | None = None) -> Path:
    """Resolve pipeline configuration path with fallback logic.

    Resolution order:
    1. If absolute path exists, use it
    2. If relative path from CWD exists, use it
    3. Try as short name in base_dir (e.g., "rx_statistical" -> "configs/pipeline/rx_statistical.yaml")
    4. Try as short name with .yaml extension in base_dir
    5. Fallback to default ./configs/pipeline even when CUVIS_CONFIGS_DIR is set

    Args:
        config_path: Pipeline configuration path (absolute, relative, or short name)
        base_dir: Base directory for pipeline configs (defaults to ./configs/pipeline)

    Returns:
        Resolved Path to pipeline YAML file

    Raises:
        FileNotFoundError: If pipeline file cannot be found
    """
    if base_dir is None:
        base_dir = get_server_base_dir() / "pipeline"

    default_base_dir = Path.cwd() / "configs" / "pipeline"

    # Try as absolute path
    path = Path(config_path)
    if path.is_absolute() and path.exists():
        return path

    # Try as relative path from CWD
    if path.exists():
        return path.resolve()

    def _try_base(dir_path: Path) -> Path | None:
        candidate = dir_path / config_path
        if candidate.exists():
            return candidate

        if not config_path.endswith(".yaml"):
            candidate_with_ext = dir_path / f"{config_path}.yaml"
            if candidate_with_ext.exists():
                return candidate_with_ext
        return None

    # Try with configured base_dir first
    resolved = _try_base(base_dir)
    if resolved:
        return resolved

    # Fallback to default base_dir even if env override is set
    if default_base_dir != base_dir:
        resolved = _try_base(default_base_dir)
        if resolved:
            return resolved

    # Not found - provide helpful error message
    tried_paths = [
        str(Path(config_path).resolve()),
        str(base_dir / config_path),
    ]
    if not config_path.endswith(".yaml"):
        tried_paths.append(str(base_dir / f"{config_path}.yaml"))
        if default_base_dir != base_dir:
            tried_paths.append(str(default_base_dir / f"{config_path}.yaml"))
    if default_base_dir != base_dir:
        tried_paths.append(str(default_base_dir / config_path))

    raise FileNotFoundError(
        f"Pipeline configuration not found: '{config_path}'\n"
        f"Tried paths:\n" + "\n".join(f"  - {p}" for p in tried_paths)
    )


def resolve_pipeline_save_path(pipeline_path: str, base_dir: Path | None = None) -> Path:
    """Resolve pipeline save path for writing.

    If the path is absolute, use it as-is.
    If the path is relative, resolve it relative to base_dir.

    Args:
        pipeline_path: Path to save pipeline (absolute or relative)
        base_dir: Base directory for relative paths (defaults to ./configs/pipeline)

    Returns:
        Resolved Path where pipeline should be saved
    """
    if base_dir is None:
        base_dir = get_server_base_dir() / "pipeline"

    path = Path(pipeline_path)
    if path.is_absolute():
        return path

    # Relative path - resolve relative to base_dir
    return base_dir / path


# ------------------------------------------------------------------
# Pipeline Discovery
# ------------------------------------------------------------------


def extract_pipeline_metadata(yaml_path: Path) -> dict[str, Any]:
    """Extract metadata from a pipeline YAML file.

    Args:
        yaml_path: Path to pipeline YAML file

    Returns:
        Dictionary with metadata fields:
        - name: str
        - description: str
        - created: str
        - cuvis_ai_version: str
        - metrics: dict[str, float]
        - epoch: int
        - tags: list[str]

    Note:
        Returns empty/default values if metadata section is missing or invalid.
    """
    try:
        with yaml_path.open("r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return _default_metadata()

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        # Extract tags from metadata or infer from node types
        tags = metadata.get("tags", [])
        if not tags:
            tags = _infer_tags_from_pipeline(data)

        return {
            "name": metadata.get("name", yaml_path.stem),
            "description": metadata.get("description", ""),
            "created": metadata.get("created", ""),
            "cuvis_ai_version": metadata.get("cuvis_ai_version", ""),
            "tags": tags,
            "author": metadata.get("author", ""),
        }
    except Exception:
        return _default_metadata()


def _default_metadata() -> dict[str, Any]:
    """Return default metadata structure."""
    return {
        "name": "",
        "description": "",
        "created": "",
        "cuvis_ai_version": "",
        "tags": [],
        "author": "",
    }


def _infer_tags_from_pipeline(pipeline_data: dict) -> list[str]:
    """Infer tags from pipeline node types.

    Args:
        pipeline_data: Parsed pipeline YAML data

    Returns:
        List of inferred tags
    """
    tags = []
    nodes = pipeline_data.get("nodes", [])

    # Check for anomaly detection nodes
    anomaly_keywords = ["rx", "anomaly", "detector", "deep_svdd", "lad"]
    for node in nodes:
        if isinstance(node, dict):
            node_type = node.get("type", "").lower()
            if any(keyword in node_type for keyword in anomaly_keywords):
                if "anomaly" not in tags:
                    tags.append("anomaly")
                break

    # Check for segmentation nodes
    segmentation_keywords = ["segment", "mask", "selector"]
    for node in nodes:
        if isinstance(node, dict):
            node_type = node.get("type", "").lower()
            if any(keyword in node_type for keyword in segmentation_keywords):
                if "segmentation" not in tags:
                    tags.append("segmentation")
                break

    return tags


def list_available_pipelinees(
    base_dir: str | Path | None = None, filter_tag: str | None = None
) -> list[dict[str, Any]]:
    """List all available pipeline configurations.

    Args:
        base_dir: Base directory to search for pipeline files (defaults to server base dir / pipeline)
        filter_tag: Optional tag to filter pipelinees (e.g., "anomaly", "segmentation")

    Returns:
        List of dictionaries with pipeline information:
        - name: str (short name without .yaml extension)
        - path: str (full path to pipeline file)
        - metadata: dict (pipeline metadata)
        - tags: list[str] (pipeline tags)
        - has_weights: bool (whether .pt file exists)
        - weights_path: str (path to .pt file if exists, empty string otherwise)

    Raises:
        FileNotFoundError: If base directory does not exist
    """
    if base_dir is None:
        base_dir_path = get_server_base_dir() / "pipeline"
    else:
        base_dir_path = Path(base_dir) if isinstance(base_dir, str) else base_dir

    if not base_dir_path.exists():
        raise FileNotFoundError(f"Pipeline base directory not found: {base_dir_path}")

    pipelinees = []

    # Find all .yaml files in base directory
    for yaml_path in base_dir_path.glob("*.yaml"):
        metadata = extract_pipeline_metadata(yaml_path)

        # Apply tag filter if provided
        if filter_tag is not None:
            if filter_tag.lower() not in [tag.lower() for tag in metadata["tags"]]:
                continue

        # Check for co-located .pt file
        pt_path = yaml_path.with_suffix(".pt")
        has_weights = pt_path.exists()

        pipelinees.append(
            {
                "name": yaml_path.stem,
                "path": str(yaml_path),
                "metadata": metadata,
                "tags": metadata["tags"],
                "has_weights": has_weights,
                "weights_path": str(pt_path) if has_weights else "",
            }
        )

    return pipelinees


def get_pipeline_info(
    pipeline_name: str,
    base_dir: str | Path | None = None,
    include_yaml_content: bool = False,
) -> dict[str, Any]:
    """Get detailed information about a specific pipeline.

    Args:
        pipeline_name: Pipeline short name (e.g., "rx_statistical")
        base_dir: Base directory for pipeline configs (defaults to server base dir / pipeline)
        include_yaml_content: Whether to include full YAML content in response

    Returns:
        Dictionary with pipeline information:
        - name: str
        - path: str
        - metadata: dict
        - tags: list[str]
        - has_weights: bool
        - weights_path: str
        - yaml_content: str (optional)

    Raises:
        FileNotFoundError: If pipeline file cannot be found
    """
    # Resolve pipeline path
    yaml_path = resolve_pipeline_path(pipeline_name, base_dir)

    # Extract metadata
    metadata = extract_pipeline_metadata(yaml_path)

    # Check for weights
    pt_path = yaml_path.with_suffix(".pt")
    has_weights = pt_path.exists()

    result = {
        "name": yaml_path.stem,
        "path": str(yaml_path),
        "metadata": metadata,
        "tags": metadata["tags"],
        "has_weights": has_weights,
        "weights_path": str(pt_path) if has_weights else "",
    }

    # Optionally include YAML content
    if include_yaml_content:
        try:
            with yaml_path.open("r") as f:
                result["yaml_content"] = f.read()
        except Exception:
            result["yaml_content"] = ""

    return result
