"""Helper functions for proto ↔ Python type conversion."""

import cuvis
import numpy as np
import torch

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
}

DTYPE_NUMPY_TO_PROTO = {
    np.dtype(np.float32): cuvis_ai_pb2.D_TYPE_FLOAT32,
    np.dtype(np.float64): cuvis_ai_pb2.D_TYPE_FLOAT64,
    np.dtype(np.int32): cuvis_ai_pb2.D_TYPE_INT32,
    np.dtype(np.int64): cuvis_ai_pb2.D_TYPE_INT64,
    np.dtype(np.uint8): cuvis_ai_pb2.D_TYPE_UINT8,
    np.dtype(bool): cuvis_ai_pb2.D_TYPE_BOOL,
    np.dtype(np.float16): cuvis_ai_pb2.D_TYPE_FLOAT16,
}

DTYPE_TORCH_TO_PROTO = {
    torch.float32: cuvis_ai_pb2.D_TYPE_FLOAT32,
    torch.float64: cuvis_ai_pb2.D_TYPE_FLOAT64,
    torch.int32: cuvis_ai_pb2.D_TYPE_INT32,
    torch.int64: cuvis_ai_pb2.D_TYPE_INT64,
    torch.uint8: cuvis_ai_pb2.D_TYPE_UINT8,
    torch.bool: cuvis_ai_pb2.D_TYPE_BOOL,
    torch.float16: cuvis_ai_pb2.D_TYPE_FLOAT16,
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
