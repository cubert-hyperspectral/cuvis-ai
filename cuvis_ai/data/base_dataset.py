from collections.abc import Callable

import numpy as np
import torch
from torchvision.datasets import VisionDataset

from cuvis_ai.data.output_format import OutputFormat

C_SUPPORTED_DTYPES = (
    np.float64,
    np.float32,
    np.float16,
    np.complex64,
    np.complex128,
    np.int64,
    np.int32,
    np.int16,
    np.int8,
    np.uint8,
    np.bool_,
)


class BaseDataset(VisionDataset):
    def __init__(
        self,
        root: str | None = None,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        output_format: OutputFormat = OutputFormat.Full,
        output_lambda: Callable | None = None,
    ):
        super().__init__(
            root, transforms=transforms, transform=transform, target_transform=target_transform
        )
        self.output_format = output_format
        self.output_lambda = output_lambda
        self.provide_datatype: np.dtype = np.float32

    def _get_return_shape(self, data: np.ndarray, labels: dict, metadata: dict):
        if self.output_format == OutputFormat.Full:
            return (data, labels, metadata)

        elif self.output_format == OutputFormat.BoundingBox:
            return (data, [l["bbox"] for l in labels], [])

        elif self.output_format == OutputFormat.SegmentationMask:
            return (data, [l["segmentation"] for l in labels], [])

        elif self.output_format == OutputFormat.Metadata:
            return (data, [], metadata)

        elif self.output_format == OutputFormat.BoundingBoxWithMeta:
            return (data, [l["bbox"] for l in labels], metadata)

        elif self.output_format == OutputFormat.SegmentationMaskWithMeta:
            return (data, [l["segmentation"] for l in labels], metadata)

        elif self.output_format == OutputFormat.CustomFilter and self.output_lambda is not None:
            return self.output_lambda(data, labels, metadata)

        else:
            raise NotImplementedError("Think about it.")

    def _apply_transform(self, d: dict | torch.Tensor, convert_to_numpy: bool | None = False):
        def unTensorify(source):
            if isinstance(source, dict):
                for k, v in source.items():
                    if isinstance(v, torch.Tensor):
                        source[k] = v.numpy()
                    elif isinstance(v, dict):
                        source[k] = unTensorify(source[k])
            if isinstance(source, torch.Tensor):
                source = source.numpy()
            return source

        ret = self.transforms(d) if self.transforms is not None else d
        return unTensorify(ret) if convert_to_numpy else ret

    def set_datatype(self, dtype: np.dtype):
        """Specify a Numpy datatype to transform the cube into before returning it.
        Valid data types are:
        np.float64, np.float32, np.float16, np.complex64, np.complex128, np.int64, np.int32, np.int16, np.int8, np.uint8, np.bool_
        """
        if dtype in C_SUPPORTED_DTYPES:
            self.provide_datatype = dtype
        else:
            raise ValueError(
                "Unsupported data type: {"
                + str(dtype.name)
                + " - use one of: "
                + str([d.name for d in C_SUPPORTED_DTYPES])
            )

    def get_datatype(self):
        """Get the current datatype set that all data will be converted into before return."""
        return self.provide_datatype
