from typing import Any

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import TVTensor


class WavelengthList(TVTensor):
    """A torchvision transforms data type which represents a list of wavelengths.
    Is used in conjunction with a HSI data cube to describe the physical wavelengths that the channels of the cube represent.

    Parameters
    ----------
    data : List(float) or np.ndarray
        The wavelength list
    dtype : torch.dtype, optional
        The data type of :arg:`data`
    device : torch.device or str or int, optional
        Where the list should be stored
    requires_grad : bool, optional
        Whether autograd should record operations.
    """

    def __new__(
        cls,
        data: list[float] | np.ndarray,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ):
        if not isinstance(data, list):
            raise ValueError("WavelengthList got invalid input data!")
        # data = np.array([float(wl) for wl in data]).reshape((len(data), 1, 1))
        data = np.array(data).reshape((1, len(data), 1, 1))
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        ).float()
        return tensor.as_subclass(cls)

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr()


@F.register_kernel(functional=F.erase_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.erase_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_brightness, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_brightness_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_brightness_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_contrast, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_contrast_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_contrast_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_gamma, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_gamma_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_gamma_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_hue, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_hue_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_hue_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_saturation, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_saturation_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_saturation_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_sharpness, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_sharpness_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.adjust_sharpness_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.autocontrast, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.autocontrast_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.autocontrast_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.equalize, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.equalize_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.equalize_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.invert_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.invert_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.posterize, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.posterize_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.posterize_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.solarize, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.solarize_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.solarize_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.affine, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.affine_bounding_boxes, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.affine_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.affine_mask, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.affine_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.center_crop, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.center_crop_bounding_boxes, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.center_crop_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.center_crop_mask, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.center_crop_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.crop, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.crop_bounding_boxes, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.crop_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.crop_mask, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.crop_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.elastic, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.elastic_bounding_boxes, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.elastic_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.elastic_mask, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.elastic_transform, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.elastic_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.five_crop, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.five_crop_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.five_crop_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.hflip, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.horizontal_flip, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.horizontal_flip_bounding_boxes, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.horizontal_flip_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.horizontal_flip_mask, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.horizontal_flip_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.pad, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.pad_bounding_boxes, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.pad_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.pad_mask, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.pad_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.perspective, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.perspective_bounding_boxes, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.perspective_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.perspective_mask, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.perspective_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.resize, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.resize_bounding_boxes, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.resize_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.resize_mask, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.resize_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.resized_crop, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.resized_crop_bounding_boxes, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.resized_crop_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.resized_crop_mask, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.resized_crop_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.rotate, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.rotate_bounding_boxes, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.rotate_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.rotate_mask, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.rotate_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.ten_crop, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.ten_crop_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.ten_crop_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.vertical_flip, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.vertical_flip_bounding_boxes, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.vertical_flip_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.vertical_flip_mask, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.vertical_flip_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.vflip, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.convert_image_dtype, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.gaussian_blur, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.gaussian_blur_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.gaussian_blur_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.normalize, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.normalize_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.normalize_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.to_dtype, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.to_dtype_image, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.to_dtype_video, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.uniform_temporal_subsample, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.uniform_temporal_subsample_video, tv_tensor_cls=WavelengthList)
def wllist_noop(d, *args, **kwargs):
    return d


@F.register_kernel(functional=F.to_grayscale, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.rgb_to_grayscale, tv_tensor_cls=WavelengthList)
@F.register_kernel(functional=F.rgb_to_grayscale_image, tv_tensor_cls=WavelengthList)
def wllist_squash(d, *args, **kwargs):
    return torch.median(d).as_subclass(type(d))
