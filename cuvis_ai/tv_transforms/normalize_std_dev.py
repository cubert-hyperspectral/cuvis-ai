from itertools import repeat
from typing import Any

import torch
from torchvision.transforms.v2 import Normalize, Transform


class NormalizeStdDev(Transform):
    """Apply a normalization operation over the input, resulting outputs normalized with a mean close to 0 and a standard deviation of 1.0.
    Assumes the input data is in NCHW memory format.
    If the input is not floating point, will convert the input to float32.

    Parameters
    ----------
    normalize_by_channel : bool, optional
        Apply the normalization for each channel individually. Default: False
    normalize_by_image : bool, optional
        Apply the normalization for each image individually. Default: True
    """

    def __init__(self, *, normalize_by_channel: bool = False, normalize_by_image: bool = True):
        super().__init__()
        self.norm_by_channel = normalize_by_channel
        self.norm_by_image = normalize_by_image

    @staticmethod
    def _norm_image(inpt, norm_by_channel):
        if norm_by_channel:
            if len(inpt.shape) < 4:
                raise RuntimeError(
                    f"NormalizeStdDev: Input has invalid dimensionality - Must be at least NCHW, but is {inpt.shape}"
                )
            # Assuming [...]NCHW dimension ordering -> Determine stddev & mean for each channel
            stds, means = map(
                list, zip(*map(torch.std_mean, torch.split(inpt, 1, dim=len(inpt.shape) - 3)))
            )
        else:
            stds, means = map(list, zip(torch.std_mean(inpt)))
        return Normalize(means, stds, inplace=False)(inpt)

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, torch.Tensor):
            if not torch.is_floating_point(inpt):
                inpt = inpt.to(torch.float32)

            if self.norm_by_image:
                if len(inpt.shape) < 4:
                    raise RuntimeError(
                        f"NormalizeStdDev: Input has invalid dimensionality - Must be at least NCHW, but is {inpt.shape}"
                    )
                img_channel = len(inpt.shape) - 4
                imgs_normed_gen = map(
                    self._norm_image,
                    torch.split(inpt, 1, dim=img_channel),
                    repeat(self.norm_by_channel),
                )
                out = torch.cat(list(imgs_normed_gen), dim=img_channel)
            else:
                out = self._norm_image(inpt, self.norm_by_channel)
            return out.as_subclass(type(inpt))

        return inpt
