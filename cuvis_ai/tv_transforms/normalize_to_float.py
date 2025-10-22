from itertools import repeat
from typing import Any

import numpy as np
import torch
from torchvision.transforms.v2 import Normalize, Transform


class NormalizeToFloat(Transform):
    """Apply a normalization operation over the input, resulting in outputs scaled to values between [0.0, 1.0].
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
            # Assuming [...]NCHW dimension ordering
            mins, maxs = map(
                list,
                zip(
                    *(
                        (torch.min(ch), torch.max(ch))
                        for ch in torch.split(inpt, 1, dim=len(inpt.shape) - 3)
                    )
                ),
            )
        else:
            mins, maxs = [torch.min(inpt)], [torch.max(inpt)]
        maxs = np.array(maxs) - np.array(mins)
        return Normalize(mins, np.array(maxs), inplace=False)(inpt)

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
