"""Standardization ``(x - mean) / std`` along a chosen axis, with fitted, per-sample or given statistics."""

from __future__ import annotations

import math
from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor, nn

from cuvis_ai_core.node import Node
from cuvis_ai_core.node.fit_utils import require_fitted

#: ImageNet channel statistics, for ``Standardize(stats="given", layout="nchw", ...)`` in front of
#: an ImageNet-pretrained backbone.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_AXES = ("channel", "spectrum", "global", "spatial", "location")
_STATS = ("fit", "sample", "given")
_LAYOUTS = ("bhwc", "nchw")
_EPS_MODES = ("floor", "add")
# Dims each named axis reduces within one sample; fitted statistics also reduce the batch dim.
# "location" reduces nothing inside a sample: the fit keeps one statistic per pixel position
# and band, so it is the one named axis whose statistics are tied to one frame size.
_AXIS_DIMS = {
    "bhwc": {
        "channel": (1, 2),
        "spatial": (1, 2),
        "spectrum": (3,),
        "global": (1, 2, 3),
        "location": (),
    },
    "nchw": {
        "channel": (2, 3),
        "spatial": (2, 3),
        "spectrum": (1,),
        "global": (1, 2, 3),
        "location": (),
    },
}
_CHANNEL_DIM = {"bhwc": 3, "nchw": 1}
_SPATIAL_DIMS = {"bhwc": {1, 2}, "nchw": {2, 3}}


class Standardize(Node):
    """Subtract a mean and divide by a standard deviation along a chosen axis.

    ``axis`` names what the statistics are taken over:

    - ``"channel"``: one mean and std per wavelength band over every pixel (autoscaling);
    - ``"spectrum"``: one mean and std per spectrum over its bands (SNV);
    - ``"global"``: one scalar mean and std over every value;
    - ``"spatial"``: one mean and std per band over the height and width of each sample;
    - ``"location"``: one mean and std per pixel position and band, fit over the training stream;
    - a list of dims (``[1, 2, 3]``, ``[3]``): reduce exactly those dims.

    ``stats`` says where the numbers come from: ``"fit"`` estimates them once over the training
    stream during statistical initialization and keeps them as buffers that travel with the
    weights (the default for ``channel``, ``global`` and ``location``); ``"sample"`` computes
    them for every sample in ``forward`` (the default for ``spectrum``, ``spatial`` and a list
    of dims); ``"given"`` takes ``mean`` and ``std`` as hparams (per channel, or one value each
    with ``axis="global"``). ``axis="location"`` takes only ``stats="fit"``; ``"sample"`` and
    ``"given"`` raise ``ValueError`` for it, because a per-sample or given statistic shaped like
    one frame has no meaning for a sample of a different size. ``with_mean=False`` only scales,
    ``with_std=False`` only centres. ``layout="nchw"`` reads the cube as ``[N, C, H, W]`` images
    instead of ``[B, H, W, C]``.

    ``ddof`` (delta degrees of freedom) is the number subtracted from the sample count in the
    divisor of the variance. ``ddof=0`` divides by N (the population standard deviation; the numpy
    and sklearn ``StandardScaler`` default). ``ddof=1`` divides by N - 1 (the Bessel-corrected
    sample standard deviation; the torch ``std`` default). The two differ by the factor
    sqrt((N - 1) / N), a uniform scalar on the data that PLS predictions and component selection
    are invariant to, but reproduced numbers depend on it. The default is 1.

    ``eps`` keeps the division finite, so a flat spectrum or a constant band maps to zero instead
    of NaN. By default (``eps_mode="floor"``) it is a floor on the standard deviation and leaves
    every other value untouched; ``eps_mode="add"`` divides by ``std + eps`` instead, the formula
    of the removed z-score nodes, for pipelines that have to reproduce their numbers bit for bit.

    A fit pools over the height and width of every frame, so ``stats="fit"`` takes ``channel``,
    ``spatial``, ``global`` or a list of dims that contains both spatial dims; per-spectrum
    statistics (``spectrum``) are computed per sample. ``axis="location"`` is the one named
    exception: it pools over nothing inside a frame, so the fitted statistic keeps the full
    ``[1, H, W, C]`` (or ``[1, C, H, W]``) shape and is tied to one frame size, which every later
    ``forward`` call must match exactly (a mismatch, including a broadcastable singleton dim,
    raises ``ValueError``). Its fit holds two float64 accumulators the size of one frame (the
    running mean and the running sum of squared deviations), 16 bytes per value of a frame:
    about 1 GB for a 1024 x 1024 x 61 frame, and a few times that at the peak while a batch
    is merged. The fitted node keeps a float32 mean and std of that size. With ``stats="fit"``,
    ``exclude_folds`` leaves the frames whose ``fold`` id is listed out of the fit (the node then
    has a ``fold [B]`` input), and ``max_initialization_frames`` stops the fit after that many
    frames.

    ``mean`` and ``std`` are listed in ``TRAINABLE_BUFFERS``, so a trainrun that unfreezes the
    node fine-tunes them by gradient.

    Replaces: ``MeanCenter`` is ``Standardize(axis="channel", with_std=False)``,
    ``UnitVarianceScaling`` is ``Standardize(axis="channel", with_mean=False)``,
    ``SNVCorrection`` is ``Standardize(axis="spectrum")``, ``ZScoreNormalizer`` is
    ``Standardize(axis="spatial", ddof=0, eps=1e-6, eps_mode="add")``, ``ChannelNormalizeNode``
    is ``Standardize(stats="given", layout="nchw", mean=IMAGENET_MEAN, std=IMAGENET_STD)`` and
    ``ZScoreNormalizerGlobal`` is ``Standardize(axis="channel", eps_mode="add")``.

    Parameters
    ----------
    axis : str or list[int], optional
        ``"channel"``, ``"spectrum"``, ``"global"``, ``"spatial"``, ``"location"`` or a list of
        dims in 1..3 (default ``"channel"``).
    with_mean : bool, optional
        Subtract the mean (default True).
    with_std : bool, optional
        Divide by the standard deviation (default True).
    stats : str or None, optional
        ``"fit"``, ``"sample"`` or ``"given"``; None picks the default for ``axis``.
    mean, std : list[float] or None, optional
        The statistics for ``stats="given"``, one value per channel or one value in total.
    ddof : int, optional
        Delta degrees of freedom, 0 or 1 (default 1).
    eps : float, optional
        Keeps the division finite (default 1e-8); how it applies is set by ``eps_mode``.
    eps_mode : str, optional
        ``"floor"`` (default) divides by ``max(std, eps)``, ``"add"`` by ``std + eps``.
    exclude_folds : list[int] or None, optional
        Fold ids left out of the fit (``stats="fit"`` only).
    max_initialization_frames : int or None, optional
        Cap on the frames the fit consumes (``stats="fit"`` only); None uses the whole stream.
    layout : str, optional
        ``"bhwc"`` (default) or ``"nchw"``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {
            NodeTag.HYPERSPECTRAL,
            NodeTag.PREPROCESSING,
            NodeTag.NORMALIZATION,
            NodeTag.TORCH,
            NodeTag.ONNX,
        }
    )

    TRAINABLE_BUFFERS = ("mean", "std")

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input cube [B, H, W, C] (or [N, C, H, W] with layout nchw)",
        ),
        "fold": PortSpec(
            dtype=torch.int32,
            shape=(-1,),
            description="Per-frame fold id; present only when exclude_folds is set",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Standardized cube, same shape and layout as the input",
        )
    }

    def __init__(
        self,
        axis: str | list[int] = "channel",
        with_mean: bool = True,
        with_std: bool = True,
        stats: str | None = None,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        ddof: int = 1,
        eps: float = 1e-8,
        eps_mode: str = "floor",
        exclude_folds: list[int] | None = None,
        max_initialization_frames: int | None = None,
        layout: str = "bhwc",
        **kwargs: Any,
    ) -> None:
        if layout not in _LAYOUTS:
            raise ValueError(f"layout must be one of {_LAYOUTS}, got {layout!r}")
        axis = self._check_axis(axis)
        if stats is None:
            stats = "fit" if axis in ("channel", "global", "location") else "sample"
        if stats not in _STATS:
            raise ValueError(f"stats must be one of {_STATS}, got {stats!r}")
        if axis == "location" and stats != "fit":
            raise ValueError(
                f"axis='location' takes only stats='fit', got {stats!r}; a per-sample or "
                "given statistic shaped like one frame has no meaning for another frame size"
            )
        if not (with_mean or with_std):
            raise ValueError("with_mean and with_std are both False; the node would do nothing")
        if ddof not in (0, 1):
            raise ValueError(f"ddof must be 0 or 1, got {ddof!r}")
        if eps_mode not in _EPS_MODES:
            raise ValueError(f"eps_mode must be one of {_EPS_MODES}, got {eps_mode!r}")
        dims = tuple(axis) if isinstance(axis, list) else _AXIS_DIMS[layout][axis]
        if stats == "fit" and axis != "location" and not _SPATIAL_DIMS[layout] <= set(dims):
            raise ValueError(
                f"stats='fit' needs an axis that reduces height and width, got {axis!r}; "
                "the statistics would be tied to one frame size (use stats='sample', or "
                "axis='location' to opt into statistics tied to one frame size)"
            )
        if stats != "fit" and (exclude_folds is not None or max_initialization_frames is not None):
            raise ValueError("exclude_folds and max_initialization_frames need stats='fit'")
        if max_initialization_frames is not None and int(max_initialization_frames) <= 0:
            raise ValueError(
                "max_initialization_frames must be None or a positive integer, "
                f"got {max_initialization_frames}"
            )
        mean, std = self._check_given(stats, axis, with_mean, with_std, mean, std)

        self.axis = axis
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)
        self.stats = stats
        self.ddof = int(ddof)
        self.eps = float(eps)
        self.eps_mode = eps_mode
        self.exclude_folds = None if exclude_folds is None else [int(f) for f in exclude_folds]
        self.max_initialization_frames = (
            None if max_initialization_frames is None else int(max_initialization_frames)
        )
        self.layout = layout
        self._dims = dims
        self._requires_initial_fit_override = stats == "fit"
        if self.exclude_folds is None:
            self.INPUT_SPECS = {"cube": type(self).INPUT_SPECS["cube"]}

        super().__init__(
            axis=axis,
            with_mean=self.with_mean,
            with_std=self.with_std,
            stats=stats,
            mean=mean,
            std=std,
            ddof=self.ddof,
            eps=self.eps,
            eps_mode=eps_mode,
            exclude_folds=self.exclude_folds,
            max_initialization_frames=self.max_initialization_frames,
            layout=layout,
            **kwargs,
        )

        if stats == "given":
            shape = [1, 1, 1, 1]
            shape[_CHANNEL_DIM[layout]] = -1
            given = mean if mean is not None else std
            assert given is not None  # _check_given guarantees one of them
            n = len(given)
            self.register_buffer("mean", torch.tensor(mean or [0.0] * n).view(shape))
            self.register_buffer("std", torch.tensor(std or [1.0] * n).view(shape))
        else:
            # Fitted statistics travel with the weights; per-sample ones are computed in forward,
            # so their placeholders are not saved.
            persistent = stats == "fit"
            self.register_buffer("mean", torch.zeros(0), persistent=persistent)
            self.register_buffer("std", torch.zeros(0), persistent=persistent)

    @staticmethod
    def _check_axis(axis: str | list[int]) -> str | list[int]:
        if isinstance(axis, str):
            if axis not in _AXES:
                raise ValueError(f"axis must be one of {_AXES} or a list of dims, got {axis!r}")
            return axis
        dims = sorted({int(d) for d in axis})
        if not dims or any(d not in (1, 2, 3) for d in dims):
            raise ValueError(
                f"axis as a list takes dims in 1..3 (dim 0 is the batch), got {axis!r}"
            )
        return dims

    @staticmethod
    def _check_given(
        stats: str,
        axis: str | list[int],
        with_mean: bool,
        with_std: bool,
        mean: list[float] | None,
        std: list[float] | None,
    ) -> tuple[list[float] | None, list[float] | None]:
        if stats != "given":
            if mean is not None or std is not None:
                raise ValueError("mean and std are hparams of stats='given' only")
            return None, None
        if axis not in ("channel", "global"):
            raise ValueError("stats='given' takes axis 'channel' (per channel) or 'global'")
        if (with_mean and mean is None) or (with_std and std is None):
            raise ValueError("stats='given' needs mean and std for the parts it applies")
        mean = None if mean is None else [float(v) for v in mean]
        std = None if std is None else [float(v) for v in std]
        lengths = {len(v) for v in (mean, std) if v is not None}
        if len(lengths) != 1 or (axis == "global" and lengths != {1}):
            raise ValueError("mean and std need the same length (1 with axis='global')")
        return mean, std

    def unfreeze(self) -> None:
        """Make fitted or given statistics trainable; a per-sample instance has none to train.

        The base class turns every ``TRAINABLE_BUFFERS`` entry into a Parameter, which would put
        the per-sample placeholders back into ``state_dict()``. They stay unsaved buffers here.
        """
        super().unfreeze()
        if self.stats == "sample":
            for name in self.TRAINABLE_BUFFERS:
                placeholder = getattr(self, name).detach()
                delattr(self, name)
                self.register_buffer(name, placeholder, persistent=False)

    def _set_stat(self, name: str, value: Tensor) -> None:
        """Write a statistic, keeping it a Parameter when the node is unfrozen."""
        value = value.detach().to(torch.float32)
        param = self._parameters.get(name)
        if param is not None:
            self._parameters[name] = nn.Parameter(
                value.to(param.device).clone(), requires_grad=param.requires_grad
            )
        else:
            self._buffers[name] = value.to(self._buffers[name].device).clone()

    @torch.no_grad()
    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Fit the mean and std over the training stream (``stats="fit"`` only).

        Parameters
        ----------
        input_stream : InputStream
            Port-keyed batches carrying ``cube``, and ``fold`` when ``exclude_folds`` is set.
        """
        if self.stats != "fit":
            return
        self._statistically_initialized = False
        reduce = (0, *self._dims)
        # Per-batch moments from one var_mean in the input precision, merged across batches with
        # Chan's parallel formula in float64: the same numbers as a float64 Welford pass without
        # upcasting the batch (3 GB less on a 2 x 1024 x 1024 x 61 cube).
        count = 0
        mean = m2 = torch.zeros(0, dtype=torch.float64)
        stat_shape: list[int] = []
        frame_shape: tuple[int, ...] | None = None
        frames = 0
        for batch in input_stream:
            x = batch.get("cube")
            if x is None:
                continue
            if self.exclude_folds is not None:
                fold = batch.get("fold")
                if fold is None:
                    raise ValueError(
                        f"{type(self).__name__} needs a 'fold' input in every batch "
                        "when exclude_folds is set"
                    )
                excluded = torch.as_tensor(self.exclude_folds, dtype=fold.dtype, device=fold.device)
                x = x[torch.isin(fold, excluded, invert=True).to(x.device)]
            if self.max_initialization_frames is not None:
                remaining = self.max_initialization_frames - frames
                if remaining <= 0:
                    break
                x = x[:remaining]
            if x.shape[0] == 0:
                continue
            if self.axis == "location":
                # The fit result is tied to one frame size, so every batch must agree on the
                # full shape after dim 0, compared before flattening: two shapes with the same
                # product (a transposed H and W, say) must not merge silently.
                if frame_shape is None:
                    frame_shape = tuple(x.shape[1:])
                elif tuple(x.shape[1:]) != frame_shape:
                    raise ValueError(
                        f"{type(self).__name__} with axis='location' needs every frame to have "
                        f"the same shape, got {frame_shape} and {tuple(x.shape[1:])}"
                    )
            frames += x.shape[0]
            keep = [d for d in range(x.ndim) if d not in reduce]
            rows = x.permute(*reduce, *keep).reshape(-1, math.prod(x.shape[d] for d in keep))
            rows = rows.to(torch.promote_types(rows.dtype, torch.float32))
            var_b, mean_b = torch.var_mean(rows, dim=0, correction=0)
            m = rows.shape[0]
            mean_b, m2_b = mean_b.double(), var_b.double() * m
            if count == 0:
                mean, m2 = mean_b, m2_b
                stat_shape = [1 if d in reduce else x.shape[d] for d in range(x.ndim)]
            else:
                # In place on the running mean and m2, so a merge does not reallocate the two
                # accumulators (each the size of one frame for axis="location").
                total = count + m
                delta = mean_b - mean
                mean += delta * (m / total)
                m2 += m2_b
                m2 += delta * delta * (count * m / total)
            count += m
        if count < 2:
            raise RuntimeError(
                f"{type(self).__name__}.statistical_initialization needs at least two values "
                "per statistic; the stream gave fewer"
            )
        self._set_stat("mean", mean.reshape(stat_shape))
        self._set_stat("std", (m2 / (count - self.ddof)).clamp_min(0.0).sqrt().reshape(stat_shape))
        self._statistically_initialized = True

    def forward(self, cube: Tensor, fold: Tensor | None = None, **_: Any) -> dict[str, Tensor]:
        """Standardize the cube.

        Parameters
        ----------
        cube : Tensor
            Input cube in the node's layout.
        fold : Tensor or None, optional
            Accepted for wiring symmetry with the fit; unused here.

        Returns
        -------
        dict[str, Tensor]
            ``{"cube": standardized}`` with the input's shape.
        """
        if self.stats == "sample":
            mean = cube.mean(dim=self._dims, keepdim=True) if self.with_mean else None
            std = (
                cube.std(dim=self._dims, correction=self.ddof, keepdim=True)
                if self.with_std
                else None
            )
        else:
            if self.stats == "fit":
                require_fitted(self)
            if self.axis == "location" and tuple(cube.shape[1:]) != tuple(self.mean.shape[1:]):
                raise ValueError(
                    f"{type(self).__name__} with axis='location' was fitted on frames of shape "
                    f"{tuple(self.mean.shape[1:])}, got {tuple(cube.shape[1:])}"
                )
            mean, std = self.mean, self.std
        out = cube
        if self.with_mean:
            out = out - mean
        if self.with_std:
            out = out / (std.clamp_min(self.eps) if self.eps_mode == "floor" else std + self.eps)
        return {"cube": out}

    def _load_from_state_dict(  # noqa: D102 (inherited torch hook)
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        keys = [prefix + name for name in self.TRAINABLE_BUFFERS]
        if self.stats == "sample":
            # Nothing to restore: an entry from a removed per-sample class (ZScoreNormalizer,
            # SNVCorrection) is empty, and 0.21.0 saved the two empty placeholders.
            for key in keys:
                state_dict.pop(key, None)
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            missing_keys[:] = [key for key in missing_keys if key not in keys]
            return
        # The statistics start as [0] placeholders until fit; size them (buffer or Parameter)
        # to the saved shape so the strict copy below succeeds.
        for name, key in zip(self.TRAINABLE_BUFFERS, keys, strict=True):
            saved = state_dict.get(key)
            current = getattr(self, name)
            if saved is not None and saved.shape != current.shape:
                self._set_stat(name, torch.zeros(saved.shape))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "Standardize"]
