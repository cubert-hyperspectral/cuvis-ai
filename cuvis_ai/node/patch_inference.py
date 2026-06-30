"""Dense per-pixel classification: tile a cube into patches, then stitch predictions back.

Two nodes form an inverse pair for patch-based per-pixel classification. ``PatchSampler`` turns a
cube plus an integer target map into a batch of center-pixel patches; ``ClassMapAccumulator``
scatters per-patch predictions back into per-frame class maps. They are coupled by a provenance
contract: when a frame is tiled for dense scoring, each patch must carry where it came from so the
prediction can be written to the right pixel::

    frame f  [H, W, C]                         per-patch provenance (one row per pixel)
    +------------------+        sample          frame_id = f
    | . . . . . . . .  |   ----------------->   y, x      = pixel coords in frame f
    | . . . (y,x) . .  |    P x P window         height   = H   (of frame f)
    | . . . . . . . .  |                         width    = W
    +------------------+                         patches  = [P, P, C]
              |
              |  classifier (external node)
              v
        logits [N, K]
              |
              v
    ClassMapAccumulator.forward(logits, frame_id, y, x, height, width)
        argmax(logits) -> scatter pred into map[frame_id][y, x]
    ClassMapAccumulator.class_maps  ->  {frame_id: [H, W] int64}   (background = background_value)

The patch tiler (a data module) produces the ``frame_id``/``y``/``x``/``height``/``width`` keys;
this module consumes them, it does not invent them. ``ClassMapAccumulator`` is a sink with the run
lifecycle ``reset() -> forward()* -> close()``; the finished maps are read from
:attr:`ClassMapAccumulator.class_maps` after the run.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_core.node import Node


class PatchSampler(Node):
    """Extract labeled center-pixel patches from a cube and an integer target map.

    For each frame, gather the pixels whose target is not ``ignore_index`` and, around each, cut a
    ``patch_size`` x ``patch_size`` window (reflect-padded at borders), emitting ``patches``
    ``[N, P, P, C]`` and integer ``labels`` ``[N]``. ``patch_size=1`` yields single-pixel spectra;
    larger odd sizes yield spatial-spectral patches.

    ``mode="train"`` draws ``samples_per_frame`` center pixels per frame (class-balanced by default,
    with replacement); ``mode="eval"`` takes every labeled pixel, optionally strided down to
    ``max_per_frame`` for dense scoring. Sampling uses torch's global RNG, so seeding torch makes a
    run reproducible while still drawing fresh patches each call.

    Parameters
    ----------
    patch_size : int
        Side of the square window; must be a positive odd integer. Default ``7``.
    samples_per_frame : int
        Center pixels drawn per frame in ``mode="train"``. Default ``256``.
    class_balanced : bool
        In ``mode="train"``, draw an equal share per present class. Default ``True``.
    ignore_index : int
        Target value marking pixels to skip (never sampled). Default ``-100``.
    mode : str
        ``"train"`` (random per-frame sample) or ``"eval"`` (every labeled pixel). Default ``"train"``.
    max_per_frame : int or None
        In ``mode="eval"``, strided cap on labeled pixels per frame (``None`` keeps all).
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.STOCHASTIC})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32, shape=(-1, -1, -1, -1), description="Hyperspectral cube [B,H,W,C]"
        ),
        "targets": PortSpec(
            dtype=torch.int64, shape=(-1, -1, -1), description="Per-pixel class targets [B,H,W]"
        ),
    }
    OUTPUT_SPECS = {
        "patches": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Center-pixel patches [N,P,P,C]",
        ),
        "labels": PortSpec(
            dtype=torch.int64, shape=(-1,), description="Center-pixel class labels [N]"
        ),
    }

    def __init__(
        self,
        patch_size: int = 7,
        samples_per_frame: int = 256,
        class_balanced: bool = True,
        ignore_index: int = -100,
        mode: str = "train",
        max_per_frame: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Validate and store the sampling hyperparameters."""
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError(f"patch_size must be a positive odd int; got {patch_size}")
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval'; got {mode!r}")
        self.patch_size = int(patch_size)
        self.samples_per_frame = int(samples_per_frame)
        self.class_balanced = bool(class_balanced)
        self.ignore_index = int(ignore_index)
        self.mode = mode
        self.max_per_frame = None if max_per_frame is None else int(max_per_frame)
        super().__init__(
            patch_size=self.patch_size,
            samples_per_frame=self.samples_per_frame,
            class_balanced=self.class_balanced,
            ignore_index=self.ignore_index,
            mode=self.mode,
            max_per_frame=self.max_per_frame,
            **kwargs,
        )

    def _sample_indices(self, labels: torch.Tensor) -> torch.Tensor:
        """Pick row indices into ``labels`` (class-balanced, with replacement) for training."""
        m = labels.shape[0]
        if not self.class_balanced:
            return torch.randint(0, m, (self.samples_per_frame,))
        present = torch.unique(labels)
        per = max(1, self.samples_per_frame // int(present.numel()))
        picks = []
        for c in present.tolist():
            pos = (labels == c).nonzero(as_tuple=False).reshape(-1)
            picks.append(pos[torch.randint(0, int(pos.numel()), (per,))])
        sel = torch.cat(picks)
        if sel.numel() > self.samples_per_frame:
            sel = sel[torch.randperm(sel.numel())[: self.samples_per_frame]]
        return sel

    @torch.no_grad()
    def forward(
        self, cube: torch.Tensor, targets: torch.Tensor, **_: Any
    ) -> dict[str, torch.Tensor]:
        """Sample patches and labels across the batch's frames.

        Parameters
        ----------
        cube : torch.Tensor
            Hyperspectral cube ``[B, H, W, C]``.
        targets : torch.Tensor
            Per-pixel integer class targets ``[B, H, W]``; ``ignore_index`` pixels are skipped.
        **_ : Any
            Additional unused keyword arguments (e.g. the pipeline ``context``).

        Returns
        -------
        dict[str, torch.Tensor]
            ``patches`` float32 ``[N, P, P, C]`` and ``labels`` int64 ``[N]``; both empty when no
            labeled pixel is present in the batch.
        """
        b, _, _, c = cube.shape
        p = self.patch_size
        r = p // 2
        out_patches: list[torch.Tensor] = []
        out_labels: list[torch.Tensor] = []
        for i in range(b):
            cube_i = cube[i]  # [H,W,C]
            tgt_i = targets[i]  # [H,W]
            coords = (tgt_i != self.ignore_index).nonzero(as_tuple=False)  # [M,2]
            if coords.shape[0] == 0:
                continue
            labels_i = tgt_i[coords[:, 0], coords[:, 1]]  # [M]
            if self.mode == "train":
                sel = self._sample_indices(labels_i)
            else:
                sel = torch.arange(coords.shape[0])
                if self.max_per_frame and sel.numel() > self.max_per_frame:
                    step = math.ceil(sel.numel() / self.max_per_frame)
                    sel = sel[::step]
            ci = coords[sel]  # [N,2]
            lab = labels_i[sel]  # [N]
            if r > 0:
                padded = F.pad(cube_i.permute(2, 0, 1).unsqueeze(0), (r, r, r, r), mode="reflect")[
                    0
                ].permute(1, 2, 0)  # [H+2r, W+2r, C]
            else:
                padded = cube_i
            rows = ci[:, 0].unsqueeze(1) + torch.arange(p, device=ci.device)  # [N,P]
            cols = ci[:, 1].unsqueeze(1) + torch.arange(p, device=ci.device)  # [N,P]
            patches = padded[rows[:, :, None], cols[:, None, :], :]  # [N,P,P,C]
            out_patches.append(patches)
            out_labels.append(lab)

        if not out_patches:
            return {
                "patches": cube.new_zeros((0, p, p, c)),
                "labels": targets.new_zeros((0,), dtype=torch.int64),
            }
        return {"patches": torch.cat(out_patches), "labels": torch.cat(out_labels).to(torch.int64)}


class ClassMapAccumulator(Node):
    """Scatter chunked patch predictions back into per-frame ``[H, W]`` class maps (sink).

    The inverse of :class:`PatchSampler`: a patch tiler streams a frame's pixels through a
    classifier in batches (so memory stays bounded), each patch tagged with its provenance
    ``(frame_id, y, x)`` plus the source frame ``height``/``width``. This sink argmaxes the
    per-batch ``logits`` and writes each prediction into the right pixel of a per-frame map. After
    the run the finished maps are read from :attr:`class_maps`.

    The run lifecycle is ``reset()`` (clear maps at the start) -> ``forward()`` per batch ->
    ``close()`` (no external resource; maps stay available via :attr:`class_maps`).

    Parameters
    ----------
    background_value : int
        Fill value for pixels no patch wrote to (default ``-1``).
    """

    _category = NodeCategory.SINK
    _tags = frozenset({NodeTag.MASK, NodeTag.CLASSIFICATION, NodeTag.POSTPROCESSING})

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="Per-patch class logits [N, num_classes]",
        ),
        "frame_id": PortSpec(
            dtype=torch.int64, shape=(-1,), description="Source frame id per patch [N]"
        ),
        "y": PortSpec(dtype=torch.int64, shape=(-1,), description="Pixel row per patch [N]"),
        "x": PortSpec(dtype=torch.int64, shape=(-1,), description="Pixel column per patch [N]"),
        "height": PortSpec(
            dtype=torch.int64, shape=(-1,), description="Source frame height per patch [N]"
        ),
        "width": PortSpec(
            dtype=torch.int64, shape=(-1,), description="Source frame width per patch [N]"
        ),
    }
    OUTPUT_SPECS: dict[str, PortSpec] = {}  # sink node

    def __init__(self, background_value: int = -1, **kwargs: Any) -> None:
        """Store the background fill and start with an empty map set."""
        self.background_value = int(background_value)
        self._maps: dict[int, torch.Tensor] = {}
        super().__init__(background_value=self.background_value, **kwargs)

    def reset(self) -> None:
        """Clear accumulated maps before a new prediction run (called by the Predictor)."""
        self._maps = {}

    @torch.no_grad()
    def forward(
        self,
        logits: torch.Tensor,
        frame_id: torch.Tensor,
        y: torch.Tensor,
        x: torch.Tensor,
        height: torch.Tensor,
        width: torch.Tensor,
        **_: Any,
    ) -> dict[str, Any]:
        """Argmax the batch's logits and scatter each prediction into its frame's class map.

        Parameters
        ----------
        logits : torch.Tensor
            Per-patch class logits ``[N, num_classes]``.
        frame_id, y, x, height, width : torch.Tensor
            Per-patch provenance ``[N]``: source frame id, pixel row/column, and the source frame
            height/width used to size a frame's map the first time it is seen.
        **_ : Any
            Additional unused keyword arguments (e.g. the pipeline ``context``).

        Returns
        -------
        dict[str, Any]
            Empty dict (sink node); results accumulate in :attr:`class_maps`.

        Raises
        ------
        IndexError
            If a patch's ``(y, x)`` falls outside its frame's ``(height, width)``.
        """
        preds = logits.argmax(dim=-1).to(torch.long).cpu()
        fid_t, y_t, x_t = frame_id.cpu(), y.cpu(), x.cpu()
        h_t, w_t = height.cpu(), width.cpu()
        for fid in torch.unique(fid_t).tolist():
            sel = fid_t == fid
            fid = int(fid)
            if fid not in self._maps:
                h, w = int(h_t[sel][0]), int(w_t[sel][0])
                self._maps[fid] = torch.full((h, w), self.background_value, dtype=torch.long)
            cmap = self._maps[fid]
            h, w = cmap.shape
            ys, xs = y_t[sel], x_t[sel]
            if (ys < 0).any() or (ys >= h).any() or (xs < 0).any() or (xs >= w).any():
                raise IndexError(
                    f"patch coordinates out of bounds for frame {fid} of size ({h}, {w})."
                )
            cmap[ys, xs] = preds[sel]
        return {}

    @property
    def class_maps(self) -> dict[int, torch.Tensor]:
        """Finished per-frame class maps ``{frame_id: [H, W] int64}`` (background = ``background_value``)."""
        return {fid: m.clone() for fid, m in self._maps.items()}

    def close(self) -> None:
        """No external resource to release; maps stay available via :attr:`class_maps`."""


__all__ = ["PatchSampler", "ClassMapAccumulator"]
