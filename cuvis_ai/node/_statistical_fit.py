"""Shared scaffolding for nodes that fit once via ``statistical_initialization``.

Several nodes in this package (clustering, blind unmixing, one-class novelty
detection, mean-centering, unit-variance scaling) share an identical lifecycle:

1. accumulate training pixels from a port-based ``InputStream`` during the fit,
2. reject an empty / too-small stream while staying uninitialized (contract gate),
3. fit some state and freeze it as torch buffers,
4. guard ``forward`` so an unfit node raises loudly instead of emitting garbage.

``_StatisticalFitNode`` centralises exactly those pieces — and nothing else.
``cuvis_ai_core.node.Node`` already provides ``requires_initial_fit``
auto-detection and ``freeze``/``unfreeze``; its ``statistical_initialization``
is a no-op stub, so subclassing this base (which *does* define
``statistical_initialization``) auto-flags every subclass as fit-required.

The leading underscore keeps the class out of the generated node manifest and
the node registry (same convention as ``_NormalizedDifferenceIndexBase``), so
it is never offered as a usable pipeline node.

Two accumulation patterns are supported:

- ``_collect_pixels`` gathers the stream into one subsampled ``[N, C]`` matrix (used by the sklearn-fit nodes: KMeans, GMM, NMF, one-class SVM).
- streaming-moment nodes (mean-center, unit-variance) override ``statistical_initialization`` to use ``WelfordAccumulator`` and only reuse the guard / empty-stream rejection / ``_initialized`` buffer here.

The fitted ``_initialized`` flag is a **persistent buffer**, so a node reloaded
from a checkpoint keeps its initialized state (a plain attribute would reset to
``False`` on reload and the guard would wrongly fire).
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_core.node import Node


class _StatisticalFitNode(Node):
    """Base for nodes fitted once during statistical initialization.

    Subclasses either implement ``_fit(pixels)`` (and let the default
    ``statistical_initialization`` collect the pixel matrix), or override
    ``statistical_initialization`` entirely (streaming-moment nodes) while
    reusing ``_require_initialized`` / ``_reject_if_insufficient`` /
    ``_mark_initialized`` from this base.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.STATEFUL, NodeTag.TORCH})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        )
    }

    #: Minimum number of training samples required to fit.
    _MIN_FIT_SAMPLES = 2

    def __init__(self, *, max_fit_pixels: int = 20000, fit_seed: int = 0, **kwargs: Any) -> None:
        """Store the fit-subsample budget and register the persistent fit flag.

        Parameters
        ----------
        max_fit_pixels : int, optional
            Upper bound on training pixels gathered by ``_collect_pixels``;
            ``0`` disables subsampling. Bounds memory and keeps sklearn fits
            tractable on full hyperspectral frames (default: 20000).
        fit_seed : int, optional
            Seed for the subsample permutation, for reproducible fits
            (default: 0).
        """
        self.max_fit_pixels = int(max_fit_pixels)
        self.fit_seed = int(fit_seed)
        super().__init__(max_fit_pixels=self.max_fit_pixels, fit_seed=self.fit_seed, **kwargs)
        self.register_buffer("_initialized", torch.zeros(1, dtype=torch.bool))

    @property
    def is_initialized(self) -> bool:
        """Whether ``statistical_initialization`` has successfully fitted state."""
        return bool(self._initialized.item())

    def _require_initialized(self) -> None:
        """Raise if ``forward`` is called before a successful fit."""
        if not self.is_initialized:
            raise RuntimeError(
                f"{type(self).__name__} is not initialized. "
                "Run statistical_initialization (or load fitted weights) before inference."
            )

    def _mark_initialized(self) -> None:
        """Flip the persistent fit flag to ``True``."""
        self._initialized.fill_(True)

    def _reject_if_insufficient(self, n_samples: int) -> None:
        """Stay uninitialized and raise when too few training samples arrived.

        Parameters
        ----------
        n_samples : int
            Number of accumulated training samples.
        """
        if n_samples < self._MIN_FIT_SAMPLES:
            self._initialized.fill_(False)
            raise RuntimeError(
                f"{type(self).__name__}.statistical_initialization received "
                f"{n_samples} sample(s); need at least {self._MIN_FIT_SAMPLES}."
            )

    @torch.no_grad()
    def _collect_pixels(
        self, input_stream: InputStream, port: str = "cube", mask_port: str = "mask"
    ) -> torch.Tensor:
        """Gather a stream of BHWC batches into one ``[N, C]`` matrix.

        Concatenates the flattened pixels from every batch on ``port`` and,
        when ``max_fit_pixels`` is set and exceeded, draws a seeded random
        subsample so the returned matrix never exceeds the budget.

        When a batch also carries ``mask_port`` (a ``[B, H, W]`` foreground
        mask), only pixels where the mask is non-zero are kept, so a node that
        declares and connects an optional ``mask`` input fits on the foreground
        alone. Subclasses without a ``mask`` port never receive one, so this is
        a no-op for them.

        Parameters
        ----------
        input_stream : InputStream
            Iterable of port-keyed batch dicts.
        port : str, optional
            Input port to read the cube from (default: ``"cube"``).
        mask_port : str, optional
            Input port to read an optional foreground mask from (default:
            ``"mask"``).

        Returns
        -------
        torch.Tensor
            ``[N, C]`` float32 pixel matrix; ``[0]``-length when no samples.
        """
        chunks: list[torch.Tensor] = []
        for batch in input_stream:
            x = batch.get(port)
            if x is None:
                continue
            flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
            mask = batch.get(mask_port)
            if mask is not None:
                flat = flat[mask.reshape(-1) > 0]
            chunks.append(flat)
        if not chunks:
            return torch.zeros(0)
        pixels = torch.cat(chunks, dim=0)
        if self.max_fit_pixels and pixels.shape[0] > self.max_fit_pixels:
            gen = torch.Generator(device="cpu").manual_seed(self.fit_seed)
            idx = torch.randperm(pixels.shape[0], generator=gen)[: self.max_fit_pixels]
            pixels = pixels[idx]
        return pixels

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Collect training pixels, reject empties, fit, and mark initialized.

        Subclasses that fit from a raw ``[N, C]`` matrix implement ``_fit``;
        streaming-moment subclasses override this method instead.

        Parameters
        ----------
        input_stream : InputStream
            Iterable of port-keyed batch dicts matching ``INPUT_SPECS``.
        """
        pixels = self._collect_pixels(input_stream)
        self._reject_if_insufficient(pixels.shape[0])
        self._fit(pixels)
        self._mark_initialized()

    def _fit(self, pixels: torch.Tensor) -> None:
        """Fit node state from a ``[N, C]`` pixel matrix (subclass implements)."""
        raise NotImplementedError

    def _load_from_state_dict(  # noqa: D102 (inherited torch hook)
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Buffers are registered with shape [0] (or a placeholder) until fit, so
        # the fitted checkpoint carries the real shapes. A strict copy_ would
        # error on the size mismatch; resize each local buffer to the saved
        # shape first so the subsequent copy succeeds.
        for name, buf in list(self._buffers.items()):
            key = prefix + name
            if key in state_dict and isinstance(buf, torch.Tensor):
                saved = state_dict[key]
                if saved.shape != buf.shape:
                    self._buffers[name] = saved.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
