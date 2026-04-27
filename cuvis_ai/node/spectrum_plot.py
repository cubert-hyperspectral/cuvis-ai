"""Per-frame spectrum-plot renderer for video export.

Renders a two-line matplotlib plot (reference vs tracked) to an RGB frame
suitable for piping into ``ToVideoNode``.  Used by the SPAM invisible-ink
pipeline to produce a secondary signature video alongside the main overlay.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import torch
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.utils.vis_helpers import fig_to_array
from cuvis_ai_core.node import Node


class SpectrumPlotNode(Node):
    """Render a per-frame line plot of tracked vs reference spectrum.

    The tracked line is plotted against ``wavelengths`` (typically the full
    cube grid), while the reference line is plotted against
    ``reference_wavelengths`` (typically the narrower bandpass subset where
    the SAM reference is defined).  Both are drawn on a single axes with a
    fixed x-range so the axis does not re-scale from frame to frame.

    Output shape is fixed to ``(plot_height, plot_width)`` so downstream
    ``ToVideoNode`` gets consistent frame dimensions.

    Parameters
    ----------
    wavelengths : np.ndarray
        Full wavelength grid in nm, shape ``[C_full]``.  Used as the x-axis
        for the tracked line.
    reference_wavelengths : np.ndarray
        Wavelength grid for the reference line in nm, shape ``[C_ref]``.
    plot_width, plot_height : int
        Pixel dimensions of the rendered frame.  Default 960 x 720.
    dpi : int
        Figure dpi.  Default 150.
    xlabel, ylabel : str
        Axis labels.
    tracked_label, reference_label : str
        Legend labels.
    tracked_color, reference_color : str
        Matplotlib colour specs for the two lines.
    bg_color, fg_color : str
        Background / foreground colours.
    y_fixed_range : tuple[float, float] | None
        If set, fixes the y-axis range.  Otherwise auto-scales per-frame on the
        union of tracked+reference values, with a small headroom.
    y_num_ticks : int
        When ``y_fixed_range`` is set, draw exactly this many evenly-spaced
        y-axis ticks spanning ``(y_min, y_max]``.  Default ``12``.
    tracked_hold_frames : int
        When ``valid=0`` for a frame, keep drawing the most recent tracked
        spectrum for up to this many frames before falling back to
        reference-only.  Smooths out brief dropouts.  ``0`` disables the
        hold.  Default ``15``.
    """

    INPUT_SPECS = {
        "tracked_spectrum": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="Per-frame masked mean spectrum [B, C_full].",
        ),
        "reference_spectrum": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Reference spectrum (any leading shape, last dim is C_ref).",
        ),
        "valid": PortSpec(
            dtype=torch.int32,
            shape=(-1,),
            description="Per-frame tracked validity [B]; 0 suppresses the tracked line.",
            optional=True,
        ),
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(-1,),
            description="Per-frame index [B] rendered in plot title.",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Rendered plot frames [B, plot_height, plot_width, 3] in [0, 1].",
        ),
    }

    def __init__(
        self,
        wavelengths: Sequence[float] | np.ndarray,
        reference_wavelengths: Sequence[float] | np.ndarray,
        plot_width: int = 960,
        plot_height: int = 720,
        dpi: int = 150,
        xlabel: str = "wavelength in [nm]",
        ylabel: str = "spectral radiance in [W/m²/sr/µm]",
        tracked_label: str = "",
        reference_label: str = "",
        tracked_color: str = "red",
        reference_color: str = "lime",
        bg_color: str = "black",
        fg_color: str = "white",
        y_fixed_range: tuple[float, float] | None = (0.0, 12.0),
        y_num_ticks: int = 12,
        tracked_hold_frames: int = 15,
        **kwargs: Any,
    ) -> None:
        if plot_width < 32 or plot_height < 32:
            raise ValueError("plot dimensions must be >= 32 px")
        if dpi <= 0:
            raise ValueError("dpi must be > 0")
        if y_num_ticks < 2:
            raise ValueError("y_num_ticks must be >= 2")
        if tracked_hold_frames < 0:
            raise ValueError("tracked_hold_frames must be >= 0")

        self._wavelengths = np.asarray(wavelengths, dtype=np.float32).ravel()
        self._ref_wavelengths = np.asarray(reference_wavelengths, dtype=np.float32).ravel()
        if self._wavelengths.size == 0:
            raise ValueError("wavelengths must be non-empty")
        if self._ref_wavelengths.size == 0:
            raise ValueError("reference_wavelengths must be non-empty")

        self.plot_width = int(plot_width)
        self.plot_height = int(plot_height)
        self.dpi = int(dpi)
        self.xlabel = str(xlabel)
        self.ylabel = str(ylabel)
        self.tracked_label = str(tracked_label)
        self.reference_label = str(reference_label)
        self.tracked_color = str(tracked_color)
        self.reference_color = str(reference_color)
        self.bg_color = str(bg_color)
        self.fg_color = str(fg_color)
        self.y_fixed_range = (
            None if y_fixed_range is None else (float(y_fixed_range[0]), float(y_fixed_range[1]))
        )
        self.y_num_ticks = int(y_num_ticks)
        self.tracked_hold_frames = int(tracked_hold_frames)

        super().__init__(
            wavelengths=self._wavelengths.tolist(),
            reference_wavelengths=self._ref_wavelengths.tolist(),
            plot_width=self.plot_width,
            plot_height=self.plot_height,
            dpi=self.dpi,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            tracked_label=self.tracked_label,
            reference_label=self.reference_label,
            tracked_color=self.tracked_color,
            reference_color=self.reference_color,
            bg_color=self.bg_color,
            fg_color=self.fg_color,
            y_fixed_range=self.y_fixed_range,
            y_num_ticks=self.y_num_ticks,
            tracked_hold_frames=self.tracked_hold_frames,
            **kwargs,
        )

        # Stateful hold across frames.
        self._last_tracked: np.ndarray | None = None
        self._hold_counter: int = 0

    def _render_frame(
        self,
        tracked: np.ndarray,
        reference: np.ndarray,
        tracked_valid: bool,
        frame_id: int | None,
    ) -> np.ndarray:
        """Render one figure and return the resulting [H, W, 3] uint8 array."""
        import matplotlib.pyplot as plt

        fig_w_in = self.plot_width / self.dpi
        fig_h_in = self.plot_height / self.dpi
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=self.dpi)
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)

        ax.plot(
            self._ref_wavelengths,
            reference,
            color=self.reference_color,
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=self.reference_label,
        )
        if tracked_valid:
            ax.plot(
                self._wavelengths,
                tracked,
                color=self.tracked_color,
                linewidth=2.0,
                marker="o",
                markersize=3,
                label=self.tracked_label,
            )

        ax.set_xlabel(self.xlabel, color=self.fg_color, fontsize=12)
        ax.set_ylabel(self.ylabel, color=self.fg_color, fontsize=12)
        ax.set_xlim(float(self._wavelengths.min()), float(self._wavelengths.max()))

        if self.y_fixed_range is not None:
            ax.set_ylim(self.y_fixed_range)
            y0, y1 = self.y_fixed_range
            ax.set_yticks(np.linspace(y0, y1, self.y_num_ticks + 1)[1:])
        else:
            series = [reference]
            if tracked_valid:
                series.append(tracked)
            y_max = max(float(np.nanmax(s)) for s in series if s.size > 0)
            y_min = min(float(np.nanmin(s)) for s in series if s.size > 0)
            span = max(y_max - y_min, 1e-6)
            ax.set_ylim(y_min - 0.05 * span, y_max + 0.10 * span)

        for spine in ax.spines.values():
            spine.set_color(self.fg_color)
        ax.tick_params(colors=self.fg_color, labelsize=10)
        ax.grid(True, color=self.fg_color, alpha=0.15, linewidth=0.6)
        if self.tracked_label or self.reference_label:
            legend = ax.legend(
                loc="upper left",
                facecolor=self.bg_color,
                edgecolor=self.fg_color,
                labelcolor=self.fg_color,
                fontsize=11,
            )
            legend.get_frame().set_alpha(0.8)

        if frame_id is not None:
            ax.set_title(f"frame {int(frame_id)}", color=self.fg_color, fontsize=11, loc="right")

        fig.tight_layout()
        arr = fig_to_array(fig, dpi=self.dpi)  # closes the figure

        # Resize to exactly the requested output size so the video frame dims
        # are deterministic regardless of bbox_inches='tight' cropping.
        if arr.shape[:2] != (self.plot_height, self.plot_width):
            arr = cv2.resize(arr, (self.plot_width, self.plot_height), interpolation=cv2.INTER_AREA)
        return arr

    @torch.no_grad()
    def forward(
        self,
        tracked_spectrum: torch.Tensor,
        reference_spectrum: torch.Tensor,
        valid: torch.Tensor | None = None,
        frame_id: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        tracked_np = tracked_spectrum.detach().cpu().numpy()
        reference_np = reference_spectrum.detach().cpu().numpy().reshape(-1)

        b = tracked_np.shape[0]
        valid_np = valid.detach().cpu().numpy() if valid is not None else np.ones(b, dtype=np.int32)
        frame_ids_np = frame_id.detach().cpu().numpy() if frame_id is not None else None

        frames = np.empty((b, self.plot_height, self.plot_width, 3), dtype=np.float32)
        for i in range(b):
            fid = int(frame_ids_np[i]) if frame_ids_np is not None else None
            if int(valid_np[i]) == 1:
                self._last_tracked = tracked_np[i].copy()
                self._hold_counter = 0
                plot_tracked = tracked_np[i]
                show_tracked = True
            elif self._last_tracked is not None and self._hold_counter < self.tracked_hold_frames:
                self._hold_counter += 1
                plot_tracked = self._last_tracked
                show_tracked = True
            else:
                self._last_tracked = None
                self._hold_counter = 0
                plot_tracked = tracked_np[i]
                show_tracked = False

            arr = self._render_frame(
                tracked=plot_tracked,
                reference=reference_np,
                tracked_valid=show_tracked,
                frame_id=fid,
            )
            frames[i] = arr.astype(np.float32) / 255.0

        device = tracked_spectrum.device
        return {"rgb_image": torch.from_numpy(frames).to(device=device, dtype=torch.float32)}
