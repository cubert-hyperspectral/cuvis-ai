"""Utility functions and helpers for cuvis.ai."""

from cuvis_ai_core.utils.restore import restore_pipeline, restore_trainrun

from cuvis_ai.utils.cli_helpers import (
    append_tracking_metrics,
    resolve_end_frame,
    resolve_run_output_dir,
    write_experiment_info,
)
from cuvis_ai.utils.color_spaces import linear_rgb_to_oklab, rgb_to_oklab, srgb_to_linear
from cuvis_ai.utils.false_rgb_sampling import initialize_false_rgb_sampled_fixed
from cuvis_ai.utils.poisson_inpaint import poisson_inpaint
from cuvis_ai.utils.vis_helpers import fig_to_array
from cuvis_ai.utils.welford import WelfordAccumulator
from cuvis_ai.utils.xml_plugin_parser import parse_numeric_text, read_xml_inputs, xml_local_name

__all__ = [
    "WelfordAccumulator",
    "append_tracking_metrics",
    "fig_to_array",
    "initialize_false_rgb_sampled_fixed",
    "linear_rgb_to_oklab",
    "poisson_inpaint",
    "resolve_end_frame",
    "resolve_run_output_dir",
    "restore_pipeline",
    "restore_trainrun",
    "rgb_to_oklab",
    "srgb_to_linear",
    "write_experiment_info",
    "xml_local_name",
    "read_xml_inputs",
    "parse_numeric_text",
]
