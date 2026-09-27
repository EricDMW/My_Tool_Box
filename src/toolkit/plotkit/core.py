"""Backward-compatible import location of the plotkit API.

The implementation now lives in :mod:`toolkit.plotkit.plots`,
:mod:`toolkit.plotkit.styles` and :mod:`toolkit.plotkit.io`; this module re-exports
it so that ``from toolkit.plotkit.core import plot_shadow_curve`` keeps working.
"""

from __future__ import annotations

from .io import save_figure
from .plots import (
    plot_bar,
    plot_gray_scale,
    plot_heatmap,
    plot_histogram,
    plot_learning_curves,
    plot_line,
    plot_scatter,
    plot_shadow_curve,
)
from .styles import (
    OKABE_ITO_COLOR_LIST,
    OKABE_ITO_COLORS,
    RESEARCH_COLOR_LIST,
    RESEARCH_COLORS,
    STYLE_PRESETS,
    get_palette,
    reset_style,
    set_research_style,
    style_context,
)

__all__ = [
    "OKABE_ITO_COLORS",
    "OKABE_ITO_COLOR_LIST",
    "RESEARCH_COLORS",
    "RESEARCH_COLOR_LIST",
    "STYLE_PRESETS",
    "get_palette",
    "plot_bar",
    "plot_gray_scale",
    "plot_heatmap",
    "plot_histogram",
    "plot_learning_curves",
    "plot_line",
    "plot_scatter",
    "plot_shadow_curve",
    "reset_style",
    "save_figure",
    "set_research_style",
    "style_context",
]
