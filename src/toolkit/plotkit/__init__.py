"""plotkit: publication-quality plotting for research and reinforcement learning.

Functions accept NumPy arrays, lists, pandas objects and PyTorch / TensorFlow / JAX
tensors, apply a style preset *locally* (the global :data:`matplotlib.rcParams` are
never modified, except by the explicit :func:`set_research_style`), draw into a given
``ax`` or a new figure, and return the :class:`~matplotlib.axes.Axes`.

Plots
-----
plot_shadow_curve
    Mean curves with std / SEM / 95 % CI / min-max bands and optional smoothing.
plot_learning_curves
    ``{method: array (n_seeds, n_steps)}`` learning curves (built on plot_shadow_curve).
plot_line, plot_scatter
    Line and scatter plots of one or several series.
plot_bar
    Grouped bars with error bars and value labels.
plot_histogram
    Overlaid histograms on common bins.
plot_heatmap, plot_gray_scale
    Annotated heatmaps with colorbar (matplotlib only).

Styles and output
-----------------
style_context, set_research_style, reset_style, STYLE_PRESETS
    Presets ``"research"``, ``"presentation"``, ``"minimal"`` and ``"default"``.
get_palette, RESEARCH_COLORS, RESEARCH_COLOR_LIST, OKABE_ITO_COLORS, OKABE_ITO_COLOR_LIST
    Colour palettes, including the colour-blind-safe Okabe-Ito palette.
save_figure
    Save a figure to several formats at once.

A gallery of all plots: ``python -m toolkit.plotkit --demo all --save renders/plotkit``.
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
