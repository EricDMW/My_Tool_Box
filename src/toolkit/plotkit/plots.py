"""Plot functions of :mod:`toolkit.plotkit`.

Every function

* accepts NumPy arrays, lists, pandas objects and PyTorch / TensorFlow / JAX tensors
  (see :mod:`toolkit.plotkit._data` for the normalisation rules);
* draws into ``ax`` when given (no new figure is created), otherwise creates a figure
  of size ``figsize`` with :func:`matplotlib.pyplot.subplots`;
* applies ``style`` locally (see :func:`toolkit.plotkit.style_context`), leaving the
  caller's global :data:`matplotlib.rcParams` untouched;
* returns the :class:`matplotlib.axes.Axes` it drew on.

A non-default ``style`` also restyles the axes it draws on (spines, tick labels,
axis-label fonts) so that axes created outside the style context match. Pass
``style="default"`` to keep an existing axes exactly as it is.
"""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Union

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from ._data import (
    as_float_array,
    as_series_list,
    broadcast_x,
    describe_shape,
    is_scalar_like,
    to_numpy,
)
from ._stats import check_band, check_smoothing, smooth, summarize
from .styles import StyleLike, is_default_style, resolve_colors, style_context

__all__ = [
    "plot_bar",
    "plot_gray_scale",
    "plot_heatmap",
    "plot_histogram",
    "plot_learning_curves",
    "plot_line",
    "plot_scatter",
    "plot_shadow_curve",
]

FigSize = tuple[float, float]
ValueFormat = Union[bool, str, Callable[[float], str]]

# Luminance above which black text is more readable than white (as in seaborn).
_LUMINANCE_THRESHOLD = 0.408

# Heatmaps with more cells than this along an axis get no cell lines / index labels.
_MAX_CELLS_FOR_LINES = 50
_MAX_CELLS_FOR_INDEX_LABELS = 30

_SERIES_HINT = "2-D arrays are read as (n_series, n_points); transpose if your series are columns"
_SAMPLES_HINT = (
    "2-D y is read with samples along `axis` (default 0: rows are samples, columns are "
    "steps); pass axis=1 if samples are columns"
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_axes(ax: Axes | None, figsize: FigSize) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        return ax
    if not isinstance(ax, Axes):
        raise TypeError(f"ax must be a matplotlib Axes, got {type(ax).__name__}")
    return ax


def _resolve_labels(labels: Any, n: int, prefix: str, func: str) -> tuple[list[str], bool]:
    """Return ``n`` legend labels and whether the caller provided any."""
    if labels is None:
        return [f"{prefix} {i + 1}" for i in range(n)], False
    if is_scalar_like(labels):
        labels = [labels]
    out = [str(label) for label in list(labels)]
    if len(out) != n:
        warnings.warn(
            f"{func}: got {len(out)} labels for {n} series; "
            + ("missing labels use defaults" if len(out) < n else "extra labels are ignored"),
            stacklevel=3,
        )
        out = out[:n] + [f"{prefix} {i + 1}" for i in range(len(out), n)]
    return out, True


def _pop_alias(kwargs: dict, alias: str, value: Any) -> Any:
    """Use ``kwargs[alias]`` (e.g. matplotlib's ``color``) when ``value`` is None."""
    alias_value = kwargs.pop(alias, None)
    return alias_value if value is None else value


def _tick_font(axis: Any, family: Any) -> None:
    try:
        axis.set_tick_params(which="both", labelfontfamily=family)
    except (TypeError, ValueError, AttributeError):  # matplotlib < 3.8
        # New ticks copy their label properties from the first tick.
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            tick.label1.set_fontfamily(family)
            tick.label2.set_fontfamily(family)


def _style_ticks(ax: Axes) -> None:
    """Freeze tick-label style from the active rcParams.

    matplotlib creates tick objects lazily, usually at draw time, when the style
    context has already been left. Storing the values as tick parameters makes them
    independent of the rcParams active at draw time.
    """
    rc = mpl.rcParams
    for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
        labelcolor = rc[f"{name}tick.labelcolor"]
        if labelcolor == "inherit":
            labelcolor = rc[f"{name}tick.color"]
        axis.set_tick_params(
            which="major",
            labelsize=rc[f"{name}tick.labelsize"],
            color=rc[f"{name}tick.color"],
            labelcolor=labelcolor,
            width=rc[f"{name}tick.major.width"],
            length=rc[f"{name}tick.major.size"],
        )
        axis.set_tick_params(which="minor", labelsize=rc[f"{name}tick.labelsize"])
        _tick_font(axis, rc["font.family"])


def _apply_axes_style(ax: Axes) -> None:
    """Apply the active rcParams to properties matplotlib fixes at axes creation."""
    rc = mpl.rcParams
    for side in ("left", "bottom", "top", "right"):
        if side in ax.spines:
            spine = ax.spines[side]
            spine.set_visible(bool(rc[f"axes.spines.{side}"]))
            spine.set_linewidth(rc["axes.linewidth"])
            spine.set_edgecolor(rc["axes.edgecolor"])
    ax.title.set_fontfamily(rc["font.family"])
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_fontfamily(rc["font.family"])
        label.set_fontsize(rc["axes.labelsize"])
        label.set_fontweight(rc["axes.labelweight"])
        label.set_color(rc["axes.labelcolor"])
    _style_ticks(ax)


def _grid(ax: Axes, axis: str = "both") -> None:
    rc = mpl.rcParams
    ax.grid(
        True,
        which="major",
        axis=axis,
        color=rc["grid.color"],
        linestyle=rc["grid.linestyle"],
        linewidth=rc["grid.linewidth"],
        alpha=rc["grid.alpha"],
    )


def _finish_axes(
    ax: Axes,
    *,
    style: StyleLike,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    grid: bool,
    legend: bool | str,
    show_legend: bool,
    grid_axis: str = "both",
) -> None:
    """Apply style, labels, grid and legend. Must run inside the style context."""
    if not is_default_style(style):
        _apply_axes_style(ax)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        _grid(ax, grid_axis)
    if legend and show_legend:
        ax.legend(loc=legend if isinstance(legend, str) else "best")


def _format_value(value: Any, fmt: ValueFormat) -> str:
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return ""
    if fmt is True:
        fmt = ".3g"
    if callable(fmt):
        return str(fmt(number))
    if "{" in fmt:
        return fmt.format(number)
    if "%" in fmt:
        return fmt % number
    return format(number, fmt)


def _set_tick_labels(ax: Axes, which: str, labels: Any, positions: np.ndarray | None) -> None:
    """Place custom tick labels (see :func:`plot_shadow_curve`)."""
    axis = ax.xaxis if which == "x" else ax.yaxis
    if isinstance(labels, Mapping):
        locs = [float(k) for k in labels.keys()]
        texts = [str(v) for v in labels.values()]
    else:
        if isinstance(labels, str):
            labels = [labels]
        elif not isinstance(labels, (list, tuple)):
            labels = to_numpy(labels).tolist()
        texts = [str(t) for t in labels]
        if positions is not None and len(positions) == len(texts):
            locs = list(positions)
        else:
            lo, hi = ax.dataLim.intervalx if which == "x" else ax.dataLim.intervaly
            locs = list(np.linspace(lo, hi, len(texts))) if len(texts) > 1 else [(lo + hi) / 2]
    axis.set_ticks(locs, labels=texts)


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def _std_arrays(y_std: Any, n_curves: int, lengths: Sequence[int]) -> list[np.ndarray | None]:
    """Normalise an explicit ``y_std`` to one array (or None) per curve."""
    if is_scalar_like(y_std):
        value = float(to_numpy(y_std))
        return [np.full(n, value) for n in lengths]
    if isinstance(y_std, (list, tuple)) and any(s is None for s in y_std):
        if len(y_std) != n_curves:
            raise ValueError(
                f"plot_shadow_curve: y_std has {len(y_std)} entries for {n_curves} curves"
            )
        items: list[np.ndarray | None] = [
            None if s is None else as_float_array(s, name="y_std").ravel() for s in y_std
        ]
    else:
        items = list(as_series_list(y_std, name="y_std"))
        if len(items) == 1 and n_curves > 1:
            items = items * n_curves
        elif len(items) != n_curves:
            raise ValueError(
                f"plot_shadow_curve: y_std provides {len(items)} arrays for {n_curves} curves "
                f"(y_std {describe_shape(y_std)})"
            )
    out: list[np.ndarray | None] = []
    for i, (s, n) in enumerate(zip(items, lengths)):
        if s is not None and s.size == 1:
            s = np.full(n, float(s.ravel()[0]))
        elif s is not None and s.shape != (n,):
            raise ValueError(
                f"plot_shadow_curve: y_std for curve {i} has shape {s.shape} but the curve "
                f"has {n} points"
            )
        out.append(s)
    return out


def plot_shadow_curve(
    y: Any,
    x: Any = None,
    y_std: Any = None,
    labels: Any = None,
    colors: Any = None,
    alpha: float = 0.2,
    ax: Axes | None = None,
    axis: int = 0,
    figsize: FigSize = (10, 6),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool | str = True,
    grid: bool = True,
    style: StyleLike = "research",
    legend_labels: Any = None,
    x_tick_labels: Any = None,
    y_tick_labels: Any = None,
    *,
    band: str | None = "std",
    smoothing: None | int | float = None,
    **kwargs: Any,
) -> Axes:
    """Plot mean curves with a shaded uncertainty band.

    Parameters
    ----------
    y : array-like or sequence of array-likes
        One curve per element. A 2-D array holds samples of one curve (e.g. seeds)
        along ``axis``; its mean is drawn with a band (see ``band``). A 1-D array (or
        flat list of numbers) is drawn as is, with a band only when ``y_std`` is
        given. A list/tuple of arrays gives several curves (use ``np.asarray`` to pass
        a nested list as a single 2-D sample matrix).
    x : array-like or sequence of array-likes, optional
        Shared x values (broadcast to every curve) or one x array per curve.
        Defaults to ``0 .. n_steps - 1``.
    y_std : float, array-like or sequence, optional
        Explicit half-width of the band: a scalar, one array shared by all curves,
        or one entry per curve (entries may be None). Overrides ``band``.
    labels : str or sequence of str, optional
        Legend labels; defaults to ``"Curve 1"``, ``"Curve 2"``, ...
    colors : colour, sequence of colours or palette name, optional
        Defaults to the style's colour cycle. A palette name such as
        ``"okabe_ito"`` (see :func:`toolkit.plotkit.get_palette`) is accepted.
    alpha : float, default 0.2
        Band opacity (0 hides the band).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created when None.
    axis : {0, 1}, default 0
        Sample axis of 2-D inputs: 0 means rows are samples and columns are steps.
    figsize : tuple, default (10, 6)
        Size of a newly created figure.
    title, xlabel, ylabel : str, optional
        Axes title and axis labels.
    legend : bool or str, default True
        Show a legend when there are several curves or labels were given. A string
        is used as the legend location.
    grid : bool, default True
        Draw a major grid.
    style : str, dict or None, default "research"
        Style preset applied locally (see :func:`toolkit.plotkit.style_context`).
    legend_labels : sequence of str, optional
        Alias of ``labels`` (takes precedence).
    x_tick_labels, y_tick_labels : sequence or mapping, optional
        Custom tick labels. A mapping ``{position: label}`` places labels exactly. A
        sequence with one label per x value is placed at the x values of the first
        curve; any other sequence is spread evenly over the data range.
    band : {"std", "sem", "ci95", "minmax"} or None, default "std"
        Band of 2-D inputs: standard deviation (``ddof=0``), standard error of the
        mean (``ddof=1``), 95 % normal-approximation confidence interval
        (``1.96 * sem``), per-step minimum/maximum, or no band. Statistics ignore NaN
        (useful for runs of different length padded with NaN).
    smoothing : int, float or None, default None
        Applied identically to the mean and both band edges. An int is a trailing
        moving-average window (same output length); a float in (0, 1) is the weight
        of a debiased exponential moving average (TensorBoard convention: larger is
        smoother).
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.plot` (e.g. ``linestyle``, ``marker``).
        ``color`` and ``label`` are accepted as aliases of ``colors`` / ``labels``.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        For invalid ``band``, ``smoothing`` or ``axis`` values and mismatched shapes.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> returns = rng.normal(size=(5, 100)).cumsum(axis=1)  # 5 seeds, 100 steps
    >>> ax = plot_shadow_curve(returns, band="ci95", smoothing=10, labels="Agent")
    """
    if legend_labels is not None:
        labels = legend_labels
    labels = _pop_alias(kwargs, "label", labels)
    colors = _pop_alias(kwargs, "color", colors)
    band = check_band(band)
    smoothing = check_smoothing(smoothing)
    if isinstance(axis, bool) or axis not in (0, 1, -1, -2):
        raise ValueError(f"axis must be 0 or 1 for 2-D inputs, got {axis!r}")
    sample_axis = axis % 2

    groups = as_series_list(y, name="y", two_d="samples", max_element_ndim=2)
    curves: list[list[np.ndarray | None]] = []
    for group in groups:
        if group.ndim == 2:
            curves.append(list(summarize(group, axis=sample_axis, band=band)))
        else:
            curves.append([group, None, None])
    n_curves = len(curves)
    lengths = [len(c[0]) for c in curves]

    if y_std is not None:
        for curve, std in zip(curves, _std_arrays(y_std, n_curves, lengths)):
            if std is not None:
                curve[1], curve[2] = curve[0] - std, curve[0] + std

    if smoothing is not None:
        curves = [[None if a is None else smooth(a, smoothing) for a in c] for c in curves]

    hint = _SAMPLES_HINT if any(g.ndim == 2 for g in groups) else None
    xs = broadcast_x(x, [c[0] for c in curves], func="plot_shadow_curve", y_input=y, hint=hint)

    with style_context(style):
        ax = _get_axes(ax, figsize)
        color_list = resolve_colors(colors, n_curves)
        label_list, labels_given = _resolve_labels(labels, n_curves, "Curve", "plot_shadow_curve")
        for (mean, lower, upper), xi, color, label in zip(curves, xs, color_list, label_list):
            (line,) = ax.plot(xi, mean, color=color, label=label, **kwargs)
            if lower is not None and alpha > 0:
                ax.fill_between(xi, lower, upper, color=line.get_color(), alpha=alpha, linewidth=0)
        if x_tick_labels is not None:
            _set_tick_labels(ax, "x", x_tick_labels, np.asarray(xs[0]))
        if y_tick_labels is not None:
            _set_tick_labels(ax, "y", y_tick_labels, None)
        _finish_axes(
            ax,
            style=style,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            grid=grid,
            legend=legend,
            show_legend=n_curves > 1 or labels_given,
        )
    return ax


def _stack_seeds(value: Any, name: str) -> np.ndarray:
    """Stack the seeds of one run into (n_seeds, n_steps), padding ragged runs with NaN."""
    seeds = as_series_list(value, name=f"runs[{name!r}]")
    length = max(len(s) for s in seeds)
    out = np.full((len(seeds), length), np.nan)
    for i, s in enumerate(seeds):
        out[i, : len(s)] = s
    return out


def plot_learning_curves(
    runs: Mapping[str, Any],
    x: Any = None,
    *,
    band: str | None = "ci95",
    smoothing: None | int | float = None,
    colors: Any = None,
    alpha: float = 0.2,
    ax: Axes | None = None,
    figsize: FigSize = (10, 6),
    title: str | None = None,
    xlabel: str | None = "Step",
    ylabel: str | None = "Return",
    legend: bool | str = True,
    grid: bool = True,
    style: StyleLike = "research",
    **kwargs: Any,
) -> Axes:
    """Plot learning curves of several methods, aggregated over seeds.

    Parameters
    ----------
    runs : mapping of str to array-like
        ``{method name: values}`` where values is an array of shape
        ``(n_seeds, n_steps)``, a 1-D array (single seed), or a list of 1-D arrays
        of possibly different lengths (padded with NaN; statistics ignore NaN).
    x : array-like or mapping, optional
        Shared x values (e.g. environment steps) or ``{method name: x}``.
    band : {"std", "sem", "ci95", "minmax"} or None, default "ci95"
        Uncertainty band across seeds (see :func:`plot_shadow_curve`).
    smoothing : int, float or None, default None
        Moving-average window (int) or EMA weight in (0, 1) (float).
    colors, alpha, ax, figsize, title, xlabel, ylabel, legend, grid, style
        As in :func:`plot_shadow_curve`. The legend is always shown when ``legend``
        is true.
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    TypeError
        If ``runs`` is not a mapping.
    ValueError
        If ``runs`` is empty or shapes do not match.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> runs = {"A": rng.normal(size=(5, 50)).cumsum(1), "B": rng.normal(size=(5, 50)).cumsum(1)}
    >>> ax = plot_learning_curves(runs, x=np.arange(50) * 1000, smoothing=5)
    """
    if not isinstance(runs, Mapping):
        raise TypeError(
            f"runs must be a mapping {{name: array (n_seeds, n_steps)}}, got {type(runs).__name__}"
        )
    if not runs:
        raise ValueError("runs must not be empty")
    names = [str(k) for k in runs]
    groups = [_stack_seeds(v, k) for k, v in zip(names, runs.values())]
    if isinstance(x, Mapping):
        missing = [k for k in runs if k not in x]
        if missing:
            raise ValueError(f"x has no entry for run(s) {missing}")
        x = [x[k] for k in runs]
    return plot_shadow_curve(
        groups,
        x=x,
        labels=names,
        colors=colors,
        alpha=alpha,
        ax=ax,
        axis=0,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=legend,
        grid=grid,
        style=style,
        band=band,
        smoothing=smoothing,
        **kwargs,
    )


def plot_line(
    x: Any,
    y: Any,
    labels: Any = None,
    colors: Any = None,
    ax: Axes | None = None,
    figsize: FigSize = (10, 6),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool | str = True,
    grid: bool = True,
    style: StyleLike = "research",
    **kwargs: Any,
) -> Axes:
    """Line plot of one or several series.

    Parameters
    ----------
    x : array-like, sequence of array-likes or None
        One x array shared by all series, one per series, or None for
        ``0 .. n - 1``.
    y : array-like or sequence of array-likes
        A 1-D array (or flat list of numbers) is one series; a list of arrays or a
        2-D array of shape ``(n_series, n_points)`` gives several series.
    labels, colors, ax, figsize, title, xlabel, ylabel, legend, grid, style
        As in :func:`plot_shadow_curve`.
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If the number or lengths of x and y series do not match.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 2 * np.pi, 100)
    >>> ax = plot_line(t, [np.sin(t), np.cos(t)], labels=["sin", "cos"])
    """
    labels = _pop_alias(kwargs, "label", labels)
    colors = _pop_alias(kwargs, "color", colors)
    ys = as_series_list(y, name="y")
    xs = broadcast_x(x, ys, func="plot_line", y_input=y, hint=_SERIES_HINT)
    with style_context(style):
        ax = _get_axes(ax, figsize)
        color_list = resolve_colors(colors, len(ys))
        label_list, labels_given = _resolve_labels(labels, len(ys), "Curve", "plot_line")
        for xi, yi, color, label in zip(xs, ys, color_list, label_list):
            ax.plot(xi, yi, color=color, label=label, **kwargs)
        _finish_axes(
            ax,
            style=style,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            grid=grid,
            legend=legend,
            show_legend=len(ys) > 1 or labels_given,
        )
    return ax


def plot_scatter(
    x: Any,
    y: Any,
    labels: Any = None,
    colors: Any = None,
    ax: Axes | None = None,
    figsize: FigSize = (10, 6),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool | str = True,
    grid: bool = True,
    style: StyleLike = "research",
    **kwargs: Any,
) -> Axes:
    """Scatter plot of one or several groups of points.

    Parameters
    ----------
    x, y : array-like or sequence of array-likes
        Same rules as :func:`plot_line`: a single x is shared by every y series and a
        2-D y of shape ``(n_series, n_points)`` gives several groups.
    labels, colors, ax, figsize, title, xlabel, ylabel, legend, grid, style
        As in :func:`plot_shadow_curve`.
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.scatter`; defaults are ``s=50`` and
        ``alpha=0.7``. Passing ``c`` (per-point values) disables the series colour.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If the number or lengths of x and y series do not match.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> ax = plot_scatter(rng.normal(size=50), rng.normal(size=(2, 50)), labels=["A", "B"])
    """
    labels = _pop_alias(kwargs, "label", labels)
    colors = _pop_alias(kwargs, "color", colors)
    ys = as_series_list(y, name="y")
    xs = broadcast_x(x, ys, func="plot_scatter", y_input=y, hint=_SERIES_HINT)
    options = {"s": 50, "alpha": 0.7, **kwargs}
    with style_context(style):
        ax = _get_axes(ax, figsize)
        color_list = resolve_colors(colors, len(ys))
        label_list, labels_given = _resolve_labels(labels, len(ys), "Series", "plot_scatter")
        for xi, yi, color, label in zip(xs, ys, color_list, label_list):
            if "c" in options:
                ax.scatter(xi, yi, label=label, **options)
            else:
                ax.scatter(xi, yi, color=color, label=label, **options)
        _finish_axes(
            ax,
            style=style,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            grid=grid,
            legend=legend,
            show_legend=len(ys) > 1 or labels_given,
        )
    return ax


# ---------------------------------------------------------------------------
# Bars and histograms
# ---------------------------------------------------------------------------


def _bar_errors(yerr: Any, n_series: int, n_cat: int) -> list[np.ndarray | None]:
    """Normalise ``yerr`` to one array of shape (n_cat,) or (2, n_cat) per series."""
    if yerr is None:
        return [None] * n_series
    if is_scalar_like(yerr):
        return [np.full(n_cat, float(to_numpy(yerr)))] * n_series
    if isinstance(yerr, (list, tuple)):
        items = as_series_list(yerr, name="yerr", max_element_ndim=2)
    else:
        arr = as_float_array(yerr, name="yerr")
        if arr.ndim == 1:
            items = [arr]
        elif arr.ndim == 2 and n_series == 1 and arr.shape == (2, n_cat):
            items = [arr]  # asymmetric (lower, upper) errors of a single series
        elif arr.ndim == 2:
            items = list(arr)
        else:
            raise ValueError(f"plot_bar: yerr must be at most 2-D, got shape {arr.shape}")
    if len(items) == 1 and n_series > 1:
        items = items * n_series
    elif len(items) != n_series:
        raise ValueError(
            f"plot_bar: yerr provides {len(items)} arrays for {n_series} series "
            f"(yerr {describe_shape(yerr)})"
        )
    out = []
    for i, err in enumerate(items):
        if err.size == 1:
            err = np.full(n_cat, float(err.ravel()[0]))
        if err.shape not in ((n_cat,), (2, n_cat)):
            raise ValueError(
                f"plot_bar: yerr for series {i} has shape {err.shape}; expected ({n_cat},) "
                f"or (2, {n_cat})"
            )
        out.append(err)
    return out


def _bar_categories(x: Any, n_cat: int, height: Any) -> list[str]:
    if x is None:
        return [str(i) for i in range(n_cat)]
    cats = as_series_list(x, name="x", numeric=False)
    if len(cats) != 1:
        raise ValueError(
            f"plot_bar: x must be one sequence of category labels, got {describe_shape(x)}"
        )
    values = cats[0].tolist()
    if len(values) != n_cat:
        msg = (
            f"plot_bar: x has {len(values)} categories but each height series has {n_cat} "
            f"values (height {describe_shape(height)})"
        )
        shape = getattr(height, "shape", None)
        if shape is not None and len(shape) == 2 and shape[0] == len(values):
            msg += "; 2-D height is read as (n_series, n_categories), transpose it"
        raise ValueError(msg)
    return [str(v) for v in values]


def plot_bar(
    x: Any,
    height: Any,
    labels: Any = None,
    colors: Any = None,
    ax: Axes | None = None,
    figsize: FigSize = (10, 6),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool | str = True,
    grid: bool = True,
    style: StyleLike = "research",
    *,
    width: float = 0.8,
    yerr: Any = None,
    value_labels: ValueFormat = False,
    horizontal: bool = False,
    capsize: float = 3.0,
    **kwargs: Any,
) -> Axes:
    """Bar chart; several series are drawn as grouped (side-by-side) bars.

    Parameters
    ----------
    x : sequence or None
        Category labels (strings or numbers). Bars are placed at positions
        ``0 .. n_categories - 1`` and labelled with ``str(category)``. None uses the
        category indices.
    height : array-like or sequence of array-likes
        A 1-D array (or flat list of numbers) is one series; a list of arrays or a
        2-D array of shape ``(n_series, n_categories)`` gives several series.
    labels, colors, ax, figsize, title, xlabel, ylabel, legend, grid, style
        As in :func:`plot_shadow_curve`. For a single series, a list with one colour
        per category colours each bar.
    width : float, default 0.8
        Total width of a group of bars (in category units); each bar is
        ``width / n_series`` wide.
    yerr : float, array-like or sequence, optional
        Error bars: a scalar, one array per series (or one shared array), or a
        ``(2, n_categories)`` array of lower/upper errors for a single series.
    value_labels : bool, str or callable, default False
        Annotate bars with their values. A string is a format such as ``".2f"``,
        ``"{:.1%}"`` or ``"%.2f"``; a callable maps the value to its label.
    horizontal : bool, default False
        Draw horizontal bars (categories on the y axis, first category on top).
    capsize : float, default 3.0
        Error-bar cap size in points.
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.bar` (default ``alpha=0.8``).

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If series lengths, categories or error bars do not match, or ``width <= 0``.

    Examples
    --------
    >>> ax = plot_bar(["A", "B", "C"], [[10, 20, 15], [12, 18, 16]], labels=["run 1", "run 2"],
    ...               yerr=[[1, 2, 1], [2, 1, 2]], value_labels=".0f")
    """
    labels = _pop_alias(kwargs, "label", labels)
    colors = _pop_alias(kwargs, "color", colors)
    if not isinstance(width, numbers.Real) or width <= 0:
        raise ValueError(f"width must be a positive number, got {width!r}")
    heights = as_series_list(height, name="height")
    n_series = len(heights)
    n_cat = len(heights[0])
    for i, h in enumerate(heights):
        if len(h) != n_cat:
            raise ValueError(
                f"plot_bar: every series needs the same number of categories; series 0 has "
                f"{n_cat}, series {i} has {len(h)}"
            )
    categories = _bar_categories(x, n_cat, height)
    errors = _bar_errors(yerr, n_series, n_cat)
    bar_width = width / n_series
    positions = np.arange(n_cat, dtype=float)
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_width
    options = {"alpha": 0.8, **kwargs}

    with style_context(style):
        ax = _get_axes(ax, figsize)
        per_bar = (
            n_series == 1
            and n_cat > 1
            and colors is not None
            and not isinstance(colors, str)
            and not mcolors.is_color_like(colors)
            and len(colors) == n_cat
        )
        color_list = [list(colors)] if per_bar else resolve_colors(colors, n_series)
        label_list, labels_given = _resolve_labels(labels, n_series, "Series", "plot_bar")
        for i in range(n_series):
            common = {"color": color_list[i], "label": label_list[i], "capsize": capsize}
            if horizontal:
                container = ax.barh(
                    positions + offsets[i],
                    heights[i],
                    height=bar_width,
                    xerr=errors[i],
                    **common,
                    **options,
                )
            else:
                container = ax.bar(
                    positions + offsets[i],
                    heights[i],
                    width=bar_width,
                    yerr=errors[i],
                    **common,
                    **options,
                )
            if value_labels is not False and value_labels is not None:
                ax.bar_label(
                    container,
                    labels=[_format_value(v, value_labels) for v in heights[i]],
                    padding=2,
                    fontsize=mpl.rcParams["xtick.labelsize"],
                )
        show_legend = n_series > 1 or labels_given
        # Head room for value labels and the legend (the bar baseline stays sticky).
        headroom = bool(value_labels) or (bool(legend) and show_legend)
        if horizontal:
            ax.set_yticks(positions, labels=categories)
            if not ax.yaxis_inverted():
                ax.invert_yaxis()
            if headroom:
                ax.margins(x=0.15)
        else:
            ax.set_xticks(positions, labels=categories)
            if headroom:
                ax.margins(y=0.15)
        if grid:
            ax.set_axisbelow(True)
        _finish_axes(
            ax,
            style=style,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            grid=grid,
            grid_axis="x" if horizontal else "y",
            legend=legend,
            show_legend=show_legend,
        )
    return ax


def plot_histogram(
    data: Any,
    bins: int | str | Sequence[float] = "auto",
    labels: Any = None,
    colors: Any = None,
    ax: Axes | None = None,
    figsize: FigSize = (10, 6),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool | str = True,
    grid: bool = True,
    style: StyleLike = "research",
    *,
    density: bool = False,
    alpha: float = 0.45,
    histtype: str = "stepfilled",
    **kwargs: Any,
) -> Axes:
    """Histogram of one or several distributions on common bins.

    Parameters
    ----------
    data : array-like or sequence of array-likes
        One distribution per series (a 2-D array has one series per row). NaN and
        infinite values are dropped.
    bins : int, str or sequence of float, default "auto"
        Bin specification of :func:`numpy.histogram_bin_edges`, computed on the
        pooled data so all series share the same bins.
    labels, colors, ax, figsize, title, xlabel, legend, grid, style
        As in :func:`plot_shadow_curve`.
    ylabel : str, optional
        Defaults to ``"Density"`` or ``"Count"``.
    density : bool, default False
        Normalise each histogram to unit area.
    alpha : float, default 0.45
        Fill opacity for filled histogram types (outlines stay opaque).
    histtype : {"stepfilled", "step", "bar"}, default "stepfilled"
        Passed to :meth:`matplotlib.axes.Axes.hist`.
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.hist`.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If a series contains no finite values.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> ax = plot_histogram([rng.normal(0, 1, 500), rng.normal(1, 1, 500)], labels=["A", "B"])
    """
    labels = _pop_alias(kwargs, "label", labels)
    colors = _pop_alias(kwargs, "color", colors)
    series = []
    for i, s in enumerate(as_series_list(data, name="data")):
        finite = s[np.isfinite(s)]
        if finite.size == 0:
            raise ValueError(f"plot_histogram: series {i} contains no finite values")
        series.append(finite)
    if isinstance(bins, (int, np.integer, str)):
        edges = np.histogram_bin_edges(np.concatenate(series), bins=bins)
    else:
        edges = as_float_array(bins, name="bins")
    if ylabel is None:
        ylabel = "Density" if density else "Count"

    with style_context(style):
        ax = _get_axes(ax, figsize)
        color_list = resolve_colors(colors, len(series))
        label_list, labels_given = _resolve_labels(labels, len(series), "Series", "plot_histogram")
        for values, color, label in zip(series, color_list, label_list):
            options = {"histtype": histtype, "density": density, **kwargs}
            if options["histtype"] in ("stepfilled", "bar"):
                options.setdefault("facecolor", mcolors.to_rgba(color, alpha))
                options.setdefault("edgecolor", color)
                options.setdefault("linewidth", 1.2)
            else:
                options.setdefault("color", color)
            ax.hist(values, bins=edges, label=label, **options)
        if grid:
            ax.set_axisbelow(True)
        _finish_axes(
            ax,
            style=style,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            grid=grid,
            grid_axis="y",
            legend=legend,
            show_legend=len(series) > 1 or labels_given,
        )
    return ax


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------


def _heatmap_norm(
    finite: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    center: float | None,
    robust: bool,
) -> mcolors.Normalize:
    if robust:
        lo_data, hi_data = np.percentile(finite, [2, 98])
    else:
        lo_data, hi_data = finite.min(), finite.max()
    lo = float(lo_data if vmin is None else vmin)
    hi = float(hi_data if vmax is None else vmax)
    if center is None:
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        return mcolors.Normalize(vmin=lo, vmax=hi)
    center = float(center)
    half = max(hi - center, center - lo)
    if half <= 0:
        half = 1.0
    lo = center - half if vmin is None else lo
    hi = center + half if vmax is None else hi
    if not lo < center < hi:
        raise ValueError(
            f"plot_heatmap: center={center} must lie strictly between vmin={lo} and vmax={hi}"
        )
    return mcolors.TwoSlopeNorm(vcenter=center, vmin=lo, vmax=hi)


def _heatmap_ticks(ax: Axes, which: str, labels: Any, n: int, data_shape: tuple) -> None:
    axis = ax.xaxis if which == "x" else ax.yaxis
    if labels is False:
        axis.set_ticks([])
        return
    if labels is None or labels is True:
        if labels is True or n <= _MAX_CELLS_FOR_INDEX_LABELS:
            axis.set_ticks(np.arange(n))
        else:
            axis.set_major_locator(MaxNLocator(integer=True))
        return
    values = to_numpy(labels).tolist() if not isinstance(labels, (list, tuple)) else labels
    texts = [str(v) for v in values]
    if len(texts) != n:
        kind = "columns" if which == "x" else "rows"
        name = "xlabels" if which == "x" else "ylabels"
        raise ValueError(
            f"plot_heatmap: {name} has {len(texts)} entries but data has {n} {kind} "
            f"(data shape {data_shape})"
        )
    axis.set_ticks(np.arange(n), labels=texts)
    if which == "x" and n > 5 and max(len(t) for t in texts) > 3:
        for text in ax.get_xticklabels():
            text.set_rotation(45)
            text.set_horizontalalignment("right")
            text.set_rotation_mode("anchor")


def _contrast_colors(rgba: np.ndarray) -> np.ndarray:
    """Black or white text colour for each RGBA background (WCAG relative luminance)."""
    rgb = np.asarray(rgba, dtype=float)[..., :3]
    linear = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    return np.where(luminance > _LUMINANCE_THRESHOLD, "black", "white")


def plot_heatmap(
    data: Any,
    xlabels: Any = None,
    ylabels: Any = None,
    cmap: Any = "viridis",
    annot: Any = False,
    ax: Axes | None = None,
    figsize: FigSize = (8, 6),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cbar: bool = True,
    cbar_label: str | None = None,
    style: StyleLike = "research",
    *,
    fmt: ValueFormat | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    robust: bool = False,
    mask: Any = None,
    nan_color: Any = None,
    aspect: str | float = "auto",
    square: bool = False,
    linewidths: float | None = None,
    linecolor: Any = "white",
    annot_kws: Mapping[str, Any] | None = None,
    cbar_kws: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Axes:
    """Heatmap of a 2-D array (matplotlib only; seaborn-compatible arguments).

    Row 0 is drawn at the top. The colorbar, if any, is available as
    ``ax.images[-1].colorbar``.

    Parameters
    ----------
    data : array-like
        2-D array (a 1-D array is drawn as a single row). NaN / infinite cells are
        left blank (see ``nan_color``).
    xlabels, ylabels : sequence, bool or None
        Tick labels for columns / rows (one per cell). None labels cells with their
        index (up to 30 cells, otherwise automatic integer ticks); False hides the
        ticks. Long x labels are rotated. seaborn's ``xticklabels`` / ``yticklabels``
        are accepted as aliases.
    cmap : str or Colormap, default "viridis"
        Colormap; use a diverging map (e.g. ``"RdBu_r"``) together with ``center``.
    annot : bool or array-like, default False
        Write the value of each cell (True) or the entries of an array of the same
        shape. The text colour switches between black and white for contrast.
    ax, figsize, title, xlabel, ylabel, style
        As in :func:`plot_shadow_curve`.
    cbar : bool, default True
        Draw a colorbar.
    cbar_label : str, optional
        Colorbar label (also accepted as ``cbar_kws={"label": ...}``).
    fmt : str or callable, optional
        Annotation format, e.g. ``".2g"`` (seaborn style), ``"{:.1%}"`` or
        ``"%.2f"``. Defaults to ``".0f"`` for integer-valued data, else ``".2g"``.
    vmin, vmax : float, optional
        Colour limits; default to the data range.
    center : float, optional
        Value mapped to the middle of the colormap (diverging norm). Missing limits
        are chosen symmetrically around ``center``.
    robust : bool, default False
        Use the 2nd and 98th percentiles instead of the data range for missing limits.
    mask : array-like of bool, optional
        Cells where ``mask`` is True are hidden (as in seaborn).
    nan_color : colour, optional
        Colour of NaN / masked cells; default transparent (axes background shows).
    aspect : {"auto", "equal"} or float, default "auto"
        Cell aspect ratio; ``square=True`` is the same as ``aspect="equal"``.
    square : bool, default False
        seaborn-compatible alias for ``aspect="equal"``.
    linewidths : float, optional
        Width of the lines between cells. Default: 0.5 for matrices of up to 50 cells
        per side, no lines for larger ones.
    linecolor : colour, default "white"
        Colour of the cell lines.
    annot_kws : dict, optional
        Text properties of the annotations (e.g. ``{"fontsize": 8}``); a ``color``
        entry disables the automatic contrast colour.
    cbar_kws : dict, optional
        Passed to :meth:`matplotlib.figure.Figure.colorbar`.
    **kwargs
        Passed to :meth:`matplotlib.axes.Axes.imshow` (e.g. ``norm``,
        ``interpolation``). seaborn / pcolormesh arguments are translated
        (``xticklabels``, ``yticklabels``, ``cbar_ax``, ``linewidth``, ``edgecolor``)
        or ignored with a warning (``shading``, ``snap``, ``antialiased``).

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, has no finite values, or labels / mask / annotations
        do not match its shape, or ``center`` is outside ``[vmin, vmax]``.

    Examples
    --------
    >>> import numpy as np
    >>> corr = np.corrcoef(np.random.default_rng(0).normal(size=(4, 30)))
    >>> ax = plot_heatmap(corr, annot=True, cmap="RdBu_r", center=0, cbar_label="r")
    """
    # seaborn / pcolormesh compatibility
    xticklabels = kwargs.pop("xticklabels", None)
    yticklabels = kwargs.pop("yticklabels", None)
    xlabels = xticklabels if xlabels is None else xlabels
    ylabels = yticklabels if ylabels is None else ylabels
    cbar_ax = kwargs.pop("cbar_ax", None)
    if "linewidth" in kwargs:
        width_alias = kwargs.pop("linewidth")
        linewidths = width_alias if linewidths is None else linewidths
    for key in ("edgecolor", "edgecolors"):
        if key in kwargs:
            linecolor = kwargs.pop(key)
    for key in ("shading", "snap", "antialiased"):
        if key in kwargs:
            kwargs.pop(key)
            warnings.warn(f"plot_heatmap: ignoring pcolormesh argument {key!r}", stacklevel=2)

    values = as_float_array(data, name="data")
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.size == 0:
        raise ValueError(
            f"plot_heatmap: data must be a non-empty 2-D array, got shape {values.shape}"
        )
    n_rows, n_cols = values.shape
    hidden = ~np.isfinite(values)
    if mask is not None:
        mask_arr = np.asarray(to_numpy(mask), dtype=bool)
        if mask_arr.shape != values.shape:
            raise ValueError(
                f"plot_heatmap: mask has shape {mask_arr.shape} but data has shape {values.shape}"
            )
        hidden = hidden | mask_arr
    finite = values[~hidden]
    if finite.size == 0:
        raise ValueError("plot_heatmap: data has no finite, unmasked values")

    norm = kwargs.pop("norm", None)
    if norm is None:
        norm = _heatmap_norm(finite, vmin, vmax, center, robust)
    elif vmin is not None or vmax is not None or center is not None:
        warnings.warn(
            "plot_heatmap: vmin, vmax and center are ignored when norm is given", stacklevel=2
        )

    annot_values = None
    if isinstance(annot, (bool, np.bool_)):
        annot_values = values if annot else None
    elif annot is not None:
        annot_values = to_numpy(annot)
        if annot_values.shape != values.shape:
            raise ValueError(
                f"plot_heatmap: annot has shape {annot_values.shape} but data has shape {values.shape}"
            )
    if fmt is None:
        integral = np.all(np.mod(finite, 1) == 0) and np.max(np.abs(finite)) < 1e15
        fmt = ".0f" if integral else ".2g"

    if isinstance(cmap, str) or cmap is None:
        cmap_obj = mpl.colormaps[cmap if cmap is not None else mpl.rcParams["image.cmap"]]
    else:
        cmap_obj = cmap
    if nan_color is not None:
        cmap_obj = cmap_obj.with_extremes(bad=nan_color)  # returns a copy
    if square:
        aspect = "equal"
    if linewidths is None:
        linewidths = 0.5 if max(n_rows, n_cols) <= _MAX_CELLS_FOR_LINES else 0.0
    image_data = np.ma.masked_array(values, mask=hidden)

    with style_context(style):
        ax = _get_axes(ax, figsize)
        if not is_default_style(style):
            _apply_axes_style(ax)
        options = {"interpolation": "nearest", **kwargs}
        image = ax.imshow(image_data, cmap=cmap_obj, norm=norm, aspect=aspect, **options)
        _heatmap_ticks(ax, "x", xlabels, n_cols, values.shape)
        _heatmap_ticks(ax, "y", ylabels, n_rows, values.shape)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(which="both", length=0)
        ax.grid(False, which="major")
        if linewidths > 0:
            ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
            ax.grid(True, which="minor", color=linecolor, linestyle="-", linewidth=linewidths)
            ax.tick_params(which="minor", labelbottom=False, labelleft=False)

        if annot_values is not None:
            text_kws = {
                "ha": "center",
                "va": "center",
                "fontsize": mpl.rcParams["xtick.labelsize"],
                **(annot_kws or {}),
            }
            fixed_color = text_kws.pop("color", None)
            auto_colors = _contrast_colors(image.cmap(image.norm(image_data)))
            for i, j in np.ndindex(n_rows, n_cols):
                if hidden[i, j]:
                    continue
                text = _format_value(annot_values[i, j], fmt)
                if text:
                    color = fixed_color if fixed_color is not None else auto_colors[i, j]
                    ax.text(j, i, text, color=color, **text_kws)

        if cbar:
            options = dict(cbar_kws or {})
            label = options.pop("label", None)
            if cbar_label is not None:
                label = cbar_label
            if cbar_ax is not None:
                options.setdefault("cax", cbar_ax)
            else:
                options.setdefault("ax", ax)
            colorbar = ax.figure.colorbar(image, **options)
            colorbar.outline.set_linewidth(0.5)
            if label:
                colorbar.set_label(label)
            if not is_default_style(style):
                _style_ticks(colorbar.ax)

        # Style was applied above; the heatmap has no data grid or legend.
        _finish_axes(
            ax,
            style=None,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            grid=False,
            legend=False,
            show_legend=False,
        )
    return ax


def plot_gray_scale(
    data: Any,
    xlabels: Any = None,
    ylabels: Any = None,
    ax: Axes | None = None,
    figsize: FigSize = (8, 6),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    style: StyleLike = "research",
    **kwargs: Any,
) -> Axes:
    """Grayscale heatmap (e.g. images, attention or occupancy maps).

    Same as :func:`plot_heatmap` with ``cmap="gray"`` (black = low, white = high);
    all keyword arguments of :func:`plot_heatmap` are accepted, including ``cmap``.

    Examples
    --------
    >>> import numpy as np
    >>> ax = plot_gray_scale(np.eye(8), title="Identity")
    """
    kwargs.setdefault("cmap", "gray")
    return plot_heatmap(
        data,
        xlabels=xlabels,
        ylabels=ylabels,
        ax=ax,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        style=style,
        **kwargs,
    )
