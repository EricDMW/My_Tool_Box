"""Saving figures to one or several file formats."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = ["save_figure"]

PathLike = Union[str, "os.PathLike[str]"]


def _root_figure(fig_or_ax: Any) -> Figure:
    obj = fig_or_ax
    if isinstance(obj, np.ndarray):
        obj = obj.flat[0] if obj.size else None
    elif isinstance(obj, (list, tuple)):
        obj = obj[0] if obj else None
    if isinstance(obj, Axes):
        obj = obj.figure
    # Sub-figures point to their parent through ``.figure``.
    for _ in range(16):
        if isinstance(obj, Figure) or obj is None:
            break
        obj = getattr(obj, "figure", None)
    if not isinstance(obj, Figure):
        raise TypeError(
            "fig_or_ax must be a matplotlib Figure, SubFigure, Axes or an array of Axes; "
            f"got {type(fig_or_ax).__name__}"
        )
    return obj


def save_figure(
    fig_or_ax: Any,
    path: PathLike,
    formats: str | Sequence[str] | None = None,
    dpi: float | str = 300,
    transparent: bool = False,
    **savefig_kwargs: Any,
) -> list[Path]:
    """Save a figure to one or several formats, creating parent directories.

    Parameters
    ----------
    fig_or_ax : Figure, SubFigure, Axes or array of Axes
        What to save; for Axes the whole parent figure is saved.
    path : str or path-like
        Output path. With ``formats`` given, the extension is replaced by (or, if the
        path has no recognised image extension, appended as) each format, so
        ``"results/curve"`` with ``formats=("pdf", "png")`` writes ``results/curve.pdf``
        and ``results/curve.png``.
    formats : str or sequence of str, optional
        File formats such as ``"pdf"``, ``"png"``, ``"svg"``. Defaults to the
        extension of ``path`` or, if it has none, ``rcParams["savefig.format"]``.
    dpi : float or "figure", default 300
        Resolution for raster formats (and rasterised parts of vector formats).
    transparent : bool, default False
        Transparent figure and axes backgrounds.
    **savefig_kwargs
        Passed to :meth:`matplotlib.figure.Figure.savefig`; defaults are
        ``bbox_inches="tight"`` and ``pad_inches=0.05``.

    Returns
    -------
    list of pathlib.Path
        The written files, in the order of ``formats``.

    Raises
    ------
    TypeError
        If ``fig_or_ax`` is not a figure or axes.
    ValueError
        If a format is not supported by the canvas.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> paths = save_figure(ax, "renders/example", formats=("pdf", "png"))  # doctest: +SKIP
    """
    fig = _root_figure(fig_or_ax)
    supported = fig.canvas.get_supported_filetypes()
    target = Path(os.fspath(path))
    suffix = target.suffix.lower().lstrip(".")

    if formats is None:
        fmts = [suffix] if suffix in supported else [str(mpl.rcParams["savefig.format"])]
    elif isinstance(formats, str):
        fmts = [formats]
    else:
        fmts = list(formats)
    if not fmts:
        raise ValueError("formats must not be empty")
    fmts = [str(f).lower().lstrip(".") for f in fmts]
    unknown = [f for f in fmts if f not in supported]
    if unknown:
        raise ValueError(f"unsupported format(s) {unknown}; supported: {sorted(supported)}")

    stem = target.with_suffix("") if suffix in supported else target
    stem.parent.mkdir(parents=True, exist_ok=True)
    options = {"bbox_inches": "tight", "pad_inches": 0.05, **savefig_kwargs}
    written: list[Path] = []
    for fmt in dict.fromkeys(fmts):  # de-duplicate, keep order
        out = stem.parent / f"{stem.name}.{fmt}"
        fig.savefig(out, format=fmt, dpi=dpi, transparent=transparent, **options)
        written.append(out)
    return written
