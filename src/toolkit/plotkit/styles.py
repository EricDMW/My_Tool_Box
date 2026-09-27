"""Style presets, colour palettes and style contexts.

Plot functions in :mod:`toolkit.plotkit` apply their ``style`` locally through
:func:`style_context`, so calling them never changes the caller's global
:data:`matplotlib.rcParams`. :func:`set_research_style` is the explicit, global opt-in
and :func:`reset_style` undoes it.

Presets (see :data:`STYLE_PRESETS`)
-----------------------------------
``"research"``
    Serif fonts, bold titles, no top/right spines, 2 pt lines, light grid; the
    paper-ready look of plotkit.
``"presentation"``
    Sans-serif with larger fonts and thicker lines for slides and posters.
``"minimal"``
    Small sans-serif fonts, thin dark-grey axes, frameless legend.
``"default"``
    No changes: plots use whatever rcParams are active.

A preset only overrides the keys it defines, on top of the currently active rcParams.
"""

from __future__ import annotations

import contextlib
import numbers
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, Union

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.style as mstyle
import numpy as np
from cycler import cycler

__all__ = [
    "OKABE_ITO_COLORS",
    "OKABE_ITO_COLOR_LIST",
    "RESEARCH_COLORS",
    "RESEARCH_COLOR_LIST",
    "STYLE_PRESETS",
    "get_palette",
    "reset_style",
    "set_research_style",
    "style_context",
]

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

#: The default categorical palette (matplotlib's ``tab10``), keyed by colour name.
RESEARCH_COLORS: dict[str, str] = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "olive": "#bcbd22",
    "cyan": "#17becf",
}

#: :data:`RESEARCH_COLORS` as an ordered list (the default series colours).
RESEARCH_COLOR_LIST: list[str] = list(RESEARCH_COLORS.values())

#: Okabe-Ito colour-blind-safe palette (Okabe and Ito, 2008), in plotting order. The
#: first six colours are well separated for all common colour-vision deficiencies;
#: yellow has low contrast on white and black carries no hue, so they come last.
OKABE_ITO_COLORS: dict[str, str] = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "bluish_green": "#009E73",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "sky_blue": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}

#: :data:`OKABE_ITO_COLORS` as an ordered list.
OKABE_ITO_COLOR_LIST: list[str] = list(OKABE_ITO_COLORS.values())

_PALETTES: dict[str, list[str]] = {
    "research": RESEARCH_COLOR_LIST,
    "tab10": RESEARCH_COLOR_LIST,
    "okabe_ito": OKABE_ITO_COLOR_LIST,
    "colorblind": OKABE_ITO_COLOR_LIST,
}

# Listed colormaps with at most this many entries are treated as qualitative palettes.
_MAX_QUALITATIVE = 32


def _palette_key(name: str) -> str:
    return name.lower().replace("-", "_")


def get_palette(name: str = "research", n: int | None = None) -> list[str]:
    """Return a list of hex colours.

    Parameters
    ----------
    name : str, default "research"
        ``"research"`` (alias ``"tab10"``), ``"okabe_ito"`` (alias ``"colorblind"``),
        or any matplotlib colormap name. Qualitative colormaps (``"tab20"``,
        ``"Set2"``, ...) return their colours; continuous colormaps (``"viridis"``,
        ...) are sampled at ``n`` evenly spaced points in [0.1, 0.9].
    n : int, optional
        Number of colours. Palettes are cycled when ``n`` exceeds their length.
        Continuous colormaps default to 8 samples.

    Returns
    -------
    list of str
        Hex colour strings.

    Raises
    ------
    ValueError
        If the palette is unknown or ``n`` is not a positive integer.

    Examples
    --------
    >>> get_palette("okabe_ito", 3)
    ['#0072B2', '#E69F00', '#009E73']
    """
    if n is not None and (isinstance(n, bool) or not isinstance(n, numbers.Integral) or int(n) < 1):
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if not isinstance(name, str):
        raise TypeError(f"palette name must be a string, got {type(name).__name__}")
    key = _palette_key(name)
    if key in _PALETTES:
        colors = list(_PALETTES[key])
    else:
        try:
            cmap = mpl.colormaps[name]
        except KeyError:
            raise ValueError(
                f"unknown palette {name!r}; use one of {sorted(_PALETTES)} or a matplotlib "
                "colormap name"
            ) from None
        if isinstance(cmap, mcolors.ListedColormap) and cmap.N <= _MAX_QUALITATIVE:
            colors = [mcolors.to_hex(c) for c in cmap(np.arange(cmap.N))]
        else:
            k = 8 if n is None else int(n)
            points = np.linspace(0.1, 0.9, k) if k > 1 else np.array([0.5])
            return [mcolors.to_hex(c) for c in cmap(points)]
    if n is None:
        return colors
    return [colors[i % len(colors)] for i in range(int(n))]


def resolve_colors(colors: Any, n: int) -> list[Any]:
    """Return ``n`` colours from a colour specification (internal helper).

    ``colors`` may be None (the active ``axes.prop_cycle``), a palette or colormap
    name, a single colour (name, hex string or RGB(A) tuple), or a sequence of colours
    (cycled when shorter than ``n``).
    """
    if colors is None:
        cycle = mpl.rcParams["axes.prop_cycle"].by_key().get("color")
        base = list(cycle) if cycle else list(RESEARCH_COLOR_LIST)
    elif isinstance(colors, str):
        if _palette_key(colors) in _PALETTES:
            base = get_palette(colors)
        elif mcolors.is_color_like(colors):
            base = [colors]
        elif colors in mpl.colormaps:
            base = get_palette(colors, n)
        else:
            raise ValueError(
                f"{colors!r} is neither a colour, a plotkit palette {sorted(_PALETTES)} "
                "nor a matplotlib colormap"
            )
    elif mcolors.is_color_like(colors):
        base = [colors]
    else:
        base = list(colors)
        if not base:
            raise ValueError("colors must not be empty")
    return [base[i % len(base)] for i in range(n)]


# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict[str, Any]] = {
    "research": {
        "font.family": "serif",
        "font.size": 12,
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.prop_cycle": cycler(color=RESEARCH_COLOR_LIST),
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "lines.linewidth": 2.0,
        "grid.alpha": 0.3,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
    },
    "presentation": {
        "font.family": "sans-serif",
        "font.size": 16,
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.labelsize": 18,
        "axes.prop_cycle": cycler(color=RESEARCH_COLOR_LIST),
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "legend.fontsize": 14,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "none",
        "lines.linewidth": 3.0,
        "lines.markersize": 9.0,
        "grid.alpha": 0.3,
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
    },
    "minimal": {
        "font.family": "sans-serif",
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "0.25",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "axes.labelsize": 10,
        "axes.labelcolor": "0.2",
        "axes.prop_cycle": cycler(color=RESEARCH_COLOR_LIST),
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "0.25",
        "ytick.color": "0.25",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.2,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
    },
    "default": {},
}

#: Read-only view of the rcParams set by each preset. ``"default"`` is empty.
STYLE_PRESETS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {name: MappingProxyType(rc) for name, rc in _PRESETS.items()}
)

# Extra settings applied only by the global opt-in set_research_style().
_RESEARCH_SAVEFIG: dict[str, Any] = {
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}

StyleLike = Union[None, str, Mapping[str, Any]]


def _style_spec(style: StyleLike) -> None | str | dict[str, Any]:
    """Resolve a style argument to None (no change), an rc dict or a matplotlib style name."""
    if style is None:
        return None
    if isinstance(style, Mapping):
        return dict(style) or None
    if isinstance(style, str):
        key = style.lower()
        if key in _PRESETS:
            return dict(_PRESETS[key]) or None
        if style in mstyle.available:
            return style
    raise ValueError(
        f"unknown style {style!r}; use one of {tuple(_PRESETS)}, a name from "
        "matplotlib.style.available, or a dict of rcParams"
    )


def is_default_style(style: StyleLike) -> bool:
    """Return True if ``style`` leaves the active rcParams unchanged (internal helper)."""
    return _style_spec(style) is None


@contextlib.contextmanager
def style_context(style: StyleLike = "research") -> Iterator[None]:
    """Context manager that applies a style preset and restores rcParams on exit.

    Parameters
    ----------
    style : str, dict or None, default "research"
        A plotkit preset (``"research"``, ``"presentation"``, ``"minimal"``,
        ``"default"``), a matplotlib style name (see ``matplotlib.style.available``),
        a dict of rcParams, or None (same as ``"default"``: no changes).

    Raises
    ------
    ValueError
        If the style name is unknown.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> with style_context("presentation"):
    ...     fig, ax = plt.subplots()
    ...     _ = ax.plot([0, 1], [0, 1])
    >>> plt.close(fig)
    """
    spec = _style_spec(style)
    with mpl.rc_context():
        if spec is not None:
            mstyle.use(spec)
        yield


# Snapshot of rcParams taken by the first set_research_style() call.
_saved_rc: dict[str, Any] | None = None


def _snapshot_rc() -> dict[str, Any]:
    snapshot = dict(mpl.rcParams.copy())
    snapshot.pop("backend", None)
    return snapshot


def set_research_style() -> None:
    """Apply the ``"research"`` preset **globally** (explicit opt-in).

    Updates :data:`matplotlib.rcParams` with :data:`STYLE_PRESETS` ``["research"]``
    plus ``savefig.dpi=300``, ``savefig.bbox="tight"`` and ``savefig.pad_inches=0.1``,
    so that plain matplotlib code also produces paper-ready figures. The previous
    rcParams are remembered (by the first call) and restored by :func:`reset_style`.
    Prefer ``style=...`` arguments or :func:`style_context` in library code.
    """
    global _saved_rc
    if _saved_rc is None:
        _saved_rc = _snapshot_rc()
    mpl.rcParams.update(_PRESETS["research"])
    mpl.rcParams.update(_RESEARCH_SAVEFIG)


def reset_style() -> None:
    """Undo :func:`set_research_style`.

    Restores the rcParams saved by the first :func:`set_research_style` call. If
    there is nothing to restore, resets matplotlib to its built-in defaults
    (:func:`matplotlib.rcdefaults`).
    """
    global _saved_rc
    if _saved_rc is not None:
        mpl.rcParams.update(_saved_rc)
        _saved_rc = None
    else:
        mpl.rcdefaults()
