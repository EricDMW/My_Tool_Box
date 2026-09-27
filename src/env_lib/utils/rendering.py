"""Shared rendering infrastructure for the matplotlib-based environments.

The environments in :mod:`env_lib` render through a small, common layer:

* :class:`Theme` -- a colour palette. Two themes ship with the package
  (``"dark"``, the default, and ``"light"``, suited for papers and slides).
  Select one globally with :func:`set_theme`.
* :class:`FigureWindow` -- owns one matplotlib figure. In ``"rgb_array"`` mode
  the figure is drawn off-screen on an Agg canvas (no GUI, no global
  ``pyplot`` state, safe on headless machines); in ``"human"`` mode it opens an
  interactive window and limits the frame rate.
* :class:`MatplotlibRenderer` -- base class for environment renderers. A
  subclass builds its artists once in :meth:`MatplotlibRenderer._build` and
  only updates their data in :meth:`MatplotlibRenderer._update`, which is an
  order of magnitude faster than clearing and redrawing the axes every frame.

Nothing in this module mutates ``matplotlib.rcParams``.
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

__all__ = [
    "DARK",
    "LIGHT",
    "RENDER_MODES",
    "FigureWindow",
    "MatplotlibRenderer",
    "Theme",
    "figure_to_rgb",
    "get_theme",
    "register_theme",
    "set_theme",
    "style_axes",
    "validate_render_mode",
    "with_alpha",
]

RENDER_MODES: tuple[str, ...] = ("human", "rgb_array")

ColorLike = Union[str, tuple[float, ...]]


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Theme:
    """Colour palette used by all environment renderers.

    Attributes
    ----------
    name:
        Identifier used by :func:`set_theme` / :func:`get_theme`.
    background, panel:
        Figure and axes face colours.
    grid, text, muted:
        Grid lines, primary text and secondary text / annotations.
    accent, positive, negative, warning:
        Semantic colours (highlight, success, failure, caution).
    categorical:
        Distinguishable colours for agents / series, in order of use.
    sequential_cmap, diverging_cmap:
        Names of matplotlib colormaps for magnitudes and signed values.
    """

    name: str
    background: str
    panel: str
    grid: str
    text: str
    muted: str
    accent: str
    positive: str
    negative: str
    warning: str
    categorical: tuple[str, ...]
    sequential_cmap: str = "viridis"
    diverging_cmap: str = "RdBu_r"

    def color(self, index: int) -> str:
        """Return the ``index``-th categorical colour (cycled)."""
        return self.categorical[index % len(self.categorical)]


DARK = Theme(
    name="dark",
    background="#0E1117",
    panel="#151A23",
    grid="#2A3140",
    text="#E6EAF2",
    muted="#8B95A7",
    accent="#4C9AFF",
    positive="#3DDC97",
    negative="#FF5C5C",
    warning="#FFC857",
    categorical=(
        "#4C9AFF",
        "#FF9F43",
        "#3DDC97",
        "#FF6B9A",
        "#A78BFA",
        "#22D3EE",
        "#FFC857",
        "#94A3B8",
    ),
    sequential_cmap="viridis",
    diverging_cmap="coolwarm",
)

LIGHT = Theme(
    name="light",
    background="#FFFFFF",
    panel="#F7F8FA",
    grid="#E3E6EB",
    text="#1F2430",
    muted="#6B7280",
    accent="#2563EB",
    positive="#059669",
    negative="#DC2626",
    warning="#D97706",
    categorical=(
        "#2563EB",
        "#EA580C",
        "#059669",
        "#DB2777",
        "#7C3AED",
        "#0891B2",
        "#CA8A04",
        "#475569",
    ),
    sequential_cmap="viridis",
    diverging_cmap="RdBu_r",
)

_THEMES: dict[str, Theme] = {DARK.name: DARK, LIGHT.name: LIGHT}
_active_theme: Theme = DARK


def register_theme(theme: Theme) -> None:
    """Make a custom :class:`Theme` available to :func:`set_theme`."""
    _THEMES[theme.name] = theme


def get_theme(theme: str | Theme | None = None) -> Theme:
    """Resolve ``theme`` (a name, a :class:`Theme` or ``None`` for the active theme)."""
    if theme is None:
        return _active_theme
    if isinstance(theme, Theme):
        return theme
    try:
        return _THEMES[theme]
    except KeyError:
        raise ValueError(f"Unknown theme {theme!r}; available: {sorted(_THEMES)}") from None


def set_theme(theme: str | Theme) -> None:
    """Set the theme used by renderers created afterwards.

    Examples
    --------
    >>> from env_lib.utils import rendering
    >>> rendering.set_theme("light")
    """
    global _active_theme
    _active_theme = get_theme(theme)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def with_alpha(color: ColorLike, alpha: float) -> tuple[float, float, float, float]:
    """Return ``color`` as an RGBA tuple with the given opacity."""
    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, float(np.clip(alpha, 0.0, 1.0)))


def style_axes(
    ax: Axes,
    theme: Theme,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    show_spines: bool = False,
    fontsize: float = 9.0,
) -> Axes:
    """Apply ``theme`` to ``ax`` (face colour, ticks, labels, grid and spines)."""
    ax.set_facecolor(theme.panel)
    ax.tick_params(colors=theme.muted, labelsize=fontsize - 1, length=3)
    for spine in ax.spines.values():
        spine.set_visible(show_spines)
        spine.set_color(theme.grid)
    if grid:
        ax.grid(True, color=theme.grid, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    if title is not None:
        ax.set_title(title, color=theme.text, fontsize=fontsize + 1, loc="left", pad=6)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=theme.muted, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=theme.muted, fontsize=fontsize)
    return ax


def figure_to_rgb(fig: Figure) -> np.ndarray:
    """Draw ``fig`` and return its pixels as a ``(H, W, 3)`` ``uint8`` array."""
    canvas = fig.canvas
    canvas.draw()
    return _rgba_to_rgb(np.asarray(canvas.buffer_rgba()))


def _rgba_to_rgb(rgba: np.ndarray) -> np.ndarray:
    """Contiguous ``(H, W, 3)`` copy of the colour channels of an ``(H, W, 4)`` buffer.

    Equivalent to ``np.ascontiguousarray(rgba[..., :3])`` but several times
    faster: one long strided copy per channel instead of a 3-element inner
    loop per pixel (about 0.6 ms instead of 2.6 ms for a 1000 x 560 frame).
    """
    height, width = rgba.shape[:2]
    rgb = np.empty((height, width, 3), dtype=rgba.dtype)
    source = rgba.reshape(height * width, 4)
    target = rgb.reshape(height * width, 3)
    for channel in range(3):
        target[:, channel] = source[:, channel]
    return rgb


def validate_render_mode(render_mode: str | None, allowed: Sequence[str] = RENDER_MODES) -> None:
    """Raise ``ValueError`` if ``render_mode`` is not ``None`` or one of ``allowed``."""
    if render_mode is not None and render_mode not in allowed:
        raise ValueError(
            f"render_mode must be one of {tuple(allowed)} or None, got {render_mode!r}"
        )


# ---------------------------------------------------------------------------
# Figure management
# ---------------------------------------------------------------------------
class _FrameClock:
    """Sleep-based frame limiter used for on-screen rendering."""

    def __init__(self, fps: float | None):
        self.period = 1.0 / fps if fps else 0.0
        self._last = None

    def remaining(self) -> float:
        now = time.perf_counter()
        if self._last is None:
            self._last = now
            return 0.0
        delay = max(0.0, self.period - (now - self._last))
        self._last = now + delay
        return delay


_NON_INTERACTIVE_BACKENDS = frozenset({"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"})
_warned_non_interactive = False


class FigureWindow:
    """A matplotlib figure bound to a render mode.

    Parameters
    ----------
    render_mode:
        ``"rgb_array"`` draws off-screen and :meth:`draw` returns an RGB array;
        ``"human"`` opens a window and :meth:`draw` returns ``None``.
    figsize, dpi:
        Figure geometry. The RGB frame is ``figsize * dpi`` pixels.
    fps:
        Frame-rate cap for ``"human"`` mode (``None`` disables the cap).
    title:
        Window title in ``"human"`` mode.
    theme:
        Palette; defaults to the active theme.
    """

    def __init__(
        self,
        render_mode: str,
        *,
        figsize: tuple[float, float] = (8.0, 6.0),
        dpi: int = 100,
        fps: float | None = 30.0,
        title: str | None = None,
        theme: str | Theme | None = None,
    ):
        validate_render_mode(render_mode)
        if render_mode is None:
            raise ValueError("FigureWindow requires a render mode")
        self.render_mode = render_mode
        self.theme = get_theme(theme)
        self._clock = _FrameClock(fps)
        self._interactive = False

        if render_mode == "human":
            import matplotlib.pyplot as plt

            self._plt = plt
            self.fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=self.theme.background)
            self._interactive = self._backend_is_interactive()
            if self._interactive:
                plt.ion()
                if title and self.fig.canvas.manager is not None:
                    self.fig.canvas.manager.set_window_title(title)
                self.fig.show()
            else:
                global _warned_non_interactive
                if not _warned_non_interactive:
                    warnings.warn(
                        "render_mode='human' requested but the active matplotlib backend is "
                        "non-interactive; no window will be shown. Use render_mode='rgb_array' "
                        "for off-screen rendering.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    _warned_non_interactive = True
        else:
            self._plt = None
            self.fig = Figure(figsize=figsize, dpi=dpi, facecolor=self.theme.background)
            FigureCanvasAgg(self.fig)

    @staticmethod
    def _backend_is_interactive() -> bool:
        import matplotlib

        backend = matplotlib.get_backend().lower()
        return backend not in _NON_INTERACTIVE_BACKENDS and "inline" not in backend

    @property
    def is_open(self) -> bool:
        """``False`` once the figure has been closed (by :meth:`close` or the user)."""
        if self.fig is None:
            return False
        if self._plt is not None:
            return self._plt.fignum_exists(self.fig.number)
        return True

    def draw(self) -> np.ndarray | None:
        """Draw the figure. Returns an RGB array in ``"rgb_array"`` mode."""
        if self.render_mode == "rgb_array":
            return figure_to_rgb(self.fig)
        if self._interactive and self.is_open:
            canvas = self.fig.canvas
            canvas.draw_idle()
            canvas.start_event_loop(max(self._clock.remaining(), 1e-3))
        return None

    def close(self) -> None:
        """Release the figure (closes the window in ``"human"`` mode)."""
        if self.fig is None:
            return
        if self._plt is not None:
            self._plt.close(self.fig)
        self.fig = None


class MatplotlibRenderer(ABC):
    """Base class for environment renderers built on persistent artists.

    Subclasses implement :meth:`_build` (create axes and artists once) and
    :meth:`_update` (push new data into the existing artists). The figure is
    created lazily on the first call to :meth:`render`.

    Blitting
    --------
    Artists that change between frames can be registered in :meth:`_build`
    with :meth:`_dynamic`. In ``"rgb_array"`` mode the static part of the
    figure (axes, ticks, labels, legends, background images) is then drawn
    once and cached, and each frame only restores that layer and redraws the
    dynamic artists, which is typically 3-6x faster than a full redraw. Call
    :meth:`_invalidate` after changing anything static (axis limits, static
    images). ``"human"`` mode always performs a full redraw. Renderers that
    register no dynamic artists are simply redrawn in full.

    Parameters
    ----------
    render_mode:
        ``"human"`` or ``"rgb_array"``.
    figsize, dpi, fps, title, theme:
        See :class:`FigureWindow`.
    blit:
        Enable blitting of registered dynamic artists in ``"rgb_array"`` mode.
    """

    def __init__(
        self,
        render_mode: str,
        *,
        figsize: tuple[float, float] = (8.0, 6.0),
        dpi: int = 100,
        fps: float | None = 30.0,
        title: str | None = None,
        theme: str | Theme | None = None,
        blit: bool = True,
    ):
        validate_render_mode(render_mode)
        self.render_mode = render_mode
        self.theme = get_theme(theme)
        self._window_kwargs: dict[str, Any] = dict(
            figsize=figsize, dpi=dpi, fps=fps, title=title, theme=self.theme
        )
        self._window: FigureWindow | None = None
        self._blit = bool(blit) and render_mode == "rgb_array"
        self._dynamic_artists: list[Artist] = []
        self._background = None

    @property
    def fig(self) -> Figure | None:
        return None if self._window is None else self._window.fig

    def render(self, *args: Any, **kwargs: Any) -> np.ndarray | None:
        """Update the artists with the given state and draw one frame."""
        if self._window is None or not self._window.is_open:
            self._window = FigureWindow(self.render_mode, **self._window_kwargs)
            self._dynamic_artists = []
            self._background = None
            self._build(self._window.fig)
        self._update(*args, **kwargs)
        if not (self._blit and self._dynamic_artists):
            return self._window.draw()

        fig = self._window.fig
        canvas = fig.canvas
        if self._background is None:
            canvas.draw()  # animated (dynamic) artists are skipped
            self._background = canvas.copy_from_bbox(fig.bbox)
        else:
            canvas.restore_region(self._background)
        for artist in self._dynamic_artists:
            fig.draw_artist(artist)
        return _rgba_to_rgb(np.asarray(canvas.buffer_rgba()))

    def reset(self) -> None:  # noqa: B027 - optional hook with a default no-op
        """Hook called when the environment is reset (clear trails, histories, ...)."""

    def close(self) -> None:
        """Close the figure. A later :meth:`render` call re-creates it."""
        self._background = None
        self._dynamic_artists = []
        if self._window is not None:
            self._window.close()
            self._window = None

    def _dynamic(self, *artists: Artist) -> None:
        """Register artists that change between frames (see *Blitting*)."""
        for artist in artists:
            if self._blit:
                artist.set_animated(True)
            self._dynamic_artists.append(artist)

    def _invalidate(self) -> None:
        """Discard the cached static layer (call after changing static artists)."""
        self._background = None

    @abstractmethod
    def _build(self, fig: Figure) -> None:
        """Create axes and artists on ``fig`` (called once per figure)."""

    @abstractmethod
    def _update(self, *args: Any, **kwargs: Any) -> None:
        """Update artist data for the current frame."""
