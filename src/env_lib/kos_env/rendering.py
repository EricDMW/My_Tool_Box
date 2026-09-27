"""Dashboard renderer shared by the NumPy and PyTorch Kuramoto environments.

The figure (1000 x 560 px by default) has two panels:

* **Phase portrait** (left): every oscillator is a dot on the unit circle at
  angle ``theta_i``, coloured by its natural frequency (with a colorbar),
  surrounded by a soft glow and followed by a short fading trail of its recent
  phases. Couplings are drawn as chords whose opacity and width scale with
  ``|K_ij|`` (repulsive couplings use the theme's negative colour). The arrow
  from the origin is the complex order parameter ``r exp(i psi)``.
* **Synchronisation** (right): time series of the order parameter ``r(t)`` and
  the phase coherence over the episode with the synchronisation threshold as a
  dashed line, and below it a raster of ``sin(theta_i)`` over the most recent
  steps.

All artists are created once in :meth:`KuramotoRenderer._build`; each frame only
updates their data. In ``"rgb_array"`` mode the static part of the figure (axes,
ticks, labels, colorbar, legend) is rasterised once and cached, and every frame
only redraws the dynamic artists on top of it (blitting). The cache is rebuilt
when the time axis has to grow, which happens a handful of times per episode.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.artist import Artist
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.text import Annotation
from matplotlib.ticker import MaxNLocator

from env_lib.kos_env._common import mean_field, phase_coherence, wrap_phases
from env_lib.utils.rendering import (
    MatplotlibRenderer,
    Theme,
    style_axes,
    with_alpha,
)

__all__ = ["KuramotoFrame", "KuramotoRenderer"]

_MAX_SERIES = 20_000  # cap on stored time-series points (long runs without reset)
_MAX_EDGES = 300  # dense networks: only the strongest couplings are drawn (Agg cost ~ count)
_TRAIL_SEGMENT_BUDGET = 320  # trails are shortened for large N to bound the draw cost


@dataclass(frozen=True)
class KuramotoFrame:
    """State of one Kuramoto system handed to :class:`KuramotoRenderer`.

    Attributes
    ----------
    phases, natural_frequencies:
        ``(N,)`` arrays of the rendered system.
    coupling_matrix:
        ``(N, N)`` coupling matrix used in the last step (or ``None``).
    history:
        The environment's ``phase_history``: one entry per step, each ``(N,)``
        or ``(B, N)`` (then ``history_index`` selects the system).
    step, max_steps, dt:
        Step counter, episode length and time step.
    reward:
        Reward of the last step (``None`` right after a reset).
    history_index:
        Row of batched history entries to display.
    """

    phases: np.ndarray
    natural_frequencies: np.ndarray
    coupling_matrix: np.ndarray | None
    history: Sequence[np.ndarray]
    step: int
    max_steps: int
    dt: float
    reward: float | None = None
    history_index: int | None = None


def _luminance(color: str) -> float:
    r, g, b = mcolors.to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


class KuramotoRenderer(MatplotlibRenderer):
    """Two-panel dashboard for a Kuramoto oscillator network.

    Parameters
    ----------
    render_mode:
        ``"rgb_array"`` or ``"human"``.
    n_oscillators:
        Number of oscillators ``N``.
    natural_freq_range:
        Range of the natural-frequency colour scale.
    coupling_scale:
        Coupling magnitude drawn with full opacity and width.
    sync_threshold:
        Order parameter above which the system counts as synchronised.
    title, subtitle:
        Header title and a muted configuration line below it.
    trail_length:
        Number of recent steps in each oscillator trail.
    raster_length:
        Number of recent steps in the ``sin(theta_i)`` raster.
    fps, theme, figsize, dpi:
        See :class:`env_lib.utils.rendering.MatplotlibRenderer`.
    """

    def __init__(
        self,
        render_mode: str,
        *,
        n_oscillators: int,
        natural_freq_range: tuple[float, float],
        coupling_scale: float = 1.0,
        sync_threshold: float = 0.99,
        title: str = "Kuramoto network",
        subtitle: str = "",
        trail_length: int = 15,
        raster_length: int = 120,
        fps: float | None = 30.0,
        theme: str | Theme | None = None,
        figsize: tuple[float, float] = (10.0, 5.6),
        dpi: int = 100,
    ):
        super().__init__(render_mode, figsize=figsize, dpi=dpi, fps=fps, title=title, theme=theme)
        self.n = int(n_oscillators)
        self.coupling_scale = float(coupling_scale) if coupling_scale > 0 else 1.0
        self.sync_threshold = float(sync_threshold)
        self.title = title
        self.subtitle = subtitle
        self.trail_length = max(2, int(trail_length))
        self.raster_length = max(2, int(raster_length))
        self._trail_keep = min(self.trail_length, max(5, _TRAIL_SEGMENT_BUDGET // self.n + 1))

        low, high = natural_freq_range
        if high <= low:
            low, high = low - 0.5, high + 0.5
        self._freq_norm = mcolors.Normalize(vmin=low, vmax=high)
        self._freq_cmap = self._sequential_cmap(self.theme)
        self._iu = np.triu_indices(self.n, 1)
        # Fixed tie-break order so that the subset of drawn edges does not flicker.
        self._edge_rank = np.random.default_rng(0).permutation(self._iu[0].size)
        self.ax_phase = None
        self.ax_series = None
        self.ax_raster = None
        self._init_buffers()

    # ------------------------------------------------------------------ state
    def _init_buffers(self) -> None:
        self._last_step = -1
        self._series = np.empty((0, 3))  # columns: t, r, coherence
        self._trail = np.empty((0, self.n))
        self._raster = np.full((self.n, self.raster_length), np.nan)
        self._color_key: bytes | None = None
        self._dot_colors = np.zeros((self.n, 4))
        self._time_span: float | None = None

    def reset(self) -> None:
        """Clear trails, the raster and the time series (called on ``env.reset()``)."""
        self._init_buffers()

    def _dynamic_artist(self, artist: Artist) -> Artist:
        """Register an artist that changes every frame (blitted in rgb_array mode)."""
        self._dynamic(artist)
        return artist

    # ------------------------------------------------------------------ colours
    @staticmethod
    def _sequential_cmap(theme: Theme) -> mcolors.Colormap:
        """Theme colormap, trimmed so that no oscillator blends into the panel."""
        base = colormaps[theme.sequential_cmap]
        low, high = (0.22, 1.0) if _luminance(theme.panel) < 0.5 else (0.0, 0.85)
        return mcolors.ListedColormap(base(np.linspace(low, high, 256)), name="kuramoto_freq")

    def _update_colors(self, natural_frequencies: np.ndarray) -> None:
        key = np.asarray(natural_frequencies, dtype=np.float64).tobytes()
        if key == self._color_key:
            return
        self._color_key = key
        self._dot_colors = self._freq_cmap(self._freq_norm(natural_frequencies))
        glow_outer = self._dot_colors.copy()
        glow_outer[:, 3] = 0.07 if self._dark else 0.10
        glow_inner = self._dot_colors.copy()
        glow_inner[:, 3] = 0.20 if self._dark else 0.24
        self._glow_outer.set_facecolors(glow_outer)
        self._glow_inner.set_facecolors(glow_inner)
        self._dots.set_facecolors(self._dot_colors)

    # ------------------------------------------------------------------ build
    def _build(self, fig: Figure) -> None:
        theme = self.theme
        self._dark = _luminance(theme.panel) < 0.5
        self._color_key = None  # force colour assignment on the new artists
        self._time_span = None
        dyn = self._dynamic_artist
        fig.set_facecolor(theme.background)

        # Header: static bold title, live metrics anchored to its right edge.
        title = fig.text(
            0.035, 0.945, self.title, color=theme.text, fontsize=12, fontweight="bold", va="center"
        )
        self._header = dyn(
            fig.add_artist(
                Annotation(
                    "",
                    xy=(1.0, 0.5),
                    xycoords=title,
                    xytext=(6, 0),
                    textcoords="offset points",
                    color=theme.text,
                    fontsize=11,
                    family="monospace",
                    va="center",
                )
            )
        )
        fig.text(0.035, 0.905, self.subtitle, color=theme.muted, fontsize=8.5, va="center")
        self._badge = dyn(
            fig.text(
                0.975,
                0.945,
                "",
                fontsize=9,
                ha="right",
                va="center",
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.35", "linewidth": 1.0},
            )
        )

        # Phase portrait: static decorations.
        ax = fig.add_axes((0.035, 0.06, 0.44, 0.785))
        self.ax_phase = ax
        style_axes(ax, theme, title="Phase portrait", grid=False)
        ax.set_xlim(-1.32, 1.32)
        ax.set_ylim(-1.32, 1.32)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        angles = np.linspace(0.0, 2.0 * np.pi, 361)
        circle = np.column_stack((np.cos(angles), np.sin(angles)))
        ax.plot(circle[:, 0], circle[:, 1], color=theme.grid, lw=1.6, zorder=0.5)
        ax.plot(0.5 * circle[:, 0], 0.5 * circle[:, 1], color=theme.grid, lw=0.7, ls=":")
        ax.plot([-1.08, 1.08], [0, 0], color=theme.grid, lw=0.6, alpha=0.6, zorder=0.5)
        ax.plot([0, 0], [-1.08, 1.08], color=theme.grid, lw=0.6, alpha=0.6, zorder=0.5)
        labels = ((0.0, "0"), (0.5 * np.pi, "pi/2"), (np.pi, "pi"), (-0.5 * np.pi, "-pi/2"))
        for angle, label in labels:
            ax.text(
                1.17 * np.cos(angle),
                1.17 * np.sin(angle),
                label,
                color=theme.muted,
                fontsize=8,
                ha="center",
                va="center",
            )
        if self.sync_threshold <= 0.95:  # a ring at ~1 would just double the unit circle
            ax.plot(
                self.sync_threshold * circle[:, 0],
                self.sync_threshold * circle[:, 1],
                color=theme.warning,
                lw=0.9,
                ls=(0, (4, 4)),
                alpha=0.45,
                zorder=0.6,
            )

        # Phase portrait: dynamic artists (drawn in this order).
        empty = np.empty((0, 2))
        self._edges = dyn(ax.add_collection(LineCollection([], zorder=1, capstyle="round")))
        self._trails = dyn(
            ax.add_collection(LineCollection([], zorder=2, capstyle="round", linewidths=2.4))
        )
        self._glow_outer = dyn(ax.scatter(empty[:, 0], empty[:, 1], s=420, linewidths=0, zorder=3))
        self._glow_inner = dyn(ax.scatter(empty[:, 0], empty[:, 1], s=170, linewidths=0, zorder=4))
        self._dots = dyn(
            ax.scatter(
                empty[:, 0],
                empty[:, 1],
                s=58,
                edgecolors=theme.background,
                linewidths=0.9,
                zorder=5,
            )
        )
        (vector_glow,) = ax.plot(
            [0, 0], [0, 0], color=theme.warning, lw=8, alpha=0.18, solid_capstyle="round", zorder=6
        )
        self._vector_glow = dyn(vector_glow)
        self._vector = dyn(
            ax.add_patch(
                FancyArrowPatch(
                    (0, 0),
                    (0, 0),
                    arrowstyle="-|>",
                    mutation_scale=15,
                    lw=2.2,
                    color=theme.warning,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=7,
                )
            )
        )
        (origin,) = ax.plot([0], [0], "o", ms=4.5, color=theme.warning, zorder=8)
        dyn(origin)
        self._vector_label = dyn(
            ax.text(
                -1.28,
                -1.27,
                "",
                color=theme.warning,
                fontsize=8.5,
                family="monospace",
                va="bottom",
            )
        )
        # Edge colours: attractive couplings in the accent colour, repulsive in negative.
        self._edge_rgb = np.array(
            [mcolors.to_rgb(theme.accent), mcolors.to_rgb(theme.negative)], dtype=np.float64
        )

        cax = fig.add_axes((0.485, 0.12, 0.011, 0.66))
        colorbar = fig.colorbar(ScalarMappable(norm=self._freq_norm, cmap=self._freq_cmap), cax=cax)
        colorbar.outline.set_edgecolor(theme.grid)
        cax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        cax.tick_params(colors=theme.muted, labelsize=7.5, length=2)
        colorbar.set_label("natural frequency omega_i", color=theme.muted, fontsize=8)

        # Order parameter time series.
        ts = fig.add_axes((0.585, 0.48, 0.39, 0.365))
        self.ax_series = ts
        style_axes(ts, theme, title="Synchronisation", xlabel="time [s]")
        ts.set_ylim(-0.02, 1.22)  # headroom above r = 1 for the legend
        ts.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ts.set_xlim(0.0, 1.0)
        ts.xaxis.set_major_locator(MaxNLocator(nbins=6))
        self._r_fill = dyn(
            ts.add_patch(
                Polygon(
                    np.zeros((3, 2)),
                    closed=True,
                    facecolor=with_alpha(theme.accent, 0.14),
                    lw=0,
                    zorder=1,
                )
            )
        )
        (c_line,) = ts.plot(
            [], [], color=theme.color(1), lw=1.3, alpha=0.9, zorder=2, label="phase coherence"
        )
        (r_glow,) = ts.plot([], [], color=theme.accent, lw=5, alpha=0.18, zorder=2.5)
        (r_line,) = ts.plot([], [], color=theme.accent, lw=1.8, zorder=3, label="order parameter r")
        (r_dot,) = ts.plot([], [], "o", ms=5, color=theme.accent, zorder=4)
        self._c_line = dyn(c_line)
        self._r_glow = dyn(r_glow)
        self._r_line = dyn(r_line)
        self._r_dot = dyn(r_dot)
        if self.sync_threshold <= 1.0:
            ts.axhline(
                self.sync_threshold,
                color=theme.warning,
                lw=1.0,
                ls=(0, (5, 4)),
                zorder=1.5,
                label=f"sync threshold {self.sync_threshold:g}",
            )
        handles, labels_ = ts.get_legend_handles_labels()
        order = [labels_.index(name) for name in labels_ if name.startswith("order")]
        order += [i for i in range(len(labels_)) if i not in order]
        legend = ts.legend(
            [handles[i] for i in order],
            [labels_[i] for i in order],
            loc="upper left",
            ncols=3,
            frameon=False,
            fontsize=7.5,
            handlelength=1.6,
            columnspacing=1.2,
            borderaxespad=0.3,
        )
        for text in legend.get_texts():
            text.set_color(theme.text)

        # Phase raster.
        raster = fig.add_axes((0.585, 0.085, 0.39, 0.25))
        self.ax_raster = raster
        style_axes(
            raster,
            theme,
            title="Phase raster sin(theta_i)",
            xlabel="time relative to now [s]",
            ylabel="oscillator",
            grid=False,
        )
        cmap = colormaps[theme.diverging_cmap].with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
        self._raster_image = dyn(
            raster.imshow(
                self._raster,
                cmap=cmap,
                vmin=-1.0,
                vmax=1.0,
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                extent=(-1.0, 0.0, -0.5, self.n - 0.5),
            )
        )
        raster.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        raster.xaxis.set_major_locator(MaxNLocator(nbins=6))
        self._raster_dt: float | None = None

    # ------------------------------------------------------------------ update
    def _ingest(self, frame: KuramotoFrame) -> None:
        """Pull the history entries recorded since the previous frame."""
        if frame.step < self._last_step:  # the environment was reset without notice
            self._init_buffers()
            self._update_colors(frame.natural_frequencies)
        n_new = frame.step - self._last_step
        if n_new <= 0:
            return
        history = frame.history
        if len(history) > 0:
            n_new = min(n_new, len(history))
            block = np.asarray(history[-n_new:], dtype=np.float64)
            if frame.history_index is not None and block.ndim == 3:
                block = block[:, frame.history_index]
        else:
            n_new = 1
            block = np.asarray(frame.phases, dtype=np.float64)[None]
        self._last_step = frame.step

        steps = np.arange(frame.step - n_new + 1, frame.step + 1)
        r, _ = mean_field(block)
        rows = np.column_stack((steps * frame.dt, r, phase_coherence(block)))
        self._series = np.concatenate((self._series, rows))[-_MAX_SERIES:]
        self._trail = np.concatenate((self._trail, block))[-self._trail_keep :]
        k = min(n_new, self.raster_length)
        self._raster = np.roll(self._raster, -k, axis=1)
        self._raster[:, -k:] = np.sin(block[-k:]).T

    def _update(self, frame: KuramotoFrame) -> None:
        theme = self.theme
        self._update_colors(frame.natural_frequencies)
        self._ingest(frame)

        phases = np.asarray(frame.phases, dtype=np.float64)
        positions = np.column_stack((np.cos(phases), np.sin(phases)))
        self._dots.set_offsets(positions)
        self._glow_inner.set_offsets(positions)
        self._glow_outer.set_offsets(positions)
        self._update_edges(frame.coupling_matrix, positions)
        self._update_trails()

        r, psi = mean_field(phases)
        r, psi = float(r), float(psi)
        tip = (r * np.cos(psi), r * np.sin(psi))
        self._vector.set_positions((0.0, 0.0), tip)
        self._vector_glow.set_data([0.0, tip[0]], [0.0, tip[1]])
        self._vector_label.set_text(f"r   = {r:.3f}\npsi = {psi:+.2f} rad")

        self._update_series(frame)
        if self._raster_dt != frame.dt:
            self._raster_dt = frame.dt
            self._raster_image.set_extent((-self.raster_length * frame.dt, 0.0, -0.5, self.n - 0.5))
            self._invalidate()
        self._raster_image.set_data(self._raster)

        reward = "--" if frame.reward is None else f"{frame.reward:.3f}"
        self._header.set_text(
            f"| N={self.n} | t={frame.step * frame.dt:.2f}s | "
            f"step {frame.step}/{frame.max_steps} | r={r:.3f} | reward={reward}"
        )
        synced = r > self.sync_threshold
        color = theme.positive if synced else theme.muted
        self._badge.set_text("SYNCHRONISED" if synced else "DRIFTING")
        self._badge.set_color(color)
        self._badge.get_bbox_patch().set_facecolor(with_alpha(color, 0.14))
        self._badge.get_bbox_patch().set_edgecolor(with_alpha(color, 0.6))

    def _update_series(self, frame: KuramotoFrame) -> None:
        series = self._series
        t, r_series, c_series = series[:, 0], series[:, 1], series[:, 2]
        self._r_line.set_data(t, r_series)
        self._r_glow.set_data(t, r_series)
        self._c_line.set_data(t, c_series)
        if t.size == 0:
            return
        self._r_dot.set_data(t[-1:], r_series[-1:])
        self._r_fill.set_xy(np.vstack(([t[0], 0.0], np.column_stack((t, r_series)), [t[-1], 0.0])))
        # The time axis grows in doublings so that the cached background stays valid.
        span = self._time_span or min(frame.max_steps, 100) * frame.dt
        while t[-1] > span:
            span *= 2.0
        if span != self._time_span:
            self._time_span = span
            self.ax_series.set_xlim(0.0, span)
            self._invalidate()

    def _update_edges(self, coupling: np.ndarray | None, positions: np.ndarray) -> None:
        """Chords for non-zero couplings; opacity and width grow with ``|K_ij|``.

        At most ``_MAX_EDGES`` chords (the strongest couplings) are drawn, which
        bounds the drawing cost of dense networks.
        """
        if coupling is None:
            self._edges.set_segments([])
            return
        coupling = np.asarray(coupling, dtype=np.float64)
        rows, cols = self._iu
        weights = 0.5 * (coupling[rows, cols] + coupling[cols, rows])
        (index,) = np.nonzero(weights)
        if index.size == 0:
            self._edges.set_segments([])
            return
        if index.size > _MAX_EDGES:
            order = np.lexsort((self._edge_rank[index], -np.abs(weights[index])))
            index = np.sort(index[order[:_MAX_EDGES]])
        rows, cols, weights = rows[index], cols[index], weights[index]
        strength = np.clip(np.abs(weights) / self.coupling_scale, 0.0, 1.0)
        alpha_max = float(np.clip(2.6 / np.sqrt(index.size), 0.08, 0.42))
        colors = np.empty((index.size, 4))
        colors[:, :3] = self._edge_rgb[(weights < 0).astype(np.intp)]
        colors[:, 3] = alpha_max * (0.12 + 0.88 * strength)
        self._edges.set_segments(np.stack((positions[rows], positions[cols]), axis=1))
        self._edges.set_color(colors)
        self._edges.set_linewidths(0.35 + 1.25 * strength)

    def _update_trails(self) -> None:
        trail = self._trail
        if trail.shape[0] < 2:
            self._trails.set_segments([])
            return
        points = np.stack((np.cos(trail), np.sin(trail)), axis=-1)  # (L, N, 2)
        segments = np.stack((points[:-1], points[1:]), axis=2).reshape(-1, 2, 2)
        n_seg = trail.shape[0] - 1
        age = np.arange(1, n_seg + 1) / (self._trail_keep - 1)  # oldest -> newest
        colors = np.broadcast_to(self._dot_colors, (n_seg, self.n, 4)).copy()
        colors[..., 3] = (0.85 * age**1.4)[:, None]
        # Hide chords of large jumps (noise, coarse dt) that would cut across the circle.
        jumps = np.abs(wrap_phases(np.diff(trail, axis=0))) > 0.6
        colors[..., 3][jumps] = 0.0
        self._trails.set_segments(segments)
        self._trails.set_color(colors.reshape(-1, 4))
