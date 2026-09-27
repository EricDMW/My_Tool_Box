"""Matplotlib renderer for :class:`env_lib.consensus_env.consensus_env.ConsensusEnv`.

The frame is a small dashboard (1000 x 560 px by default):

* left -- the arena: agents as coloured markers with glow halos, communication
  links whose opacity decreases with their length, fading trails of the last
  positions, the formation target slots (hollow markers around the current
  centroid, formation task only) and the centroid (cross);
* right, top -- the task error on a log scale, with the success tolerance as a
  dashed line over a shaded success band;
* right, bottom -- the algebraic connectivity ``lambda_2`` of the graph;
* header -- task, team size, step counter, key metrics and a status badge.

All artists are created once in :meth:`ConsensusRenderer._build`; each frame
only updates their data. Text is the dominant drawing cost in Agg, so the
figure keeps the number of text artists small (no arena tick labels, one
shared step axis, fixed title and label positions, a hand-built key instead of
a :class:`~matplotlib.legend.Legend`). In ``"rgb_array"`` mode the renderer
also blits: the static layer (axes, grid, ticks, titles, key) is rasterised
once and cached, and only the animated artists are redrawn on top of it. The
cache is rebuilt when an axis range changes (a few times per episode).
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.ticker import LogLocator, MaxNLocator, MultipleLocator, NullLocator
from matplotlib.transforms import ScaledTranslation

from env_lib.utils.rendering import (
    MatplotlibRenderer,
    Theme,
    style_axes,
    with_alpha,
)

__all__ = ["ConsensusRenderer"]

_EDGE_ALPHA = (0.12, 0.70)
_TRAIL_ALPHA = 0.75
_TRAIL_WIDTH = (0.4, 2.2)
_ERROR_FLOOR = 1e-12
_KEY_FONTSIZE = 7.5


class ConsensusRenderer(MatplotlibRenderer):
    """Dashboard renderer for the consensus / formation environment.

    Parameters
    ----------
    render_mode:
        ``"human"`` or ``"rgb_array"``.
    n_agents:
        Number of agents.
    arena_size:
        Half width of the square arena.
    task, formation_shape, topology, dynamics:
        Environment settings shown in the header.
    max_steps:
        Episode length (x range of the metric panels).
    tolerance:
        Success threshold on the task error (dashed line).
    comm_radius:
        Communication range of a proximity graph, used as the reference length
        of the link opacity ramp. ``None`` (static graphs) uses ``arena_size``.
    trail_length:
        Number of past positions drawn as a fading trail (``>= 2``).
    figsize, dpi, fps, theme:
        See :class:`env_lib.utils.rendering.MatplotlibRenderer`.
    """

    def __init__(
        self,
        render_mode: str,
        *,
        n_agents: int,
        arena_size: float,
        task: str = "consensus",
        formation_shape: str = "circle",
        topology: str = "ring",
        dynamics: str = "single",
        max_steps: int = 200,
        tolerance: float = 0.05,
        comm_radius: float | None = None,
        trail_length: int = 40,
        figsize: tuple[float, float] = (10.0, 5.6),
        dpi: int = 100,
        fps: float | None = 20.0,
        theme: str | Theme | None = None,
    ):
        super().__init__(
            render_mode, figsize=figsize, dpi=dpi, fps=fps, title="Consensus", theme=theme
        )
        if trail_length < 2:
            raise ValueError(f"trail_length must be >= 2, got {trail_length}")
        self.n_agents = int(n_agents)
        self.arena_size = float(arena_size)
        self.task = task
        self.formation_shape = formation_shape
        self.topology = topology
        self.dynamics = dynamics
        self.max_steps = int(max_steps)
        self.tolerance = float(tolerance)
        self.edge_reference = float(comm_radius) if comm_radius else self.arena_size
        self.trail_length = int(trail_length)

        n, t = self.n_agents, self.trail_length
        self._task_label = f"formation({formation_shape})" if task == "formation" else task
        self._agent_rgba = np.array([with_alpha(self.theme.color(i), 1.0) for i in range(n)])
        self._edge_rgba = np.array(with_alpha(self.theme.muted, 1.0))
        self._pairs = np.triu_indices(n, k=1)

        # Trail colour and width tables, oldest segment first.
        ramp = np.arange(1, t) / (t - 1)
        self._trail_rgba = np.repeat(self._agent_rgba[:, None, :], t - 1, axis=1)
        self._trail_rgba[..., 3] = _TRAIL_ALPHA * ramp**1.6
        self._trail_width = _TRAIL_WIDTH[0] + (_TRAIL_WIDTH[1] - _TRAIL_WIDTH[0]) * ramp

        # Bounded histories: trail ring (oldest first) and metrics indexed by step.
        self._trail = np.zeros((t, n, 2), dtype=np.float64)
        self._trail_count = 0
        self._last_step = -1
        self._error_hist = np.full(self.max_steps + 1, np.nan)
        self._lambda_hist = np.full(self.max_steps + 1, np.nan)
        self._steps = np.arange(self.max_steps + 1, dtype=np.float64)
        self._error_limits: tuple[float, float] | None = None
        self._lambda_top = 0.0

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear trails and metric histories."""
        self._trail_count = 0
        self._last_step = -1
        self._error_hist.fill(np.nan)
        self._lambda_hist.fill(np.nan)
        self._error_limits = None
        self._lambda_top = 0.0

    # ------------------------------------------------------------------
    def _build(self, fig: Figure) -> None:
        theme = self.theme
        self._build_header(fig)
        self._build_arena(fig)

        # Task error panel (log scale).
        ax_err = fig.add_axes((0.585, 0.415, 0.39, 0.445))
        style_axes(ax_err, theme)
        _fixed_title(ax_err, "Task error", theme)
        ax_err.set_yscale("log")
        ax_err.set_xlim(0, self.max_steps)
        ax_err.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax_err.yaxis.set_minor_locator(NullLocator())
        ax_err.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax_err.tick_params(labelbottom=False)
        _fix_label_positions(ax_err)
        ax_err.axhspan(
            _ERROR_FLOOR, self.tolerance, color=with_alpha(theme.positive, 0.08), lw=0, zorder=0
        )
        ax_err.axhline(self.tolerance, color=theme.positive, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax_err.text(
            0.99,
            self.tolerance,
            f"tolerance {self.tolerance:g}",
            transform=ax_err.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color=theme.positive,
            fontsize=7.5,
        )
        (self._err_glow,) = ax_err.plot([], [], color=theme.accent, lw=5.0, alpha=0.18)
        (self._err_line,) = ax_err.plot([], [], color=theme.accent, lw=1.8)
        (self._err_dot,) = ax_err.plot(
            [], [], ls="none", marker="o", markersize=5.5, color=theme.accent,
            markeredgecolor=theme.panel, markeredgewidth=1.2,
        )  # fmt: skip
        ax_err.set_ylim(self.tolerance * 0.1, self.tolerance * 1e3)
        self._ax_err = ax_err

        # Algebraic connectivity panel (shares the step axis with the error panel).
        ax_lam = fig.add_axes((0.585, 0.09, 0.39, 0.215))
        style_axes(ax_lam, theme, xlabel="step")
        _fixed_title(ax_lam, r"Algebraic connectivity $\lambda_2$", theme)
        ax_lam.set_xlim(0, self.max_steps)
        ax_lam.set_ylim(0.0, 1.0)
        ax_lam.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax_lam.yaxis.set_major_locator(MaxNLocator(3))
        _fix_label_positions(ax_lam, xlabel_y=-0.3)
        (self._lam_glow,) = ax_lam.plot(
            [], [], color=theme.accent, lw=4.5, alpha=0.16, drawstyle="steps-post"
        )
        (self._lam_line,) = ax_lam.plot([], [], color=theme.accent, lw=1.6, drawstyle="steps-post")
        (self._lam_dot,) = ax_lam.plot(
            [], [], ls="none", marker="o", markersize=5.0, color=theme.accent,
            markeredgecolor=theme.panel, markeredgewidth=1.1,
        )  # fmt: skip
        self._lam_warning = ax_lam.text(
            0.99,
            0.92,
            "",
            transform=ax_lam.transAxes,
            ha="right",
            va="top",
            color=theme.negative,
            fontsize=8,
            fontweight="bold",
        )
        self._ax_lam = ax_lam

        # Artists redrawn every frame (blitted in rgb_array mode), in drawing order.
        self._dynamic(
            self._edges,
            self._links,
            self._trails,
            self._slots,
            self._halo_outer,
            self._halo_inner,
            self._agents,
            self._centroid,
            self._err_glow,
            self._err_line,
            self._err_dot,
            self._lam_glow,
            self._lam_line,
            self._lam_dot,
            self._lam_warning,
            self._header,
            self._badge,
        )

    def _build_header(self, fig: Figure) -> None:
        theme = self.theme
        self._header = fig.text(
            0.025, 0.955, "", color=theme.text, fontsize=12, fontweight="bold", va="center"
        )
        fig.text(
            0.025,
            0.91,
            f"topology={self.topology}  |  dynamics={self.dynamics}  |  "
            f"arena=[-{self.arena_size:g}, {self.arena_size:g}]^2",
            color=theme.muted,
            fontsize=8.5,
            va="center",
        )
        self._badge = fig.text(
            0.975,
            0.955,
            "RUNNING",
            color=theme.muted,
            fontsize=8.5,
            fontweight="bold",
            ha="right",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=with_alpha(theme.muted, 0.12),
                edgecolor=with_alpha(theme.muted, 0.6),
                linewidth=0.8,
            ),
        )

    def _build_arena(self, fig: Figure) -> None:
        theme = self.theme
        n, a = self.n_agents, self.arena_size
        formation = self.task == "formation"

        ax = fig.add_axes((0.025, 0.035, 0.49, 0.845))
        style_axes(ax, theme, grid=True)
        margin = 0.03 * a
        ax.set_xlim(-a - margin, a + margin)
        ax.set_ylim(-a - margin, a + margin)
        ax.set_aspect("equal", adjustable="box")
        # Grid only: coordinates are summarised in the sub-header.
        grid_step = a / 4.0
        ax.xaxis.set_major_locator(MultipleLocator(grid_step))
        ax.yaxis.set_major_locator(MultipleLocator(grid_step))
        ax.tick_params(length=0, labelbottom=False, labelleft=False)
        _fix_label_positions(ax)
        ax.add_patch(
            Rectangle(
                (-a, -a), 2 * a, 2 * a, fill=False, edgecolor=theme.muted, linewidth=1.0, alpha=0.6
            )
        )
        self._ax_arena = ax

        self._edges = LineCollection([], linewidths=1.1, zorder=1, capstyle="round")
        ax.add_collection(self._edges)
        self._links = LineCollection(
            [], linewidths=0.8, linestyles=(0, (2.0, 2.5)), zorder=1.5, colors=theme.muted
        )
        self._links.set_alpha(0.45)
        self._links.set_visible(formation)
        ax.add_collection(self._links)
        self._trails = LineCollection([], zorder=2, capstyle="butt")
        ax.add_collection(self._trails)

        empty = np.zeros((n, 2))
        self._slots = ax.scatter(
            empty[:, 0],
            empty[:, 1],
            s=120,
            facecolors="none",
            edgecolors=[with_alpha(c, 0.85) for c in self._agent_rgba],
            linewidths=1.3,
            zorder=2.5,
        )
        self._slots.set_visible(formation)
        self._halo_outer = ax.scatter(
            empty[:, 0],
            empty[:, 1],
            s=520,
            c=[with_alpha(c, 0.09) for c in self._agent_rgba],
            linewidths=0,
            zorder=3,
        )
        self._halo_inner = ax.scatter(
            empty[:, 0],
            empty[:, 1],
            s=210,
            c=[with_alpha(c, 0.22) for c in self._agent_rgba],
            linewidths=0,
            zorder=3.1,
        )
        self._agents = ax.scatter(
            empty[:, 0],
            empty[:, 1],
            s=64,
            c=self._agent_rgba,
            edgecolors=theme.panel,
            linewidths=1.4,
            zorder=4,
        )
        (self._centroid,) = ax.plot(
            [0.0], [0.0], marker="+", markersize=11, markeredgewidth=2.0, color=theme.text, zorder=5
        )
        self._build_key(fig, ax, formation)

    def _build_key(self, fig: Figure, ax, formation: bool) -> None:
        """Compact one-row key centred at the top of the arena (cheaper than a Legend)."""
        theme = self.theme
        entries: list[tuple[str, dict]] = [
            ("agent", dict(marker="o", markersize=6, markerfacecolor=theme.color(0),
                           markeredgecolor=theme.panel, ls="none")),
            ("link", dict(color=theme.muted, lw=1.2)),
        ]  # fmt: skip
        if formation:
            entries.append(
                ("target slot", dict(marker="o", markersize=7, markerfacecolor="none",
                                     markeredgecolor=theme.muted, ls="none"))
            )  # fmt: skip
        entries.append(
            ("centroid", dict(marker="+", markersize=8, markeredgewidth=1.8, color=theme.text,
                              ls="none"))
        )  # fmt: skip

        # Geometry in inches relative to the anchor (top centre of the arena).
        handle_w, gap, spacing, pad = 0.16, 0.05, 0.16, 0.08
        text_w = [
            TextPath((0, 0), label, size=_KEY_FONTSIZE).get_extents().width / 72.0
            for label, _ in entries
        ]
        total = sum(handle_w + gap + w for w in text_w) + spacing * (len(entries) - 1)
        height = 0.2
        anchor = fig.dpi_scale_trans + ScaledTranslation(0.5, 0.985, ax.transAxes)
        ax.add_patch(
            FancyBboxPatch(
                (-total / 2 - pad, -height - 0.02),
                total + 2 * pad,
                height,
                boxstyle="round,pad=0,rounding_size=0.05",
                transform=anchor,
                facecolor=with_alpha(theme.panel, 0.85),
                edgecolor=theme.grid,
                linewidth=0.8,
                zorder=6,
                clip_on=False,
            )
        )
        x, y = -total / 2, -0.02 - height / 2
        for (label, style), width in zip(entries, text_w):
            if style.get("ls") == "none":
                ax.plot([x + handle_w / 2], [y], transform=anchor, zorder=7, clip_on=False, **style)
            else:
                ax.plot(
                    [x, x + handle_w], [y, y], transform=anchor, zorder=7, clip_on=False, **style
                )
            ax.text(
                x + handle_w + gap,
                y,
                label,
                transform=anchor,
                va="center_baseline",
                color=theme.text,
                fontsize=_KEY_FONTSIZE,
                zorder=7,
            )
            x += handle_w + gap + width + spacing

    # ------------------------------------------------------------------
    def _update(
        self,
        *,
        positions: np.ndarray,
        adjacency: np.ndarray,
        targets: np.ndarray,
        centroid: np.ndarray,
        step: int,
        error: float,
        lambda2: float,
        success: bool,
    ) -> None:
        pos = np.asarray(positions, dtype=np.float64)
        step = int(step)

        # Trail history: a repeated step overwrites, a step back restarts.
        if step < self._last_step:
            self.reset()
        if step != self._last_step or self._trail_count == 0:
            self._trail[:-1] = self._trail[1:]
            self._trail_count = min(self._trail_count + 1, self.trail_length)
        self._trail[-1] = pos
        self._last_step = step

        # Agents, halos, centroid and formation slots.
        self._agents.set_offsets(pos)
        self._halo_outer.set_offsets(pos)
        self._halo_inner.set_offsets(pos)
        self._centroid.set_data([centroid[0]], [centroid[1]])
        if self.task == "formation":
            self._slots.set_offsets(targets)
            self._links.set_segments(np.stack([pos, np.asarray(targets)], axis=1))

        # Communication links, more opaque when short.
        mask = np.asarray(adjacency, dtype=bool)[self._pairs]
        start, end = pos[self._pairs[0][mask]], pos[self._pairs[1][mask]]
        closeness = np.clip(1.0 - np.linalg.norm(end - start, axis=1) / self.edge_reference, 0, 1)
        colors = np.tile(self._edge_rgba, (start.shape[0], 1))
        colors[:, 3] = _EDGE_ALPHA[0] + (_EDGE_ALPHA[1] - _EDGE_ALPHA[0]) * closeness
        self._edges.set_segments(np.stack([start, end], axis=1))
        self._edges.set_color(colors)

        # Fading trails (vectorised over agents and segments).
        count = self._trail_count
        if count >= 2:
            points = self._trail[-count:].transpose(1, 0, 2)  # (n, count, 2)
            segments = np.stack([points[:, :-1], points[:, 1:]], axis=2).reshape(-1, 2, 2)
            self._trails.set_segments(segments)
            self._trails.set_color(self._trail_rgba[:, -(count - 1) :].reshape(-1, 4))
            self._trails.set_linewidth(np.tile(self._trail_width[-(count - 1) :], self.n_agents))
        else:
            self._trails.set_segments([])

        self._update_metrics(step, float(error), float(lambda2))
        self._update_header(step, float(error), float(lambda2), bool(success))

    def _update_metrics(self, step: int, error: float, lambda2: float) -> None:
        if 0 <= step <= self.max_steps:
            self._error_hist[step] = max(error, _ERROR_FLOOR)
            self._lambda_hist[step] = lambda2
        valid = ~np.isnan(self._error_hist)
        steps, errors = self._steps[valid], self._error_hist[valid]
        lambdas = self._lambda_hist[valid]
        self._err_line.set_data(steps, errors)
        self._err_glow.set_data(steps, errors)
        self._lam_line.set_data(steps, lambdas)
        self._lam_glow.set_data(steps, lambdas)
        if steps.size == 0:
            self._err_dot.set_data([], [])
            self._lam_dot.set_data([], [])
            return
        self._err_dot.set_data(steps[-1:], errors[-1:])
        self._lam_dot.set_data(steps[-1:], lambdas[-1:])

        # Decade-aligned limits that only grow within an episode (stable axes).
        low = 10.0 ** math.floor(math.log10(min(float(errors.min()), 0.3 * self.tolerance)))
        high = 10.0 ** math.ceil(math.log10(max(float(errors.max()) * 1.5, 10 * self.tolerance)))
        if self._error_limits is not None:
            low, high = min(low, self._error_limits[0]), max(high, self._error_limits[1])
        if (low, high) != self._error_limits:
            self._ax_err.set_ylim(low, high)
            self._error_limits = (low, high)
            self._invalidate()
        top = _nice_ceil(1.2 * float(np.max(lambdas)))
        if top > self._lambda_top:
            self._lambda_top = top
            self._ax_lam.set_ylim(0.0, top)
            self._invalidate()
        self._lam_warning.set_text("graph disconnected" if lambda2 < 1e-9 else "")

    def _update_header(self, step: int, error: float, lambda2: float, success: bool) -> None:
        theme = self.theme
        self._header.set_text(
            f"Consensus | task={self._task_label} | N={self.n_agents} | "
            f"step {step}/{self.max_steps} | error={_fmt(error)} | lambda2={lambda2:.3f}"
        )
        badge_color = theme.positive if success else theme.muted
        self._badge.set_text("SOLVED" if success else "RUNNING")
        self._badge.set_color(badge_color)
        patch = self._badge.get_bbox_patch()
        patch.set_facecolor(with_alpha(badge_color, 0.14))
        patch.set_edgecolor(with_alpha(badge_color, 0.7))


def _fixed_title(ax, title: str, theme: Theme) -> None:
    """Left-aligned title at a fixed position (skips matplotlib's per-draw title layout)."""
    ax.set_title(title, color=theme.text, fontsize=10, loc="left", pad=6, y=1.0)


def _fix_label_positions(ax, xlabel_y: float = -0.12, ylabel_x: float = -0.1) -> None:
    """Pin axis-label positions (skips the per-draw tick-label bounding-box search)."""
    ax.xaxis.set_label_coords(0.5, xlabel_y)
    ax.yaxis.set_label_coords(ylabel_x, 0.5)


def _nice_ceil(value: float) -> float:
    """Smallest number of the form {1, 2, 2.5, 5} * 10**k that is >= ``value`` (at least 0.1)."""
    value = max(value, 0.1)
    scale = 10.0 ** math.floor(math.log10(value))
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        if factor * scale >= value * (1 - 1e-12):
            return factor * scale
    return 10.0 * scale  # pragma: no cover - unreachable


def _fmt(value: float) -> str:
    """Compact number formatting for the header."""
    if not math.isfinite(value):
        return str(value)
    if value == 0.0 or 1e-3 <= abs(value) < 1e4:
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.3f}"
    return f"{value:.3e}"
