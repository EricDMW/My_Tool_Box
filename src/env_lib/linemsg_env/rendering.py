"""Matplotlib dashboard for :class:`env_lib.linemsg_env.LineMsgEnv`.

The frame has three parts:

* the line of agents (top): nodes coloured by state (message / no message,
  boundary cells in the grid colour), a glow under agents whose action is
  ``1`` and arrows on the links that are active this step (message flow
  towards the sink on the left);
* a space-time raster (bottom left): one row per step, one column per agent,
  so message fronts travelling towards the sink are visible;
* the team reward trace (bottom right).
"""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator

from env_lib.utils.rendering import (
    MatplotlibRenderer,
    Theme,
    style_axes,
    with_alpha,
)

__all__ = ["LineMsgRenderer"]

_EMPTY = np.zeros((0, 2))


class LineMsgRenderer(MatplotlibRenderer):
    """Persistent-artist renderer for the line message-passing environment.

    Parameters
    ----------
    render_mode : {"human", "rgb_array"}
        See :class:`env_lib.utils.rendering.FigureWindow`.
    num_agents : int
        Number of agents on the line.
    n_obs_neighbors : int, default 1
        Number of boundary cells drawn on each side of the line.
    window : int, default 51
        Number of time steps shown by the raster and the reward trace
        (normally ``max_iter + 1``).
    fps : float, optional
        Frame-rate cap in ``"human"`` mode.
    theme : str or Theme, optional
        Colour theme (defaults to the active theme).
    figsize, dpi :
        Figure geometry (default 1000 x 560 pixels).
    """

    def __init__(
        self,
        render_mode: str,
        *,
        num_agents: int,
        n_obs_neighbors: int = 1,
        window: int = 51,
        fps: float | None = 10.0,
        theme: str | Theme | None = None,
        figsize: tuple[float, float] = (10.0, 5.6),
        dpi: int = 100,
    ):
        super().__init__(
            render_mode, figsize=figsize, dpi=dpi, fps=fps, title="LineMsg", theme=theme
        )
        self.num_agents = int(num_agents)
        self.n_pad = int(n_obs_neighbors)
        self.window = max(2, int(window))
        self._first = None

    # ------------------------------------------------------------------
    def _build(self, fig: Figure) -> None:
        th = self.theme
        n_agents, n_pad = self.num_agents, self.n_pad
        xs = np.arange(n_agents, dtype=float)
        node_size = float(np.clip(2600.0 / n_agents, 18.0, 300.0))
        self._node_size = node_size
        max_reward = 1.0 + 0.1 * (n_agents - 1)
        self._max_reward = max_reward

        # Header ------------------------------------------------------------
        fig.text(0.045, 0.945, "LineMsg", color=th.text, fontsize=15, fontweight="bold")
        fig.text(
            0.045,
            0.905,
            f"{n_agents} agents relay a message from the source (right) to the sink (left)",
            color=th.muted,
            fontsize=9,
        )
        self._metric_values = {}
        for k, label in enumerate(("step", "reward", "return", "holding")):
            x = 0.555 + 0.112 * k
            fig.text(x, 0.955, label.upper(), color=th.muted, fontsize=7.5)
            self._metric_values[label] = fig.text(
                x, 0.905, "", color=th.text, fontsize=12.5, fontweight="bold"
            )

        # Line of agents ------------------------------------------------------
        ax = fig.add_axes((0.045, 0.66, 0.635, 0.19))
        style_axes(ax, th, grid=False)
        ax.set_xlim(-n_pad - 0.7, n_agents - 1 + n_pad + 0.7)
        ax.set_ylim(-1.0, 0.85)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(-n_pad - 0.55, 0.62, "Line of agents", color=th.text, fontsize=10, va="center")
        pad_x = np.concatenate([np.arange(-n_pad, 0), np.arange(n_agents, n_agents + n_pad)])
        self._pads = ax.scatter(
            pad_x,
            np.zeros_like(pad_x, dtype=float),
            s=node_size * 0.55,
            marker="s",
            color=th.grid,
            zorder=2,
        )
        base = [((i, 0.0), (i + 1, 0.0)) for i in range(-1, n_agents)]
        ax.add_collection(
            LineCollection(base, colors=[th.grid], linewidths=2.0, zorder=1, capstyle="round")
        )
        link_width = 3.0 if n_agents <= 40 else 1.8
        self._link_glow = LineCollection(
            [], colors=[with_alpha(th.accent, 0.25)], linewidths=link_width * 3, zorder=1.2
        )
        self._links = LineCollection([], colors=[th.accent], linewidths=link_width, zorder=1.4)
        ax.add_collection(self._link_glow)
        ax.add_collection(self._links)
        self._arrows = ax.scatter(
            [], [], s=node_size * 0.35, marker="<", color=th.accent, linewidths=0, zorder=3
        )
        self._glow = ax.scatter(
            [], [], s=node_size * 2.6, color=with_alpha(th.accent, 0.22), linewidths=0, zorder=2
        )
        self._nodes = ax.scatter(
            xs,
            np.zeros(n_agents),
            s=node_size,
            c=[th.grid] * n_agents,
            edgecolors=th.panel,
            linewidths=1.5,
            zorder=4,
        )
        self._node_rgba = {
            1: np.array(with_alpha(th.positive, 1.0)),
            0: np.array(with_alpha(th.grid, 1.0)),
        }
        # Muted ring marks agents without the message (visible in both themes).
        self._rings = ax.scatter(
            xs,
            np.zeros(n_agents),
            s=node_size,
            facecolors="none",
            edgecolors=th.muted,
            linewidths=1.0,
            zorder=5,
        )
        label_y = -0.62
        ax.text(0, label_y, "sink", color=th.muted, fontsize=8, ha="center", va="center")
        ax.text(
            n_agents - 1, label_y, "source", color=th.muted, fontsize=8, ha="center", va="center"
        )
        self._ax_line = ax

        # Space-time raster ---------------------------------------------------
        ax_r = fig.add_axes((0.045, 0.1, 0.635, 0.5))
        style_axes(ax_r, th, xlabel="agent", ylabel="step", grid=False)
        cmap = ListedColormap([th.grid, th.positive]).with_extremes(bad=th.panel)
        self._raster_data = np.full((self.window, n_agents), np.nan)
        self._raster = ax_r.imshow(
            self._raster_data,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=(-0.5, n_agents - 0.5, self.window - 0.5, -0.5),
        )
        self._now_line = ax_r.axhline(0.0, color=th.accent, linewidth=1.0, alpha=0.9)
        ax_r.set_xlim(-n_pad - 0.7, n_agents - 1 + n_pad + 0.7)
        if n_agents <= 16:
            ax_r.set_xticks(xs)
        else:
            ax_r.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax_r.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax_r.set_title(
            "Space-time raster of agent states", color=th.text, fontsize=10, loc="left", pad=6
        )
        self._ax_raster = ax_r

        # Legend -------------------------------------------------------------
        ax_l = fig.add_axes((0.72, 0.64, 0.26, 0.22))
        ax_l.set_axis_off()
        handles = [
            Line2D([], [], ls="", marker="o", ms=9, mfc=th.positive, mec=th.panel, label="message"),
            Line2D([], [], ls="", marker="o", ms=9, mfc=th.grid, mec=th.muted, label="no message"),
            Line2D([], [], ls="", marker="s", ms=7, mfc=th.grid, mec=th.grid, label="boundary"),
            Line2D([], [], color=th.accent, lw=2.5, marker="<", ms=7, label="active link (a = 1)"),
        ]
        legend = ax_l.legend(
            handles=handles,
            loc="center left",
            frameon=False,
            fontsize=9,
            labelcolor=th.text,
            handlelength=2.2,
            borderaxespad=0.0,
        )
        legend.set_in_layout(False)

        # Reward trace ------------------------------------------------------
        ax_w = fig.add_axes((0.735, 0.1, 0.245, 0.5))
        style_axes(ax_w, th, xlabel="step")
        ax_w.set_title("Team reward", color=th.text, fontsize=10, loc="left", pad=6)
        ax_w.axhline(max_reward, color=th.muted, linewidth=0.9, linestyle=(0, (4, 3)))
        ax_w.text(
            0.02,
            max_reward,
            " max",
            color=th.muted,
            fontsize=8,
            va="bottom",
            transform=ax_w.get_yaxis_transform(),
        )
        self._fill = Polygon(
            np.zeros((0, 2)), closed=True, facecolor=with_alpha(th.accent, 0.16), edgecolor="none"
        )
        ax_w.add_patch(self._fill)
        (self._reward_glow,) = ax_w.plot([], [], color=with_alpha(th.accent, 0.25), linewidth=5)
        (self._reward_line,) = ax_w.plot([], [], color=th.accent, linewidth=1.8)
        (self._reward_dot,) = ax_w.plot([], [], "o", color=th.accent, ms=5, mec=th.panel, mew=1.2)
        ax_w.set_ylim(0.0, max_reward * 1.12)
        ax_w.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax_w.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax_w.set_xlim(0, self.window - 1)
        self._ax_reward = ax_w
        self._first = None

        # Artists that change every frame, in drawing order.
        self._dynamic(
            self._link_glow,
            self._links,
            self._pads,
            self._glow,
            self._arrows,
            self._nodes,
            self._rings,
            self._raster,
            self._now_line,
            self._fill,
            self._reward_glow,
            self._reward_line,
            self._reward_dot,
            *self._metric_values.values(),
        )

    # ------------------------------------------------------------------
    def _update(
        self,
        *,
        state: np.ndarray,
        actions: np.ndarray,
        state_history: np.ndarray,
        reward_history: np.ndarray,
        first_step: int,
        step: int,
        max_iter: int,
        reward: float,
        episode_return: float,
    ) -> None:
        n_agents, n_pad = self.num_agents, self.n_pad
        agent_state = np.asarray(state[n_pad : n_pad + n_agents])
        agent_action = np.asarray(actions[n_pad : n_pad + n_agents])
        has_msg = agent_state == 1

        # Nodes and links.
        colors = np.where(has_msg[:, None], self._node_rgba[1], self._node_rgba[0])
        self._nodes.set_facecolors(colors)
        ring_alpha = np.where(has_msg, 0.0, 1.0)
        ring = np.array(with_alpha(self.theme.muted, 1.0))
        ring_colors = np.tile(ring, (n_agents, 1))
        ring_colors[:, 3] = ring_alpha
        self._rings.set_edgecolors(ring_colors)

        active = np.flatnonzero(agent_action == 1)
        xs = active.astype(float)
        self._glow.set_offsets(np.column_stack([xs, np.zeros_like(xs)]) if active.size else _EMPTY)
        pulling = active[active >= 1]  # the sink's action has no effect
        segs = [((i, 0.0), (i + 1, 0.0)) for i in pulling]
        self._links.set_segments(segs)
        self._link_glow.set_segments(segs)
        if pulling.size:
            self._arrows.set_offsets(np.column_stack([pulling + 0.5, np.zeros(pulling.size)]))
        else:
            self._arrows.set_offsets(_EMPTY)

        # Raster.
        rows = min(len(state_history), self.window)
        data = self._raster_data
        data.fill(np.nan)
        data[:rows] = state_history[:rows]
        self._raster.set_data(data)
        if first_step != self._first:
            self._first = first_step
            top, bottom = first_step - 0.5, first_step + self.window - 0.5
            self._raster.set_extent((-0.5, n_agents - 0.5, bottom, top))
            self._ax_raster.set_ylim(bottom, top)
            self._ax_reward.set_xlim(first_step, first_step + self.window - 1)
            self._invalidate()
        self._now_line.set_ydata([step + 0.5, step + 0.5])  # below the current row

        # Reward trace (steps >= 1; row 0 of an episode is the reset state).
        steps = first_step + np.arange(len(reward_history))
        keep = steps >= 1
        sx, sy = steps[keep].astype(float), np.asarray(reward_history)[keep]
        self._reward_line.set_data(sx, sy)
        self._reward_glow.set_data(sx, sy)
        if sx.size:
            self._reward_dot.set_data(sx[-1:], sy[-1:])
            self._fill.set_xy(
                np.concatenate([[[sx[0], 0.0]], np.column_stack([sx, sy]), [[sx[-1], 0.0]]])
            )
        else:
            self._reward_dot.set_data([], [])
            self._fill.set_xy(np.zeros((1, 2)))

        # Header.
        mv = self._metric_values
        mv["step"].set_text(f"{step} / {max_iter}")
        mv["reward"].set_text(f"{reward:.2f}")
        mv["return"].set_text(f"{episode_return:.2f}")
        mv["holding"].set_text(f"{int(has_msg.sum())} / {n_agents}")
