"""Matplotlib dashboard for :class:`env_lib.wireless_comm_env.WirelessCommEnv`.

Left: the agent grid. Every agent is a circle coloured by the outcome of its
last action (idle, success, collision, lost) with a mini queue bar below it
(one cell per deadline slot, the left cell expires first; filled cells are
packets, darker = more urgent). Access points are squares at the interior
grid corners, coloured by their load (idle, one transmitter, collision), and
a line joins every transmitting agent to the access point it chose, coloured
by the outcome.

Right: packets delivered per step with a rolling mean (plus the rolling
collision count) and the cumulative return of the episode.
"""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
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
from env_lib.wireless_comm_env.wireless_comm_env import ACTION_OFFSETS, OUTCOME_COLLISION

__all__ = ["WirelessCommRenderer"]

_ROLLING = 10


def _nice_ceiling(value: float, minimum: float) -> float:
    """Smallest "round" number (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8 times 10**k) >= ``value``."""
    value = max(float(value), minimum)
    exponent = np.floor(np.log10(value))
    for mantissa in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        candidate = mantissa * 10.0**exponent
        if candidate >= value - 1e-9:
            return float(candidate)
    return float(10.0 ** (exponent + 1))  # pragma: no cover


def _rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    csum = np.cumsum(np.concatenate([[0.0], values]))
    idx = np.arange(1, values.size + 1)
    lo = np.maximum(idx - width, 0)
    return (csum[idx] - csum[lo]) / (idx - lo)


class WirelessCommRenderer(MatplotlibRenderer):
    """Persistent-artist renderer for the wireless multiple-access environment.

    Parameters
    ----------
    render_mode : {"human", "rgb_array"}
        See :class:`env_lib.utils.rendering.FigureWindow`.
    grid_x, grid_y : int
        Number of agent rows and columns.
    ddl : int
        Number of queue slots per agent.
    window : int, default 51
        Number of time steps shown by the right-hand panels (normally
        ``max_iter + 1``).
    packet_arrival_probability, success_transmission_probability : float, optional
        Shown in the header when given.
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
        grid_x: int,
        grid_y: int,
        ddl: int,
        window: int = 51,
        packet_arrival_probability: float | None = None,
        success_transmission_probability: float | None = None,
        fps: float | None = 10.0,
        theme: str | Theme | None = None,
        figsize: tuple[float, float] = (10.0, 5.6),
        dpi: int = 100,
    ):
        super().__init__(
            render_mode, figsize=figsize, dpi=dpi, fps=fps, title="WirelessComm", theme=theme
        )
        self.grid_x = int(grid_x)
        self.grid_y = int(grid_y)
        self.ddl = int(ddl)
        self.window = max(2, int(window))
        self.p = packet_arrival_probability
        self.q = success_transmission_probability
        self.n_agents = self.grid_x * self.grid_y
        rows, cols = np.divmod(np.arange(self.n_agents), self.grid_y)
        self._agent_xy = np.column_stack([cols, rows]).astype(float)
        # End point of the transmission line for every (action, agent).
        self._target_xy = np.stack(
            [np.column_stack([cols + dc + 0.5, rows + dr + 0.5]) for dr, dc in ACTION_OFFSETS]
        ).astype(float)
        self._first: int | None = None
        self._ylims = (0.0, 0.0)

    # ------------------------------------------------------------------
    def _build(self, fig: Figure) -> None:
        th = self.theme
        gx, gy, ddl = self.grid_x, self.grid_y, self.ddl
        n_ap = (gx - 1) * (gy - 1)

        # Header ------------------------------------------------------------
        fig.text(0.03, 0.945, "WirelessComm", color=th.text, fontsize=15, fontweight="bold")
        subtitle = f"{gx} x {gy} agents, {n_ap} access points, deadline {ddl}"
        if self.p is not None and self.q is not None:
            subtitle += f", p = {self.p:.2f}, q = {self.q:.2f}"
        fig.text(0.03, 0.905, subtitle, color=th.muted, fontsize=9)
        self._metric_values = {}
        for k, label in enumerate(("step", "delivered", "return", "collisions")):
            x = 0.6 + 0.1 * k
            fig.text(x, 0.955, label.upper(), color=th.muted, fontsize=7.5)
            self._metric_values[label] = fig.text(
                x, 0.905, "", color=th.text, fontsize=12.5, fontweight="bold"
            )

        # Agent grid ----------------------------------------------------------
        ax = fig.add_axes((0.02, 0.03, 0.54, 0.83))
        style_axes(ax, th, grid=False)
        ax.set_xlim(-0.8, gy - 0.2)
        ax.set_ylim(gx - 0.2, -0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        fig_w, fig_h = fig.get_size_inches() * fig.dpi
        pos = ax.get_position()
        cell_px = min(pos.width * fig_w / (gy + 0.6), pos.height * fig_h / (gx + 0.6))
        pt = cell_px * 72.0 / fig.dpi  # points per grid cell

        lattice = [((0, i), (gy - 1, i)) for i in range(gx)]
        lattice += [((j, 0), (j, gx - 1)) for j in range(gy)]
        ax.add_collection(
            LineCollection(lattice, colors=[with_alpha(th.grid, 0.9)], linewidths=0.8, zorder=0)
        )

        self._tx_glow = LineCollection([], linewidths=max(2.0, 0.09 * pt), zorder=1)
        self._tx_lines = LineCollection([], linewidths=max(1.0, 0.035 * pt), zorder=1.5)
        ax.add_collection(self._tx_glow)
        ax.add_collection(self._tx_lines)

        ap_rows, ap_cols = np.divmod(np.arange(n_ap), gy - 1)
        self._ap_base_size = (0.2 * pt) ** 2
        self._aps = ax.scatter(
            ap_cols + 0.5,
            ap_rows + 0.5,
            s=self._ap_base_size,
            marker="s",
            c=[th.grid] * n_ap,
            edgecolors=th.muted,
            linewidths=0.8,
            zorder=2,
        )
        self._agents = ax.scatter(
            self._agent_xy[:, 0],
            self._agent_xy[:, 1],
            s=(0.4 * pt) ** 2,
            c=[th.panel] * self.n_agents,
            edgecolors=[th.muted] * self.n_agents,
            linewidths=1.2,
            zorder=3,
        )

        slot_w = min(0.44 / ddl, 0.15)
        offsets = (np.arange(ddl) - (ddl - 1) / 2.0) * slot_w
        slot_x = (self._agent_xy[:, 0][:, None] + offsets[None, :]).ravel()
        slot_y = np.repeat(self._agent_xy[:, 1] + 0.34, ddl)
        self._slots = ax.scatter(
            slot_x,
            slot_y,
            s=(0.86 * slot_w * pt) ** 2,
            marker="s",
            c=[th.grid] * (self.n_agents * ddl),
            linewidths=0,
            zorder=3,
        )
        self._ax_grid = ax

        # Colour tables.
        self._outcome_rgba = np.array(
            [
                to_rgba(th.panel),  # idle (hollow)
                to_rgba(th.positive),
                to_rgba(th.negative),
                to_rgba(th.warning),
            ]
        )
        self._outcome_edge = np.array(
            [to_rgba(th.muted), to_rgba(th.panel), to_rgba(th.panel), to_rgba(th.panel)]
        )
        self._ap_rgba = np.array([to_rgba(th.grid), to_rgba(th.positive), to_rgba(th.negative)])
        # Queue urgency: one hue, most urgent slot (index 0) fully saturated.
        panel = np.array(to_rgba(th.panel))
        accent = np.array(to_rgba(th.accent))
        mix = np.linspace(1.0, 0.45, ddl) if ddl > 1 else np.ones(1)
        self._slot_rgba = mix[:, None] * accent + (1.0 - mix[:, None]) * panel
        self._slot_rgba[:, 3] = 1.0
        self._slot_empty = np.array(to_rgba(th.grid))

        # Legend -------------------------------------------------------------
        ax_l = fig.add_axes((0.6, 0.75, 0.39, 0.12))
        ax_l.set_axis_off()
        handles = [
            Line2D([], [], ls="", marker="o", ms=8, mfc=th.positive, mec=th.panel, label="success"),
            Line2D(
                [], [], ls="", marker="o", ms=8, mfc=th.negative, mec=th.panel, label="collision"
            ),
            Line2D([], [], ls="", marker="o", ms=8, mfc=th.warning, mec=th.panel, label="lost"),
            Line2D([], [], ls="", marker="o", ms=8, mfc=th.panel, mec=th.muted, label="idle"),
            Line2D(
                [], [], ls="", marker="s", ms=7, mfc=th.grid, mec=th.muted, label="access point"
            ),
            Line2D([], [], ls="", marker="s", ms=6, mfc=th.accent, mec=th.accent, label="packet"),
        ]
        ax_l.legend(
            handles=handles,
            loc="center left",
            ncol=3,
            frameon=False,
            fontsize=8.5,
            labelcolor=th.text,
            handletextpad=0.3,
            columnspacing=1.2,
            borderaxespad=0.0,
        )

        # Throughput -------------------------------------------------------
        ax_t = fig.add_axes((0.625, 0.44, 0.355, 0.25))
        style_axes(ax_t, th)
        ax_t.set_title("Packets per step", color=th.text, fontsize=10, loc="left", pad=5)
        ax_t.tick_params(labelbottom=False)
        (self._raw,) = ax_t.plot(
            [], [], color=with_alpha(th.positive, 0.45), linewidth=1.0, drawstyle="steps-mid"
        )
        (self._mean_glow,) = ax_t.plot([], [], color=with_alpha(th.positive, 0.22), linewidth=5)
        (self._mean,) = ax_t.plot(
            [], [], color=th.positive, linewidth=2.0, label=f"delivered ({_ROLLING}-step mean)"
        )
        (self._coll,) = ax_t.plot(
            [], [], color=th.negative, linewidth=1.5, label=f"collisions ({_ROLLING}-step mean)"
        )
        ax_t.legend(
            loc="upper left",
            frameon=False,
            fontsize=7.5,
            labelcolor=th.text,
            ncol=2,
            handlelength=1.4,
            borderaxespad=0.2,
        )
        ax_t.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax_t.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        self._ax_tput = ax_t

        # Cumulative return ---------------------------------------------------
        ax_r = fig.add_axes((0.625, 0.08, 0.355, 0.25))
        style_axes(ax_r, th, xlabel="step")
        ax_r.set_title("Cumulative return", color=th.text, fontsize=10, loc="left", pad=5)
        self._fill = Polygon(
            np.zeros((1, 2)), closed=True, facecolor=with_alpha(th.accent, 0.16), edgecolor="none"
        )
        ax_r.add_patch(self._fill)
        (self._ret,) = ax_r.plot([], [], color=th.accent, linewidth=2.0)
        (self._ret_dot,) = ax_r.plot([], [], "o", color=th.accent, ms=5, mec=th.panel, mew=1.2)
        ax_r.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax_r.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        self._ax_ret = ax_r
        self._first = None
        self._ylims = (0.0, 0.0)

        self._dynamic(
            self._tx_glow,
            self._tx_lines,
            self._aps,
            self._agents,
            self._slots,
            self._raw,
            self._mean_glow,
            self._mean,
            self._coll,
            self._fill,
            self._ret,
            self._ret_dot,
            *self._metric_values.values(),
        )

    def reset(self) -> None:
        """Let the y-limits shrink again for the new episode."""
        self._ylims = (0.0, 0.0)

    def _update_ylims(
        self, sx: np.ndarray, delivered: np.ndarray, collisions: np.ndarray, returns: np.ndarray
    ) -> None:
        """Grow the y-limits in large steps (every change forces a full redraw)."""
        top_t, top_r = self._ylims
        peak_t = max(delivered.max(), collisions.max()) if sx.size else 0.0
        peak_r = returns.max() if sx.size else 0.0
        if top_t == 0.0 or peak_t > top_t:
            top_t = _nice_ceiling(1.5 * peak_t, 4.0)
        if top_r == 0.0 or peak_r > top_r:
            # Project the return at the right edge of the window from the mean rate.
            projected = peak_r / sx[-1] * (self._first + self.window - 1) if sx.size else 0.0
            top_r = _nice_ceiling(max(1.1 * projected, 1.25 * peak_r), 10.0)
        if (top_t, top_r) != self._ylims:
            self._ylims = (top_t, top_r)
            self._ax_tput.set_ylim(0.0, top_t)
            self._ax_ret.set_ylim(0.0, top_r)
            self._invalidate()

    # ------------------------------------------------------------------
    def _update(
        self,
        *,
        queues: np.ndarray,
        actions: np.ndarray,
        outcomes: np.ndarray,
        ap_load: np.ndarray,
        history: np.ndarray,
        first_step: int,
        step: int,
        max_iter: int,
        reward: float,
        episode_return: float,
    ) -> None:
        n_agents, ddl = self.n_agents, self.ddl
        actions = np.asarray(actions).reshape(-1)
        outcomes = np.asarray(outcomes).reshape(-1)

        # Agents, queues and access points.
        self._agents.set_facecolors(self._outcome_rgba[outcomes])
        self._agents.set_edgecolors(self._outcome_edge[outcomes])
        packets = np.asarray(queues).reshape(ddl, n_agents).T > 0.5  # (n_agents, ddl)
        slot_colors = np.where(
            packets[..., None], self._slot_rgba[None, :, :], self._slot_empty[None, None, :]
        )
        self._slots.set_facecolors(slot_colors.reshape(-1, 4))
        load = np.asarray(ap_load).reshape(-1)
        self._aps.set_facecolors(self._ap_rgba[np.minimum(load, 2)])
        self._aps.set_sizes(self._ap_base_size * (1.0 + 0.35 * np.minimum(load, 4)))

        tx = np.flatnonzero(actions != 0)
        if tx.size:
            start = self._agent_xy[tx]
            end = self._target_xy[actions[tx], tx]
            segs = np.stack([start, end], axis=1)
            colors = self._outcome_rgba[outcomes[tx]]
            glow = colors.copy()
            glow[:, 3] = 0.22
            self._tx_lines.set_segments(segs)
            self._tx_lines.set_color(colors)
            self._tx_glow.set_segments(segs)
            self._tx_glow.set_color(glow)
        else:
            self._tx_lines.set_segments([])
            self._tx_glow.set_segments([])

        # Time series.
        if first_step != self._first:
            self._first = first_step
            xlim = (first_step, first_step + self.window - 1)
            self._ax_tput.set_xlim(*xlim)
            self._ax_ret.set_xlim(*xlim)
            self._invalidate()
        steps = first_step + np.arange(history.shape[0])
        keep = steps >= 1
        sx = steps[keep].astype(float)
        delivered = history[keep, 1]
        collisions = history[keep, 2]
        csum = np.cumsum(history[:, 0])
        returns = (episode_return - (csum[-1] - csum))[keep]
        mean = _rolling_mean(delivered, _ROLLING)
        coll = _rolling_mean(collisions, _ROLLING)
        self._raw.set_data(sx, delivered)
        self._mean.set_data(sx, mean)
        self._mean_glow.set_data(sx, mean)
        self._coll.set_data(sx, coll)
        self._ret.set_data(sx, returns)
        if sx.size:
            self._ret_dot.set_data(sx[-1:], returns[-1:])
            self._fill.set_xy(
                np.concatenate([[[sx[0], 0.0]], np.column_stack([sx, returns]), [[sx[-1], 0.0]]])
            )
        else:
            self._ret_dot.set_data([], [])
            self._fill.set_xy(np.zeros((1, 2)))
        self._update_ylims(sx, delivered, collisions, returns)

        # Header.
        mv = self._metric_values
        mv["step"].set_text(f"{step} / {max_iter}")
        mv["delivered"].set_text(f"{reward:.0f}")
        mv["return"].set_text(f"{episode_return:.0f}")
        mv["collisions"].set_text(f"{int(np.sum(outcomes == OUTCOME_COLLISION))}")
