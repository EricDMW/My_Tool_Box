"""Matplotlib renderer for :class:`env_lib.power_grid_env.power_grid_env.PowerGridEnv`.

The frame is a dashboard (1000 x 560 px by default):

* left -- the transmission network: buses coloured by their frequency
  deviation (diverging colour map with a colour bar), a ring around every bus
  whose size shows its control effort ``|u_i| / u_max_i`` (green: injecting,
  violet: absorbing), lines coloured and weighted by their loading
  ``|sin(theta_i - theta_j)|`` and triangles at buses with an active step
  load change (down: load increase, up: load decrease);
* right, top -- the frequency deviation of every bus over time (thin traces),
  the centre-of-inertia frequency (bold), the frequency nadir and, when in
  range, the trip limits over a shaded band;
* right, bottom -- the power balance: net control injection ``sum_i u_i``
  and the load imbalance ``-sum_i Delta P_i`` it has to cover;
* header -- system size, step, time, largest deviation, nadir and a status
  badge (STABLE, ALERT above half the limit, TRIPPED).

All artists are created once in :meth:`PowerGridRenderer._build`; each frame
only updates their data. In ``"rgb_array"`` mode the static layer (axes, grid,
colour bar, key, limit lines) is cached and only the animated artists are
redrawn (blitting). Axis ranges and the colour scale only grow within an
episode, so the cache is rebuilt a few times per episode at most.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib import colormaps
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.textpath import TextPath
from matplotlib.ticker import FixedLocator, MaxNLocator
from matplotlib.transforms import ScaledTranslation

from env_lib.utils.rendering import MatplotlibRenderer, Theme, style_axes, with_alpha

__all__ = ["PowerGridRenderer"]

_TWO_PI = 2.0 * math.pi
_KEY_FONTSIZE = 7.5
_NICE = (1.0, 2.0, 2.5, 5.0, 10.0)
_MIN_FREQ_SCALE = 0.01  # Hz
_MIN_POWER_SCALE = 0.05  # pu
_NADIR_MIN = 5e-4  # Hz; shallower nadirs are shown as zero


class PowerGridRenderer(MatplotlibRenderer):
    """Dashboard renderer for the power grid frequency-control environment.

    Parameters
    ----------
    render_mode:
        ``"human"`` or ``"rgb_array"``.
    n_buses:
        Number of buses.
    max_steps, dt:
        Episode length and control interval (time axis of the right panels).
    frequency_limit:
        Trip limit in Hz (dashed lines, status badge).
    topology, nominal_frequency:
        Shown in the sub-header.
    figsize, dpi, fps, theme:
        See :class:`env_lib.utils.rendering.MatplotlibRenderer`.
    """

    def __init__(
        self,
        render_mode: str,
        *,
        n_buses: int,
        max_steps: int,
        dt: float,
        frequency_limit: float,
        topology: str = "small_world",
        nominal_frequency: float = 50.0,
        figsize: tuple[float, float] = (10.0, 5.6),
        dpi: int = 100,
        fps: float | None = 20.0,
        theme: str | Theme | None = None,
    ):
        super().__init__(
            render_mode, figsize=figsize, dpi=dpi, fps=fps, title="Power grid", theme=theme
        )
        self.n_buses = int(n_buses)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.frequency_limit = float(frequency_limit)
        self.topology = topology
        self.nominal_frequency = float(nominal_frequency)

        n = self.n_buses
        self._times = np.arange(self.max_steps + 1) * self.dt
        self._freq_hist = np.full((self.max_steps + 1, n), np.nan)
        self._coi_hist = np.full(self.max_steps + 1, np.nan)
        self._control_hist = np.full(self.max_steps + 1, np.nan)
        self._imbalance_hist = np.full(self.max_steps + 1, np.nan)
        self._marker_size = float(np.clip(2400.0 / n, 14.0, 130.0))
        self._freq_cmap = colormaps[self.theme.diverging_cmap]
        self._loading_cmap = mcolors.LinearSegmentedColormap.from_list(
            "power_grid_loading",
            [
                (0.0, self.theme.muted),
                (0.5, _blend(self.theme.muted, self.theme.warning, 0.5)),
                (0.8, self.theme.warning),
                (1.0, self.theme.negative),
            ],
        )
        self._inject_rgba = np.array(with_alpha(self.theme.positive, 1.0))
        self._absorb_rgba = np.array(with_alpha(self.theme.color(4), 1.0))
        self._reset_state()

    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
        self._freq_hist.fill(np.nan)
        self._coi_hist.fill(np.nan)
        self._control_hist.fill(np.nan)
        self._imbalance_hist.fill(np.nan)
        self._last_step = -1
        self._freq_scale = 0.0
        self._power_scale = 0.0
        self._nadir_point: tuple[float, float] | None = None
        self._network_key: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self) -> None:
        """Clear the time histories and scales (called on ``env.reset()``)."""
        self._reset_state()

    # ------------------------------------------------------------------
    def _build(self, fig: Figure) -> None:
        self._build_header(fig)
        self._build_network(fig)
        self._build_frequency_panel(fig)
        self._build_power_panel(fig)
        self._network_key = None
        self._freq_scale = 0.0
        self._power_scale = 0.0
        self._dynamic(
            self._lines,
            self._rings,
            self._buses,
            self._increase,
            self._decrease,
            self._traces,
            self._coi_glow,
            self._coi_line,
            self._nadir_line,
            self._nadir_dot,
            self._nadir_text,
            self._control_line,
            self._imbalance_line,
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
            f"topology={self.topology}  |  f0={self.nominal_frequency:g} Hz  |  "
            f"trip limit +/-{self.frequency_limit:g} Hz  |  dt={self.dt:g} s",
            color=theme.muted,
            fontsize=8.5,
            va="center",
        )
        self._badge = fig.text(
            0.975,
            0.955,
            "STABLE",
            color=theme.positive,
            fontsize=8.5,
            fontweight="bold",
            ha="right",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=with_alpha(theme.positive, 0.14),
                edgecolor=with_alpha(theme.positive, 0.7),
                linewidth=0.8,
            ),
        )

    def _build_network(self, fig: Figure) -> None:
        theme = self.theme
        n = self.n_buses
        ax = fig.add_axes((0.015, 0.035, 0.44, 0.845))
        style_axes(ax, theme, grid=False)
        ax.set_xlim(-1.18, 1.18)
        ax.set_ylim(-1.14, 1.22)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        self._ax_net = ax

        self._lines = LineCollection([], zorder=1, capstyle="round")
        ax.add_collection(self._lines)
        empty = np.zeros((n, 2))
        size = self._marker_size
        self._rings = ax.scatter(
            empty[:, 0],
            empty[:, 1],
            s=np.full(n, size),
            facecolors="none",
            edgecolors=[with_alpha(theme.positive, 0.0)] * n,
            linewidths=1.6,
            zorder=2,
        )
        self._freq_norm = mcolors.Normalize(vmin=-_MIN_FREQ_SCALE, vmax=_MIN_FREQ_SCALE)
        self._buses = ax.scatter(
            empty[:, 0],
            empty[:, 1],
            s=size,
            c=np.zeros(n),
            cmap=self._freq_cmap,
            norm=self._freq_norm,
            edgecolors=theme.text,
            linewidths=0.8,
            zorder=3,
        )
        # Load-step markers, placed just above the bus (offset in points).
        lift = (0.5 * math.sqrt(size) + 5.0) / 72.0  # bus radius plus 5 pt, in inches
        offset = ScaledTranslation(0.0, lift, fig.dpi_scale_trans)
        style = dict(
            color=theme.warning,
            edgecolors=theme.panel,
            linewidths=0.6,
            zorder=4,
            transform=ax.transData + offset,
        )
        self._increase = ax.scatter([], [], marker="v", s=36, **style)
        self._decrease = ax.scatter([], [], marker="^", s=36, **style)

        cax = fig.add_axes((0.452, 0.16, 0.009, 0.58))
        colorbar = fig.colorbar(ScalarMappable(norm=self._freq_norm, cmap=self._freq_cmap), cax=cax)
        colorbar.outline.set_edgecolor(theme.grid)
        cax.tick_params(colors=theme.muted, labelsize=7.5, length=2)
        colorbar.set_label("frequency deviation [Hz]", color=theme.muted, fontsize=8)
        self._colorbar = colorbar
        self._build_key(fig, ax)

    def _build_key(self, fig: Figure, ax) -> None:
        """Compact one-row key centred at the top of the network panel."""
        theme = self.theme
        entries: list[tuple[str, dict]] = [
            ("bus", dict(marker="o", markersize=6, markerfacecolor=self._freq_cmap(0.2),
                         markeredgecolor=theme.text, markeredgewidth=0.7, ls="none")),
            ("control", dict(marker="o", markersize=8, markerfacecolor="none",
                             markeredgecolor=theme.positive, markeredgewidth=1.4, ls="none")),
            ("load step", dict(marker="v", markersize=6, color=theme.warning, ls="none")),
            ("line loading", dict(color=theme.warning, lw=2.0)),
        ]  # fmt: skip
        _draw_key(fig, ax, theme, entries, anchor=(0.5, 0.985), align="center")

    def _build_frequency_panel(self, fig: Figure) -> None:
        theme = self.theme
        horizon = self.max_steps * self.dt
        ax = fig.add_axes((0.575, 0.435, 0.4, 0.43))
        style_axes(ax, theme)
        _fixed_title(ax, "Bus frequency deviation [Hz]", theme)
        ax.set_xlim(0.0, horizon)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(labelbottom=False)
        _fix_label_positions(ax)
        limit = self.frequency_limit
        big = 1e3 * max(limit, 1.0)
        for sign in (1.0, -1.0):
            ax.axhspan(
                sign * limit, sign * big, color=with_alpha(theme.negative, 0.08), lw=0, zorder=0
            )
            ax.axhline(sign * limit, color=theme.negative, lw=1.1, ls=(0, (4, 3)), zorder=1)
        ax.axhline(0.0, color=theme.muted, lw=0.8, alpha=0.6, zorder=1)
        ax.text(
            0.99,
            0.965,
            f"trip limit +/-{limit:g} Hz",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=theme.negative,
            fontsize=7.5,
            zorder=6,
        )
        self._traces = LineCollection(
            [], colors=[with_alpha(theme.accent, 0.4)], linewidths=0.9, zorder=2
        )
        ax.add_collection(self._traces)
        (self._coi_glow,) = ax.plot([], [], color=theme.text, lw=5.0, alpha=0.12, zorder=3)
        (self._coi_line,) = ax.plot([], [], color=theme.text, lw=1.8, zorder=3.1)
        self._nadir_line = ax.axhline(
            0.0, color=theme.warning, lw=1.0, ls=(0, (1.5, 2.0)), zorder=2.5
        )
        (self._nadir_dot,) = ax.plot(
            [], [], ls="none", marker="o", markersize=5.5, color=theme.warning,
            markeredgecolor=theme.panel, markeredgewidth=1.0, zorder=4,
        )  # fmt: skip
        self._nadir_text = ax.text(
            0.99, 0.0, "", transform=ax.get_yaxis_transform(), ha="right", va="top",
            color=theme.warning, fontsize=7.5, zorder=4,
        )  # fmt: skip
        entries = [
            ("centre of inertia", dict(color=theme.text, lw=1.8)),
            ("buses", dict(color=theme.accent, lw=1.0, alpha=0.8)),
        ]
        _draw_key(fig, ax, theme, entries, anchor=(0.012, 0.975), align="left")
        self._ax_freq = ax

    def _build_power_panel(self, fig: Figure) -> None:
        theme = self.theme
        horizon = self.max_steps * self.dt
        ax = fig.add_axes((0.575, 0.09, 0.4, 0.235))
        style_axes(ax, theme, xlabel="time [s]")
        _fixed_title(ax, "Power balance [pu]", theme)
        ax.set_xlim(0.0, horizon)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        _fix_label_positions(ax, xlabel_y=-0.28)
        ax.axhline(0.0, color=theme.muted, lw=0.8, alpha=0.6, zorder=1)
        (self._imbalance_line,) = ax.plot(
            [], [], color=theme.warning, lw=1.5, ls=(0, (4, 2)), drawstyle="steps-post", zorder=2
        )
        (self._control_line,) = ax.plot([], [], color=theme.positive, lw=1.8, zorder=3)
        entries = [
            ("control sum u", dict(color=theme.positive, lw=1.8)),
            ("imbalance -sum dP", dict(color=theme.warning, lw=1.5, ls=(0, (4, 2)))),
        ]
        _draw_key(fig, ax, theme, entries, anchor=(0.012, 0.955), align="left")
        self._ax_power = ax

    # ------------------------------------------------------------------
    def _update(
        self,
        *,
        positions: np.ndarray,
        edges: np.ndarray,
        line_loading: np.ndarray,
        omega: np.ndarray,
        action: np.ndarray,
        u_max: np.ndarray,
        step_disturbance: np.ndarray,
        total_disturbance: float,
        coi_omega: float,
        step: int,
        nadir: float,
        tripped: bool,
    ) -> None:
        step = int(step)
        if step < self._last_step:
            self._reset_state()
        self._last_step = step
        freq = np.asarray(omega, dtype=np.float64) / _TWO_PI
        control = np.asarray(action, dtype=np.float64)
        if 0 <= step <= self.max_steps:
            self._freq_hist[step] = freq
            self._coi_hist[step] = coi_omega / _TWO_PI
            self._control_hist[step] = control.sum()
            self._imbalance_hist[step] = -total_disturbance

        self._update_network(positions, edges, line_loading, freq, control, u_max,
                             step_disturbance)  # fmt: skip
        self._update_frequency(step, float(nadir) / _TWO_PI)
        self._update_power()
        self._update_header(step, freq, float(nadir) / _TWO_PI, bool(tripped))

    def _update_network(
        self,
        positions: np.ndarray,
        edges: np.ndarray,
        loading: np.ndarray,
        freq: np.ndarray,
        control: np.ndarray,
        u_max: np.ndarray,
        steps: np.ndarray,
    ) -> None:
        pos = np.asarray(positions, dtype=np.float64)
        key = self._network_key
        if key is None or not (np.array_equal(key[0], pos) and np.array_equal(key[1], edges)):
            self._lines.set_segments(np.stack([pos[edges[:, 0]], pos[edges[:, 1]]], axis=1))
            self._buses.set_offsets(pos)
            self._rings.set_offsets(pos)
            self._network_key = (pos.copy(), np.array(edges, copy=True))
        loading = np.clip(np.asarray(loading, dtype=np.float64), 0.0, 1.0)
        colors = self._loading_cmap(loading)
        colors[:, 3] = 0.45 + 0.55 * loading
        self._lines.set_color(colors)
        self._lines.set_linewidth(0.7 + 2.6 * loading)

        self._buses.set_array(freq)
        effort = np.clip(np.abs(control) / np.asarray(u_max, dtype=np.float64), 0.0, 1.0)
        ring = np.where((control >= 0.0)[:, None], self._inject_rgba, self._absorb_rgba)
        ring[:, 3] = np.where(effort > 1e-3, 0.35 + 0.65 * effort, 0.0)
        self._rings.set_edgecolors(ring)
        self._rings.set_sizes(self._marker_size * (1.5 + 3.5 * effort) ** 2 / 2.0)

        active = np.abs(steps) > 1e-12
        increase = active & (steps < 0.0)
        decrease = active & (steps > 0.0)
        magnitude = 28.0 + 260.0 * np.abs(steps)
        self._increase.set_offsets(pos[increase] if increase.any() else np.empty((0, 2)))
        self._increase.set_sizes(magnitude[increase])
        self._decrease.set_offsets(pos[decrease] if decrease.any() else np.empty((0, 2)))
        self._decrease.set_sizes(magnitude[decrease])

    def _update_frequency(self, step: int, nadir: float) -> None:
        valid = ~np.isnan(self._coi_hist)
        times = self._times[valid]
        history = self._freq_hist[valid]  # (T, n)
        if times.size:
            segments = np.empty((self.n_buses, times.size, 2))
            segments[:, :, 0] = times
            segments[:, :, 1] = history.T
            self._traces.set_segments(segments)
        else:
            self._traces.set_segments([])
        coi = self._coi_hist[valid]
        self._coi_line.set_data(times, coi)
        self._coi_glow.set_data(times, coi)

        # Nadir: lowest bus frequency so far, marked where it was reached.
        if times.size and np.isfinite(history).any():
            flat = int(np.nanargmin(history))
            row, _ = divmod(flat, self.n_buses)
            point = (float(times[row]), float(history.flat[flat]))
            if self._nadir_point is None or point[1] < self._nadir_point[1]:
                self._nadir_point = point
        low = nadir if nadir < -_NADIR_MIN else 0.0
        self._nadir_line.set_ydata([low, low])
        if self._nadir_point is not None and self._nadir_point[1] < -_NADIR_MIN:
            self._nadir_dot.set_data([self._nadir_point[0]], [self._nadir_point[1]])
            self._nadir_text.set_text(f"nadir {low:.3f} Hz")
            self._nadir_text.set_y(low)
            self._nadir_line.set_visible(True)
        else:
            self._nadir_dot.set_data([], [])
            self._nadir_text.set_text("")
            self._nadir_line.set_visible(False)

        peak = float(np.nanmax(np.abs(history))) if times.size else 0.0
        scale = _nice_ceil(max(1.15 * peak, 1.2 * abs(low), _MIN_FREQ_SCALE))
        if scale > self._freq_scale:
            self._freq_scale = scale
            self._ax_freq.set_ylim(-scale, scale)
            self._ax_freq.yaxis.set_major_locator(FixedLocator(np.linspace(-scale, scale, 5)))
            self._freq_norm.vmin, self._freq_norm.vmax = -scale, scale
            self._colorbar.set_ticks(np.linspace(-scale, scale, 5))
            self._buses.set_norm(self._freq_norm)
            self._invalidate()

    def _update_power(self) -> None:
        valid = ~np.isnan(self._control_hist)
        times = self._times[valid]
        control = self._control_hist[valid]
        imbalance = self._imbalance_hist[valid]
        self._control_line.set_data(times, control)
        self._imbalance_line.set_data(times, imbalance)
        if times.size:
            peak = max(float(np.abs(control).max()), float(np.abs(imbalance).max()))
            scale = _nice_ceil(max(1.2 * peak, _MIN_POWER_SCALE))
            if scale > self._power_scale:
                self._power_scale = scale
                self._ax_power.set_ylim(-scale, scale)
                self._ax_power.yaxis.set_major_locator(FixedLocator([-scale, 0.0, scale]))
                self._invalidate()

    def _update_header(self, step: int, freq: np.ndarray, nadir: float, tripped: bool) -> None:
        theme = self.theme
        peak = float(np.abs(freq).max())
        self._header.set_text(
            f"Power grid | N={self.n_buses} | step {step}/{self.max_steps} | "
            f"t={step * self.dt:.2f} s | max dev {peak:.3f} Hz | "
            f"nadir {nadir if nadir < -_NADIR_MIN else 0.0:.3f} Hz"
        )
        if tripped:
            label, color = "TRIPPED", theme.negative
        elif peak > 0.5 * self.frequency_limit:
            label, color = "ALERT", theme.warning
        else:
            label, color = "STABLE", theme.positive
        self._badge.set_text(label)
        self._badge.set_color(color)
        patch = self._badge.get_bbox_patch()
        patch.set_facecolor(with_alpha(color, 0.14))
        patch.set_edgecolor(with_alpha(color, 0.7))


def _draw_key(
    fig: Figure,
    ax,
    theme: Theme,
    entries: list[tuple[str, dict]],
    *,
    anchor: tuple[float, float],
    align: str,
) -> None:
    """One-row key on a rounded panel (cheaper than a Legend, laid out from text widths).

    ``anchor`` is the top edge (centre or left end, see ``align``) in axes coordinates.
    """
    handle_w, gap, spacing, pad, height = 0.16, 0.05, 0.16, 0.08, 0.2
    text_w = [
        TextPath((0, 0), label, size=_KEY_FONTSIZE).get_extents().width / 72.0
        for label, _ in entries
    ]
    total = sum(handle_w + gap + w for w in text_w) + spacing * (len(entries) - 1)
    frame = fig.dpi_scale_trans + ScaledTranslation(anchor[0], anchor[1], ax.transAxes)
    left = -total / 2 if align == "center" else pad
    ax.add_patch(
        FancyBboxPatch(
            (left - pad, -height - 0.02),
            total + 2 * pad,
            height,
            boxstyle="round,pad=0,rounding_size=0.05",
            transform=frame,
            facecolor=with_alpha(theme.panel, 0.85),
            edgecolor=theme.grid,
            linewidth=0.8,
            zorder=6,
            clip_on=False,
        )
    )
    x, y = left, -0.02 - height / 2
    for (label, style), width in zip(entries, text_w):
        if style.get("ls") == "none":
            ax.plot([x + handle_w / 2], [y], transform=frame, zorder=7, clip_on=False, **style)
        else:
            ax.plot([x, x + handle_w], [y, y], transform=frame, zorder=7, clip_on=False, **style)
        ax.text(
            x + handle_w + gap,
            y,
            label,
            transform=frame,
            va="center_baseline",
            color=theme.text,
            fontsize=_KEY_FONTSIZE,
            zorder=7,
        )
        x += handle_w + gap + width + spacing


def _blend(first: str, second: str, weight: float) -> tuple[float, float, float]:
    a, b = np.array(mcolors.to_rgb(first)), np.array(mcolors.to_rgb(second))
    return tuple((1.0 - weight) * a + weight * b)


def _fixed_title(ax, title: str, theme: Theme) -> None:
    """Left-aligned title at a fixed position (skips matplotlib's per-draw title layout)."""
    ax.set_title(title, color=theme.text, fontsize=10, loc="left", pad=6, y=1.0)


def _fix_label_positions(ax, xlabel_y: float = -0.12, ylabel_x: float = -0.1) -> None:
    """Pin axis-label positions (skips the per-draw tick-label bounding-box search)."""
    ax.xaxis.set_label_coords(0.5, xlabel_y)
    ax.yaxis.set_label_coords(ylabel_x, 0.5)


def _nice_ceil(value: float) -> float:
    """Smallest number of the form {1, 2, 2.5, 5} * 10**k that is >= ``value``."""
    scale = 10.0 ** math.floor(math.log10(value))
    for factor in _NICE:
        if factor * scale >= value * (1 - 1e-12):
            return factor * scale
    return 10.0 * scale  # pragma: no cover - unreachable
