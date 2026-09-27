"""Matplotlib renderer for :class:`env_lib.platoon_env.platoon_env.PlatoonEnv`.

The frame is a dashboard (1000 x 560 px by default):

* top -- the road: a two-lane strip seen from above with a camera that follows
  the platoon (moving lane markings and distance posts). Vehicles are rounded
  rectangles to scale in length, the leader drawn in the text colour and
  labelled, the followers in a sequential colour ramp from the first to the
  last follower. Brake lights glow when a vehicle decelerates. The gap in
  front of every follower is shaded by its spacing error (red: too close,
  blue: too far) with a tick at the desired position, and the V2V links of
  the topology are drawn as arcs with moving data packets;
* bottom left -- the spacing errors of all followers over time (error
  propagation, i.e. string (in)stability, is visible as growing curves);
* bottom centre -- the speeds of the leader (with its reference profile) and
  of the followers;
* bottom right -- the peak spacing error of every follower so far, with the
  ratio of the last to the first follower;
* header -- team size, topology, step, time, leader speed, minimum gap and a
  status badge (CRUISING, COLLISION, BREAK-UP or COMPLETE).

All artists are created once in :meth:`PlatoonRenderer._build`; each frame
only updates their data. In ``"rgb_array"`` mode the static layer is cached and
only the animated artists are redrawn (blitting); the cache is rebuilt when an
axis range changes (a few times per episode).
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath
from matplotlib.ticker import MaxNLocator, NullLocator

from env_lib.utils.rendering import MatplotlibRenderer, Theme, style_axes, with_alpha

__all__ = ["PlatoonRenderer"]

_TOPOLOGY_LABELS = {
    "predecessor": ("PF", "predecessor following"),
    "predecessor_leader": ("PLF", "predecessor-leader following"),
    "bidirectional": ("BD", "bidirectional"),
    "none": ("no V2V", "no communication, sensors only"),
}
_ERROR_COLOR_RANGE = 2.0  # m, saturation of the gap shading
_BRAKE_THRESHOLD = 0.3  # m/s^2 of deceleration before the brake lights glow
_BRAKE_FULL = 3.0  # m/s^2 of deceleration for full brightness
_DASH, _DASH_PERIOD = 3.0, 12.0  # m, lane marking
_CAR_HEIGHT = 0.3  # lane units
_LANE_Y = 0.5  # centre line of the platoon lane
_ARC_BASE = _LANE_Y + 0.5 * _CAR_HEIGHT + 0.04
_ARC_POINTS = 17
_PACKET_SPEED = 0.07  # fraction of an arc per step
_CORNER_PX = 3.5
_KEY_FONTSIZE = 7.0


class PlatoonRenderer(MatplotlibRenderer):
    """Dashboard renderer for the vehicle platoon environment.

    Parameters
    ----------
    render_mode:
        ``"human"`` or ``"rgb_array"``.
    n_followers:
        Number of followers.
    topology, scenario:
        Environment settings shown in the header.
    max_steps, dt:
        Episode length and time step (x range of the time plots).
    headway, standstill_distance:
        Spacing policy (desired-position ticks and header).
    tau_range:
        Range of the actuator lags (header).
    adjacency, leader_links:
        V2V graph over the followers and links from the leader (see
        :func:`env_lib.platoon_env.platoon_adjacency`).
    figsize, dpi, fps, theme:
        See :class:`env_lib.utils.rendering.MatplotlibRenderer`.
    """

    def __init__(
        self,
        render_mode: str,
        *,
        n_followers: int,
        topology: str = "predecessor",
        scenario: str = "mixed",
        max_steps: int = 600,
        dt: float = 0.1,
        headway: float = 0.6,
        standstill_distance: float = 2.0,
        tau_range: tuple[float, float] = (0.2, 0.4),
        adjacency: np.ndarray | None = None,
        leader_links: np.ndarray | None = None,
        figsize: tuple[float, float] = (10.0, 5.6),
        dpi: int = 100,
        fps: float | None = 10.0,
        theme: str | Theme | None = None,
    ):
        super().__init__(
            render_mode, figsize=figsize, dpi=dpi, fps=fps, title="Platoon", theme=theme
        )
        self.n = int(n_followers)
        self.topology = topology
        self.scenario = scenario
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.headway = float(headway)
        self.standstill_distance = float(standstill_distance)
        self.tau_range = tuple(float(t) for t in tau_range)
        n = self.n
        if adjacency is None:
            adjacency = np.zeros((n, n), dtype=bool)
            adjacency[np.arange(1, n), np.arange(n - 1)] = True
        if leader_links is None:
            leader_links = np.zeros(n, dtype=bool)
            leader_links[0] = True
        self._links = self._link_list(np.asarray(adjacency, bool), np.asarray(leader_links, bool))

        # Colours: sequential ramp over the followers (lightness monotone), leader in text colour.
        cmap = colormaps[self.theme.sequential_cmap]
        low, high = (0.0, 0.78) if _is_light(self.theme) else (0.28, 1.0)
        ramp = np.linspace(low, high, n) if n > 1 else np.array([low])
        self._follower_rgba = cmap(ramp)
        self._leader_rgba = np.array(with_alpha(self.theme.text, 1.0))
        self._vehicle_rgba = np.vstack([self._leader_rgba, self._follower_rgba])
        diverging = colormaps[self.theme.diverging_cmap].reversed()  # negative error -> red
        self._gap_cmap = ListedColormap(diverging(np.linspace(0.0, 1.0, 256)))
        self._gap_norm = Normalize(-_ERROR_COLOR_RANGE, _ERROR_COLOR_RANGE)

        # Histories indexed by step.
        self._times = np.arange(self.max_steps + 1) * self.dt
        self._err_hist = np.full((self.max_steps + 1, n), np.nan)
        self._speed_hist = np.full((self.max_steps + 1, n + 1), np.nan)
        self._last_step = -1
        self._episode = None
        self._err_limit = 0.0
        self._peak_limit = 0.0
        self._speed_limit = 0.0
        self._view_width = 0.0
        self._lengths: np.ndarray | None = None
        self._templates: np.ndarray | None = None
        self._windows: np.ndarray | None = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the histories (called when the environment is reset)."""
        self._err_hist.fill(np.nan)
        self._speed_hist.fill(np.nan)
        self._last_step = -1
        self._err_limit = 0.0
        self._peak_limit = 0.0

    @staticmethod
    def _link_list(adjacency: np.ndarray, leader_links: np.ndarray) -> list[tuple[int, int, bool]]:
        """Directed links ``(sender, receiver, from_leader)`` in vehicle indices (leader = 0)."""
        links = [(0, k + 1, k > 0) for k in np.flatnonzero(leader_links)]
        receivers, senders = np.nonzero(adjacency)
        links += [(int(s) + 1, int(r) + 1, False) for r, s in zip(receivers, senders)]
        return links

    # ------------------------------------------------------------------
    def _build(self, fig: Figure) -> None:
        self._build_header(fig)
        self._build_road(fig)
        self._build_error_panel(fig)
        self._build_speed_panel(fig)
        self._build_peak_panel(fig)
        self._dynamic(
            self._dashes,
            self._posts,
            self._gaps,
            self._targets,
            self._arcs,
            self._packets,
            self._brake_glow,
            self._vehicles,
            self._cabins,
            self._brake_lights,
            self._lead_label,
            self._err_lines,
            self._err_dots,
            self._err_cursor,
            self._speed_lines,
            self._leader_line,
            self._speed_cursor,
            self._bars,
            self._ratio_text,
            self._header,
            self._badge,
        )

    def _build_header(self, fig: Figure) -> None:
        theme = self.theme
        short, long = _TOPOLOGY_LABELS.get(self.topology, (self.topology, self.topology))
        # Static prefix (cached with the background) and dynamic remainder: text is the
        # dominant drawing cost, so only the changing part is redrawn every frame.
        bold = FontProperties(size=12, weight="bold")
        prefix = fig.text(
            0.025, 0.955, f"Platoon | N={self.n} | {short}", color=theme.text,
            fontproperties=bold, va="center",
        )  # fmt: skip
        self._header = fig.text(
            0.025 + _text_width(fig, prefix),
            0.955,
            "",
            color=theme.text,
            fontproperties=bold,
            va="center",
        )
        low, high = self.tau_range
        fig.text(
            0.025,
            0.91,
            f"{long}  |  scenario={self.scenario}  |  headway h={self.headway:g} s  |  "
            f"standstill r={self.standstill_distance:g} m  |  lag tau in [{low:.2f}, {high:.2f}] s"
            f"  |  dt={self.dt:g} s",
            color=theme.muted,
            fontsize=8.5,
            va="center",
        )
        self._badge = fig.text(
            0.975,
            0.955,
            "CRUISING",
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

    def _build_road(self, fig: Figure) -> None:
        theme = self.theme
        n = self.n
        ax = fig.add_axes((0.025, 0.6, 0.95, 0.265))
        style_axes(ax, theme, grid=False)
        ax.set_facecolor(theme.panel)
        ax.set_ylim(-0.6, 2.5)
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        self._ax_road = ax
        # Road surface and edges (static; the camera moves the markings instead).
        surface = with_alpha(theme.grid, 0.55 if _is_light(theme) else 0.45)
        ax.axhspan(0.0, 2.0, color=surface, lw=0, zorder=0)
        for y in (0.0, 2.0):
            ax.axhline(y, color=theme.muted, lw=1.4, alpha=0.8, zorder=1)

        self._dashes = LineCollection([], colors=[with_alpha(theme.muted, 0.7)], linewidths=1.6)
        self._dashes.set_capstyle("butt")
        ax.add_collection(self._dashes)
        self._posts = LineCollection([], colors=[with_alpha(theme.muted, 0.8)], linewidths=1.0)
        ax.add_collection(self._posts)
        self._gaps = PolyCollection([], linewidths=0, zorder=2)
        ax.add_collection(self._gaps)
        self._targets = LineCollection([], colors=[with_alpha(theme.text, 0.55)], linewidths=1.1)
        self._targets.set_zorder(2.5)
        ax.add_collection(self._targets)
        self._arcs = LineCollection([], linewidths=1.0, zorder=3)
        ax.add_collection(self._arcs)
        self._packets = ax.scatter(
            np.zeros(max(len(self._links), 1)), np.zeros(max(len(self._links), 1)),
            s=9, c=[with_alpha(theme.accent, 0.95)], linewidths=0, zorder=3.5,
        )  # fmt: skip
        self._packets.set_visible(bool(self._links))
        self._arcs.set_visible(bool(self._links))
        empty = np.zeros((2 * (n + 1), 2))
        self._brake_glow = ax.scatter(
            empty[:, 0], empty[:, 1], s=34, c=[with_alpha(theme.negative, 0.0)],
            linewidths=0, zorder=3.8,
        )  # fmt: skip
        self._vehicles = PolyCollection(
            [], facecolors=self._vehicle_rgba, edgecolors=theme.panel, linewidths=0.9, zorder=4
        )
        ax.add_collection(self._vehicles)
        self._cabins = PolyCollection(
            [], facecolors=[with_alpha(theme.background, 0.55)], linewidths=0, zorder=4.2
        )
        ax.add_collection(self._cabins)
        self._brake_lights = ax.scatter(
            empty[:, 0], empty[:, 1], s=6, c=[with_alpha(theme.negative, 0.0)],
            linewidths=0, zorder=4.4,
        )  # fmt: skip
        self._lead_label = ax.text(
            0.0, _LANE_Y - 0.5 * _CAR_HEIGHT - 0.14, "LEAD", color=theme.text,
            fontsize=6.5, fontweight="bold", ha="center", va="center", zorder=5,
        )  # fmt: skip

        # Spacing-error colour key (static).
        cax = fig.add_axes((0.815, 0.842, 0.14, 0.012))
        bar = fig.colorbar(
            ScalarMappable(norm=self._gap_norm, cmap=self._gap_cmap),
            cax=cax,
            orientation="horizontal",
        )
        bar.outline.set_visible(False)
        bar.set_ticks([-_ERROR_COLOR_RANGE, 0.0, _ERROR_COLOR_RANGE])
        cax.tick_params(labelsize=6.5, colors=theme.muted, length=2, pad=1)
        cax.set_title("spacing error [m]", fontsize=7, color=theme.muted, pad=2)
        self._build_road_key(ax)

    def _build_road_key(self, ax) -> None:
        """Static key of the road view (top left)."""
        theme = self.theme
        x0, y = 0.012, 0.9
        entries = [
            ("leader", dict(color=theme.text)),
            ("followers 1..n", dict(color=self._follower_rgba[len(self._follower_rgba) // 2])),
            ("V2V link", dict(color=theme.accent)),
            ("desired position", dict(color=theme.text, alpha=0.55)),
        ]
        if not self._links:
            entries.pop(2)
        for label, style in entries:
            if label == "desired position":
                ax.plot([x0 + 0.006, x0 + 0.006], [y - 0.045, y + 0.045], transform=ax.transAxes,
                        lw=1.1, **style)  # fmt: skip
            elif label == "V2V link":
                ax.plot([x0, x0 + 0.014], [y, y], transform=ax.transAxes, lw=1.0, **style)
            else:
                ax.add_patch(Rectangle((x0, y - 0.035), 0.014, 0.07, transform=ax.transAxes,
                                       lw=0, zorder=6, **style))  # fmt: skip
            ax.text(x0 + 0.02, y, label, transform=ax.transAxes, va="center", color=theme.muted,
                    fontsize=_KEY_FONTSIZE)  # fmt: skip
            x0 += 0.03 + 0.0052 * len(label)

    def _build_error_panel(self, fig: Figure) -> None:
        theme = self.theme
        ax = fig.add_axes((0.06, 0.1, 0.415, 0.39))
        style_axes(ax, theme, xlabel="time [s]")
        _fixed_title(ax, "Spacing error $e_i$ [m]", theme)
        ax.set_xlim(0.0, self.max_steps * self.dt)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(5, symmetric=True))
        ax.axhline(0.0, color=theme.muted, lw=0.9, alpha=0.8, zorder=1)
        _fix_label_positions(ax, xlabel_y=-0.13)
        self._err_lines = LineCollection([], colors=self._follower_rgba, linewidths=1.3, zorder=2)
        ax.add_collection(self._err_lines)
        self._err_dots = ax.scatter(
            np.zeros(self.n), np.zeros(self.n), s=14, c=self._follower_rgba,
            edgecolors=theme.panel, linewidths=0.6, zorder=3,
        )  # fmt: skip
        self._err_cursor = ax.axvline(0.0, color=theme.muted, lw=0.8, alpha=0.6, zorder=1)
        self._ax_err = ax
        self._set_err_limit(0.5)
        # Key: first and last follower.
        first, last = self._follower_rgba[0], self._follower_rgba[-1]
        for k, (label, color) in enumerate((("follower 1", first), (f"follower {self.n}", last))):
            x = 0.6 + 0.21 * k
            ax.plot([x, x + 0.04], [1.045, 1.045], transform=ax.transAxes, color=color, lw=2.0,
                    clip_on=False)  # fmt: skip
            ax.text(x + 0.05, 1.045, label, transform=ax.transAxes, va="center",
                    color=theme.muted, fontsize=_KEY_FONTSIZE)  # fmt: skip

    def _build_speed_panel(self, fig: Figure) -> None:
        theme = self.theme
        ax = fig.add_axes((0.535, 0.1, 0.235, 0.39))
        style_axes(ax, theme, xlabel="time [s]")
        _fixed_title(ax, "Speed [m/s]", theme)
        ax.set_xlim(0.0, self.max_steps * self.dt)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        _fix_label_positions(ax, xlabel_y=-0.13)
        (self._reference_line,) = ax.plot(
            [], [], color=theme.muted, lw=1.0, ls=(0, (3, 2)), alpha=0.9, zorder=1
        )
        self._speed_lines = LineCollection([], colors=self._follower_rgba, linewidths=1.0, zorder=2)
        self._speed_lines.set_alpha(0.85)
        ax.add_collection(self._speed_lines)
        (self._leader_line,) = ax.plot([], [], color=theme.text, lw=1.8, zorder=3)
        self._speed_cursor = ax.axvline(0.0, color=theme.muted, lw=0.8, alpha=0.6, zorder=1)
        self._ax_speed = ax
        self._set_speed_limit(30.0)
        for k, (label, style) in enumerate(
            (("leader", dict(color=theme.text, lw=1.8)),
             ("reference", dict(color=theme.muted, lw=1.0, ls=(0, (3, 2)))))
        ):  # fmt: skip
            x = 0.4 + 0.29 * k
            ax.plot([x, x + 0.07], [1.045, 1.045], transform=ax.transAxes, clip_on=False, **style)
            ax.text(x + 0.09, 1.045, label, transform=ax.transAxes, va="center",
                    color=theme.muted, fontsize=_KEY_FONTSIZE)  # fmt: skip

    def _build_peak_panel(self, fig: Figure) -> None:
        theme = self.theme
        ax = fig.add_axes((0.825, 0.1, 0.15, 0.39))
        style_axes(ax, theme, xlabel="follower")
        _fixed_title(ax, "Peak $|e_i|$ [m]", theme)
        ax.set_xlim(0.4, self.n + 0.6)
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True, min_n_ticks=1))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(False, axis="x")
        _fix_label_positions(ax, xlabel_y=-0.13)
        self._bars = PolyCollection([], facecolors=self._follower_rgba, linewidths=0, zorder=2)
        ax.add_collection(self._bars)
        self._ratio_text = ax.text(
            0.96, 0.95, "", transform=ax.transAxes, ha="right", va="top",
            color=theme.text, fontsize=7.5,
        )  # fmt: skip
        self._ax_peak = ax
        self._set_peak_limit(0.5)
        width = 0.36 if self.n <= 16 else 0.45
        index = np.arange(1, self.n + 1, dtype=float)
        self._bar_x = np.stack([index - width, index + width], axis=1)

    # ------------------------------------------------------------------
    # Axis-range helpers (grow-only; each change invalidates the blit cache)
    # ------------------------------------------------------------------
    def _set_err_limit(self, limit: float) -> None:
        self._err_limit = limit
        self._ax_err.set_ylim(-limit, limit)
        self._invalidate()

    def _set_peak_limit(self, limit: float) -> None:
        self._peak_limit = limit
        self._ax_peak.set_ylim(0.0, limit)
        self._invalidate()

    def _set_speed_limit(self, limit: float) -> None:
        self._speed_limit = limit
        self._ax_speed.set_ylim(0.0, limit)
        self._invalidate()

    def _set_view(self, width: float) -> None:
        """Width of the road view in metres (camera zoom) and the derived geometry."""
        self._view_width = width
        self._ax_road.set_xlim(-0.5 * width, 0.5 * width)
        self._post_spacing = _nice_ceil(width / 10.0)
        self._templates = None  # corner radii depend on the zoom
        self._invalidate()

    def _vehicle_templates(self, lengths: np.ndarray) -> None:
        """Rounded-rectangle outlines (front at x = 0) for the current zoom and lengths."""
        ax = self._ax_road
        bbox = ax.get_position()
        fig_w, fig_h = ax.figure.get_size_inches() * ax.figure.dpi
        px_per_m = bbox.width * fig_w / self._view_width
        ylim = ax.get_ylim()
        px_per_unit = bbox.height * fig_h / (ylim[1] - ylim[0])
        rx, ry = _CORNER_PX / px_per_m, _CORNER_PX / px_per_unit
        self._templates = np.stack(
            [_rounded_rect(length, _CAR_HEIGHT, rx, ry) for length in lengths]
        )
        self._windows = np.stack(
            [
                _rounded_rect(0.2 * length, 0.72 * _CAR_HEIGHT, 0.6 * rx, 0.6 * ry)
                for length in lengths
            ]
        )
        self._windows[..., 0] -= 0.22 * lengths[:, None]
        self._lengths = lengths.copy()

    # ------------------------------------------------------------------
    def _update(
        self,
        *,
        positions: np.ndarray,
        lengths: np.ndarray,
        speeds: np.ndarray,
        accelerations: np.ndarray,
        spacing_errors: np.ndarray,
        peak_errors: np.ndarray,
        step: int,
        collision: bool,
        breakup: bool,
        episode: int | None = None,
        leader_reference: np.ndarray | None = None,
    ) -> None:
        step = int(step)
        if episode != self._episode or step < self._last_step:
            self.reset()
            self._episode = episode
            if leader_reference is not None:
                ref = np.asarray(leader_reference, dtype=np.float64)
                self._reference_line.set_data(self._times[: ref.size], ref)
                top = _nice_ceil(max(float(ref.max()), float(np.max(speeds))) + 2.0, 5.0)
                if top != self._speed_limit:
                    self._set_speed_limit(top)
                else:
                    self._invalidate()  # the reference line is part of the static layer
        self._last_step = step
        if 0 <= step <= self.max_steps:
            self._err_hist[step] = spacing_errors
            self._speed_hist[step] = speeds

        self._update_road(positions, lengths, accelerations, spacing_errors, speeds, step)
        self._update_plots(step, spacing_errors, speeds, peak_errors)
        self._update_header(step, positions, lengths, speeds, collision, breakup)

    def _update_road(self, pos, lengths, acc, errors, speeds, step) -> None:
        n = self.n
        pos = np.asarray(pos, dtype=np.float64)
        lengths = np.asarray(lengths, dtype=np.float64)
        rear = pos - lengths
        extent = float(pos[0] - rear[-1])
        if self._view_width == 0.0 or extent * 1.08 + 10.0 > self._view_width:
            base = n * (4.5 + self.standstill_distance + self.headway * 25.0) + 10.0
            self._set_view(_nice_ceil(max(base, extent * 1.08 + 10.0), 10.0))
        if self._templates is None or not np.array_equal(lengths, self._lengths):
            self._vehicle_templates(lengths)
        camera = 0.5 * (pos[0] + rear[-1])
        half = 0.5 * self._view_width
        x_front, x_rear = pos - camera, rear - camera

        # Lane markings and distance posts move with the road.
        start = math.floor((camera - half) / _DASH_PERIOD) * _DASH_PERIOD
        marks = np.arange(start, camera + half + _DASH_PERIOD, _DASH_PERIOD) - camera
        self._dashes.set_segments(
            np.stack([np.stack([marks, np.ones_like(marks)], 1),
                      np.stack([marks + _DASH, np.ones_like(marks)], 1)], axis=1)
        )  # fmt: skip
        spacing = self._post_spacing
        first = math.ceil((camera - half) / spacing) * spacing
        posts = np.arange(first, camera + half, spacing)
        rel = posts - camera
        self._posts.set_segments(
            np.stack([np.stack([rel, np.full_like(rel, -0.02)], 1),
                      np.stack([rel, np.full_like(rel, -0.14)], 1)], axis=1)
        )  # fmt: skip

        # Vehicles, cabins, gaps and desired positions.
        offsets = np.stack([x_front, np.full(n + 1, _LANE_Y)], axis=1)
        self._vehicles.set_verts(self._templates + offsets[:, None, :])
        self._cabins.set_verts(self._windows + offsets[:, None, :])
        y0, y1 = _LANE_Y - 0.3 * _CAR_HEIGHT, _LANE_Y + 0.3 * _CAR_HEIGHT
        gap_front, gap_rear = x_rear[:-1], x_front[1:]
        quads = np.empty((n, 4, 2))
        quads[:, 0, 0] = quads[:, 1, 0] = gap_rear
        quads[:, 2, 0] = quads[:, 3, 0] = np.maximum(gap_front, gap_rear)
        quads[:, [0, 3], 1] = y0
        quads[:, [1, 2], 1] = y1
        self._gaps.set_verts(quads)
        errors = np.asarray(errors, dtype=np.float64)
        colors = self._gap_cmap(self._gap_norm(errors))
        colors[:, 3] = 0.25 + 0.6 * np.clip(np.abs(errors) / _ERROR_COLOR_RANGE, 0.0, 1.0)
        self._gaps.set_facecolor(colors)
        desired = gap_front - (self.standstill_distance + self.headway * np.asarray(speeds)[1:])
        self._targets.set_segments(
            np.stack([np.stack([desired, np.full(n, y0 - 0.06)], 1),
                      np.stack([desired, np.full(n, y1 + 0.06)], 1)], axis=1)
        )  # fmt: skip

        # Brake lights: intensity grows with the deceleration.
        braking = np.clip(
            (-np.asarray(acc, dtype=np.float64) - _BRAKE_THRESHOLD) / _BRAKE_FULL, 0, 1
        )
        lights = np.empty((2 * (n + 1), 2))
        lights[:, 0] = np.repeat(x_rear + 0.25, 2)
        lights[0::2, 1] = _LANE_Y - 0.3 * _CAR_HEIGHT
        lights[1::2, 1] = _LANE_Y + 0.3 * _CAR_HEIGHT
        light_rgba = np.tile(np.array(with_alpha(self.theme.negative, 1.0)), (2 * (n + 1), 1))
        light_rgba[:, 3] = np.repeat(braking, 2)
        glow_rgba = light_rgba.copy()
        glow_rgba[:, 3] *= 0.35
        self._brake_lights.set_offsets(lights)
        self._brake_lights.set_facecolor(light_rgba)
        self._brake_glow.set_offsets(lights)
        self._brake_glow.set_facecolor(glow_rgba)
        self._lead_label.set_x(x_front[0] - 0.5 * lengths[0])

        # V2V arcs with packets travelling from sender to receiver.
        if self._links:
            centres = x_front - 0.5 * lengths
            senders = np.array([s for s, _, _ in self._links])
            receivers = np.array([r for _, r, _ in self._links])
            leader_arc = np.array([flag for _, _, flag in self._links])
            x_s, x_r = centres[senders], centres[receivers]
            span = np.abs(x_r - x_s)
            height = 0.25 + 1.1 * np.clip(span / (0.6 * self._view_width), 0.0, 1.0)
            height = np.where(leader_arc, height + 0.15, height)
            # Opposite directions of a bidirectional pair bend on the same side but
            # at slightly different heights.
            height = np.where(senders > receivers, 0.8 * height, height)
            t = np.linspace(0.0, 1.0, _ARC_POINTS)[None, :]
            xs = (1 - t) * x_s[:, None] + t * x_r[:, None]
            ys = _ARC_BASE + 4.0 * height[:, None] * t * (1 - t)
            self._arcs.set_segments(np.stack([xs, ys], axis=-1))
            accent = np.array(with_alpha(self.theme.accent, 1.0))
            rgba = np.tile(accent, (len(self._links), 1))
            rgba[:, 3] = np.where(leader_arc, 0.35, 0.75)
            self._arcs.set_color(rgba)
            phase = (step * _PACKET_SPEED + 0.37 * np.arange(len(self._links))) % 1.0
            px = (1 - phase) * x_s + phase * x_r
            py = _ARC_BASE + 4.0 * height * phase * (1 - phase)
            self._packets.set_offsets(np.stack([px, py], axis=1))

    def _update_plots(self, step, errors, speeds, peaks) -> None:
        n = self.n
        valid = ~np.isnan(self._err_hist[:, 0])
        times = self._times[valid]
        err = self._err_hist[valid]
        spd = self._speed_hist[valid]
        count = times.size
        if count:
            tt = np.broadcast_to(times, (n, count))
            self._err_lines.set_segments(np.stack([tt, err.T], axis=-1))
            self._speed_lines.set_segments(np.stack([tt, spd[:, 1:].T], axis=-1))
            self._leader_line.set_data(times, spd[:, 0])
        now = step * self.dt
        self._err_dots.set_offsets(np.stack([np.full(n, now), np.asarray(errors)], axis=1))
        self._err_cursor.set_xdata([now, now])
        self._speed_cursor.set_xdata([now, now])

        worst = float(np.nanmax(np.abs(err))) if count else 0.0
        if worst * 1.15 > self._err_limit:
            self._set_err_limit(_nice_ceil(worst * 1.15))
        top_speed = float(np.nanmax(spd)) if count else 0.0
        if top_speed + 1.0 > self._speed_limit:
            self._set_speed_limit(_nice_ceil(top_speed + 2.0, 5.0))

        peaks = np.asarray(peaks, dtype=np.float64)
        bars = np.empty((n, 4, 2))
        bars[:, 0, 0] = bars[:, 1, 0] = self._bar_x[:, 0]
        bars[:, 2, 0] = bars[:, 3, 0] = self._bar_x[:, 1]
        bars[:, [0, 3], 1] = 0.0
        bars[:, 1, 1] = bars[:, 2, 1] = peaks
        self._bars.set_verts(bars)
        if float(peaks.max()) * 1.15 > self._peak_limit:
            self._set_peak_limit(_nice_ceil(float(peaks.max()) * 1.15))
        ratio = peaks[-1] / max(float(peaks[0]), 0.01)
        self._ratio_text.set_text(f"last/first {ratio:.2f}" if n > 1 else "")

    def _update_header(self, step, pos, lengths, speeds, collision, breakup) -> None:
        theme = self.theme
        pos = np.asarray(pos, dtype=np.float64)
        gaps = pos[:-1] - pos[1:] - np.asarray(lengths)[:-1]
        self._header.set_text(
            f" | step {step}/{self.max_steps} | t={step * self.dt:.1f} s | "
            f"v_lead={float(speeds[0]):.1f} m/s | min gap {float(gaps.min()):.1f} m"
        )
        if collision:
            label, color = "COLLISION", theme.negative
        elif breakup:
            label, color = "BREAK-UP", theme.warning
        elif step >= self.max_steps:
            label, color = "COMPLETE", theme.positive
        else:
            label, color = "CRUISING", theme.accent
        self._badge.set_text(label)
        self._badge.set_color(color)
        patch = self._badge.get_bbox_patch()
        patch.set_facecolor(with_alpha(color, 0.14))
        patch.set_edgecolor(with_alpha(color, 0.7))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _text_width(fig: Figure, text) -> float:
    """Width of a text artist in figure coordinates."""
    get_renderer = getattr(fig.canvas, "get_renderer", None)
    if get_renderer is not None:
        return text.get_window_extent(renderer=get_renderer()).width / fig.bbox.width
    # Fallback: ink extent of the glyph outlines (slightly narrower than the advance).
    path = TextPath((0, 0), text.get_text(), prop=text.get_fontproperties())
    return path.get_extents().width / 72.0 / fig.get_size_inches()[0]


def _is_light(theme: Theme) -> bool:
    """``True`` for themes with a light background."""
    r, g, b, _ = with_alpha(theme.background, 1.0)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5


def _rounded_rect(length: float, height: float, rx: float, ry: float) -> np.ndarray:
    """Polygon of a rectangle ``[-length, 0] x [-height/2, height/2]`` with rounded corners."""
    rx = min(rx, 0.5 * length)
    ry = min(ry, 0.5 * height)
    angle = np.linspace(0.0, 0.5 * np.pi, 5)
    cos, sin = np.cos(angle), np.sin(angle)
    half = 0.5 * height
    corners = [
        (-rx + rx * cos, half - ry + ry * sin),  # front top
        (-length + rx - rx * sin, half - ry + ry * cos),  # rear top
        (-length + rx - rx * cos, -half + ry - ry * sin),  # rear bottom
        (-rx + rx * sin, -half + ry - ry * cos),  # front bottom
    ]
    return np.concatenate([np.stack(c, axis=1) for c in corners])


def _fixed_title(ax, title: str, theme: Theme) -> None:
    """Left-aligned title at a fixed position (skips matplotlib's per-draw title layout)."""
    ax.set_title(title, color=theme.text, fontsize=10, loc="left", pad=6, y=1.0)


def _fix_label_positions(ax, xlabel_y: float = -0.12, ylabel_x: float = -0.1) -> None:
    """Pin axis-label positions (skips the per-draw tick-label bounding-box search)."""
    ax.xaxis.set_label_coords(0.5, xlabel_y)
    ax.yaxis.set_label_coords(ylabel_x, 0.5)


def _nice_ceil(value: float, minimum: float = 0.5) -> float:
    """Smallest number of the form {1, 2, 2.5, 5} * 10**k that is >= ``value`` (at least ``minimum``)."""
    value = max(value, minimum)
    scale = 10.0 ** math.floor(math.log10(value))
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        if factor * scale >= value * (1 - 1e-12):
            return factor * scale
    return 10.0 * scale  # pragma: no cover - unreachable
