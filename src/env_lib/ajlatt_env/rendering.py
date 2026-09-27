"""Dashboard renderer for the AJLATT environment.

Layout: the map panel on the left shows the occupancy grid, robots (true pose
as oriented arrowheads, belief as a covariance ellipse), their sensing
sectors, trajectories with fading trails, communication links, active
target measurements and every robot's belief about the target. Two panels on
the right track the trace of the target and self-localisation covariances of
each robot over the episode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import ticker as mticker
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon, Wedge

from env_lib.utils.rendering import MatplotlibRenderer, style_axes, with_alpha

if TYPE_CHECKING:  # pragma: no cover
    from env_lib.ajlatt_env.env import AJLATTEnv

__all__ = ["AJLATTRenderer"]

_ARROW = np.array([[1.0, 0.0], [-0.65, 0.6], [-0.3, 0.0], [-0.65, -0.6]])


def _mix(color_a: str, color_b: str, weight: float) -> tuple:
    a = np.array(mcolors.to_rgb(color_a))
    b = np.array(mcolors.to_rgb(color_b))
    return tuple((1 - weight) * a + weight * b)


def _ellipse_params(cov: np.ndarray, scale: float):
    """Width, height and angle (degrees) of the confidence ellipse of a 2x2 covariance."""
    values, vectors = np.linalg.eigh(0.5 * (cov + cov.T))
    values = np.clip(values, 0.0, None)
    major = vectors[:, 1]
    angle = np.degrees(np.arctan2(major[1], major[0]))
    return 2 * scale * np.sqrt(values[1]), 2 * scale * np.sqrt(values[0]), angle


class AJLATTRenderer(MatplotlibRenderer):
    """Persistent-artist renderer for :class:`~env_lib.ajlatt_env.env.AJLATTEnv`.

    Parameters
    ----------
    env:
        The environment to draw.
    render_mode:
        ``"human"`` or ``"rgb_array"``.
    confidence:
        Probability mass of the drawn covariance ellipses.
    trail_length:
        Number of past positions drawn as a fading robot trail.
    """

    def __init__(
        self,
        env: AJLATTEnv,
        render_mode: str,
        *,
        confidence: float = 0.95,
        trail_length: int = 160,
        figsize=(11.0, 6.2),
        dpi: int = 100,
        fps: float | None = 8,
        theme=None,
    ):
        super().__init__(
            render_mode, figsize=figsize, dpi=dpi, fps=fps, title="AJLATT", theme=theme
        )
        self.env = env
        self.scale = float(np.sqrt(-2.0 * np.log(1.0 - confidence)))
        self.trail_length = trail_length
        self._robot_trail: list[list[np.ndarray]] = [[] for _ in range(env.nR)]
        self._target_trail: list[np.ndarray] = []

    def reset(self) -> None:
        self._robot_trail = [[] for _ in range(self.env.nR)]
        self._target_trail = []
        if self.fig is not None:
            self._draw_map()
            self._limits = {key: None for key in self._limits}
            self._invalidate()

    # ------------------------------------------------------------------
    def _build(self, fig) -> None:
        env, theme = self.env, self.theme
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.55, 1.0],
            left=0.04,
            right=0.975,
            top=0.845,
            bottom=0.08,
            wspace=0.16,
            hspace=0.42,
        )
        self.ax_map = fig.add_subplot(gs[:, 0])
        self.ax_target = fig.add_subplot(gs[0, 1])
        self.ax_robot = fig.add_subplot(gs[1, 1])

        grid = env.MAP
        extent = [grid.mapmin[0], grid.mapmax[0], grid.mapmin[1], grid.mapmax[1]]
        style_axes(self.ax_map, theme, grid=False, show_spines=True)
        self.ax_map.set_xticks([])
        self.ax_map.set_yticks([])
        obstacle = _mix(theme.panel, theme.text, 0.22)
        self._cmap = mcolors.ListedColormap([theme.panel, obstacle])
        nx, ny = grid.mapdim
        self._image = self.ax_map.imshow(
            np.zeros((ny, nx)),
            cmap=self._cmap,
            vmin=0,
            vmax=1,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            zorder=0,
        )
        self.ax_map.set_xlim(extent[0], extent[1])
        self.ax_map.set_ylim(extent[2], extent[3])
        self.ax_map.set_aspect("equal")
        self._draw_map()

        size = 0.02 * max(extent[1] - extent[0], extent[3] - extent[2])
        self._arrow = _ARROW * size
        cfg = env.config
        colors = [theme.color(i) for i in range(env.nR)]
        target_color = theme.negative

        self._wedges, self._robot_glow, self._robot_body, self._robot_ellipses = [], [], [], []
        self._target_ellipses = []
        self._trails = []
        for color in colors:
            wedge = Wedge(
                (0, 0),
                cfg.sensor_r_max,
                0,
                cfg.fov,
                facecolor=with_alpha(color, 0.10),
                edgecolor=with_alpha(color, 0.45),
                linewidth=0.8,
                zorder=2,
            )
            self.ax_map.add_patch(wedge)
            self._wedges.append(wedge)
            trail = LineCollection([], linewidths=1.6, zorder=3, capstyle="round")
            self.ax_map.add_collection(trail)
            self._trails.append(trail)
            ellipse = Ellipse(
                (0, 0),
                0,
                0,
                facecolor=with_alpha(color, 0.25),
                edgecolor=color,
                linewidth=1.0,
                zorder=4,
            )
            self.ax_map.add_patch(ellipse)
            self._robot_ellipses.append(ellipse)
            target_ellipse = Ellipse(
                (0, 0),
                0,
                0,
                facecolor=with_alpha(color, 0.10),
                edgecolor=with_alpha(color, 0.8),
                linewidth=0.9,
                linestyle="--",
                zorder=5,
            )
            self.ax_map.add_patch(target_ellipse)
            self._target_ellipses.append(target_ellipse)
            glow = Polygon(
                self._arrow * 1.7,
                closed=True,
                facecolor=with_alpha(color, 0.22),
                edgecolor="none",
                zorder=6,
            )
            body = Polygon(
                self._arrow,
                closed=True,
                facecolor=color,
                edgecolor=theme.background,
                linewidth=0.8,
                zorder=7,
            )
            self.ax_map.add_patch(glow)
            self.ax_map.add_patch(body)
            self._robot_glow.append(glow)
            self._robot_body.append(body)

        self._links = LineCollection(
            [],
            colors=[with_alpha(theme.muted, 0.55)],
            linewidths=0.9,
            linestyles=(0, (4, 3)),
            zorder=3,
        )
        self.ax_map.add_collection(self._links)
        self._rays = LineCollection(
            [], colors=[with_alpha(theme.positive, 0.85)], linewidths=1.3, zorder=5
        )
        self.ax_map.add_collection(self._rays)
        (self._target_path,) = self.ax_map.plot(
            [],
            [],
            color=with_alpha(target_color, 0.7),
            linewidth=1.2,
            linestyle=(0, (2, 2)),
            zorder=3,
        )
        (self._target_glow,) = self.ax_map.plot(
            [],
            [],
            marker="*",
            markersize=24,
            linestyle="none",
            color=with_alpha(target_color, 0.25),
            zorder=8,
        )
        (self._target_marker,) = self.ax_map.plot(
            [],
            [],
            marker="*",
            markersize=15,
            linestyle="none",
            color=target_color,
            markeredgecolor=theme.background,
            markeredgewidth=0.8,
            zorder=9,
        )
        self._target_estimates = self.ax_map.scatter(
            np.zeros(env.nR), np.zeros(env.nR), s=26, marker="x", c=colors, linewidths=1.4, zorder=8
        )
        landmarks = env._target_init_pose[1:, :2]
        if len(landmarks):
            self.ax_map.scatter(
                landmarks[:, 0],
                landmarks[:, 1],
                s=40,
                marker="D",
                color=theme.warning,
                edgecolors=theme.background,
                zorder=6,
            )

        # Header: title on the left, live status on the right, key underneath.
        fig.text(
            0.04,
            0.955,
            "AJLATT",
            color=theme.text,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="center",
        )
        fig.text(
            0.112,
            0.955,
            "multi-robot localisation and target tracking",
            color=theme.muted,
            fontsize=9.5,
            ha="left",
            va="center",
        )
        self._status = fig.text(
            0.975,
            0.955,
            "",
            color=theme.text,
            fontsize=9.5,
            ha="right",
            va="center",
            family="monospace",
        )
        handles = [
            Line2D(
                [],
                [],
                marker=(3, 0, -90),
                markersize=8,
                linestyle="none",
                color=color,
                label=f"robot {i}",
            )
            for i, color in enumerate(colors)
        ]
        handles += [
            Line2D(
                [],
                [],
                marker="*",
                markersize=10,
                linestyle="none",
                color=target_color,
                label="target",
            ),
            Line2D([], [], marker="x", linestyle="none", color=theme.muted, label="target belief"),
            Line2D([], [], color=theme.positive, label="measurement"),
            Line2D([], [], color=theme.muted, linestyle=(0, (4, 3)), label="comm. link"),
        ]
        fig.legend(
            handles=handles,
            loc="center left",
            ncol=len(handles),
            bbox_to_anchor=(0.035, 0.908),
            frameon=False,
            fontsize=8.5,
            labelcolor=theme.muted,
            handlelength=1.5,
            columnspacing=1.2,
        )

        # Uncertainty panels.
        horizon = env.config.max_episode_steps
        for ax, title in (
            (self.ax_target, "target covariance trace"),
            (self.ax_robot, "self-localisation covariance trace"),
        ):
            style_axes(ax, theme, title=title, xlabel="step")
            ax.set_yscale("log")
            ax.set_xlim(0, horizon)
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        self._target_lines = [
            self.ax_target.plot([], [], color=c, linewidth=1.5)[0] for c in colors
        ]
        self._robot_lines = [self.ax_robot.plot([], [], color=c, linewidth=1.5)[0] for c in colors]
        self._limits = {id(self.ax_target): None, id(self.ax_robot): None}

        self._dynamic(
            *self._wedges,
            *self._trails,
            self._links,
            *self._robot_ellipses,
            self._target_path,
            *self._target_ellipses,
            self._rays,
            *self._robot_glow,
            *self._robot_body,
            self._target_estimates,
            self._target_glow,
            self._target_marker,
            self._status,
            *self._target_lines,
            *self._robot_lines,
        )

    def _draw_map(self) -> None:
        grid = self.env.MAP
        nx, ny = grid.mapdim
        occupancy = np.zeros((ny, nx)) if grid.map is None else (np.asarray(grid.map) == 1)
        self._image.set_data(occupancy.astype(float))

    # ------------------------------------------------------------------
    def _update(self) -> None:
        env = self.env
        cfg = env.config
        robots = np.array([agent.state for agent in env.robot_true])
        estimates = [est for est in env.robot_est]
        target = env.target_true[0].state

        for i in range(env.nR):
            self._robot_trail[i].append(robots[i, :2].copy())
            del self._robot_trail[i][: -self.trail_length]
        self._target_trail.append(target[:2].copy())

        hide_threshold = 2.0 * max(cfg.target_init_cov)
        for i in range(env.nR):
            x, y, theta = robots[i]
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, -s], [s, c]])
            self._robot_body[i].set_xy(self._arrow @ rot.T + (x, y))
            self._robot_glow[i].set_xy(1.7 * self._arrow @ rot.T + (x, y))

            wedge = self._wedges[i]
            wedge.set_center((x, y))
            heading = np.degrees(theta)
            wedge.set_theta1(heading - cfg.fov / 2)
            wedge.set_theta2(heading + cfg.fov / 2)

            est = estimates[i]
            width, height, angle = _ellipse_params(est.cov[:2, :2], self.scale)
            ellipse = self._robot_ellipses[i]
            ellipse.set_center(est.state[:2])
            ellipse.set_width(width)
            ellipse.set_height(height)
            ellipse.set_angle(angle)

            belief = env.target_est[i][0]
            target_ellipse = self._target_ellipses[i]
            visible = bool(np.trace(belief.cov) < hide_threshold)
            target_ellipse.set_visible(visible)
            if visible:
                width, height, angle = _ellipse_params(belief.cov[:2, :2], self.scale)
                target_ellipse.set_center(belief.state[:2])
                target_ellipse.set_width(width)
                target_ellipse.set_height(height)
                target_ellipse.set_angle(angle)

            trail = np.asarray(self._robot_trail[i])
            if len(trail) > 1:
                segments = np.stack([trail[:-1], trail[1:]], axis=1)
                alphas = np.linspace(0.05, 0.9, len(segments))
                rgba = np.tile(mcolors.to_rgba(self.theme.color(i)), (len(segments), 1))
                rgba[:, 3] = alphas
                self._trails[i].set_segments(segments)
                self._trails[i].set_color(rgba)
            else:
                self._trails[i].set_segments([])

        links = [
            (robots[i, :2], robots[j, :2])
            for i in range(env.nR)
            for j in range(i + 1, env.nR)
            if env.com_plot[i, j] or env.com_plot[j, i]
        ]
        self._links.set_segments(links)
        observed = np.flatnonzero(env.RT_obs[:, env.nR])
        self._rays.set_segments([(robots[i, :2], target[:2]) for i in observed])

        path = np.asarray(self._target_trail)
        self._target_path.set_data(path[:, 0], path[:, 1])
        self._target_marker.set_data([target[0]], [target[1]])
        self._target_glow.set_data([target[0]], [target[1]])
        self._target_estimates.set_offsets(
            np.array([env.target_est[i][0].state[:2] for i in range(env.nR)])
        )

        stats = env.episode_statistics()
        steps = np.arange(len(stats["target_cov_trace"]))
        self._set_series(self.ax_target, self._target_lines, steps, stats["target_cov_trace"])
        self._set_series(self.ax_robot, self._robot_lines, steps, stats["robot_cov_trace"])

        reward = float(np.sum(stats["reward"][-1])) if len(stats["reward"]) else 0.0
        seen = int(len(observed))
        self._status.set_text(
            f"{cfg.map_name} | step {env.step_count:>3d}/{cfg.max_episode_steps} | "
            f"t {env.step_count * env.dt:5.1f} s | reward {reward:8.2f} | seen {seen}/{env.nR}"
        )

    def _set_series(self, ax, lines, steps, values) -> None:
        """Update one metric panel; y-limits snap to half-decades and only grow."""
        values = np.asarray(values, dtype=float)
        for i, line in enumerate(lines):
            line.set_data(steps, values[:, i])
        positive = values[np.isfinite(values) & (values > 0)]
        if not positive.size:
            return
        low = 10.0 ** (np.floor(2 * np.log10(positive.min())) / 2)
        high = 10.0 ** (np.ceil(2 * np.log10(positive.max() * 1.05)) / 2)
        current = self._limits[id(ax)]
        if current is not None:
            low, high = min(low, current[0]), max(high, current[1])
        if (low, high) != current:
            ax.set_ylim(low, high)
            self._limits[id(ax)] = (low, high)
            self._invalidate()
