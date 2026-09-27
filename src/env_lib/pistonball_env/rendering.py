"""Vector renderer for :class:`~env_lib.pistonball_env.PistonballEnv` (pygame).

The renderer draws the arena with anti-aliased sprites generated from signed
distance fields, so no image assets are needed:

* a vertical gradient background, shaded walls and an accent-coloured goal
  strip on the left wall (pre-rendered once into a static layer);
* pistons as rounded rectangles with a lighter head cap; the pistons that
  observe the ball (the ``kappa``-hop neighbourhood) are drawn in the accent
  colour with a soft glow;
* the ball with radial shading, a rotation marker and a fading motion trail;
* a HUD with the step counter, episode return, ball x-velocity and number of
  observing pistons, plus a bar showing the ball's distance to the left wall.

Colours come from the active :class:`env_lib.utils.rendering.Theme`, so the
``"dark"`` and ``"light"`` themes both apply. ``pygame`` is imported when the
renderer draws its first frame. ``"rgb_array"`` rendering never touches the
display subsystem and therefore works on headless machines; ``"human"``
rendering opens a window capped at ``fps`` frames per second.
"""

from __future__ import annotations

import logging
import math
import sys
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from env_lib.utils.rendering import Theme, get_theme, validate_render_mode

__all__ = ["PistonballLayout", "PistonballRenderer", "hex_to_rgb"]

_logger = logging.getLogger(__name__)

RGB = tuple[int, int, int]
_WHITE: RGB = (255, 255, 255)
_OPEN_WINDOWS = 0  # number of renderers currently holding the shared pygame display
_BLACK: RGB = (0, 0, 0)


def hex_to_rgb(color: str) -> RGB:
    """Convert a colour to an 8-bit ``(r, g, b)`` tuple.

    Accepts hex strings (``"#RRGGBB"``, ``"#RGB"``, ``"#RRGGBBAA"``) and any
    other matplotlib colour specification (``"white"``, ``"tab:blue"``, RGB
    tuples in ``[0, 1]``).

    Examples
    --------
    >>> hex_to_rgb("#4C9AFF")
    (76, 154, 255)
    >>> hex_to_rgb("white")
    (255, 255, 255)
    """
    from matplotlib import colors as mcolors

    try:
        rgb = mcolors.to_rgb(color.strip() if isinstance(color, str) else color)
    except ValueError:
        if isinstance(color, str) and len(color.strip().lstrip("#")) == 3:
            value = "".join(ch * 2 for ch in color.strip().lstrip("#"))
            return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
        raise ValueError(f"Invalid colour {color!r}") from None
    return tuple(int(round(255 * channel)) for channel in rgb)  # type: ignore[return-value]


def _mix(a: Sequence[float], b: Sequence[float], t: float) -> RGB:
    """Linear interpolation between two RGB colours (``t = 0`` gives ``a``)."""
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def _luminance(color: RGB) -> float:
    return (0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]) / 255.0


@dataclass(frozen=True)
class PistonballLayout:
    """Screen geometry of a Pistonball environment (pixels, ``y`` down).

    Attributes
    ----------
    n_pistons:
        Number of pistons.
    screen_width, screen_height:
        Frame size.
    wall_width:
        Distance from the screen border to the inner face of each wall.
    piston_width, piston_radius, piston_body_height:
        Piston lane width, head half-thickness and height of the base housing.
    ball_radius:
        Ball radius.
    minimum_piston_y, maximum_piston_y:
        Highest and lowest piston head positions.
    """

    n_pistons: int
    screen_width: int
    screen_height: int
    wall_width: int = 80
    piston_width: int = 40
    piston_radius: int = 5
    piston_body_height: int = 23
    ball_radius: int = 40
    minimum_piston_y: float = 387.0
    maximum_piston_y: float = 451.0

    @property
    def arena_left(self) -> int:
        return self.wall_width

    @property
    def arena_right(self) -> int:
        return self.screen_width - self.wall_width

    @property
    def arena_top(self) -> int:
        return self.wall_width

    @property
    def floor(self) -> int:
        """y of the bottom wall (the pistons' base plate)."""
        return self.screen_height - self.wall_width

    @property
    def housing_top(self) -> int:
        """y where the piston rods enter the base housing."""
        return self.floor - self.piston_body_height


@dataclass(frozen=True)
class _Palette:
    """RGB colours derived from a :class:`Theme`."""

    dark: bool
    text: RGB
    muted: RGB
    accent: RGB
    positive: RGB
    warning: RGB
    background: RGB
    canvas_top: RGB
    canvas_bottom: RGB
    arena_top: RGB
    arena_bottom: RGB
    lane: RGB
    wall_outer: RGB
    wall_inner: RGB
    wall_edge: RGB
    housing_top: RGB
    housing_bottom: RGB
    housing_edge: RGB
    slot: RGB
    track: RGB
    piston: RGB
    piston_cap: RGB
    piston_edge: RGB
    active: RGB
    active_cap: RGB
    active_edge: RGB
    ball: RGB
    ball_edge: RGB
    marker: RGB
    shadow_alpha: float

    @classmethod
    def from_theme(cls, theme: Theme) -> _Palette:
        bg = hex_to_rgb(theme.background)
        panel = hex_to_rgb(theme.panel)
        grid = hex_to_rgb(theme.grid)
        text = hex_to_rgb(theme.text)
        muted = hex_to_rgb(theme.muted)
        accent = hex_to_rgb(theme.accent)
        ball = hex_to_rgb(theme.warning)
        dark = _luminance(bg) < 0.5
        deep = bg if dark else text  # direction of "darker than the surroundings"
        piston = _mix(muted, panel, 0.45 if dark else 0.25)
        arena_top = _mix(panel, grid, 0.3)
        return cls(
            dark=dark,
            text=text,
            muted=muted,
            accent=accent,
            positive=hex_to_rgb(theme.positive),
            warning=ball,
            background=bg,
            canvas_top=_mix(bg, panel, 0.8),
            canvas_bottom=bg,
            arena_top=arena_top,
            arena_bottom=_mix(panel, bg, 0.35),
            lane=_mix(arena_top, grid, 0.55),
            wall_outer=_mix(grid, bg, 0.35),
            wall_inner=_mix(grid, text, 0.14),
            wall_edge=_mix(grid, text, 0.32),
            housing_top=_mix(grid, text, 0.10),
            housing_bottom=_mix(grid, bg, 0.30),
            housing_edge=_mix(grid, text, 0.30),
            slot=_mix(grid, deep, 0.55),
            track=_mix(grid, bg, 0.15) if dark else grid,
            piston=piston,
            piston_cap=_mix(piston, _WHITE, 0.35),
            piston_edge=_mix(piston, deep, 0.35),
            active=accent,
            active_cap=_mix(accent, _WHITE, 0.40),
            active_edge=_mix(accent, deep, 0.35),
            ball=ball,
            ball_edge=_mix(ball, _BLACK, 0.35),
            marker=_mix(ball, _BLACK, 0.62),
            shadow_alpha=0.45 if dark else 0.22,
        )


# ---------------------------------------------------------------------------
# Signed-distance helpers (pixel centres at +0.5; negative inside)
# ---------------------------------------------------------------------------
def _grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    ys = np.arange(height, dtype=np.float64)[:, None] + 0.5
    xs = np.arange(width, dtype=np.float64)[None, :] + 0.5
    return xs, ys


def _sdf_round_rect(
    xs: np.ndarray, ys: np.ndarray, x0: float, y0: float, w: float, h: float, r: float
) -> np.ndarray:
    qx = np.abs(xs - (x0 + w / 2)) - (w / 2 - r)
    qy = np.abs(ys - (y0 + h / 2)) - (h / 2 - r)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside - r


def _sdf_circle(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    return np.hypot(xs - cx, ys - cy) - r


def _sdf_capsule(
    xs: np.ndarray, ys: np.ndarray, p0: tuple[float, float], p1: tuple[float, float], r: float
) -> np.ndarray:
    ax, ay = p0
    bx, by = p1
    dx, dy = bx - ax, by - ay
    t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    return np.hypot(xs - ax - t * dx, ys - ay - t * dy) - r


def _coverage(sdf: np.ndarray) -> np.ndarray:
    """Anti-aliased coverage of the region ``sdf < 0``."""
    return np.clip(0.5 - sdf, 0.0, 1.0)


class _Layer:
    """Premultiplied RGBA accumulator used to compose sprites with numpy."""

    def __init__(self, height: int, width: int):
        self.rgb = np.zeros((height, width, 3), dtype=np.float64)
        self.alpha = np.zeros((height, width), dtype=np.float64)

    def over(self, color: RGB | np.ndarray, alpha: np.ndarray) -> None:
        color = np.asarray(color, dtype=np.float64)
        a = alpha[..., None]
        self.rgb = color * a + self.rgb * (1.0 - a)
        self.alpha = alpha + self.alpha * (1.0 - alpha)

    def to_rgba(self) -> np.ndarray:
        a = self.alpha[..., None]
        rgb = np.where(a > 1e-6, self.rgb / np.maximum(a, 1e-6), 0.0)
        out = np.empty(self.alpha.shape + (4,), dtype=np.uint8)
        out[..., :3] = np.clip(np.rint(rgb), 0, 255)
        out[..., 3] = np.clip(np.rint(self.alpha * 255.0), 0, 255)
        return out


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class PistonballRenderer:
    """Draws Pistonball frames with pygame.

    Parameters
    ----------
    layout:
        Screen geometry of the environment.
    render_mode:
        ``"rgb_array"`` (off-screen, returns frames) or ``"human"`` (window).
    fps:
        Frame-rate cap in ``"human"`` mode.
    theme:
        Theme name or :class:`Theme`; defaults to the active theme.
    trail_length:
        Number of past ball positions drawn as a fading trail.

    Examples
    --------
    The environment creates the renderer on its first ``render()`` call::

        env = PistonballEnv(render_mode="rgb_array")
        env.reset(seed=0)
        frame = env.render()  # (560, 960, 3) uint8
    """

    _GAP = 3  # horizontal gap between a piston sprite and its lane border
    _WALL = 10  # drawn wall thickness
    _GLOW = 14  # glow margin around observing pistons
    _MARGIN = 16  # HUD side margin

    def __init__(
        self,
        layout: PistonballLayout,
        render_mode: str = "rgb_array",
        *,
        fps: float | None = 20,
        theme: str | Theme | None = None,
        trail_length: int = 12,
    ):
        validate_render_mode(render_mode)
        if render_mode is None:
            raise ValueError("PistonballRenderer requires render_mode 'human' or 'rgb_array'")
        if trail_length < 0:
            raise ValueError(f"trail_length must be >= 0, got {trail_length}")
        self.layout = layout
        self.render_mode = render_mode
        self.fps = fps
        self.theme = get_theme(theme)
        self.palette = _Palette.from_theme(self.theme)
        self.trail_length = int(trail_length)
        self._trail: deque[tuple[float, float]] = deque(maxlen=max(self.trail_length, 1))

        self._pg: Any = None
        self._canvas: Any = None
        self._window: Any = None
        self._clock: Any = None
        self._window_closed_by_user = False
        self._fonts: dict[str, Any] = {}
        self._text_cache: dict[tuple[str, str, RGB], Any] = {}
        self._static: Any = None
        self._sprites: dict[str, Any] = {}
        self._labels: list[tuple[int, Any, Any, tuple[int, int]]] = []

        self._piston_w = layout.piston_width - 2 * self._GAP
        self._piston_x0 = (
            layout.wall_width + layout.piston_width * np.arange(layout.n_pistons) + self._GAP
        )
        # Tallest visible piston: head at minimum_piston_y, rod down to the housing.
        self._piston_h = int(
            math.ceil(layout.housing_top - (layout.minimum_piston_y - layout.piston_radius))
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        """``False`` in ``"human"`` mode once the window was closed."""
        if self.render_mode == "human":
            return self._window is not None
        return True

    def reset(self) -> None:
        """Clear the ball trail (called when the environment is reset)."""
        self._trail.clear()

    def render(
        self,
        *,
        piston_y: np.ndarray,
        observable: np.ndarray,
        ball_position: tuple[float, float],
        ball_angle: float,
        ball_velocity: tuple[float, float],
        step: int,
        max_cycles: int,
        episode_return: float,
        terminated: bool = False,
        truncated: bool = False,
    ) -> np.ndarray | None:
        """Draw one frame.

        Parameters
        ----------
        piston_y:
            Piston head positions, shape ``(n_pistons,)``.
        observable:
            Boolean mask of the pistons that observe the ball.
        ball_position, ball_angle, ball_velocity:
            Ball state in screen pixels, radians and px/s.
        step, max_cycles:
            Step counter and episode limit.
        episode_return:
            Team return accumulated since the last reset.
        terminated, truncated:
            Episode status shown in the HUD.

        Returns
        -------
        numpy.ndarray or None
            ``(screen_height, screen_width, 3)`` ``uint8`` frame in
            ``"rgb_array"`` mode, ``None`` in ``"human"`` mode.
        """
        if self.render_mode == "human" and self._window_closed_by_user:
            return None
        self._ensure_initialised()
        pg = self._pg
        if self.render_mode == "human":
            pg.event.pump()
            if pg.event.get(pg.QUIT):
                _logger.info("Pistonball window closed by the user; human rendering stops.")
                self._window_closed_by_user = True
                self.close()
                return None
            canvas = self._window
        else:
            canvas = self._canvas

        self._draw(
            canvas,
            np.asarray(piston_y, dtype=np.float64),
            np.asarray(observable, dtype=bool),
            (float(ball_position[0]), float(ball_position[1])),
            float(ball_angle),
            (float(ball_velocity[0]), float(ball_velocity[1])),
            int(step),
            int(max_cycles),
            float(episode_return),
            "goal reached" if terminated else ("time limit" if truncated else None),
        )

        if self.render_mode == "human":
            pg.display.flip()
            if self.fps:
                self._clock.tick(self.fps)
            return None
        return self._to_array(canvas)

    def _to_array(self, surface: Any) -> np.ndarray:
        """Copy a 32-bit surface into an ``(H, W, 3)`` ``uint8`` RGB array."""
        width, height = surface.get_size()
        if surface.get_bytesize() != 4:
            to_bytes = getattr(self._pg.image, "tobytes", None) or self._pg.image.tostring
            frame = np.frombuffer(to_bytes(surface, "RGB"), dtype=np.uint8)
            return frame.reshape(height, width, 3).copy()
        pixels = np.frombuffer(surface.get_buffer(), dtype=np.uint8)
        pixels = pixels.reshape(height, surface.get_pitch() // 4, 4)[:, :width]
        frame = np.empty((height, width, 3), dtype=np.uint8)
        for channel, shift in enumerate(surface.get_shifts()[:3]):
            byte = shift // 8 if sys.byteorder == "little" else 3 - shift // 8
            frame[..., channel] = pixels[..., byte]
        return frame

    def close(self) -> None:
        """Close the window (``"human"``) and release the drawing surfaces. Idempotent.

        The pygame display is shared by every renderer in the process; it is
        only shut down when the last open window is closed.
        """
        global _OPEN_WINDOWS
        if self._window is not None and self._pg is not None:
            _OPEN_WINDOWS = max(0, _OPEN_WINDOWS - 1)
            if _OPEN_WINDOWS == 0:
                self._pg.display.quit()
        self._window = None
        self._canvas = None
        self._static = None
        self._sprites = {}
        self._labels = []
        self._text_cache = {}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _ensure_initialised(self) -> None:
        if self._pg is None:
            try:
                import pygame
            except ImportError as exc:  # pragma: no cover - depends on the installation
                raise ImportError(
                    'Pistonball rendering requires pygame: pip install "my-tool-box[pistonball]"'
                ) from exc
            self._pg = pygame
        pg = self._pg
        if not pg.font.get_init():
            pg.font.init()
        size = (self.layout.screen_width, self.layout.screen_height)
        if self.render_mode == "human" and self._window is None:
            global _OPEN_WINDOWS
            pg.display.init()
            self._window = pg.display.set_mode(size)
            _OPEN_WINDOWS += 1
            pg.display.set_caption("Pistonball")
            self._clock = pg.time.Clock()
            self._static = None  # rebuild and convert for the display format
        if self.render_mode == "rgb_array" and self._canvas is None:
            self._canvas = pg.Surface(size, 0, 32)
        if self._static is None:
            self._fonts = self._load_fonts()
            self._sprites = self._build_sprites()
            self._static = self._build_static()
            self._labels = self._build_labels()
            if self._window is not None:
                self._static = self._static.convert()
                self._sprites = {k: v.convert_alpha() for k, v in self._sprites.items()}

    def _load_fonts(self) -> dict[str, Any]:
        pg = self._pg
        width = self.layout.screen_width
        scale = 1.0 if width >= 480 else 0.8
        sizes = {"title": 36, "label": 19, "small": 17}
        fonts = {}
        for key, size in sizes.items():
            size = max(10, int(round(size * scale)))
            try:
                fonts[key] = pg.font.Font(None, size)
            except (OSError, RuntimeError):  # pragma: no cover - missing default font
                fonts[key] = pg.font.SysFont("dejavusans,arial,helvetica", size)
        return fonts

    def _surface_from_rgba(self, rgba: np.ndarray) -> Any:
        pg = self._pg
        height, width = rgba.shape[:2]
        from_bytes = getattr(pg.image, "frombytes", None) or pg.image.fromstring
        return from_bytes(np.ascontiguousarray(rgba).tobytes(), (width, height), "RGBA")

    def _text(self, font: str, text: str, color: RGB) -> Any:
        key = (font, text, color)
        surface = self._text_cache.get(key)
        if surface is None:
            if len(self._text_cache) > 512:
                self._text_cache.clear()
            surface = self._fonts[font].render(text, True, color)
            self._text_cache[key] = surface
        return surface

    # ------------------------------------------------------------------
    # Sprites (anti-aliased, built once with numpy)
    # ------------------------------------------------------------------
    def _build_sprites(self) -> dict[str, Any]:
        pal = self.palette
        sprites = {
            "piston": self._surface_from_rgba(
                self._piston_rgba(pal.piston, pal.piston_cap, pal.piston_edge)
            ),
            "active": self._surface_from_rgba(
                self._piston_rgba(pal.active, pal.active_cap, pal.active_edge)
            ),
            "glow": self._surface_from_rgba(self._glow_rgba()),
            "ball": self._surface_from_rgba(self._ball_rgba()),
            "marker": self._surface_from_rgba(self._marker_rgba()),
        }
        for k, rgba in enumerate(self._trail_rgba()):
            sprites[f"trail{k}"] = self._surface_from_rgba(rgba)
        return sprites

    def _piston_silhouette(self, xs: np.ndarray, ys: np.ndarray, x0: float, y0: float):
        cap_h = 2 * self.layout.piston_radius
        w = self._piston_w
        body_w = w - 8
        cap = _sdf_round_rect(xs, ys, x0, y0, w, cap_h, 4.0)
        body = _sdf_round_rect(
            xs, ys, x0 + (w - body_w) / 2, y0 + cap_h - 3, body_w, self._piston_h + 20, 3.0
        )
        return cap, body, body_w

    def _piston_rgba(self, fill: RGB, cap_color: RGB, edge: RGB) -> np.ndarray:
        w, h = self._piston_w, self._piston_h
        xs, ys = _grid(h, w)
        cap, body, body_w = self._piston_silhouette(xs, ys, 0.0, 0.0)
        layer = _Layer(h, w)

        # Rod: cylindrical shading across its width plus a darker outline.
        u = np.clip((xs - (w - body_w) / 2) / body_w, 0.0, 1.0)
        shade = 0.72 + 0.34 * np.sin(np.pi * u) ** 0.7 + 0.18 * np.exp(-(((u - 0.32) / 0.1) ** 2))
        rod = np.clip(np.asarray(fill, dtype=np.float64) * shade[..., None], 0, 255)
        rod = np.broadcast_to(rod, (h, w, 3))
        rim = np.clip((body + 1.6) / 1.6, 0.0, 1.0)[..., None]
        rod = rod * (1.0 - 0.6 * rim) + np.asarray(edge, dtype=np.float64) * 0.6 * rim
        layer.over(rod, _coverage(body))

        # Head cap: lighter, with a soft top highlight and a darker lower edge.
        cap_h = 2 * self.layout.piston_radius
        v = np.clip(ys / cap_h, 0.0, 1.0)[..., None]  # (h, 1, 1)
        cap_rgb = np.clip(np.asarray(cap_color, dtype=np.float64) * (1.08 - 0.22 * v), 0, 255)
        cap_rgb = np.where(ys[..., None] > cap_h - 2, np.asarray(edge, dtype=np.float64), cap_rgb)
        layer.over(np.broadcast_to(cap_rgb, (h, w, 3)), _coverage(cap))
        return layer.to_rgba()

    def _glow_rgba(self) -> np.ndarray:
        g = self._GLOW
        w, h = self._piston_w + 2 * g, self._piston_h + g
        xs, ys = _grid(h, w)
        cap, body, _ = self._piston_silhouette(xs, ys, float(g), float(g))
        dist = np.maximum(np.minimum(cap, body), 0.0)
        alpha = 0.55 * np.exp(-dist / 4.5) * (dist < g)
        layer = _Layer(h, w)
        layer.over(self.palette.accent, alpha)
        return layer.to_rgba()

    def _ball_rgba(self) -> np.ndarray:
        pal = self.palette
        r = float(self.layout.ball_radius)
        pad = 14
        size = int(2 * r + 2 * pad)
        c = size / 2
        xs, ys = _grid(size, size)
        layer = _Layer(size, size)

        # Soft drop shadow below the ball.
        d_shadow = _sdf_circle(xs, ys, c + 1.0, c + 6.0, r - 3.0)
        shadow = pal.shadow_alpha * (1.0 - np.clip((d_shadow + 3.0) / 11.0, 0.0, 1.0)) ** 2
        layer.over(_BLACK, shadow)

        # Lambert + specular shading of a sphere lit from the upper left.
        nx, ny = (xs - c) / r, (ys - c) / r
        nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
        light = np.array([-0.45, -0.6, 0.66])
        light /= np.linalg.norm(light)
        lambert = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0)
        base = np.asarray(pal.ball, dtype=np.float64)
        rgb = base * (0.45 + 0.65 * lambert[..., None]) + 255.0 * 0.5 * lambert[..., None] ** 30
        d_ball = _sdf_circle(xs, ys, c, c, r)
        rim = np.clip((d_ball + 2.2) / 2.2, 0.0, 1.0)[..., None]
        rgb = rgb * (1.0 - 0.7 * rim) + np.asarray(pal.ball_edge, dtype=np.float64) * 0.7 * rim
        layer.over(np.clip(rgb, 0, 255), _coverage(d_ball))
        return layer.to_rgba()

    def _marker_rgba(self) -> np.ndarray:
        r = float(self.layout.ball_radius)
        size = int(2 * r + 4)
        c = size / 2
        xs, ys = _grid(size, size)
        line = _sdf_capsule(xs, ys, (c + 0.18 * r, c), (c + 0.74 * r, c), 3.0)
        hub = _sdf_circle(xs, ys, c, c, 4.0)
        layer = _Layer(size, size)
        layer.over(self.palette.marker, 0.9 * _coverage(np.minimum(line, hub)))
        return layer.to_rgba()

    def _trail_rgba(self) -> list[np.ndarray]:
        r = float(self.layout.ball_radius)
        out = []
        for k in range(self.trail_length):
            t = (k + 1) / self.trail_length  # 1 = most recent
            radius = r * (0.35 + 0.5 * t)
            size = int(math.ceil(2 * radius + 4))
            c = size / 2
            xs, ys = _grid(size, size)
            d = _sdf_circle(xs, ys, c, c, radius)
            alpha = 0.32 * t**1.6 * np.clip(-d / (0.5 * radius) + 0.15, 0.0, 1.0)
            layer = _Layer(size, size)
            layer.over(self.palette.ball, alpha)
            out.append(layer.to_rgba())
        return out

    # ------------------------------------------------------------------
    # Static layer (background, arena, walls, housing, title)
    # ------------------------------------------------------------------
    def _build_static(self) -> Any:
        pg = self._pg
        lay, pal = self.layout, self.palette
        width, height = lay.screen_width, lay.screen_height
        wall = self._WALL
        img = np.empty((height, width, 3), dtype=np.float64)

        def vgradient(top: RGB, bottom: RGB, n: int) -> np.ndarray:
            t = np.linspace(0.0, 1.0, max(n, 1))[:, None]
            return np.asarray(top, float) * (1.0 - t) + np.asarray(bottom, float) * t

        img[:] = vgradient(pal.canvas_top, pal.canvas_bottom, height)[:, None, :]

        left, right, top = lay.arena_left, lay.arena_right, lay.arena_top
        housing, floor = lay.housing_top, lay.floor
        img[top:housing, left:right] = vgradient(pal.arena_top, pal.arena_bottom, housing - top)[
            :, None, :
        ]
        for i in range(1, lay.n_pistons):  # lane separators
            x = left + i * lay.piston_width
            img[top:housing, x] = pal.lane

        # Walls: shaded slabs with a highlighted inner edge.
        across = vgradient(pal.wall_outer, pal.wall_inner, wall)
        img[top - wall : top, left - wall : right + wall] = across[:, None, :]
        img[top - 1, left:right] = pal.wall_edge
        img[top - wall : floor, left - wall : left] = across[None, :, :]
        img[top:floor, left - 1] = pal.wall_edge
        img[top - wall : floor, right : right + wall] = across[None, ::-1, :]
        img[top:floor, right] = pal.wall_edge

        # Goal: accent strip on the left wall with a soft glow into the arena.
        accent = np.asarray(pal.accent, dtype=np.float64)
        img[top:housing, left - 4 : left] = accent
        glow_w = min(36, right - left)
        fade = (0.3 * (1.0 - np.linspace(0.0, 1.0, glow_w)) ** 2)[None, :, None]
        region = img[top:housing, left : left + glow_w]
        img[top:housing, left : left + glow_w] = region * (1.0 - fade) + accent * fade

        # Base housing that the piston rods slide into.
        bottom = min(floor + 4, height)
        img[housing:bottom, left - wall : right + wall] = vgradient(
            pal.housing_top, pal.housing_bottom, bottom - housing
        )[:, None, :]
        img[housing, left - wall : right + wall] = pal.housing_edge
        for x0 in self._piston_x0.tolist():
            img[housing + 1 : housing + 5, x0 + 3 : x0 + self._piston_w - 3] = pal.slot

        rgb = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        surface = pg.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))

        # Title and goal label.
        m = self._MARGIN
        title = self._text("title", "Pistonball", pal.text)
        baseline = 36
        icon = 14
        pg.draw.rect(surface, pal.accent, (m, baseline - icon - 4, icon, icon), border_radius=4)
        surface.blit(title, (m + icon + 8, baseline - self._fonts["title"].get_ascent()))
        goal = pg.transform.rotate(self._text("small", "GOAL", pal.accent), 90)
        surface.blit(goal, goal.get_rect(center=(left - wall - 16, (top + housing) // 2)))

        # Distance track under the arena.
        pg.draw.rect(surface, pal.track, (left, self._track_y, right - left, 8), border_radius=4)
        pg.draw.rect(surface, pal.positive, (left - 2, self._track_y - 3, 3, 14), border_radius=1)
        for _, idle, _, pos in self._build_labels():
            surface.blit(idle, pos)
        return surface

    @property
    def _track_y(self) -> int:
        return self.layout.floor + 40

    def _build_labels(self) -> list[tuple[int, Any, Any, tuple[int, int]]]:
        """Piston index labels: (index, muted surface, accent surface, position)."""
        lay, pal = self.layout, self.palette
        font = self._fonts["small"]
        widest = font.size(str(lay.n_pistons - 1))[0]
        every = max(1, int(math.ceil((widest + 6) / lay.piston_width)))
        labels = []
        y = lay.floor + 10
        for i in range(0, lay.n_pistons, every):
            idle = self._text("small", str(i), pal.muted)
            active = self._text("small", str(i), pal.accent)
            cx = lay.wall_width + lay.piston_width * i + lay.piston_width // 2
            labels.append((i, idle, active, (cx - idle.get_width() // 2, y)))
        return labels

    # ------------------------------------------------------------------
    # Per-frame drawing
    # ------------------------------------------------------------------
    def _draw(
        self,
        canvas: Any,
        piston_y: np.ndarray,
        observable: np.ndarray,
        ball: tuple[float, float],
        angle: float,
        velocity: tuple[float, float],
        step: int,
        max_cycles: int,
        episode_return: float,
        status: str | None,
    ) -> None:
        pg = self._pg
        lay, pal, sprites = self.layout, self.palette, self._sprites
        canvas.blit(self._static, (0, 0))
        bx, by = ball

        # Fading trail (oldest first), then the current position joins the history.
        if self.trail_length:
            history = list(self._trail)
            n_hist = len(history)
            blits = []
            for age, (tx, ty) in enumerate(history):
                sprite = sprites[f"trail{self.trail_length - n_hist + age}"]
                half = sprite.get_width() / 2
                blits.append((sprite, (int(round(tx - half)), int(round(ty - half)))))
            canvas.blits(blits, False)
            self._trail.append((bx, by))

        # Pistons, observers highlighted with a glow.
        housing = lay.housing_top
        tops = np.rint(piston_y - lay.piston_radius).astype(int)
        heights = np.clip(housing - tops, 0, self._piston_h)
        x0 = self._piston_x0
        g = self._GLOW
        glow = sprites["glow"]
        blits = [
            (
                glow,
                (int(x0[i]) - g, int(tops[i]) - g),
                (0, 0, glow.get_width(), int(heights[i]) + g),
            )
            for i in np.flatnonzero(observable).tolist()
        ]
        idle, active = sprites["piston"], sprites["active"]
        blits.extend(
            (active if obs else idle, (x, t), (0, 0, self._piston_w, h))
            for x, t, h, obs in zip(
                x0.tolist(), tops.tolist(), heights.tolist(), observable.tolist()
            )
        )
        canvas.blits(blits, False)
        canvas.blits([(label, pos) for i, _, label, pos in self._labels if observable[i]], False)

        # Ball with a rotating marker.
        ball_sprite = sprites["ball"]
        half = ball_sprite.get_width() / 2
        canvas.blit(ball_sprite, (int(round(bx - half)), int(round(by - half))))
        marker = pg.transform.rotozoom(sprites["marker"], -math.degrees(angle), 1.0)
        canvas.blit(marker, marker.get_rect(center=(int(round(bx)), int(round(by)))))

        # Distance-to-goal bar aligned with the arena.
        left, right = lay.arena_left, lay.arena_right
        ball_left = min(max(bx - lay.ball_radius, left), right)
        distance = max(bx - lay.ball_radius - left, 0.0)
        track_y = self._track_y
        fill = int(round(ball_left - left))
        if fill > 0:
            pg.draw.rect(canvas, pal.accent, (left, track_y, fill, 8), border_radius=4)
        mx = int(round(ball_left))
        pg.draw.circle(canvas, pal.marker, (mx, track_y + 4), 7)
        pg.draw.circle(canvas, pal.ball, (mx, track_y + 4), 5)
        self._draw_row(
            canvas,
            [("to goal ", pal.muted), (f"{distance:.0f} px", pal.text)],
            left,
            track_y + 26,
            font="small",
        )

        self._draw_hud(canvas, step, max_cycles, episode_return, velocity[0], observable, status)

    def _draw_row(
        self,
        canvas: Any,
        parts: Sequence[tuple[str, RGB]],
        x: int,
        baseline: int,
        font: str = "label",
        align: str = "left",
    ) -> int:
        """Blit text parts on one baseline; returns the total width."""
        surfaces = [self._text(font, text, color) for text, color in parts]
        total = sum(s.get_width() for s in surfaces)
        if align == "right":
            x -= total
        y = baseline - self._fonts[font].get_ascent()
        for surface in surfaces:
            canvas.blit(surface, (x, y))
            x += surface.get_width()
        return total

    def _draw_hud(
        self,
        canvas: Any,
        step: int,
        max_cycles: int,
        episode_return: float,
        ball_vx: float,
        observable: np.ndarray,
        status: str | None,
    ) -> None:
        pg = self._pg
        pal, lay = self.palette, self.layout
        m = self._MARGIN
        width = lay.screen_width
        title_right = m + 22 + self._fonts["title"].size("Pistonball")[0]

        step_parts = [("step ", pal.muted), (f"{step}/{max_cycles}", pal.text)]
        step_width = sum(self._fonts["label"].size(t)[0] for t, _ in step_parts)
        chips = [
            [("return ", pal.muted), (f"{episode_return:.2f}", pal.text)],
            [("ball vx ", pal.muted), (f"{ball_vx:+.1f} px/s", pal.text)],
            [("observing ", pal.muted), (f"{int(observable.sum())}/{lay.n_pistons}", pal.accent)],
        ]
        if width - m - step_width > title_right + 16:
            self._draw_row(canvas, step_parts, width - m, 36, align="right")
            free_right = width - m - step_width - 16
        else:
            chips.insert(0, step_parts)
            free_right = width - m

        if status is not None:
            color = pal.positive if status == "goal reached" else pal.warning
            text = self._text("small", status, color)
            pill_w, pill_h = text.get_width() + 16, text.get_height() + 6
            if title_right + 12 + pill_w <= free_right:
                rect = pg.Rect(title_right + 12, 36 - pill_h + 3, pill_w, pill_h)
                pg.draw.rect(
                    canvas, _mix(color, pal.background, 0.78), rect, border_radius=pill_h // 2
                )
                pg.draw.rect(canvas, color, rect, width=1, border_radius=pill_h // 2)
                canvas.blit(text, text.get_rect(center=rect.center))

        font = self._fonts["label"]
        gap = 18
        widths = [sum(font.size(t)[0] for t, _ in chip) for chip in chips]
        if sum(widths) + gap * (len(chips) - 1) > width - 2 * m:
            gap = 10
            short = {"return ": "R ", "ball vx ": "vx ", "observing ": "obs ", "step ": ""}
            chips = [[(short.get(t, t), c) for t, c in chip] for chip in chips]
        x = m
        for chip in chips:
            x += self._draw_row(canvas, chip, x, 62) + gap
