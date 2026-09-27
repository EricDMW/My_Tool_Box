"""Target motion policies for the AJLATT environment.

Each policy maps the true target pose ``(x, y, theta)`` to a unicycle command
``(v, omega)``. Policies are small stateful objects (one per environment
instance) with a :meth:`TargetPolicy.reset` hook, so several environments can
run side by side without sharing state.

Available policies (``AJLATTConfig.target_policy``):

``"auto"``
    Chosen from the map name: ``"sine"`` for ``empty``, the matching
    waypoint route for ``obstacles04`` / ``obstacles05``, ``"circle"``
    otherwise.
``"sine"``
    Follows the slope field ``dy/dx = magnitude * cos(freq * x)`` until
    ``x >= border``, then circles.
``"waypoints"``
    Heading control towards piecewise waypoints (``obstacles04``/``05`` routes
    are built in; custom routes via :class:`WaypointPolicy`).
``"circle"``
    Constant ``(v, omega) = (0.25, 0.15)``.
``"random"``
    Constant speed, uniformly random turn rate each step.
``"static"``
    The target does not move.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

__all__ = [
    "CirclePolicy",
    "RandomTurnPolicy",
    "SinePolicy",
    "StaticPolicy",
    "TargetPolicy",
    "WAYPOINT_ROUTES",
    "WaypointPolicy",
    "make_target_policy",
]

Segment = tuple[tuple[float, float], tuple[float, float]]

#: Built-in waypoint routes: ``[((x_min, x_max), (x_goal, y_goal)), ...]`` and nominal speed.
WAYPOINT_ROUTES = {
    "obstacles04": (
        (
            ((-np.inf, 20.0), (20.0, 10.0)),
            ((20.0, 22.5), (22.5, 15.5)),
            ((22.5, 32.5), (32.5, 15.8)),
            ((32.5, np.inf), (33.5, 32.5)),
        ),
        0.1,
    ),
    "obstacles05": (
        (
            ((-np.inf, 35.0), (35.0, 10.0)),
            ((35.0, 38.0), (38.0, 40.0)),
            ((38.0, 40.0), (40.0, 66.0)),
            ((40.0, np.inf), (69.0, 67.0)),
        ),
        0.25,
    ),
}


class TargetPolicy:
    """Base class of target motion policies."""

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Clear internal state at the start of an episode."""
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, pose: np.ndarray) -> tuple[float, float]:
        raise NotImplementedError


class StaticPolicy(TargetPolicy):
    """The target stays where it is."""

    def __call__(self, pose):
        return 0.0, 0.0


class CirclePolicy(TargetPolicy):
    """Constant forward and angular velocity."""

    def __init__(self, velocity: float = 0.25, omega: float = 0.15):
        self.velocity = velocity
        self.omega = omega

    def __call__(self, pose):
        return self.velocity, self.omega


class RandomTurnPolicy(TargetPolicy):
    """Constant speed with a random turn rate in ``(-omega_max, omega_max)``."""

    def __init__(self, velocity: float = 0.25, omega_max: float = np.pi / 4):
        self.velocity = velocity
        self.omega_max = omega_max
        self.rng = np.random.default_rng()

    def __call__(self, pose):
        return self.velocity, float(self.rng.uniform(-self.omega_max, self.omega_max))


class SinePolicy(TargetPolicy):
    """Track the slope field ``dy/dx = magnitude * cos(freq * x)``, then circle.

    This is the original policy for the ``empty`` map.
    """

    def __init__(
        self,
        velocity: float = 0.3,
        k_theta: float = 0.5,
        magnitude: float = 5.0,
        freq: float = 1.0,
        border: float = 32.0,
        circle_radius: float = 4.0,
        circle_velocity: float = 0.25,
    ):
        self.velocity = velocity
        self.k_theta = k_theta
        self.magnitude = magnitude
        self.freq = freq
        self.border = border
        self.circle_radius = circle_radius
        self.circle_velocity = circle_velocity
        self.circling = False

    def reset(self, rng=None):
        super().reset(rng)
        self.circling = False

    def __call__(self, pose):
        if pose[0] < self.border and not self.circling:
            slope = self.magnitude * np.cos(self.freq * pose[0])
            omega = self.k_theta * (np.arctan(slope) - pose[2])
            return self.velocity, float(omega)
        self.circling = True
        return self.circle_velocity, -self.circle_velocity / self.circle_radius


class WaypointPolicy(TargetPolicy):
    """Proportional heading control towards the waypoint of the active x-segment.

    Parameters
    ----------
    segments:
        ``[((x_min, x_max), (x_goal, y_goal)), ...]``; the first segment whose
        ``x_min <= x < x_max`` is active.
    velocity:
        Nominal forward speed.
    k_theta:
        Heading gain.
    """

    def __init__(self, segments: Sequence[Segment], velocity: float, k_theta: float = 0.5):
        self.segments = [
            ((float(a), float(b)), (float(x), float(y))) for (a, b), (x, y) in segments
        ]
        self.velocity = velocity
        self.k_theta = k_theta

    def __call__(self, pose):
        x, y, theta = pose
        for (x_min, x_max), (x_goal, y_goal) in self.segments:
            if x_min <= x < x_max:
                dx, dy = x_goal - x, y_goal - y
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    return 0.0, 0.0
                error = (np.arctan2(dy, dx) - theta + np.pi) % (2 * np.pi) - np.pi
                return self.velocity, float(self.k_theta * error)
        return 0.0, 0.0


def make_target_policy(
    name: str,
    map_name: str = "",
    *,
    velocity: float = 0.3,
    k_theta: float = 0.5,
    omega_max: float = np.pi / 4,
) -> TargetPolicy:
    """Create a target policy by name (see the module docstring).

    A :class:`TargetPolicy` instance is returned unchanged, so custom policies
    (e.g. a :class:`WaypointPolicy` with your own route) can be passed as
    ``AJLATTConfig.target_policy``.
    """
    if isinstance(name, TargetPolicy):
        return name
    map_name = Path(str(map_name)).name
    for suffix in (".yaml", ".cfg"):
        map_name = map_name[: -len(suffix)] if map_name.endswith(suffix) else map_name
    if name == "auto":
        if map_name == "empty":
            name = "sine"
        elif map_name in WAYPOINT_ROUTES:
            name = "waypoints"
        else:
            name = "circle"
    if name == "sine":
        return SinePolicy(velocity=velocity, k_theta=k_theta)
    if name == "waypoints":
        if map_name not in WAYPOINT_ROUTES:
            raise ValueError(
                f"No built-in waypoint route for map {map_name!r}; available: "
                f"{sorted(WAYPOINT_ROUTES)}. Pass a WaypointPolicy instance instead."
            )
        segments, speed = WAYPOINT_ROUTES[map_name]
        return WaypointPolicy(segments, speed, k_theta=k_theta)
    if name == "circle":
        return CirclePolicy()
    if name == "random":
        return RandomTurnPolicy(omega_max=omega_max)
    if name == "static":
        return StaticPolicy()
    raise ValueError(
        f"Unknown target policy {name!r}; use 'auto', 'sine', 'waypoints', 'circle', "
        "'random' or 'static'"
    )
