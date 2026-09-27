"""Configuration of the AJLATT environment.

:class:`AJLATTConfig` gathers every scenario, sensing, noise, reward and
simulation parameter in one validated dataclass. Legacy keyword names from the
original argparse-based configuration (``num_Robot``, ``T_steps``,
``SIGPERCENT``, ...) are still accepted by :meth:`AJLATTConfig.from_kwargs`.

For command-line experiments, :meth:`AJLATTConfig.add_arguments` registers all
fields on an ``argparse.ArgumentParser`` and :meth:`AJLATTConfig.from_namespace`
builds a config from the parsed arguments.
"""

from __future__ import annotations

import argparse
import dataclasses
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

__all__ = ["DEFAULT_ROBOT_POSES", "DEFAULT_TARGET_POSES", "AJLATTConfig"]

DEFAULT_ROBOT_POSES: tuple[tuple[float, float, float], ...] = (
    (7.5, 12.5, 0.0),
    (7.5, 10.5, 0.0),
    (10.0, 12.5, 0.0),
    (10.0, 10.5, 0.0),
    (80.0, 10.0, -np.pi),
    (40.0, -20.0, np.pi / 2),
)

DEFAULT_TARGET_POSES: tuple[tuple[float, float, float], ...] = (
    (10.5, 11.5, 0.0),
    (13.5, 6.0, 0.0),
    (20.0, 10.0, 0.0),
    (10.0, 3.0, 0.0),
    (15.0, 23.0, 0.0),
    (25.0, 3.0, 0.0),
    (30.0, 20.0, 0.0),
    (10.0, 15.0, 0.0),
    (20.0, 20.0, 0.0),
    (25.0, 10.0, 0.0),
    (30.0, 10.0, 0.0),
    (10.0, 10.0, 0.0),
    (12.0, 30.0, 0.0),
)

# Legacy name -> new field name.
_ALIASES: dict[str, str] = {
    "num_Robot": "num_robots",
    "num_Target": "num_targets",
    "T_steps": "max_episode_steps",
    "SIGPERCENT": "range_noise_proportional",
    "useupdate": "use_update",
}

# Legacy parameters that no longer have an effect.
_IGNORED = frozenset(
    {
        "version",
        "episode_num",
        "env_name",
        "record",
        "ros",
        "log_dir",
        "repeat",
        "im_size",
        "robot_est_init_pos",
        "figID",
        "ssh_debug",
        "model_name",
        "cen_decen_framework",
        "linear_speed",
        "dialogue_history_method",
        "save_path",
        "is_training",
        "known_noise",
    }
)


@dataclass
class AJLATTConfig:
    """Parameters of the AJLATT environment (defaults reproduce the original setup).

    Scenario
    --------
    map_name: Bundled map name (see ``env_lib.ajlatt_env.maps.available_maps()``) or a
        path to a user map (``.yaml`` header next to a ``.cfg`` grid).
    num_robots, num_targets: Team size and number of targets. Target 0 moves; the
        others are static landmarks with 2-D (position-only) estimates.
    max_episode_steps: Episode length (``truncated`` is set afterwards).
    sampling_period: Duration of one step in seconds.
    margin2wall: Safety margin used by the map queries.
    robot_init_pose, target_init_pose: ``(x, y, theta)`` per robot / target.
    robot_init_cov, target_init_cov: Initial position variance per robot / target
        (heading variance is fixed to 1e-3 for robots and 0.1 for target 0).

    Sensing and communication
    -------------------------
    sensor_r_min, sensor_r_max, fov: Range limits (m) and field of view (degrees).
    sigma_p: Range noise as a fraction of the range (``range_noise_proportional``).
    sigma_r: Absolute range noise when ``range_noise_proportional`` is False.
    sigma_th: Bearing noise (rad).
    use_update: Run measurement updates (False = dead reckoning only).
    commu_r_max: Communication range.
    ci_solver: Covariance-intersection weight solver, ``"newton"`` or ``"slsqp"``.

    Motion noise
    ------------
    process_noise_fixed: Use the fixed velocity noise below; otherwise the noise
        scales with the commanded speed (``sigmapR``, ``sigmapT``).
    sigma_vR, sigma_wR, sigmapR: Robot velocity noise.
    sigma_vT, sigma_wT, sigmapT: Target velocity noise (used by the estimators).

    Target motion
    -------------
    target_policy: ``"auto"``, ``"sine"``, ``"waypoints"``, ``"circle"``, ``"random"``
        or ``"static"`` (see :mod:`env_lib.ajlatt_env.controllers`).
    target_linear_velocity: Speed of the ``sine`` policy (and of the speed-dependent
        target noise); the other policies use their own fixed speeds.
    target_omega_bound: Turn-rate bound of the ``random`` policy.
    k_theta: Heading gain of the ``sine`` and ``waypoints`` policies.
    k_p: Unused; kept for compatibility with the original parameter set.

    Actions
    -------
    max_linear_velocity, max_angular_velocity: Action-space bounds
        ``v in [0, max_linear_velocity]``, ``omega in [-max_angular_velocity, ...]``.
    clip_actions: Clip actions to the action space (the original did not).

    Rewards
    -------
    target_cov_weight, robot_cov_weight: Weights of ``trace`` of the target and
        self-localisation covariances.
    boundary_penalty: Penalty when the estimated pose leaves the map (0.1 m margin).
    obstacle_penalty, obstacle_collision_distance: Penalty (and per-agent
        termination) when an obstacle is closer than the distance.
    mutual_collision_penalty, mutual_collision_distance: Robot-robot proximity penalty.
    terminate_on_collision: Report obstacle collisions in ``terminated``.
    obstacle_sensing_margin: Robots closer than this to the map border report a
        zero obstacle reading (original behaviour).

    Misc
    ----
    seed: Default seed for the first ``reset()``.
    """

    # Scenario
    map_name: str = "obstacles04"
    num_robots: int = 4
    num_targets: int = 1
    max_episode_steps: int = 120
    sampling_period: float = 0.5
    margin2wall: float = 1.0
    robot_init_pose: Sequence[Sequence[float]] = DEFAULT_ROBOT_POSES
    target_init_pose: Sequence[Sequence[float]] = DEFAULT_TARGET_POSES
    robot_init_cov: Sequence[float] = (0.1,) * 6
    target_init_cov: Sequence[float] = (1.0,) + (10.0,) * 10

    # Sensing and communication
    sensor_r_min: float = 0.0
    sensor_r_max: float = 3.0
    fov: float = 180.0
    sigma_p: float = 0.0
    sigma_r: float = 0.0
    sigma_th: float = 0.0
    range_noise_proportional: bool = True
    use_update: bool = True
    commu_r_max: float = 6.0
    ci_solver: str = "newton"

    # Motion noise
    process_noise_fixed: bool = True
    sigmapR: float = 0.1
    sigma_vR: float = 0.1
    sigma_wR: float = 0.5 / 180.0 * np.pi
    sigmapT: float = 0.1
    sigma_vT: float = 0.2
    sigma_wT: float = 0.5 / 180.0 * np.pi

    # Target motion
    target_policy: str = "auto"
    target_linear_velocity: float = 0.3
    target_omega_bound: float = np.pi / 4
    k_theta: float = 0.5
    k_p: float = 1.0

    # Actions
    max_linear_velocity: float = 2.0
    max_angular_velocity: float = np.pi / 4
    clip_actions: bool = False

    # Rewards and termination
    target_cov_weight: float = 10.0
    robot_cov_weight: float = 5.0
    boundary_penalty: float = 20.0
    obstacle_penalty: float = 1.3
    obstacle_collision_distance: float = 0.2
    mutual_collision_penalty: float = 0.5
    mutual_collision_distance: float = 0.4
    terminate_on_collision: bool = True
    obstacle_sensing_margin: float = 1.0

    # Misc
    seed: int | None = None

    def __post_init__(self) -> None:
        # Store poses and variances as immutable tuples (lists and arrays accepted).
        self.robot_init_pose = tuple(
            tuple(float(v) for v in pose)
            for pose in np.asarray(self.robot_init_pose, dtype=float).reshape(-1, 3)
        )
        self.target_init_pose = tuple(
            tuple(float(v) for v in pose)
            for pose in np.asarray(self.target_init_pose, dtype=float).reshape(-1, 3)
        )
        self.robot_init_cov = tuple(float(v) for v in np.ravel(self.robot_init_cov))
        self.target_init_cov = tuple(float(v) for v in np.ravel(self.target_init_cov))
        self.validate()

    # -- validation ---------------------------------------------------------
    def validate(self) -> None:
        """Raise ``ValueError`` if a parameter is out of range."""
        if self.num_robots < 1:
            raise ValueError("num_robots must be >= 1")
        if self.num_targets < 1:
            raise ValueError("num_targets must be >= 1")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be >= 1")
        if self.sampling_period <= 0:
            raise ValueError("sampling_period must be positive")
        if not 0 <= self.sensor_r_min < self.sensor_r_max:
            raise ValueError("require 0 <= sensor_r_min < sensor_r_max")
        if not 0 < self.fov <= 360:
            raise ValueError("fov must be in (0, 360] degrees")
        if self.commu_r_max < 0:
            raise ValueError("commu_r_max must be >= 0")
        if self.ci_solver not in ("newton", "slsqp"):
            raise ValueError("ci_solver must be 'newton' or 'slsqp'")
        for name in (
            "sigma_p",
            "sigma_r",
            "sigma_th",
            "sigma_vR",
            "sigma_wR",
            "sigma_vT",
            "sigma_wT",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        robots = np.asarray(self.robot_init_pose, dtype=float)
        targets = np.asarray(self.target_init_pose, dtype=float)
        if robots.ndim != 2 or robots.shape[1] != 3 or len(robots) < self.num_robots:
            raise ValueError(
                f"robot_init_pose must provide (x, y, theta) for at least {self.num_robots} robots"
            )
        if targets.ndim != 2 or targets.shape[1] != 3 or len(targets) < self.num_targets:
            raise ValueError(
                f"target_init_pose must provide (x, y, theta) for at least {self.num_targets} targets"
            )
        if len(self.robot_init_cov) < self.num_robots:
            raise ValueError(f"robot_init_cov needs at least {self.num_robots} values")
        if len(self.target_init_cov) < self.num_targets:
            raise ValueError(f"target_init_cov needs at least {self.num_targets} values")

    # -- derived quantities --------------------------------------------------
    @property
    def observation_dim(self) -> int:
        """Per-robot observation size: ``7 + 4 (num_robots - 1) + 5``."""
        return 7 + 4 * (self.num_robots - 1) + 5

    def robot_poses(self) -> np.ndarray:
        return np.asarray(self.robot_init_pose, dtype=float)[: self.num_robots].copy()

    def target_poses(self) -> np.ndarray:
        return np.asarray(self.target_init_pose, dtype=float)[: self.num_targets].copy()

    def replace(self, **changes: Any) -> AJLATTConfig:
        """Return a copy with some fields replaced (validated)."""
        return dataclasses.replace(self, **changes)

    # -- construction helpers -----------------------------------------------
    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def from_kwargs(cls, strict: bool = True, **kwargs: Any) -> AJLATTConfig:
        """Build a config from keyword arguments, accepting legacy names.

        Parameters
        ----------
        strict:
            Raise ``TypeError`` for unknown keys (otherwise warn and ignore).
        **kwargs:
            Field values. Legacy names (``num_Robot``, ``num_Target``,
            ``T_steps``, ``maxmum_run``, ``SIGPERCENT``, ``useupdate``,
            ``fix_seed``) are translated with a
            ``DeprecationWarning``; obsolete keys (``figID``, ``ssh_debug``,
            ``episode_num``, ...) are ignored with a warning.
        """
        known = set(cls.field_names())
        values: dict[str, Any] = {}
        legacy_limit: int | None = None
        fix_seed = kwargs.pop("fix_seed", None)
        if fix_seed is not None:
            warnings.warn(
                "AJLATT parameter 'fix_seed' is deprecated; pass seed=<int> (or reset(seed=...))",
                DeprecationWarning,
                stacklevel=3,
            )
        for key, value in kwargs.items():
            if key in known:
                values[key] = value
            elif key in _ALIASES:
                new = _ALIASES[key]
                warnings.warn(
                    f"AJLATT parameter {key!r} is deprecated; use {new!r}",
                    DeprecationWarning,
                    stacklevel=3,
                )
                values.setdefault(new, value)
            elif key == "maxmum_run":
                legacy_limit = int(value)
            elif key in _IGNORED:
                warnings.warn(
                    f"AJLATT parameter {key!r} no longer has any effect and is ignored",
                    DeprecationWarning,
                    stacklevel=3,
                )
            elif strict:
                raise TypeError(f"Unknown AJLATT parameter {key!r}")
            else:
                warnings.warn(f"Ignoring unknown AJLATT parameter {key!r}", stacklevel=3)
        for key in ("range_noise_proportional", "use_update", "process_noise_fixed"):
            if key in values:
                values[key] = bool(values[key])
        if fix_seed is not None:
            if fix_seed:
                values.setdefault("seed", 1)
            else:
                values.pop("seed", None)
        if legacy_limit is not None:
            values["max_episode_steps"] = min(
                int(values.get("max_episode_steps", cls.max_episode_steps)), legacy_limit + 1
            )
        return cls(**values)

    @classmethod
    def add_arguments(
        cls, parser: argparse.ArgumentParser, prefix: str = ""
    ) -> argparse.ArgumentParser:
        """Register every scalar field as ``--<prefix><name>`` on ``parser``."""
        defaults = cls()
        for f in fields(cls):
            value = getattr(defaults, f.name)
            flag = f"--{prefix}{f.name}"
            if isinstance(value, bool):
                parser.add_argument(
                    flag, type=_str2bool, default=value, metavar="BOOL", help=f"(default: {value})"
                )
            elif isinstance(value, (int, float, str)) or value is None:
                kind = type(value) if value is not None else int
                parser.add_argument(flag, type=kind, default=value, help=f"(default: {value})")
        return parser

    @classmethod
    def from_namespace(
        cls, namespace: argparse.Namespace, prefix: str = "", strict: bool = False
    ) -> AJLATTConfig:
        """Build a config from parsed arguments (fields not present keep defaults)."""
        values = {}
        for name in cls.field_names():
            key = f"{prefix}{name}"
            if hasattr(namespace, key):
                values[name] = getattr(namespace, key)
        return cls.from_kwargs(strict=strict, **values)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict representation (tuples of poses converted to lists)."""
        out = {}
        for name in self.field_names():
            value = getattr(self, name)
            out[name] = (
                np.asarray(value).tolist()
                if isinstance(value, (tuple, list, np.ndarray))
                else value
            )
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], strict: bool = True) -> AJLATTConfig:
        """Inverse of :meth:`to_dict` (legacy names accepted)."""
        return cls.from_kwargs(strict=strict, **dict(data))


def _str2bool(text: str) -> bool:
    lowered = str(text).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {text!r}")
