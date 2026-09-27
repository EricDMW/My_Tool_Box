"""Cooperative adaptive cruise control (CACC) of a vehicle platoon.

A leader vehicle is followed by ``n`` follower vehicles on a single lane. The
followers are the agents: each commands its own acceleration and must keep a
speed-dependent distance to its predecessor (constant time-headway spacing
policy) while the leader drives an exogenous speed profile (cruising,
stop-and-go waves, random speed changes). The followers exchange data over a
vehicle-to-vehicle (V2V) communication topology; which communicated quantities
an agent observes depends on that topology.

The benchmark reproduces the central phenomenon of platoon control, *string
(in)stability*: with sensor-only adaptive cruise control (ACC) and a short
time headway, spacing errors grow from vehicle to vehicle along the platoon
until vehicles collide, while cooperative ACC with the communicated predecessor
acceleration attenuates them.

The dynamics are written once for a batch of platoons (leading batch
dimension ``B``); :class:`PlatoonEnv` simulates ``B = 1`` and
:class:`PlatoonVectorEnv` ``B = num_envs`` copies with the same code.
:func:`cacc_policy` is the decentralised linear CACC baseline and
:func:`string_stability_gain` evaluates the string-stability transfer function
of the closed loop.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env_lib.errors import ResetNeededError
from env_lib.utils.rendering import state_without_renderer, validate_render_mode
from env_lib.utils.vector import BatchedVectorEnv

__all__ = [
    "CACC_GAINS",
    "FEATURE_SCALES",
    "OBSERVATION_FEATURES",
    "SCENARIOS",
    "TOPOLOGIES",
    "PlatoonConfig",
    "PlatoonEnv",
    "PlatoonVectorEnv",
    "cacc_policy",
    "platoon_adjacency",
    "string_stability_gain",
]

#: Communication topologies: predecessor following (PF), predecessor-leader
#: following (PLF), bidirectional (BD) and no communication (sensors only).
TOPOLOGIES: tuple[str, ...] = ("predecessor", "predecessor_leader", "bidirectional", "none")

#: Leader speed profiles. ``"mixed"`` starts with a stop-and-go wave and then
#: draws every further event uniformly from the three basic event types.
SCENARIOS: tuple[str, ...] = ("mixed", "cruise", "stop_and_go", "random")

#: Features of one observation row, in column order.
OBSERVATION_FEATURES: tuple[str, ...] = (
    "spacing_error",
    "relative_speed",
    "speed",
    "acceleration",
    "predecessor_available",
    "predecessor_acceleration",
    "predecessor_command",
    "leader_available",
    "leader_speed_error",
    "leader_acceleration_error",
    "follower_available",
    "follower_spacing_error",
    "follower_relative_speed",
)

_SPACING_SCALE = 5.0  # m
_SPEED_DIFF_SCALE = 5.0  # m/s
_SPEED_SCALE = 30.0  # m/s
_ACCEL_SCALE = 3.0  # m/s^2

#: Physical unit of every observation feature: ``obs = physical value / scale``
#: (availability flags are 0 or 1).
FEATURE_SCALES: dict[str, float] = {
    "spacing_error": _SPACING_SCALE,
    "relative_speed": _SPEED_DIFF_SCALE,
    "speed": _SPEED_SCALE,
    "acceleration": _ACCEL_SCALE,
    "predecessor_available": 1.0,
    "predecessor_acceleration": _ACCEL_SCALE,
    "predecessor_command": _ACCEL_SCALE,
    "leader_available": 1.0,
    "leader_speed_error": _SPEED_DIFF_SCALE,
    "leader_acceleration_error": _ACCEL_SCALE,
    "follower_available": 1.0,
    "follower_spacing_error": _SPACING_SCALE,
    "follower_relative_speed": _SPEED_DIFF_SCALE,
}

#: Default gains of :func:`cacc_policy` (k_p in 1/s^2, k_d in 1/s, k_a dimensionless).
CACC_GAINS: dict[str, float] = {"k_p": 0.5, "k_d": 0.6, "k_a": 0.65}

_OBS_DIM = len(OBSERVATION_FEATURES)
_COL = {name: k for k, name in enumerate(OBSERVATION_FEATURES)}

# Leader profile generator. Event types: 0 cruise, 1 stop-and-go, 2 random.
# Every event has two segments (ramp to a target speed, then hold it). Tables
# are indexed [event type, segment]; targets marked relative are offsets from
# the initial cruise speed.
_INITIAL_SPEED = (16.0, 25.0)  # m/s, initial cruise speed of the platoon
_INITIAL_HOLD = (2.0, 5.0)  # s, cruise before the first event
_TARGET_LO = np.array([[-2.0, -2.0], [0.0, -3.0], [5.0, 5.0]])
_TARGET_HI = np.array([[2.0, 2.0], [4.0, 3.0], [30.0, 30.0]])
_TARGET_RELATIVE = np.array([[True, True], [False, True], [False, False]])
_RATE_LO = np.array([[0.3, 0.3], [2.0, 1.0], [0.5, 0.5]])  # m/s^2
_RATE_HI = np.array([[0.8, 0.8], [4.0, 2.0], [2.5, 2.5]])
_HOLD_LO = np.array([[4.0, 4.0], [1.0, 4.0], [1.0, 1.0]])  # s
_HOLD_HI = np.array([[10.0, 10.0], [5.0, 10.0], [6.0, 6.0]])
_MIN_EVENT_TIME = float((_HOLD_LO.sum(axis=1)).min())
_SCENARIO_EVENT = {"cruise": 0, "stop_and_go": 1, "random": 2}

_NOISE_CLIP = 3.0  # initial perturbations are truncated at 3 standard deviations
_PEAK_FLOOR = 0.01  # m, floor of the first follower's peak error in error_amplification

_OPTION_KEYS = (
    "positions",
    "velocities",
    "accelerations",
    "leader_command",
    "time_constants",
    "lengths",
)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
def _check_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _check_float(
    name: str, value: Any, *, minimum: float = -math.inf, strict: bool = False
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < minimum or (strict and value == minimum):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be {relation} {minimum}, got {value}")
    return value


def _check_range(name: str, value: Any, *, minimum: float) -> tuple[float, float]:
    try:
        low, high = value
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a (low, high) pair, got {value!r}") from None
    low = _check_float(f"{name}[0]", low, minimum=minimum, strict=True)
    high = _check_float(f"{name}[1]", high, minimum=minimum, strict=True)
    if high < low:
        raise ValueError(f"{name} must satisfy low <= high, got {value!r}")
    return low, high


def _check_choice(name: str, value: Any, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return str(value)


@dataclass(frozen=True)
class PlatoonConfig:
    """Validated, immutable parameters of a platoon environment.

    See :class:`PlatoonEnv` for the meaning of every field.
    """

    n_followers: int = 8
    topology: str = "predecessor"
    scenario: str = "mixed"
    dt: float = 0.1
    max_steps: int = 600
    headway: float = 0.6
    standstill_distance: float = 2.0
    accel_min: float = -6.0
    accel_max: float = 3.0
    tau_range: tuple[float, float] = (0.2, 0.4)
    length_range: tuple[float, float] = (4.0, 5.0)
    vehicle_seed: int | None = 0
    randomize_vehicles: bool = False
    init_spacing_noise: float = 0.1
    init_speed_noise: float = 0.05
    spacing_weight: float = 1.0
    speed_weight: float = 0.1
    control_weight: float = 0.02
    jerk_weight: float = 0.01
    collision_penalty: float = 500.0
    breakup_distance: float = 100.0
    breakup_penalty: float = 250.0

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> PlatoonConfig:
        """Validate constructor arguments and build a configuration.

        Raises
        ------
        TypeError
            For unknown arguments or arguments of the wrong type.
        ValueError
            For out-of-range values.
        """
        known = set(cls.__dataclass_fields__)
        unknown = set(kwargs) - known
        if unknown:
            raise TypeError(f"unexpected argument(s) {sorted(unknown)}; valid: {sorted(known)}")
        d = {**{k: f.default for k, f in cls.__dataclass_fields__.items()}, **kwargs}
        seed = d["vehicle_seed"]
        if seed is not None:
            seed = _check_int("vehicle_seed", seed, 0)
        if not isinstance(d["randomize_vehicles"], (bool, np.bool_)):
            raise TypeError("randomize_vehicles must be a bool")
        config = cls(
            n_followers=_check_int("n_followers", d["n_followers"], 1),
            topology=_check_choice("topology", d["topology"], TOPOLOGIES),
            scenario=_check_choice("scenario", d["scenario"], SCENARIOS),
            dt=_check_float("dt", d["dt"], minimum=0.0, strict=True),
            max_steps=_check_int("max_steps", d["max_steps"], 1),
            headway=_check_float("headway", d["headway"], minimum=0.0),
            standstill_distance=_check_float(
                "standstill_distance", d["standstill_distance"], minimum=0.0, strict=True
            ),
            accel_min=_check_float("accel_min", d["accel_min"]),
            accel_max=_check_float("accel_max", d["accel_max"]),
            tau_range=_check_range("tau_range", d["tau_range"], minimum=0.0),
            length_range=_check_range("length_range", d["length_range"], minimum=0.0),
            vehicle_seed=seed,
            randomize_vehicles=bool(d["randomize_vehicles"]),
            init_spacing_noise=_check_float(
                "init_spacing_noise", d["init_spacing_noise"], minimum=0.0
            ),
            init_speed_noise=_check_float("init_speed_noise", d["init_speed_noise"], minimum=0.0),
            spacing_weight=_check_float("spacing_weight", d["spacing_weight"], minimum=0.0),
            speed_weight=_check_float("speed_weight", d["speed_weight"], minimum=0.0),
            control_weight=_check_float("control_weight", d["control_weight"], minimum=0.0),
            jerk_weight=_check_float("jerk_weight", d["jerk_weight"], minimum=0.0),
            collision_penalty=_check_float(
                "collision_penalty", d["collision_penalty"], minimum=0.0
            ),
            breakup_distance=_check_float(
                "breakup_distance", d["breakup_distance"], minimum=0.0, strict=True
            ),
            breakup_penalty=_check_float("breakup_penalty", d["breakup_penalty"], minimum=0.0),
        )
        if not config.accel_min < 0.0 < config.accel_max:
            raise ValueError(
                f"need accel_min < 0 < accel_max, got [{config.accel_min}, {config.accel_max}]"
            )
        max_gap = config.standstill_distance + config.headway * _INITIAL_SPEED[1]
        if config.breakup_distance <= max_gap + _NOISE_CLIP * config.init_spacing_noise:
            raise ValueError(
                f"breakup_distance={config.breakup_distance} m must exceed the largest initial "
                f"gap (about {max_gap + _NOISE_CLIP * config.init_spacing_noise:.1f} m)"
            )
        return config


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def _lag_coefficients(tau: np.ndarray, dt: float) -> tuple[np.ndarray, ...]:
    """Exact zero-order-hold discretisation of ``p' = v, v' = a, a' = (u - a) / tau``.

    Returns ``(alpha, c_va, c_vu, c_pa, c_pu)`` such that over one step with a
    constant command ``u``::

        a+ = alpha a + (1 - alpha) u
        v+ = v + c_va a + c_vu u
        p+ = p + dt v + c_pa a + c_pu u
    """
    tau = np.asarray(tau, dtype=np.float64)
    alpha = np.exp(-dt / tau)
    c_va = tau * (1.0 - alpha)
    c_vu = dt - c_va
    c_pa = tau * dt - tau * c_va
    c_pu = 0.5 * dt * dt - c_pa
    return alpha, c_va, c_vu, c_pa, c_pu


def platoon_adjacency(topology: str, n_followers: int) -> tuple[np.ndarray, np.ndarray]:
    """V2V communication graph of a platoon.

    Parameters
    ----------
    topology:
        One of :data:`TOPOLOGIES`.
    n_followers:
        Number of followers ``n`` (``>= 1``).

    Returns
    -------
    adjacency : numpy.ndarray
        Boolean ``(n, n)`` matrix over the followers; ``adjacency[i, j]`` is
        ``True`` when follower ``i`` receives data from follower ``j``
        (rows are receivers). ``"predecessor"`` and ``"predecessor_leader"``
        link every follower to its predecessor (``j = i - 1``);
        ``"bidirectional"`` adds the link from the follower behind
        (``j = i + 1``); ``"none"`` has no links.
    leader_links : numpy.ndarray
        Boolean ``(n,)`` vector; ``leader_links[i]`` is ``True`` when follower
        ``i`` receives data from the leader: the first follower (whose
        predecessor is the leader) for ``"predecessor"`` and
        ``"bidirectional"``, every follower for ``"predecessor_leader"``.
    """
    topology = _check_choice("topology", topology, TOPOLOGIES)
    n = _check_int("n_followers", n_followers, 1)
    adjacency = np.zeros((n, n), dtype=bool)
    leader_links = np.zeros(n, dtype=bool)
    idx = np.arange(1, n)
    if topology != "none":
        adjacency[idx, idx - 1] = True
        leader_links[0] = True
    if topology == "bidirectional":
        adjacency[idx - 1, idx] = True
    if topology == "predecessor_leader":
        leader_links[:] = True
    return adjacency, leader_links


def cacc_policy(
    obs: Any,
    *,
    k_p: float = CACC_GAINS["k_p"],
    k_d: float = CACC_GAINS["k_d"],
    k_a: float = CACC_GAINS["k_a"],
    accel_min: float = -6.0,
    accel_max: float = 3.0,
) -> np.ndarray:
    """Decentralised linear CACC baseline, a pure function of the observation.

    Each follower computes from its own observation row

    ``u_i = k_p e_i + k_d (v_{i-1} - v_i) + k_a a_{i-1}``

    where ``a_{i-1}`` is the communicated predecessor acceleration. Followers
    without a predecessor link (``topology="none"``) receive no acceleration
    and the law reduces to sensor-only ACC; ``k_a=0`` gives ACC for every
    topology. The result is clipped to ``[accel_min, accel_max]``.

    The default gains are string stable (:func:`string_stability_gain` equals
    one) for the default time headway of 0.6 s, ``dt = 0.1`` s and actuator
    lags up to about 0.43 s, which covers the default range [0.2, 0.4] s; ACC
    with the same ``k_p`` and ``k_d`` needs a headway of about 1.2 s.

    Parameters
    ----------
    obs:
        Observation(s) of shape ``(..., n_followers, obs_dim)`` in the layout
        of :data:`OBSERVATION_FEATURES` (any leading batch shape).
    k_p, k_d, k_a:
        Spacing-error, relative-speed and acceleration feed-forward gains.
    accel_min, accel_max:
        Command bounds in m/s^2.

    Returns
    -------
    numpy.ndarray
        Commanded accelerations of shape ``(..., n_followers)``, ``float32``.

    Examples
    --------
    >>> env = PlatoonEnv()
    >>> obs, _ = env.reset(seed=0)
    >>> cacc_policy(obs).shape
    (8,)
    """
    array = np.asarray(obs, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] != _OBS_DIM:
        raise ValueError(f"obs must have shape (..., n_followers, {_OBS_DIM}), got {np.shape(obs)}")
    spacing = array[..., _COL["spacing_error"]] * _SPACING_SCALE
    rel_speed = array[..., _COL["relative_speed"]] * _SPEED_DIFF_SCALE
    pred_accel = (
        array[..., _COL["predecessor_acceleration"]]
        * _ACCEL_SCALE
        * array[..., _COL["predecessor_available"]]
    )
    command = k_p * spacing + k_d * rel_speed + k_a * pred_accel
    return np.clip(command, accel_min, accel_max).astype(np.float32)


def string_stability_gain(
    k_p: float = CACC_GAINS["k_p"],
    k_d: float = CACC_GAINS["k_d"],
    k_a: float = CACC_GAINS["k_a"],
    *,
    headway: float = 0.6,
    time_constant: float = 0.3,
    dt: float = 0.1,
    n_frequencies: int = 2000,
) -> float:
    """Peak gain of the string-stability transfer function of the linear controller.

    For identical vehicles with the exact discretisation used by the
    environment and the (unsaturated) law of :func:`cacc_policy`, the
    command of follower ``i`` relates to that of its predecessor by
    ``U_i(z) = Gamma(z) U_{i-1}(z)`` with

    ``Gamma = (k_p H_p + k_d H_v + k_a H_a) / (1 + k_p H_p + (k_p h + k_d) H_v)``,

    where ``H_p``, ``H_v`` and ``H_a`` are the transfer functions from the
    command to position, speed and acceleration of one vehicle. The same
    ratio holds for positions, speeds and accelerations. The platoon is
    string stable when ``max_w |Gamma(exp(j w dt))| <= 1``: disturbances are
    not amplified along the platoon. ``Gamma(1) = 1``, so the peak is never
    below one.

    Parameters
    ----------
    k_p, k_d, k_a:
        Controller gains.
    headway:
        Time headway ``h`` of the spacing policy (s).
    time_constant:
        Actuator lag ``tau`` (s).
    dt:
        Sampling time (s).
    n_frequencies:
        Number of log-spaced frequencies between ``1e-3`` rad/s and the
        Nyquist frequency ``pi / dt``.

    Returns
    -------
    float
        ``max_w |Gamma|``, or ``inf`` if the vehicle's own control loop is
        unstable (the ratio is then meaningless).
    """
    h = _check_float("headway", headway, minimum=0.0)
    tau = _check_float("time_constant", time_constant, minimum=0.0, strict=True)
    dt = _check_float("dt", dt, minimum=0.0, strict=True)
    n_frequencies = _check_int("n_frequencies", n_frequencies, 2)
    alpha, c_va, c_vu, c_pa, c_pu = (float(c) for c in _lag_coefficients(np.array(tau), dt))
    # Closed loop of one vehicle following a fixed predecessor.
    a_mat = np.array([[1.0, dt, c_pa], [0.0, 1.0, c_va], [0.0, 0.0, alpha]])
    b_vec = np.array([c_pu, c_vu, 1.0 - alpha])
    gain = np.array([-k_p, -(k_p * h + k_d), 0.0])
    if np.max(np.abs(np.linalg.eigvals(a_mat + np.outer(b_vec, gain)))) >= 1.0:
        return math.inf
    omega = np.logspace(-3.0, math.log10(math.pi / dt), n_frequencies)
    z = np.exp(1j * omega * dt)
    h_a = (1.0 - alpha) / (z - alpha)
    h_v = (c_va * h_a + c_vu) / (z - 1.0)
    h_p = (dt * h_v + c_pa * h_a + c_pu) / (z - 1.0)
    ratio = (k_p * h_p + k_d * h_v + k_a * h_a) / (1.0 + k_p * h_p + (k_p * h + k_d) * h_v)
    return float(np.max(np.abs(ratio)))


def _sample_leader_commands(
    rng: np.random.Generator,
    scenario: str,
    start_speed: np.ndarray,
    n_steps: int,
    dt: float,
) -> np.ndarray:
    """Per-step commanded accelerations of the leader, shape ``(m, n_steps)``.

    The profile is a sequence of events of two segments each; a segment ramps
    the reference speed to a target with constant acceleration and then holds
    it. Segment boundaries are aligned with the time grid (the ramp rate is
    adjusted so that the target is met exactly). After the last event the
    reference speed is held.
    """
    m = start_speed.shape[0]
    n_events = int(math.ceil(n_steps * dt / _MIN_EVENT_TIME)) + 1
    if scenario == "mixed":
        types = rng.integers(0, 3, size=(m, n_events))
        types[:, 0] = _SCENARIO_EVENT["stop_and_go"]
    else:
        types = np.full((m, n_events), _SCENARIO_EVENT[scenario])
    draws = rng.random((m, n_events, 2, 3))
    initial_hold = rng.uniform(*_INITIAL_HOLD, size=m)

    kind, seg = types[..., None], np.arange(2)  # (m, E, 1) and (2,) -> (m, E, 2)
    lo, hi = _TARGET_LO[kind, seg], _TARGET_HI[kind, seg]
    targets = lo + (hi - lo) * draws[..., 0]
    targets += np.where(_TARGET_RELATIVE[kind, seg], start_speed[:, None, None], 0.0)
    targets = np.maximum(targets, 0.0).reshape(m, -1)
    rates = _RATE_LO[kind, seg] + (_RATE_HI - _RATE_LO)[kind, seg] * draws[..., 1]
    holds = _HOLD_LO[kind, seg] + (_HOLD_HI - _HOLD_LO)[kind, seg] * draws[..., 2]
    rates, holds = rates.reshape(m, -1), holds.reshape(m, -1)

    previous = np.concatenate([start_speed[:, None], targets[:, :-1]], axis=1)
    delta = targets - previous
    ramp_steps = np.maximum(1, np.rint(np.abs(delta) / (rates * dt))).astype(np.int64)
    hold_steps = np.maximum(1, np.rint(holds / dt)).astype(np.int64)
    ramp_accel = delta / (ramp_steps * dt)

    # Pieces: initial hold, then (ramp, hold) per segment, then a filler hold.
    n_seg = targets.shape[1]
    values = np.zeros((m, 2 + 2 * n_seg))
    counts = np.zeros((m, 2 + 2 * n_seg), dtype=np.int64)
    counts[:, 0] = np.maximum(1, np.rint(initial_hold / dt))
    values[:, 1:-1:2] = ramp_accel
    counts[:, 1:-1:2] = ramp_steps
    counts[:, 2:-1:2] = hold_steps
    cumulative = np.minimum(np.cumsum(counts[:, :-1], axis=1), n_steps)
    counts[:, :-1] = np.diff(cumulative, axis=1, prepend=0)
    counts[:, -1] = n_steps - cumulative[:, -1]
    return np.repeat(values.ravel(), counts.ravel()).reshape(m, n_steps)


def _known_options(options: dict[str, Any] | None, stacklevel: int) -> dict[str, Any] | None:
    """Drop unknown reset options with a warning (their values are not validated)."""
    if not options:
        return options
    unknown = sorted(set(options) - set(_OPTION_KEYS), key=str)
    if not unknown:
        return options
    warnings.warn(
        f"ignoring unknown reset option(s) {unknown}; valid options: {list(_OPTION_KEYS)}",
        UserWarning,
        stacklevel=stacklevel,
    )
    return {key: value for key, value in options.items() if key in _OPTION_KEYS}


# ---------------------------------------------------------------------------
# Batched simulation core (shared by the single and the vector environment)
# ---------------------------------------------------------------------------
class _PlatoonCore:
    """State and dynamics of ``B`` platoons, advanced with array operations.

    Vehicle index ``0`` is the leader and ``1..n`` are the followers, so the
    state arrays have shape ``(B, n + 1)``; per-follower quantities have shape
    ``(B, n)`` (follower ``k`` is vehicle ``k + 1``). The hot path updates
    preallocated arrays in place; everything handed out is a copy.
    """

    def __init__(self, config: PlatoonConfig, batch_size: int):
        self.cfg = config
        self.batch = int(batch_size)
        b, n = self.batch, config.n_followers
        v = n + 1
        self._rows = np.arange(b)
        self._adjacency, self._leader_links = platoon_adjacency(config.topology, n)

        rng = np.random.default_rng(config.vehicle_seed)
        self._fixed_tau = rng.uniform(*config.tau_range, size=v)
        self._fixed_length = rng.uniform(*config.length_range, size=v)
        self._fixed_coeffs = _lag_coefficients(self._fixed_tau, config.dt)

        # Vehicle state (leader first) and parameters.
        self.pos = np.zeros((b, v))
        self.vel = np.zeros((b, v))
        self.acc = np.zeros((b, v))
        self.cmd = np.zeros((b, v))
        self.tau = np.tile(self._fixed_tau, (b, 1))
        self.length = np.tile(self._fixed_length, (b, 1))
        self.alpha, self.c_va, self.c_vu, self.c_pa, self.c_pu = (
            np.tile(c, (b, 1)) for c in self._fixed_coeffs
        )
        self.beta = 1.0 - self.alpha
        self.leader_cmd = np.zeros((b, config.max_steps))
        self.leader_v0 = np.zeros(b)
        self.steps = np.zeros(b, dtype=np.int64)

        # Derived per-follower quantities and episode statistics.
        self.gap = np.zeros((b, n))
        self.err = np.zeros((b, n))
        self.dv = np.zeros((b, n))
        self.jerk = np.zeros((b, n))
        self.peak = np.zeros((b, n))
        self.min_gap = np.zeros(b)
        self.collided = np.zeros((b, n), dtype=bool)
        self.broken = np.zeros((b, n), dtype=bool)
        self._any_failed = False

        # Scratch buffers of the hot path.
        self._dp = np.zeros((b, v))
        self._tmp = np.zeros((b, v))
        self._tmp_n = np.zeros((b, n))

        # Observation buffer in feature-major layout (the availability flags are
        # constant); observe() returns a C-contiguous (B, n, obs_dim) copy.
        self._has_pred = config.topology != "none"
        self._has_leader = config.topology == "predecessor_leader"
        self._has_follower = config.topology == "bidirectional"
        self._obs_buf = np.zeros((_OBS_DIM, b, n), dtype=np.float32)
        self._obs_buf[_COL["predecessor_available"]] = float(self._has_pred)
        self._obs_buf[_COL["leader_available"]] = float(self._has_leader)
        if self._has_follower:
            self._obs_buf[_COL["follower_available"], :, :-1] = 1.0
        self._derive()

    # ------------------------------------------------------------------
    def _derive(self) -> None:
        """Gaps, spacing errors and relative speeds of the followers (in place)."""
        cfg = self.cfg
        pos, vel = self.pos, self.vel
        np.subtract(pos[:, :-1], pos[:, 1:], out=self.gap)
        self.gap -= self.length[:, :-1]
        np.multiply(vel[:, 1:], -cfg.headway, out=self.err)
        self.err += self.gap
        self.err -= cfg.standstill_distance
        np.subtract(vel[:, :-1], vel[:, 1:], out=self.dv)
        self.gap.min(axis=1, out=self.min_gap)

    def _option(self, options: dict[str, Any], key: str, size: int, idx: np.ndarray):
        """Option array for the copies ``idx``: shape ``(size,)`` or ``(B, size)``."""
        array = np.array(options[key], dtype=np.float64)
        if array.shape == (size,):
            array = np.broadcast_to(array, (idx.size, size))
        elif array.shape == (self.batch, size):
            array = array[idx]
        else:
            expected = f"({size},)" if self.batch == 1 else f"({size},) or ({self.batch}, {size})"
            raise ValueError(f"options[{key!r}] must have shape {expected}, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"options[{key!r}] must be finite")
        return array

    def reset(
        self, mask: np.ndarray, rng: np.random.Generator, options: dict[str, Any] | None
    ) -> None:
        """Re-initialise the copies where ``mask`` is true (see ``PlatoonEnv.reset``)."""
        cfg = self.cfg
        n, v = cfg.n_followers, cfg.n_followers + 1
        # Unknown keys were reported by the environment (see _known_options).
        options = {
            k: val
            for k, val in dict(options or {}).items()
            if k in _OPTION_KEYS and val is not None
        }
        if ("positions" in options) != ("velocities" in options):
            raise ValueError("options 'positions' and 'velocities' must be given together")
        idx = np.flatnonzero(mask)
        m = idx.size
        if m == 0:
            return

        # Vehicle parameters.
        if cfg.randomize_vehicles:
            tau = rng.uniform(*cfg.tau_range, size=(m, v))
            length = rng.uniform(*cfg.length_range, size=(m, v))
        else:
            tau = np.broadcast_to(self._fixed_tau, (m, v))
            length = np.broadcast_to(self._fixed_length, (m, v))
        custom_tau = "time_constants" in options
        if custom_tau:
            tau = self._option(options, "time_constants", v, idx)
            if np.any(tau <= 0.0):
                raise ValueError("options['time_constants'] must be positive")
        if "lengths" in options:
            length = self._option(options, "lengths", v, idx)
            if np.any(length <= 0.0):
                raise ValueError("options['lengths'] must be positive")

        # Initial state: equilibrium at the cruise speed plus small perturbations.
        if "positions" in options:
            pos = self._option(options, "positions", v, idx)
            vel = self._option(options, "velocities", v, idx)
            gaps = pos[:, :-1] - pos[:, 1:] - length[:, :-1]
            if np.any(gaps <= 0.0):
                raise ValueError(
                    "options['positions'] must place every vehicle behind its predecessor "
                    "(p_{i-1} - p_i - L_{i-1} > 0)"
                )
            if np.any(vel < 0.0):
                raise ValueError("options['velocities'] must be non-negative")
        else:
            speed = rng.uniform(*_INITIAL_SPEED, size=m)
            noise_v = np.clip(rng.standard_normal((m, n)), -_NOISE_CLIP, _NOISE_CLIP)
            noise_g = np.clip(rng.standard_normal((m, n)), -_NOISE_CLIP, _NOISE_CLIP)
            vel = np.empty((m, v))
            vel[:, 0] = speed
            vel[:, 1:] = np.maximum(speed[:, None] + cfg.init_speed_noise * noise_v, 0.0)
            desired = cfg.standstill_distance + cfg.headway * speed[:, None]
            gaps = np.maximum(desired + cfg.init_spacing_noise * noise_g, 0.5 * desired)
            pos = np.zeros((m, v))
            pos[:, 1:] = -np.cumsum(gaps + length[:, :-1], axis=1)
        if "accelerations" in options:
            acc = self._option(options, "accelerations", v, idx)
        else:
            acc = np.zeros((m, v))

        # Leader profile.
        if "leader_command" in options:
            commands = self._option(options, "leader_command", cfg.max_steps, idx)
            commands = np.clip(commands, cfg.accel_min, cfg.accel_max)
        else:
            commands = _sample_leader_commands(rng, cfg.scenario, vel[:, 0], cfg.max_steps, cfg.dt)

        # Commit (after all validation, so a bad option leaves the state untouched).
        self.tau[idx] = tau
        self.length[idx] = length
        if cfg.randomize_vehicles or custom_tau:
            coeffs = _lag_coefficients(tau, cfg.dt)
        else:
            coeffs = self._fixed_coeffs
        for target, value in zip((self.alpha, self.c_va, self.c_vu, self.c_pa, self.c_pu), coeffs):
            target[idx] = value
        self.beta[idx] = 1.0 - self.alpha[idx]
        self.pos[idx], self.vel[idx], self.acc[idx] = pos, vel, acc
        self.leader_cmd[idx] = commands
        self.leader_v0[idx] = vel[:, 0]
        self.cmd[idx] = 0.0
        self.steps[idx] = 0
        self.jerk[idx] = 0.0
        self.collided[idx] = False
        self.broken[idx] = False
        self._derive()
        self.peak[idx] = np.abs(self.err[idx])

    def step(self, commands: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance every copy by one step.

        Parameters
        ----------
        commands:
            Follower commands of shape ``(B, n)`` (float64, already clipped).

        Returns
        -------
        tuple
            ``(agent_rewards (B, n), terminated (B,), truncated (B,))``.
        """
        cfg = self.cfg
        dt = cfg.dt
        pos, vel, acc, cmd = self.pos, self.vel, self.acc, self.cmd
        dp, tmp = self._dp, self._tmp
        cmd[:, 0] = self.leader_cmd[self._rows, np.minimum(self.steps, cfg.max_steps - 1)]
        cmd[:, 1:] = commands

        # Exact zero-order-hold update (uses the old v and a, so p first, then v, then a).
        np.multiply(vel, dt, out=dp)
        np.multiply(self.c_pa, acc, out=tmp)
        dp += tmp
        np.multiply(self.c_pu, cmd, out=tmp)
        dp += tmp
        np.multiply(self.c_va, acc, out=tmp)
        vel += tmp
        np.multiply(self.c_vu, cmd, out=tmp)
        vel += tmp
        np.subtract(cmd, acc, out=tmp)
        tmp *= self.beta  # a+ - a = (1 - alpha) (u - a)
        acc += tmp
        stopped = vel < 0.0
        if stopped.any():
            # Vehicles do not reverse: a vehicle that would stop within the step is held at
            # standstill (the position error of this is below v * dt / 2 at the stop).
            old_acc = acc - tmp
            vel[stopped] = 0.0
            np.maximum(acc, 0.0, out=acc, where=stopped)
            np.maximum(dp, 0.0, out=dp, where=stopped)
            np.subtract(acc, old_acc, out=tmp)
        pos += dp
        np.multiply(tmp[:, 1:], 1.0 / dt, out=self.jerk)
        self.steps += 1
        self._derive()

        # Per-follower rewards.
        scratch = self._tmp_n
        rewards = np.multiply(self.err, self.err)
        rewards *= cfg.spacing_weight
        np.multiply(self.dv, self.dv, out=scratch)
        scratch *= cfg.speed_weight
        rewards += scratch
        np.multiply(commands, commands, out=scratch)
        scratch *= cfg.control_weight
        rewards += scratch
        np.multiply(self.jerk, self.jerk, out=scratch)
        scratch *= cfg.jerk_weight
        rewards += scratch
        rewards *= -dt

        # Collisions and break-ups (the per-follower flags are only rebuilt when needed).
        terminated = (self.min_gap <= 0.0) | (self.gap.max(axis=1) > cfg.breakup_distance)
        if terminated.any():
            np.less_equal(self.gap, 0.0, out=self.collided)
            np.greater(self.gap, cfg.breakup_distance, out=self.broken)
            rewards -= cfg.collision_penalty * self.collided
            rewards -= cfg.breakup_penalty * self.broken
            self._any_failed = True
        elif self._any_failed:
            self.collided.fill(False)
            self.broken.fill(False)
            self._any_failed = False

        np.abs(self.err, out=scratch)
        np.maximum(self.peak, scratch, out=self.peak)
        truncated = self.steps >= cfg.max_steps
        return rewards, terminated, truncated

    def observe(self) -> np.ndarray:
        """Joint observation of shape ``(B, n, obs_dim)``, ``float32`` (a new array)."""
        buf, col, unsafe = self._obs_buf, _COL, "unsafe"
        vel, acc = self.vel, self.acc
        np.multiply(self.err, 1.0 / _SPACING_SCALE, out=buf[col["spacing_error"]], casting=unsafe)
        np.multiply(
            self.dv, 1.0 / _SPEED_DIFF_SCALE, out=buf[col["relative_speed"]], casting=unsafe
        )
        np.multiply(vel[:, 1:], 1.0 / _SPEED_SCALE, out=buf[col["speed"]], casting=unsafe)
        np.multiply(acc[:, 1:], 1.0 / _ACCEL_SCALE, out=buf[col["acceleration"]], casting=unsafe)
        if self._has_pred:
            np.multiply(
                acc[:, :-1],
                1.0 / _ACCEL_SCALE,
                out=buf[col["predecessor_acceleration"]],
                casting=unsafe,
            )
            np.multiply(
                self.cmd[:, :-1],
                1.0 / _ACCEL_SCALE,
                out=buf[col["predecessor_command"]],
                casting=unsafe,
            )
        if self._has_leader:
            speed_error = vel[:, :1] - vel[:, 1:]
            speed_error *= 1.0 / _SPEED_DIFF_SCALE
            buf[col["leader_speed_error"]] = speed_error
            accel_error = acc[:, :1] - acc[:, 1:]
            accel_error *= 1.0 / _ACCEL_SCALE
            buf[col["leader_acceleration_error"]] = accel_error
        if self._has_follower:
            np.multiply(
                self.err[:, 1:],
                1.0 / _SPACING_SCALE,
                out=buf[col["follower_spacing_error"], :, :-1],
                casting=unsafe,
            )
            np.multiply(
                self.dv[:, 1:],
                1.0 / _SPEED_DIFF_SCALE,
                out=buf[col["follower_relative_speed"], :, :-1],
                casting=unsafe,
            )
        return buf.transpose(1, 2, 0).copy(order="C")

    def infos(self, agent_rewards: np.ndarray) -> dict[str, np.ndarray]:
        """Batched info arrays (leading dimension ``B``, all copies)."""
        peak = self.peak
        return {
            "agent_rewards": np.array(agent_rewards, dtype=np.float64),
            "spacing_errors": self.err.copy(),
            "gaps": self.gap.copy(),
            "relative_speeds": self.dv.copy(),
            "speeds": self.vel[:, 1:].copy(),
            "accelerations": self.acc[:, 1:].copy(),
            "commands": self.cmd[:, 1:].copy(),
            "leader_speed": self.vel[:, 0].copy(),
            "leader_acceleration": self.acc[:, 0].copy(),
            "min_gap": self.min_gap.copy(),
            "collision": self.collided.any(axis=1),
            "breakup": self.broken.any(axis=1),
            "peak_spacing_errors": peak.copy(),
            "error_amplification": peak[:, -1] / np.maximum(peak[:, 0], _PEAK_FLOOR),
            "step": self.steps.copy(),
            "time": self.steps * self.cfg.dt,
        }

    def row_info(self, row: int, agent_rewards: np.ndarray) -> dict[str, Any]:
        """Info of one copy (same keys as :meth:`infos`, Python scalars for scalars)."""
        peak = self.peak[row]
        steps = int(self.steps[row])
        return {
            "agent_rewards": np.array(agent_rewards, dtype=np.float64),
            "spacing_errors": self.err[row].copy(),
            "gaps": self.gap[row].copy(),
            "relative_speeds": self.dv[row].copy(),
            "speeds": self.vel[row, 1:].copy(),
            "accelerations": self.acc[row, 1:].copy(),
            "commands": self.cmd[row, 1:].copy(),
            "leader_speed": float(self.vel[row, 0]),
            "leader_acceleration": float(self.acc[row, 0]),
            "min_gap": float(self.min_gap[row]),
            "collision": bool(self.collided[row].any()),
            "breakup": bool(self.broken[row].any()),
            "peak_spacing_errors": peak.copy(),
            "error_amplification": float(peak[-1] / max(float(peak[0]), _PEAK_FLOOR)),
            "step": steps,
            "time": steps * self.cfg.dt,
        }

    def leader_reference(self, row: int) -> np.ndarray:
        """Reference speed of the leader profile (without actuator lag), ``(max_steps + 1,)``."""
        ref = np.empty(self.cfg.max_steps + 1)
        ref[0] = self.leader_v0[row]
        np.cumsum(self.leader_cmd[row] * self.cfg.dt, out=ref[1:])
        ref[1:] += ref[0]
        return ref

    def render_state(self, row: int) -> dict[str, Any]:
        """Snapshot (copies) of copy ``row`` for the renderer."""
        return {
            "positions": self.pos[row].copy(),
            "lengths": self.length[row].copy(),
            "speeds": self.vel[row].copy(),
            "accelerations": self.acc[row].copy(),
            "spacing_errors": self.err[row].copy(),
            "peak_errors": self.peak[row].copy(),
            "step": int(self.steps[row]),
            "collision": bool(self.collided[row].any()),
            "breakup": bool(self.broken[row].any()),
        }


class _PlatoonSpecMixin:
    """Read-only views shared by the single and the vector environment."""

    _cfg: PlatoonConfig
    _core: _PlatoonCore

    @property
    def config(self) -> PlatoonConfig:
        """The validated, immutable environment parameters."""
        return self._cfg

    @property
    def n_followers(self) -> int:
        """Number of followers (agents)."""
        return self._cfg.n_followers

    @property
    def n_agents(self) -> int:
        """Number of agents (the followers)."""
        return self._cfg.n_followers

    @property
    def topology(self) -> str:
        """V2V communication topology."""
        return self._cfg.topology

    @property
    def scenario(self) -> str:
        """Leader speed-profile scenario."""
        return self._cfg.scenario

    @property
    def dt(self) -> float:
        """Time step in seconds."""
        return self._cfg.dt

    @property
    def max_steps(self) -> int:
        """Episode length in steps (truncation)."""
        return self._cfg.max_steps

    @property
    def headway(self) -> float:
        """Time headway ``h`` of the spacing policy in seconds."""
        return self._cfg.headway

    @property
    def standstill_distance(self) -> float:
        """Standstill distance ``r`` of the spacing policy in metres."""
        return self._cfg.standstill_distance

    @property
    def obs_dim(self) -> int:
        """Width of one observation row."""
        return _OBS_DIM

    @property
    def adjacency(self) -> np.ndarray:
        """V2V graph over the followers, bool ``(n, n)`` (rows receive from columns; a copy)."""
        return self._core._adjacency.copy()

    @property
    def leader_links(self) -> np.ndarray:
        """Followers that receive data from the leader, bool ``(n,)`` (a copy)."""
        return self._core._leader_links.copy()

    @property
    def observation_layout(self) -> dict[str, slice]:
        """Column slice of every feature of an observation row (see :data:`OBSERVATION_FEATURES`)."""
        return {name: slice(k, k + 1) for k, name in enumerate(OBSERVATION_FEATURES)}


def _spaces(config: PlatoonConfig) -> tuple[spaces.Box, spaces.Box]:
    n = config.n_followers
    observation_space = spaces.Box(-np.inf, np.inf, shape=(n, _OBS_DIM), dtype=np.float32)
    action_space = spaces.Box(config.accel_min, config.accel_max, shape=(n,), dtype=np.float32)
    return observation_space, action_space


def _make_renderer(render_mode: str, config: PlatoonConfig, core: _PlatoonCore, fps: float):
    from env_lib.platoon_env.rendering import PlatoonRenderer

    return PlatoonRenderer(
        render_mode,
        n_followers=config.n_followers,
        topology=config.topology,
        scenario=config.scenario,
        max_steps=config.max_steps,
        dt=config.dt,
        headway=config.headway,
        standstill_distance=config.standstill_distance,
        tau_range=config.tau_range,
        adjacency=core._adjacency,
        leader_links=core._leader_links,
        fps=fps,
    )


# ---------------------------------------------------------------------------
# Single environment
# ---------------------------------------------------------------------------
class PlatoonEnv(_PlatoonSpecMixin, gym.Env):
    """Cooperative adaptive cruise control of a vehicle platoon.

    **Vehicles.** Vehicle ``0`` is the leader and vehicles ``i = 1..n`` are the
    followers, which are the agents (agent index ``k = i - 1`` in all arrays).
    Every vehicle has position ``p_i`` (front bumper), speed ``v_i``,
    acceleration ``a_i`` and length ``L_i``, and a first-order actuator
    (drivetrain) lag with time constant ``tau_i``::

        dp_i/dt = v_i,    dv_i/dt = a_i,    da_i/dt = (u_i - a_i) / tau_i

    where ``u_i`` is the commanded acceleration (the action of follower ``i``,
    clipped to ``[accel_min, accel_max]``). The command is held constant over
    a step and the model is discretised exactly (zero-order hold), with
    ``alpha_i = exp(-dt / tau_i)``::

        a_i+ = alpha_i a_i + (1 - alpha_i) u_i
        v_i+ = v_i + tau_i (1 - alpha_i) a_i + (dt - tau_i (1 - alpha_i)) u_i
        p_i+ = p_i + dt v_i + c_pa a_i + (dt^2 / 2 - c_pa) u_i,
               c_pa = tau_i dt - tau_i^2 (1 - alpha_i)

    Vehicles do not reverse: a vehicle whose speed would become negative is
    held at standstill (``v = 0``, ``a = max(a, 0)``, position not decreased).
    The lags ``tau_i`` and lengths ``L_i`` are drawn uniformly from
    ``tau_range`` and ``length_range`` with a generator seeded by
    ``vehicle_seed`` (fixed platoon), or from ``self.np_random`` at every reset
    when ``randomize_vehicles=True``.

    **Spacing policy.** Constant time headway: follower ``i`` should keep the
    gap ``g_i = p_{i-1} - p_i - L_{i-1}`` at ``d_i* = r + h v_i`` (standstill
    distance ``r``, headway ``h``). The spacing error is ``e_i = g_i - d_i*``
    and the relative speed ``dv_i = v_{i-1} - v_i``.

    **Leader.** The leader has the same dynamics; its command follows a speed
    profile drawn per episode from ``self.np_random``: after 2-5 s at the
    initial cruise speed ``v_c ~ U[16, 25]`` m/s, a sequence of events, each
    ramping the reference speed to a target with constant acceleration and
    holding it: ``"cruise"`` (targets ``v_c +- 2`` m/s at 0.3-0.8 m/s^2, held
    4-10 s), ``"stop_and_go"`` (brake at 2-4 m/s^2 to 0-4 m/s, hold 1-5 s,
    accelerate at 1-2 m/s^2 back to ``v_c +- 3`` m/s, hold 4-10 s) and
    ``"random"`` (targets in [5, 30] m/s at 0.5-2.5 m/s^2, held 1-6 s).
    ``"mixed"`` (default) starts with a stop-and-go wave and draws every
    further event uniformly from the three types.

    **Initial state.** Equilibrium at ``v_c``: zero accelerations, follower
    speeds ``v_c + init_speed_noise * xi`` and gaps ``r + h v_c +
    init_spacing_noise * xi'`` with standard normal ``xi, xi'`` truncated at
    three standard deviations (gaps are kept above half the desired gap).
    The leader starts at ``p_0 = 0``.

    **Communication.** ``topology`` selects the V2V links (see
    :func:`platoon_adjacency`, :attr:`adjacency` and :attr:`leader_links`):
    ``"predecessor"`` (PF), ``"predecessor_leader"`` (PLF), ``"bidirectional"``
    (BD) or ``"none"`` (sensors only).

    **Observation.** ``Box(-inf, inf, (n, 13), float32)``; row ``k`` belongs to
    follower ``i = k + 1``. Every feature is a physical quantity divided by a
    fixed scale (:data:`FEATURE_SCALES`); communicated channels that the
    topology does not provide are zero with availability flag 0:

    ====  ===========================  ================================  ======
    col   layout key                   content                           scale
    ====  ===========================  ================================  ======
    0     spacing_error                ``e_i`` (radar)                   5 m
    1     relative_speed               ``v_{i-1} - v_i`` (radar)         5 m/s
    2     speed                        ``v_i``                           30 m/s
    3     acceleration                 ``a_i``                           3 m/s^2
    4     predecessor_available        1 for PF, PLF, BD
    5     predecessor_acceleration     ``a_{i-1}`` (V2V)                 3 m/s^2
    6     predecessor_command          ``u_{i-1}`` of the last step      3 m/s^2
    7     leader_available             1 for PLF
    8     leader_speed_error           ``v_0 - v_i`` (V2V)               5 m/s
    9     leader_acceleration_error    ``a_0 - a_i`` (V2V)               3 m/s^2
    10    follower_available           1 for BD, except the last one
    11    follower_spacing_error       ``e_{i+1}`` (V2V)                 5 m
    12    follower_relative_speed      ``v_i - v_{i+1}`` (V2V)           5 m/s
    ====  ===========================  ================================  ======

    **Action.** ``Box(accel_min, accel_max, (n,), float32)``: commanded
    accelerations in m/s^2, clipped to the bounds.

    **Reward.** Per follower, evaluated after the step,

    ``r_i = -(w_e e_i^2 + w_v dv_i^2 + w_u u_i^2 + w_j j_i^2) * dt``

    with the jerk ``j_i = (a_i+ - a_i) / dt`` and the weights
    ``spacing_weight``, ``speed_weight``, ``control_weight`` and
    ``jerk_weight``; a follower whose gap is ``<= 0`` (collision) receives
    ``-collision_penalty`` and one whose gap exceeds ``breakup_distance``
    receives ``-breakup_penalty`` in addition. The team reward returned by
    :meth:`step` is ``sum_i r_i``.

    **Episode end.** ``terminated`` on a collision or a platoon break-up (any
    follower); ``truncated`` after ``max_steps`` steps.

    **Info.** ``agent_rewards`` ``(n,)``; per follower ``(n,)``:
    ``spacing_errors``, ``gaps``, ``relative_speeds``, ``speeds``,
    ``accelerations``, ``commands`` (applied, clipped) and
    ``peak_spacing_errors`` (``max |e_i|`` since the reset); scalars:
    ``leader_speed``, ``leader_acceleration``, ``min_gap``, ``collision``,
    ``breakup``, ``error_amplification`` (peak error of the last follower
    divided by that of the first, the latter floored at 0.01 m; values above
    one indicate amplification along the platoon), ``step`` and ``time`` (s).

    Parameters
    ----------
    n_followers:
        Number of followers (agents), ``>= 1``.
    topology:
        One of :data:`TOPOLOGIES`.
    scenario:
        Leader profile, one of :data:`SCENARIOS`.
    dt:
        Time step in seconds (``> 0``).
    max_steps:
        Episode length in steps (``>= 1``); 600 steps are 60 s.
    headway:
        Time headway ``h`` in seconds (``>= 0``).
    standstill_distance:
        Standstill distance ``r`` in metres (``> 0``).
    accel_min, accel_max:
        Command bounds in m/s^2 (``accel_min < 0 < accel_max``).
    tau_range:
        Range of the actuator lags in seconds (``0 < low <= high``).
    length_range:
        Range of the vehicle lengths in metres.
    vehicle_seed:
        Seed of the fixed vehicle parameters (``None``: fresh OS entropy).
    randomize_vehicles:
        Resample lags and lengths from ``self.np_random`` at every reset.
    init_spacing_noise, init_speed_noise:
        Standard deviations of the initial gap (m) and speed (m/s) perturbations.
    spacing_weight, speed_weight, control_weight, jerk_weight:
        Reward weights ``w_e``, ``w_v``, ``w_u``, ``w_j`` (``>= 0``).
    collision_penalty:
        Penalty of a follower that collides with its predecessor (``>= 0``).
    breakup_distance:
        Gap in metres above which the platoon counts as broken up.
    breakup_penalty:
        Penalty of a follower whose gap exceeds ``breakup_distance`` (``>= 0``).
    render_mode:
        ``None``, ``"human"`` or ``"rgb_array"``.

    Raises
    ------
    ValueError
        For out-of-range or unknown argument values.
    TypeError
        For arguments of the wrong type.

    Examples
    --------
    >>> env = PlatoonEnv(scenario="stop_and_go")
    >>> obs, info = env.reset(seed=0)
    >>> obs.shape
    (8, 13)
    >>> while True:
    ...     obs, reward, terminated, truncated, info = env.step(cacc_policy(obs))
    ...     if terminated or truncated:
    ...         break
    >>> bool(info["collision"]), info["error_amplification"] < 1.0
    (False, True)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __getstate__(self) -> dict[str, Any]:
        # Pickle and deep-copy without the renderer (rebuilt on the next render()).
        return state_without_renderer(self)

    def __init__(
        self,
        *,
        n_followers: int = 8,
        topology: str = "predecessor",
        scenario: str = "mixed",
        dt: float = 0.1,
        max_steps: int = 600,
        headway: float = 0.6,
        standstill_distance: float = 2.0,
        accel_min: float = -6.0,
        accel_max: float = 3.0,
        tau_range: tuple[float, float] = (0.2, 0.4),
        length_range: tuple[float, float] = (4.0, 5.0),
        vehicle_seed: int | None = 0,
        randomize_vehicles: bool = False,
        init_spacing_noise: float = 0.1,
        init_speed_noise: float = 0.05,
        spacing_weight: float = 1.0,
        speed_weight: float = 0.1,
        control_weight: float = 0.02,
        jerk_weight: float = 0.01,
        collision_penalty: float = 500.0,
        breakup_distance: float = 100.0,
        breakup_penalty: float = 250.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        validate_render_mode(render_mode, self.metadata["render_modes"])
        self._cfg = PlatoonConfig.from_kwargs(
            n_followers=n_followers,
            topology=topology,
            scenario=scenario,
            dt=dt,
            max_steps=max_steps,
            headway=headway,
            standstill_distance=standstill_distance,
            accel_min=accel_min,
            accel_max=accel_max,
            tau_range=tau_range,
            length_range=length_range,
            vehicle_seed=vehicle_seed,
            randomize_vehicles=randomize_vehicles,
            init_spacing_noise=init_spacing_noise,
            init_speed_noise=init_speed_noise,
            spacing_weight=spacing_weight,
            speed_weight=speed_weight,
            control_weight=control_weight,
            jerk_weight=jerk_weight,
            collision_penalty=collision_penalty,
            breakup_distance=breakup_distance,
            breakup_penalty=breakup_penalty,
        )
        self.render_mode = render_mode
        self._core = _PlatoonCore(self._cfg, 1)
        self.observation_space, self.action_space = _spaces(self._cfg)
        self._reset_done = False
        self._renderer = None
        self._episode = 0
        self._warned_no_render_mode = False

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------
    @property
    def positions(self) -> np.ndarray:
        """Front-bumper positions of all vehicles (leader first), ``(n + 1,)`` (a copy)."""
        self._require_reset()
        return self._core.pos[0].copy()

    @property
    def velocities(self) -> np.ndarray:
        """Speeds of all vehicles (leader first), ``(n + 1,)`` (a copy)."""
        self._require_reset()
        return self._core.vel[0].copy()

    @property
    def accelerations(self) -> np.ndarray:
        """Accelerations of all vehicles (leader first), ``(n + 1,)`` (a copy)."""
        self._require_reset()
        return self._core.acc[0].copy()

    @property
    def gaps(self) -> np.ndarray:
        """Gaps ``g_i`` of the followers, ``(n,)`` (a copy)."""
        self._require_reset()
        return self._core.gap[0].copy()

    @property
    def spacing_errors(self) -> np.ndarray:
        """Spacing errors ``e_i`` of the followers, ``(n,)`` (a copy)."""
        self._require_reset()
        return self._core.err[0].copy()

    @property
    def time_constants(self) -> np.ndarray:
        """Actuator lags ``tau_i`` of all vehicles (leader first), ``(n + 1,)`` (a copy)."""
        return self._core.tau[0].copy()

    @property
    def lengths(self) -> np.ndarray:
        """Vehicle lengths ``L_i`` (leader first), ``(n + 1,)`` (a copy)."""
        return self._core.length[0].copy()

    @property
    def leader_command(self) -> np.ndarray:
        """Commanded acceleration of the leader per step, ``(max_steps,)`` (a copy)."""
        self._require_reset()
        return self._core.leader_cmd[0].copy()

    @property
    def step_count(self) -> int:
        """Number of steps taken in the current episode."""
        return int(self._core.steps[0])

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        Parameters
        ----------
        seed:
            Seed of ``self.np_random`` (leader profile, initial perturbations
            and, with ``randomize_vehicles=True``, the vehicle parameters).
        options:
            Optional overrides of the sampled episode, all arrays with the
            leader first: ``"positions"`` and ``"velocities"`` (``(n + 1,)``,
            given together), ``"accelerations"`` (``(n + 1,)``, default zeros),
            ``"leader_command"`` (``(max_steps,)`` commanded accelerations of
            the leader, clipped to the action bounds), ``"time_constants"`` and
            ``"lengths"`` (``(n + 1,)``, for this episode only). Unknown keys
            are ignored with a ``UserWarning``; the values of known keys are
            validated strictly.

        Returns
        -------
        observation, info

        Raises
        ------
        ValueError
            For option values of the wrong shape, non-finite values, vehicles
            that overlap or negative speeds.
        """
        super().reset(seed=seed)
        options = _known_options(options, stacklevel=3)  # warn at the caller of reset()
        self._core.reset(np.ones(1, dtype=bool), self.np_random, options)
        self._reset_done = True
        self._episode += 1
        observation = self._core.observe()[0]
        info = self._info(np.zeros((1, self._cfg.n_followers)))
        if self._renderer is not None:
            self._renderer.reset()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the platoon by one time step.

        Parameters
        ----------
        action:
            Commanded accelerations, shape ``(n_followers,)`` (anything with
            ``n_followers`` entries); cast to ``float32`` like the action space
            and clipped to the bounds.

        Returns
        -------
        observation, reward, terminated, truncated, info

        Raises
        ------
        ValueError
            For a wrong number of entries or non-finite values.
        ResetNeededError
            If called before :meth:`reset`.
        """
        self._require_reset()
        commands = self._validate_action(action)
        rewards, terminated, truncated = self._core.step(commands[None])
        observation = self._core.observe()[0]
        info = self._info(rewards)
        if self.render_mode == "human":
            self.render()
        return observation, float(rewards.sum()), bool(terminated[0]), bool(truncated[0]), info

    def render(self) -> np.ndarray | None:
        """Render the current state.

        Returns
        -------
        numpy.ndarray or None
            ``(H, W, 3)`` ``uint8`` frame in ``"rgb_array"`` mode, else ``None``.
        """
        if self.render_mode is None:
            if not self._warned_no_render_mode:
                warnings.warn(
                    "render() called without a render_mode; create the environment with "
                    "render_mode='rgb_array' or 'human'",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_no_render_mode = True
            return None
        self._require_reset()
        if self._renderer is None:
            self._renderer = _make_renderer(
                self.render_mode, self._cfg, self._core, self.metadata["render_fps"]
            )
        return self._renderer.render(
            **self._core.render_state(0),
            episode=self._episode,
            leader_reference=self._core.leader_reference(0),
        )

    def close(self) -> None:
        """Release the rendering resources (idempotent)."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_reset(self) -> None:
        if not self._reset_done:
            raise ResetNeededError("Call reset() before using the environment")

    def _validate_action(self, action: Any) -> np.ndarray:
        n = self._cfg.n_followers
        commands = np.asarray(action, dtype=np.float32)
        if commands.size != n:
            raise ValueError(f"action must have shape ({n},), got {np.shape(action)}")
        commands = commands.reshape(n).astype(np.float64)
        if not np.all(np.isfinite(commands)):
            raise ValueError("action contains NaN or inf")
        return np.clip(commands, self._cfg.accel_min, self._cfg.accel_max)

    def _info(self, rewards: np.ndarray) -> dict[str, Any]:
        return self._core.row_info(0, rewards[0])


# ---------------------------------------------------------------------------
# Vector environment
# ---------------------------------------------------------------------------
class PlatoonVectorEnv(_PlatoonSpecMixin, BatchedVectorEnv):
    """``num_envs`` platoons simulated as one batch.

    Uses the same batched core as :class:`PlatoonEnv`, so copy ``b`` evolves
    exactly like a single environment started from the same state. Spaces:
    observations ``(num_envs, n, 13)`` ``float32``, actions
    ``(num_envs, n)`` ``float32``. Rewards are the team rewards of the copies;
    ``infos`` hold the keys of :class:`PlatoonEnv` with a leading
    ``num_envs`` dimension (``agent_rewards`` has shape ``(num_envs, n)``).

    Parameters
    ----------
    num_envs:
        Number of platoons ``B``.
    autoreset_mode:
        ``"next_step"`` (default), ``"same_step"`` or ``"disabled"``; see
        :class:`env_lib.utils.vector.BatchedVectorEnv`.
    render_mode:
        ``None``, ``"human"`` or ``"rgb_array"``; :meth:`render` draws copy 0.
    **kwargs:
        The keyword arguments of :class:`PlatoonEnv`.

    Notes
    -----
    ``reset(options=...)`` accepts the options of :meth:`PlatoonEnv.reset`;
    every array may have the single-environment shape (applied to all reset
    copies) or an extra leading ``num_envs`` dimension (one row per copy).

    Examples
    --------
    >>> envs = PlatoonVectorEnv(num_envs=256, n_followers=16)
    >>> obs, infos = envs.reset(seed=0)
    >>> obs.shape
    (256, 16, 13)
    >>> obs, rewards, terminated, truncated, infos = envs.step(cacc_policy(obs))
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __getstate__(self) -> dict[str, Any]:
        # Pickle and deep-copy without the renderer (rebuilt on the next render()).
        return state_without_renderer(self)

    def __init__(
        self,
        num_envs: int = 1,
        *,
        autoreset_mode: str = "next_step",
        render_mode: str | None = None,
        **kwargs: Any,
    ):
        validate_render_mode(render_mode, self.metadata["render_modes"])
        self._cfg = PlatoonConfig.from_kwargs(**kwargs)
        observation_space, action_space = _spaces(self._cfg)
        super().__init__(
            num_envs,
            observation_space,
            action_space,
            autoreset_mode=autoreset_mode,
            render_mode=render_mode,
        )
        self._core = _PlatoonCore(self._cfg, self.num_envs)
        self._renderer = None
        self._episode = 0
        self._last_step0 = 0

    # ------------------------------------------------------------------
    def _reset_envs(self, mask: np.ndarray, options: dict[str, Any] | None) -> None:
        options = _known_options(options, stacklevel=4)  # warn at the caller of reset()
        self._core.reset(mask, self.np_random, options)
        if mask[0]:
            self._episode += 1

    def _step_envs(
        self, actions: np.ndarray, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        cfg = self._cfg
        commands = np.clip(actions.astype(np.float64), cfg.accel_min, cfg.accel_max)
        rewards, terminated, truncated = self._core.step(commands)
        infos = self._core.infos(rewards)
        return rewards.sum(axis=1), terminated, truncated, infos

    def _observe(self) -> np.ndarray:
        return self._core.observe()

    def _reset_infos(self, mask: np.ndarray) -> dict[str, Any]:
        return self._core.infos(np.zeros((self.num_envs, self._cfg.n_followers)))

    # ------------------------------------------------------------------
    def render(self) -> np.ndarray | None:
        """Render copy 0 (``None`` without a render mode or in ``"human"`` mode)."""
        if self.render_mode is None:
            return None
        if self._needs_reset:
            raise ResetNeededError("Call reset() before render().")
        if self._renderer is None:
            self._renderer = _make_renderer(
                self.render_mode, self._cfg, self._core, self.metadata["render_fps"]
            )
        return self._renderer.render(
            **self._core.render_state(0),
            episode=self._episode,
            leader_reference=self._core.leader_reference(0),
        )

    def close_extras(self, **kwargs: Any) -> None:
        """Release the rendering resources (called by :meth:`close`)."""
        if getattr(self, "_renderer", None) is not None:  # absent if __init__ failed early
            self._renderer.close()
            self._renderer = None
