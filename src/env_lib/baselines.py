"""Classical baseline controllers for every ``env_lib`` environment.

:func:`baseline_policy` returns a ready-to-use controller for any environment
of the package -- a single environment, a vector environment (native or
Gymnasium's Sync/Async), wrapped or not -- by dispatching on the type of the
unwrapped environment::

    import env_lib

    env = env_lib.make("Formation-v0")
    policy = env_lib.baseline_policy(env)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(policy(obs))

Every controller is a pure function of the observation that accepts any
leading batch shape, ``policy(obs[..., *observation_shape]) ->
action[..., *action_shape]``, so the same policy drives a single environment
and ``num_envs`` copies of it. The pure functions are public as well
(:func:`laplacian_consensus`, :func:`kuramoto_feedback`,
:func:`linemsg_relay`, :func:`wireless_schedule`, :func:`pistonball_ramp`,
:func:`ajlatt_encircle`); :func:`baseline_policy` binds their parameters from
the environment configuration.

The returned :class:`BaselinePolicy` also accepts

* PyTorch tensors (the action is returned as a tensor on the same device),
* observation dictionaries of :class:`env_lib.wrappers.ParallelEnvAdapter`
  (pass the adapter to :func:`baseline_policy`), and
* observations flattened by :class:`env_lib.wrappers.FlattenJointSpaces`.

:func:`list_baselines` lists the controller of every environment family.
"""

from __future__ import annotations

import importlib
import inspect
import math
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv

__all__ = [
    "BaselinePolicy",
    "ajlatt_encircle",
    "baseline_policy",
    "family_of",
    "kuramoto_feedback",
    "laplacian_consensus",
    "linemsg_relay",
    "list_baselines",
    "pistonball_ramp",
    "wireless_schedule",
]

#: Short description of the baseline controller of every environment family.
_DESCRIPTIONS: dict[str, str] = {
    "consensus": "Laplacian protocol u_i = k sum_j (y_j - y_i) - k_v v_i",
    "kuramoto": "frequency compensation + mean-field phase feedback, maximal coupling",
    "linemsg": "always relay: every agent keeps its link active",
    "wireless_comm": "collision-free access-point schedule with idle-slot borrowing",
    "pistonball": "decentralised ramp: pistons left of the ball down, right of it up",
    "ajlatt": "encircle the target belief at evenly spaced slots",
    "power_grid": "droop control (env_lib.power_grid_env.droop_policy)",
    "platoon": "cooperative adaptive cruise control (env_lib.platoon_env.cacc_policy)",
}

#: Environment class name -> family (subclasses are matched through the MRO).
_FAMILY_BY_CLASS: dict[str, str] = {
    "ConsensusEnv": "consensus",
    "ConsensusVectorEnv": "consensus",
    "KuramotoOscillatorEnv": "kuramoto",
    "KuramotoOscillatorEnvTorch": "kuramoto",
    "KuramotoOscillatorVectorEnv": "kuramoto",
    "KuramotoEnvBase": "kuramoto",
    "LineMsgEnv": "linemsg",
    "WirelessCommEnv": "wireless_comm",
    "PistonballEnv": "pistonball",
    "AJLATTEnv": "ajlatt",
    "PowerGridEnv": "power_grid",
    "PowerGridVectorEnv": "power_grid",
    "PlatoonEnv": "platoon",
    "PlatoonVectorEnv": "platoon",
}

#: ``env_lib`` subpackage -> family, for classes not listed above.
_FAMILY_BY_PACKAGE: dict[str, str] = {
    "consensus_env": "consensus",
    "kos_env": "kuramoto",
    "linemsg_env": "linemsg",
    "wireless_comm_env": "wireless_comm",
    "pistonball_env": "pistonball",
    "ajlatt_env": "ajlatt",
    "power_grid_env": "power_grid",
    "platoon_env": "platoon",
}


def list_baselines() -> dict[str, str]:
    """Baseline controller of every environment family.

    Returns
    -------
    dict
        ``{family: short description}``; the families are those of
        :attr:`env_lib.registration.EnvSpec.family`.
    """
    return dict(_DESCRIPTIONS)


# ---------------------------------------------------------------------------
# Pure controllers (any leading batch shape)
# ---------------------------------------------------------------------------
def laplacian_consensus(
    observation: Any,
    *,
    gain: float = 1.0,
    velocity_gain: float = 0.0,
    max_control: float = math.inf,
) -> np.ndarray:
    """Distributed Laplacian consensus / formation protocol from the observation.

    With the neighbour slots of :class:`~env_lib.consensus_env.ConsensusEnv`
    (``y_j - y_i`` and a validity mask per slot, ``y = x - d``) the action of
    agent ``i`` is

    ``u_i = gain * sum_{valid slots} (y_j - y_i) - velocity_gain * v_i``,

    i.e. ``u = -gain L (x - d) - velocity_gain v``, clipped to
    ``[-max_control, max_control]``. It equals
    :meth:`ConsensusEnv.laplacian_policy` whenever every neighbour fits into
    the observation slots (always true for static graphs with the default
    ``max_neighbors``).

    Parameters
    ----------
    observation:
        Array of shape ``(..., n_agents, 6 + 5 * max_neighbors)``.
    gain, velocity_gain:
        Position and velocity feedback gains.
    max_control:
        Bound of every action component.

    Returns
    -------
    numpy.ndarray
        ``float32`` action of shape ``(..., n_agents, 2)``.
    """
    obs = np.asarray(observation, dtype=np.float64)
    width = obs.shape[-1] - 6
    if obs.ndim < 2 or width < 5 or width % 5:
        raise ValueError(
            f"expected a consensus observation of shape (..., n_agents, 6 + 5 k), got {obs.shape}"
        )
    slots = obs[..., 6:].reshape(obs.shape[:-1] + (width // 5, 5))
    control = gain * np.sum(slots[..., 0:2] * slots[..., 4:5], axis=-2)
    if velocity_gain:
        control -= velocity_gain * obs[..., 2:4]
    return np.clip(control, -max_control, max_control).astype(np.float32)


def kuramoto_feedback(
    observation: Any,
    *,
    n_oscillators: int,
    n_couplings: int = 0,
    control_low: Any = -1.0,
    control_high: Any = 1.0,
    coupling: Any = 5.0,
    target_frequency: float | None = None,
    gain: float = 3.0,
) -> np.ndarray:
    """Frequency compensation plus mean-field phase feedback for the Kuramoto network.

    With phases ``theta_i``, natural frequencies ``omega_i`` and the mean field
    ``r exp(i psi) = mean_j exp(i theta_j)`` (all read from the observation),
    the control input of oscillator ``i`` is

    ``a_i = (Omega - omega_i) + gain * sin(psi - theta_i)``,

    clipped to the control bounds, where ``Omega`` is the mean natural
    frequency (phase synchronisation) or ``target_frequency`` (frequency
    tracking). The first term gives every oscillator the same effective
    frequency, so that the network can lock with zero phase differences; the
    second pulls every phase towards the mean phase ``psi``. In dynamic
    coupling mode every coupling strength is set to ``coupling`` (by default
    the upper bound, which locks the phases fastest). The mean field is a
    global quantity, which matches the centralised (single-agent) interface of
    the Kuramoto environments.

    Parameters
    ----------
    observation:
        Array of shape ``(..., obs_dim)`` with the layout
        ``[phases, natural_frequencies, (couplings,) controls]``.
    n_oscillators, n_couplings:
        ``N`` and the number of coupling actions ``M`` (0 in constant mode).
    control_low, control_high:
        Bounds of the control inputs (scalars or arrays of length ``N``).
    coupling:
        Coupling strength(s) commanded in dynamic mode (scalar or length ``M``).
    target_frequency:
        ``None`` for phase synchronisation, else the frequency to track.
    gain:
        Phase feedback gain.

    Returns
    -------
    numpy.ndarray
        ``float32`` action of shape ``(..., N + M)``.
    """
    obs = np.asarray(observation, dtype=np.float64)
    n = int(n_oscillators)
    theta, omega = obs[..., :n], obs[..., n : 2 * n]
    psi = np.arctan2(
        np.sin(theta).mean(axis=-1, keepdims=True), np.cos(theta).mean(axis=-1, keepdims=True)
    )
    reference = omega.mean(axis=-1, keepdims=True) if target_frequency is None else target_frequency
    control = np.clip(reference - omega + gain * np.sin(psi - theta), control_low, control_high)
    if n_couplings:
        strengths = np.broadcast_to(np.asarray(coupling, dtype=np.float64), (int(n_couplings),))
        strengths = np.broadcast_to(strengths, obs.shape[:-1] + (int(n_couplings),))
        control = np.concatenate((control, strengths), axis=-1)
    return control.astype(np.float32)


def linemsg_relay(observation: Any, *, joint_discrete: bool = True) -> np.ndarray:
    """Always relay: every agent chooses action 1 (keep the link to its right neighbour).

    An active middle agent obtains the message for sure when its right
    neighbour holds it and with probability 0.8 otherwise, and the source keeps
    it, so this fixed rule keeps the whole line informed (it maximises the
    expected reward of :class:`~env_lib.linemsg_env.LineMsgEnv`).

    Parameters
    ----------
    observation:
        Array of shape ``(..., num_agents, window)``.
    joint_discrete:
        Return the joint action as the integer ``2 ** num_agents - 1`` of the
        ``Discrete(2 ** num_agents)`` space (default space), else an array of
        ``num_agents`` ones (``MultiBinary``).

    Returns
    -------
    numpy.ndarray
        ``int64`` array of shape ``(...)`` (joint discrete) or ``int8`` array of
        shape ``(..., num_agents)``.
    """
    obs = np.asarray(observation)
    if obs.ndim < 2:
        raise ValueError(
            f"expected an observation of shape (..., num_agents, window), got {obs.shape}"
        )
    batch, n = obs.shape[:-2], obs.shape[-2]
    if joint_discrete:
        return np.full(batch, 2**n - 1, dtype=np.int64)
    return np.ones(batch + (n,), dtype=np.int8)


def wireless_schedule(
    observation: Any, *, grid_x: int, grid_y: int, ddl: int, n_obs_neighbors: int = 1
) -> np.ndarray:
    """Collision-free access schedule with borrowing of idle access points.

    Access point ``(a, b)`` is *owned* by agent ``(a, b)``: that agent sends
    to it (action 4, down-right) whenever it holds a packet, so no two owners
    ever collide. Agents without an access point of their own borrow an
    idle one; an access point is idle when its owner's queue is empty, which
    the borrower sees in its observation window:

    * last column ``(i, grid_y - 1)``: action 2 (down-left) to ``(i, grid_y - 2)``;
    * last row ``(grid_x - 1, j)``: action 3 (up-right) to ``(grid_x - 2, j)``;
      for ``j = grid_y - 2`` only if agent ``(grid_x - 2, grid_y - 1)``, which
      borrows the same access point, is empty as well;
    * corner ``(grid_x - 1, grid_y - 1)``: action 1 (up-left) if all three
      other agents around access point ``(grid_x - 2, grid_y - 2)`` are empty.

    Borrowing needs ``n_obs_neighbors >= 1``; with 0 only owners transmit.
    Agents without a packet stay idle.

    Parameters
    ----------
    observation:
        Array of shape ``(..., grid_x * grid_y, ddl * (2 n + 1) ** 2)``.
    grid_x, grid_y, ddl, n_obs_neighbors:
        Configuration of :class:`~env_lib.wireless_comm_env.WirelessCommEnv`.

    Returns
    -------
    numpy.ndarray
        ``int64`` actions of shape ``(..., grid_x * grid_y)``.
    """
    obs = np.asarray(observation)
    n, width = int(n_obs_neighbors), 2 * int(n_obs_neighbors) + 1
    n_agents = int(grid_x) * int(grid_y)
    if obs.ndim < 2 or obs.shape[-2:] != (n_agents, ddl * width * width):
        raise ValueError(
            f"expected an observation of shape (..., {n_agents}, {ddl * width * width}), "
            f"got {obs.shape}"
        )
    window = obs.reshape(obs.shape[:-1] + (ddl, width, width))
    occupied = np.any(window == 1, axis=-3)  # (..., n_agents, width, width)

    def holds(di: int, dj: int) -> np.ndarray:
        return occupied[..., n + di, n + dj]

    row, col = np.divmod(np.arange(n_agents), grid_y)
    own = holds(0, 0)
    actions = np.where(own & (row < grid_x - 1) & (col < grid_y - 1), 4, 0)
    if n >= 1:
        last_col = (col == grid_y - 1) & (row < grid_x - 1)
        actions = np.where(last_col & own & ~holds(0, -1), 2, actions)
        last_row = (row == grid_x - 1) & (col < grid_y - 1)
        shared = np.where(col == grid_y - 2, ~holds(-1, 1), True)
        actions = np.where(last_row & own & ~holds(-1, 0) & shared, 3, actions)
        corner = (row == grid_x - 1) & (col == grid_y - 1)
        free = ~holds(-1, -1) & ~holds(-1, 0) & ~holds(0, -1)
        actions = np.where(corner & own & free, 1, actions)
    return actions.astype(np.int64)


def pistonball_ramp(observation: Any, *, continuous: bool = True, gain: float = 8.0) -> np.ndarray:
    """Decentralised ramp heuristic for Pistonball.

    Column 1 of a piston's observation is its x position ``(5 + 40 i) / (40 n)``
    and column 2 the ball x position ``(x - 80) / (40 n)`` (zero when the piston
    does not see the ball), so ``d_i = (obs[1] - obs[2]) * n`` is the signed
    distance from piston ``i`` to the ball in piston widths. Every piston that
    sees the ball tracks a target height that rises from fully down (half a
    piston left of the ball) to fully up (half a piston right of it), which
    tilts the surface under the ball towards the goal; pistons that do not see
    the ball move down so that they never block it.

    Parameters
    ----------
    observation:
        Array of shape ``(..., n_pistons, 7)``.
    continuous:
        Continuous actions in ``[-1, 1]`` (else ``{0: down, 1: stay, 2: up}``).
    gain:
        Proportional gain on the height error.

    Returns
    -------
    numpy.ndarray
        ``float32`` (continuous) or ``int64`` actions of shape ``(..., n_pistons)``.
    """
    obs = np.asarray(observation, dtype=np.float64)
    if obs.ndim < 2 or obs.shape[-1] != 7:
        raise ValueError(f"expected an observation of shape (..., n_pistons, 7), got {obs.shape}")
    n_pistons = obs.shape[-2]
    sees_ball = np.any(obs[..., 2:] != 0.0, axis=-1)
    offset = (obs[..., 1] - obs[..., 2]) * n_pistons + 15.0 / 40.0
    raised = np.clip(0.5 + offset, 0.0, 1.0)  # 1 = fully up
    target = 1.0 - 2.0 * raised  # in units of observation column 0 (+1 = lowest)
    action = np.clip(gain * (obs[..., 0] - target), -1.0, 1.0)  # +1 moves up
    action = np.where(sees_ball, action, -1.0)
    if continuous:
        return action.astype(np.float32)
    return np.rint(action).astype(np.int64) + 1


def ajlatt_encircle(
    observation: Any,
    *,
    radius: float = 1.8,
    gain: float = 1.2,
    max_linear_velocity: float = 2.0,
    max_angular_velocity: float = math.pi / 4,
    heading_column: int = -2,
) -> np.ndarray:
    """Drive robot ``i`` to the ``i``-th of evenly spaced slots around its target belief.

    Slot ``i`` lies at distance ``radius`` from the believed target position,
    at the world angle ``2 pi i / n_robots``, so the team views the target from
    several directions. In the robot frame (columns 0-1 of the observation
    hold the target belief ``p``, the heading ``theta`` is part of the robot's
    own pose) the slot is ``g = p + radius (cos(a_i - theta), sin(a_i - theta))``.
    The robot turns towards the slot (towards the target once within 0.3 m of
    the slot) with ``omega = clip(gain * e, +-max_angular_velocity)``, ``e`` the
    bearing error, and drives with
    ``v = min(max_linear_velocity, 0.5 |g|) * max(0, cos e)``.

    Parameters
    ----------
    observation:
        Array of shape ``(..., n_robots, obs_dim)`` of
        :class:`~env_lib.ajlatt_env.AJLATTEnv`.
    radius, gain:
        Slot radius (metres) and heading gain.
    max_linear_velocity, max_angular_velocity:
        Action bounds.
    heading_column:
        Column of the robot's own heading (``self_pose`` is ``[-4:-1]``).

    Returns
    -------
    numpy.ndarray
        ``float32`` actions ``(v, omega)`` of shape ``(..., n_robots, 2)``.
    """
    obs = np.asarray(observation, dtype=np.float64)
    if obs.ndim < 2:
        raise ValueError(
            f"expected an observation of shape (..., n_robots, obs_dim), got {obs.shape}"
        )
    n_robots = obs.shape[-2]
    target = obs[..., 0:2]
    theta = obs[..., heading_column]
    angle = 2.0 * np.pi * np.arange(n_robots) / n_robots - theta
    goal = target + radius * np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    distance = np.hypot(goal[..., 0], goal[..., 1])
    error = np.where(
        distance < 0.3,
        np.arctan2(target[..., 1], target[..., 0]),
        np.arctan2(goal[..., 1], goal[..., 0]),
    )
    speed = np.minimum(max_linear_velocity, 0.5 * distance) * np.maximum(0.0, np.cos(error))
    turn = np.clip(gain * error, -max_angular_velocity, max_angular_velocity)
    return np.stack((speed, turn), axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Policy object
# ---------------------------------------------------------------------------
class BaselinePolicy:
    """A baseline controller bound to one environment configuration.

    Calling the object maps an observation (any leading batch shape) to an
    action. It is created by :func:`baseline_policy`.

    Attributes
    ----------
    family:
        Environment family.
    description:
        One-line description of the controller.
    parameters:
        Keyword arguments bound to the underlying pure function.
    """

    def __init__(
        self,
        function: Callable[..., Any],
        parameters: dict[str, Any],
        *,
        family: str,
        description: str,
        observation_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
        action_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    ) -> None:
        self.function = function
        self.parameters = dict(parameters)
        self.family = family
        self.description = description
        # (outer, core) shapes when a wrapper flattened the spaces.
        self._observation_shapes = observation_shapes
        self._action_shapes = action_shapes

    @property
    def name(self) -> str:
        """Label used by :func:`env_lib.evaluate` (``"baseline:<family>"``)."""
        return f"baseline:{self.family}"

    def __call__(self, observation: Any) -> Any:
        if hasattr(observation, "detach") and hasattr(observation, "cpu"):  # torch.Tensor
            import torch

            action = self._numpy_call(observation.detach().cpu().numpy())
            return torch.as_tensor(action, device=observation.device)
        return self._numpy_call(observation)

    def _numpy_call(self, observation: Any) -> np.ndarray:
        obs = np.asarray(observation)
        if self._observation_shapes is not None:
            outer, core = self._observation_shapes
            obs = obs.reshape(obs.shape[: obs.ndim - len(outer)] + core)
        action = np.asarray(self.function(obs, **self.parameters))
        if self._action_shapes is not None:
            outer, core = self._action_shapes
            action = action.reshape(action.shape[: action.ndim - len(core)] + outer)
        return action

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={_short_repr(v)}" for k, v in self.parameters.items())
        return f"BaselinePolicy({self.family}: {self.function.__name__}({params}))"


class _ParallelBaselinePolicy:
    """Baseline for a :class:`~env_lib.wrappers.ParallelEnvAdapter` (dict in, dict out)."""

    def __init__(self, adapter: Any, joint: BaselinePolicy) -> None:
        self.adapter = adapter
        self.joint = joint
        self.family = joint.family
        self.description = joint.description
        self.name = joint.name

    def __call__(self, observations: Mapping[str, Any]) -> dict[str, Any]:
        joint_obs = self.adapter.stack_observations(observations)
        return self.adapter.split_actions(self.joint(joint_obs), agents=list(observations))

    def __repr__(self) -> str:
        return f"ParallelBaselinePolicy({self.joint!r})"


def _short_repr(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size and np.all(value == value.flat[0]):
            return f"{float(value.flat[0]):g}"
        return f"array(shape={value.shape})"
    if isinstance(value, float):
        return f"{value:g}"
    return repr(value)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def family_of(env: Any) -> str:
    """Environment family of a (possibly wrapped or vectorised) ``env_lib`` environment.

    Raises
    ------
    TypeError
        If the environment is not an ``env_lib`` environment.
    """
    if isinstance(env, str):
        from env_lib.registration import get_spec

        return get_spec(env).family
    if hasattr(env, "possible_agents") and hasattr(env, "split_actions"):
        env = env.env
    return _prototype(env)[1]


def _family_of_core(core: Any) -> str:
    for cls in type(core).__mro__:
        family = _FAMILY_BY_CLASS.get(cls.__name__)
        if family is not None:
            return family
    parts = type(core).__module__.split(".")
    if len(parts) > 1 and parts[0] == "env_lib" and parts[1] in _FAMILY_BY_PACKAGE:
        return _FAMILY_BY_PACKAGE[parts[1]]
    raise TypeError(
        f"no baseline controller for {type(core).__module__}.{type(core).__name__}; "
        f"baselines exist for the env_lib families {sorted(_DESCRIPTIONS)}"
    )


def _prototype(env: Any) -> tuple[Any, str, spaces.Space, spaces.Space, spaces.Space, spaces.Space]:
    """``(core, family, core_obs_space, core_action_space, outer_obs_space, outer_action_space)``.

    ``core`` is the unwrapped (single or native vector) environment whose
    attributes configure the controller; the *outer* spaces are those of one
    copy as seen by the caller (after wrappers), the *core* spaces those of
    one copy of the unwrapped environment.
    """
    if isinstance(env, VectorEnv):
        outer_obs, outer_act = env.single_observation_space, env.single_action_space
        base = env.unwrapped
        if getattr(base, "envs", None):  # SyncVectorEnv
            core = base.envs[0].unwrapped
            family = _family_of_core(core)
            return core, family, core.observation_space, core.action_space, outer_obs, outer_act
        if getattr(base, "env_fns", None):  # AsyncVectorEnv: build one copy locally
            temporary = base.env_fns[0]()
            try:
                core = temporary.unwrapped
                family = _family_of_core(core)
                core_spaces = (core.observation_space, core.action_space)
                snapshot = _Snapshot(core)
            finally:
                temporary.close()
            return snapshot, family, *core_spaces, outer_obs, outer_act
        family = _family_of_core(base)  # native batched implementation
        return (
            base,
            family,
            base.single_observation_space,
            base.single_action_space,
            outer_obs,
            outer_act,
        )
    core = getattr(env, "unwrapped", None) or env
    family = _family_of_core(core)
    return (
        core,
        family,
        core.observation_space,
        core.action_space,
        env.observation_space,
        env.action_space,
    )


class _Snapshot:
    """Attributes of a closed environment copy (AsyncVectorEnv prototype)."""

    _ATTRIBUTES = (
        "dynamics",
        "damping",
        "reward_type",
        "target_frequency",
        "grid_x",
        "grid_y",
        "ddl",
        "n_obs_nghbr",
        "continuous",
    )

    def __init__(self, core: Any) -> None:
        self.core_type = type(core)
        for name in self._ATTRIBUTES:
            if hasattr(core, name):
                setattr(self, name, getattr(core, name))
        if hasattr(core, "observation_layout"):
            layout = core.observation_layout
            self.observation_layout = layout() if callable(layout) else layout


def baseline_policy(env: Any, **kwargs: Any) -> Any:
    """Classical baseline controller for an ``env_lib`` environment.

    Parameters
    ----------
    env:
        An ``env_lib`` environment (single or vector, native or
        Sync/Async, wrapped or not), a
        :class:`~env_lib.wrappers.ParallelEnvAdapter`, or a registered id
        (the default configuration is used).
    **kwargs:
        Controller parameters (see :func:`list_baselines` and the pure
        functions): ``gain``, ``velocity_gain`` (consensus); ``gain``,
        ``coupling`` (Kuramoto); ``gain`` (Pistonball); ``radius``, ``gain``
        (AJLATT); ``gain`` (PowerGrid, see ``droop_policy``); ``k_p``,
        ``k_d``, ``k_a``, ``accel_min``, ``accel_max`` (Platoon, see
        ``cacc_policy``; the bounds default to the action bounds); none for
        LineMsg and WirelessComm.

    Returns
    -------
    BaselinePolicy
        Callable ``policy(observation) -> action``. For a parallel adapter the
        policy maps observation dictionaries to action dictionaries.

    Raises
    ------
    TypeError
        For environments without a baseline or unknown keyword arguments.
    NotImplementedError
        If the environment's baseline module is not available.

    Examples
    --------
    >>> import env_lib
    >>> envs = env_lib.make_vec("Consensus-v0", num_envs=8, vectorization_mode="sync")
    >>> policy = env_lib.baseline_policy(envs)
    >>> obs, infos = envs.reset(seed=0)
    >>> policy(obs).shape
    (8, 8, 2)
    """
    if isinstance(env, str):
        from env_lib.registration import make

        created = make(env)
        try:
            return baseline_policy(created, **kwargs)
        finally:
            created.close()
    if hasattr(env, "possible_agents") and hasattr(env, "split_actions"):
        return _ParallelBaselinePolicy(env, baseline_policy(env.env, **kwargs))

    core, family, core_obs, core_act, outer_obs, outer_act = _prototype(env)
    builder = _BUILDERS[family]
    function, parameters = builder(core, core_obs, core_act, dict(kwargs))
    return BaselinePolicy(
        function,
        parameters,
        family=family,
        description=_DESCRIPTIONS[family],
        observation_shapes=_shape_adapter(outer_obs, core_obs, "observation"),
        action_shapes=_shape_adapter(outer_act, core_act, "action"),
    )


def _shape_adapter(
    outer: spaces.Space, core: spaces.Space, what: str
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    outer_shape, core_shape = outer.shape, core.shape
    if outer_shape == core_shape:
        return None
    if outer_shape is None or core_shape is None or math.prod(outer_shape) != math.prod(core_shape):
        raise TypeError(
            f"the environment's {what} space {outer} is not a reshaped version of the "
            f"unwrapped {what} space {core}; the baseline cannot be applied through this wrapper"
        )
    return tuple(outer_shape), tuple(core_shape)


def _take(kwargs: dict[str, Any], family: str, **defaults: Any) -> dict[str, Any]:
    unknown = set(kwargs) - set(defaults)
    if unknown:
        raise TypeError(
            f"unknown parameter(s) {sorted(unknown)} for the {family} baseline; "
            f"valid: {sorted(defaults)}"
        )
    return {**defaults, **kwargs}


def _build_consensus(core: Any, obs_space: Any, act_space: Any, kwargs: dict) -> tuple:
    params = _take(kwargs, "consensus", gain=1.0, velocity_gain=None)
    gain = float(params["gain"])
    if not gain > 0.0:
        raise ValueError(f"gain must be > 0, got {gain}")
    velocity_gain = params["velocity_gain"]
    if getattr(core, "dynamics", "single") == "double":
        if velocity_gain is None:
            velocity_gain = max(0.0, math.sqrt(gain) - float(getattr(core, "damping", 0.5)))
    elif velocity_gain is None:
        velocity_gain = 0.0
    max_control = float(np.max(act_space.high))
    return laplacian_consensus, {
        "gain": gain,
        "velocity_gain": float(velocity_gain),
        "max_control": max_control,
    }


def _build_kuramoto(core: Any, obs_space: Any, act_space: Any, kwargs: dict) -> tuple:
    params = _take(kwargs, "kuramoto", gain=3.0, coupling=None)
    obs_dim, act_dim = int(obs_space.shape[-1]), int(act_space.shape[-1])
    n = (obs_dim - act_dim) // 2
    m = act_dim - n
    if n < 1 or m < 0 or 3 * n + m != obs_dim:
        raise TypeError(
            f"cannot infer the Kuramoto layout from observation dim {obs_dim} and action dim "
            f"{act_dim}"
        )
    low = np.asarray(act_space.low, dtype=np.float64).reshape(-1)
    high = np.asarray(act_space.high, dtype=np.float64).reshape(-1)
    coupling = high[n:] if params["coupling"] is None else params["coupling"]
    reward_type = getattr(core, "reward_type", "order_parameter")
    target = None
    if reward_type == "frequency_synchronization":
        target = float(getattr(core, "target_frequency", 1.0))
    return kuramoto_feedback, {
        "n_oscillators": n,
        "n_couplings": m,
        "control_low": low[:n],
        "control_high": high[:n],
        "coupling": np.asarray(coupling, dtype=np.float64),
        "target_frequency": target,
        "gain": float(params["gain"]),
    }


def _build_linemsg(core: Any, obs_space: Any, act_space: Any, kwargs: dict) -> tuple:
    _take(kwargs, "linemsg")
    return linemsg_relay, {"joint_discrete": isinstance(act_space, spaces.Discrete)}


def _build_wireless(core: Any, obs_space: Any, act_space: Any, kwargs: dict) -> tuple:
    _take(kwargs, "wireless_comm")
    return wireless_schedule, {
        "grid_x": int(core.grid_x),
        "grid_y": int(core.grid_y),
        "ddl": int(core.ddl),
        "n_obs_neighbors": int(core.n_obs_nghbr),
    }


def _build_pistonball(core: Any, obs_space: Any, act_space: Any, kwargs: dict) -> tuple:
    params = _take(kwargs, "pistonball", gain=8.0)
    return pistonball_ramp, {
        "continuous": isinstance(act_space, spaces.Box),
        "gain": float(params["gain"]),
    }


def _build_ajlatt(core: Any, obs_space: Any, act_space: Any, kwargs: dict) -> tuple:
    params = _take(kwargs, "ajlatt", radius=1.8, gain=1.2)
    high = np.asarray(act_space.high, dtype=np.float64).reshape(-1, 2)
    heading = -2
    layout = getattr(core, "observation_layout", None)
    if layout is not None:
        layout = layout() if callable(layout) else layout
        heading = int(layout["self_pose"].start) + 2
    return ajlatt_encircle, {
        "radius": float(params["radius"]),
        "gain": float(params["gain"]),
        "max_linear_velocity": float(high[0, 0]),
        "max_angular_velocity": float(high[0, 1]),
        "heading_column": heading,
    }


def _import_baseline(family: str, name: str, modules: tuple[str, ...]) -> Callable[..., Any]:
    """Import a baseline function shipped by an environment package (lazily)."""
    errors = []
    for module_name in modules:
        try:
            return getattr(importlib.import_module(module_name), name)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise NotImplementedError(
        f"the {family} baseline {name} is not available ({'; '.join(errors)})"
    )


def _build_power_grid(core: Any, obs_space: Any, act_space: Any, kwargs: dict) -> tuple:
    function = _import_baseline(
        "power_grid",
        "droop_policy",
        ("env_lib.power_grid_env", "env_lib.power_grid_env.power_grid_env"),
    )
    default_gain = inspect.signature(function).parameters["gain"].default
    params = _take(kwargs, "power_grid", gain=default_gain)
    return function, {"gain": float(params["gain"])}


def _build_platoon(core: Any, obs_space: Any, act_space: Any, kwargs: dict) -> tuple:
    function = _import_baseline(
        "platoon", "cacc_policy", ("env_lib.platoon_env", "env_lib.platoon_env.platoon_env")
    )
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind == parameter.KEYWORD_ONLY
    }
    # Clip to the environment's actual command bounds unless given explicitly.
    if "accel_min" in defaults:
        defaults["accel_min"] = float(np.min(act_space.low))
    if "accel_max" in defaults:
        defaults["accel_max"] = float(np.max(act_space.high))
    return function, _take(kwargs, "platoon", **defaults)


_BUILDERS: dict[str, Callable[..., tuple]] = {
    "consensus": _build_consensus,
    "kuramoto": _build_kuramoto,
    "linemsg": _build_linemsg,
    "wireless_comm": _build_wireless,
    "pistonball": _build_pistonball,
    "ajlatt": _build_ajlatt,
    "power_grid": _build_power_grid,
    "platoon": _build_platoon,
}
