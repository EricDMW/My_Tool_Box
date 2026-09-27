"""Gymnasium registration for every environment shipped with ``env_lib``.

All environments are registered with string entry points, so registering them
does not import optional heavy dependencies (``pygame``, ``pymunk``,
``torch``). The dependency is only imported when the environment is created.

None of the registrations set ``max_episode_steps``: every environment enforces
its own episode limit through a constructor argument (``max_steps``,
``max_cycles``, ``max_iter``, ...). A second ``TimeLimit`` wrapper would
silently override a user-supplied limit.

Version suffixes follow the historical ids of this project: ``-v1``/``-v2``
denote preset *configurations* (e.g. ``WirelessComm-v1`` is the 4x4 grid), not
newer revisions of ``-v0``. :func:`make` suppresses Gymnasium's misleading
"out of date" ``DeprecationWarning`` for the ``-v0`` ids (``gymnasium.make``
still emits it; it can be ignored).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym

__all__ = ["ENV_SPECS", "EnvSpec", "get_spec", "list_envs", "make", "make_vec", "register_envs"]


@dataclass(frozen=True)
class EnvSpec:
    """Registration record of one environment id.

    Attributes
    ----------
    id:
        Gymnasium id.
    entry_point:
        ``"module:Class"`` of the single environment.
    kwargs:
        Constructor arguments of this configuration.
    description:
        One-line summary.
    family:
        Environment family (the ``env_lib`` subpackage without ``_env``).
    observation_type, action_type:
        ``"continuous"``, ``"discrete"`` or, for actions that can be either,
        ``"continuous|discrete"``.
    requires:
        Optional-dependency extra needed by the environment (``"torch"``,
        ``"pistonball"``) or ``None``.
    vector_entry_point:
        ``"module:Class"`` of the native batched implementation used by
        :func:`make_vec`, or ``None``.
    disable_env_checker:
        Skip Gymnasium's single-agent passive checker (for environments that
        return per-agent reward arrays).
    """

    id: str
    entry_point: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    family: str = ""
    observation_type: str = "continuous"
    action_type: str = "continuous"
    requires: str | None = None
    vector_entry_point: str | None = None
    disable_env_checker: bool = False


# Backwards-compatible private alias.
_Spec = EnvSpec


_KURAMOTO_NP = "env_lib.kos_env.kuramoto_env:KuramotoOscillatorEnv"
_KURAMOTO_NP_VEC = "env_lib.kos_env.vector:KuramotoOscillatorVectorEnv"
_KURAMOTO_TORCH = "env_lib.kos_env.kuramoto_env_torch:KuramotoOscillatorEnvTorch"
_CONSENSUS = "env_lib.consensus_env.consensus_env:ConsensusEnv"
_CONSENSUS_VEC = "env_lib.consensus_env.vector:ConsensusVectorEnv"
_POWER_GRID = "env_lib.power_grid_env.power_grid_env:PowerGridEnv"
_POWER_GRID_VEC = "env_lib.power_grid_env.power_grid_env:PowerGridVectorEnv"
_PLATOON = "env_lib.platoon_env.platoon_env:PlatoonEnv"
_PLATOON_VEC = "env_lib.platoon_env.platoon_env:PlatoonVectorEnv"

_KURAMOTO = {"family": "kuramoto", "vector_entry_point": _KURAMOTO_NP_VEC}
_KURAMOTO_T = {"family": "kuramoto", "requires": "torch"}
_DISCRETE = {"observation_type": "discrete", "action_type": "discrete"}

ENV_SPECS: list[EnvSpec] = [
    # Kuramoto oscillator networks (NumPy backend).
    _Spec(
        "KuramotoOscillator-v0", _KURAMOTO_NP, {}, "10 oscillators, dynamic coupling", **_KURAMOTO
    ),
    _Spec(
        "KuramotoOscillator-v1",
        _KURAMOTO_NP,
        {"n_oscillators": 5},
        "5 oscillators, dynamic coupling",
        **_KURAMOTO,
    ),
    _Spec(
        "KuramotoOscillator-Constant-v0",
        _KURAMOTO_NP,
        {"n_oscillators": 6, "coupling_mode": "constant"},
        "6 oscillators, fixed coupling matrix",
        **_KURAMOTO,
    ),
    _Spec(
        "KuramotoOscillator-FreqSync-Constant-v0",
        _KURAMOTO_NP,
        {
            "n_oscillators": 6,
            "coupling_mode": "constant",
            "reward_type": "frequency_synchronization",
            "target_frequency": 2.0,
        },
        "frequency tracking with fixed coupling",
        **_KURAMOTO,
    ),
    # Kuramoto oscillator networks (PyTorch backend, batched).
    _Spec(
        "KuramotoOscillatorTorch-v0",
        _KURAMOTO_TORCH,
        {},
        "PyTorch backend, 10 oscillators",
        **_KURAMOTO_T,
    ),
    _Spec(
        "KuramotoOscillatorTorch-v1",
        _KURAMOTO_TORCH,
        {"n_oscillators": 8, "n_agents": 4, "device": "cpu"},
        "PyTorch backend, 4 parallel systems",
        **_KURAMOTO_T,
    ),
    _Spec(
        "KuramotoOscillatorTorch-v2",
        _KURAMOTO_TORCH,
        {"n_oscillators": 10, "n_agents": 1, "device": "auto", "integration_method": "rk4"},
        "PyTorch backend, RK4, GPU when available",
        **_KURAMOTO_T,
    ),
    _Spec(
        "KuramotoOscillatorTorch-Constant-v0",
        _KURAMOTO_TORCH,
        {"n_oscillators": 6, "coupling_mode": "constant"},
        "PyTorch backend, fixed coupling matrix",
        **_KURAMOTO_T,
    ),
    _Spec(
        "KuramotoOscillatorTorch-FreqSync-Constant-v0",
        _KURAMOTO_TORCH,
        {
            "n_oscillators": 6,
            "coupling_mode": "constant",
            "reward_type": "frequency_synchronization",
            "target_frequency": 2.0,
        },
        "PyTorch backend, frequency tracking",
        **_KURAMOTO_T,
    ),
    # Networked multi-agent benchmarks with discrete actions.
    _Spec(
        "LineMsg-v0",
        "env_lib.linemsg_env.linemsg_env:LineMsgEnv",
        {},
        "message passing on a line",
        family="linemsg",
        **_DISCRETE,
    ),
    _Spec(
        "WirelessComm-v0",
        "env_lib.wireless_comm_env.wireless_comm_env:WirelessCommEnv",
        {},
        "6x6 wireless access grid",
        family="wireless_comm",
        **_DISCRETE,
    ),
    _Spec(
        "WirelessComm-v1",
        "env_lib.wireless_comm_env.wireless_comm_env:WirelessCommEnv",
        {"grid_x": 4, "grid_y": 4},
        "4x4 wireless access grid",
        family="wireless_comm",
        **_DISCRETE,
    ),
    _Spec(
        "Pistonball-v0",
        "env_lib.pistonball_env.pistonball_env:PistonballEnv",
        {},
        "cooperative physics game (requires pygame, pymunk)",
        family="pistonball",
        action_type="continuous|discrete",
        requires="pistonball",
    ),
    # Networked control with continuous states and actions.
    _Spec(
        "Consensus-v0",
        _CONSENSUS,
        {},
        "networked consensus / rendezvous",
        family="consensus",
        vector_entry_point=_CONSENSUS_VEC,
    ),
    _Spec(
        "Formation-v0",
        _CONSENSUS,
        {"task": "formation"},
        "networked formation control",
        family="consensus",
        vector_entry_point=_CONSENSUS_VEC,
    ),
    _Spec(
        "PowerGrid-v0",
        _POWER_GRID,
        {},
        "frequency control of a networked power system",
        family="power_grid",
        vector_entry_point=_POWER_GRID_VEC,
    ),
    _Spec(
        "Platoon-v0",
        _PLATOON,
        {},
        "cooperative adaptive cruise control of a vehicle platoon",
        family="platoon",
        vector_entry_point=_PLATOON_VEC,
    ),
    # Multi-robot active localisation and target tracking. The environment
    # returns per-robot reward and termination arrays (joint multi-agent API),
    # which Gymnasium's single-agent passive checker would flag.
    _Spec(
        "AJLATT-v0",
        "env_lib.ajlatt_env.env:AJLATTEnv",
        {},
        "multi-robot target tracking",
        family="ajlatt",
        disable_env_checker=True,
    ),
]


def register_envs() -> None:
    """Register all ``env_lib`` environments with Gymnasium (idempotent)."""
    for spec in ENV_SPECS:
        if spec.id in gym.registry:
            continue
        gym.register(
            id=spec.id,
            entry_point=spec.entry_point,
            vector_entry_point=spec.vector_entry_point,
            kwargs=dict(spec.kwargs),
            disable_env_checker=spec.disable_env_checker,
        )


def list_envs() -> list[str]:
    """Return the ids of all environments provided by ``env_lib``."""
    return [spec.id for spec in ENV_SPECS]


def get_spec(env_id: str) -> EnvSpec:
    """Registration record of ``env_id`` (see :class:`EnvSpec`)."""
    for spec in ENV_SPECS:
        if spec.id == env_id:
            return spec
    raise KeyError(f"unknown env_lib environment {env_id!r}; see env_lib.list_envs()")


def make(env_id: str, **kwargs: Any) -> gym.Env:
    """Create a registered environment.

    Thin convenience wrapper around :func:`gymnasium.make` that guarantees the
    ``env_lib`` environments are registered first.

    Parameters
    ----------
    env_id:
        One of :func:`list_envs`.
    **kwargs:
        Forwarded to the environment constructor (and to ``gymnasium.make``
        for options such as ``render_mode``).
    """
    register_envs()
    with warnings.catch_warnings():
        # Version suffixes of env_lib ids denote configurations, not revisions;
        # Gymnasium's "out of date" notice for -v0 ids is therefore misleading.
        warnings.filterwarnings("ignore", message=r".*is out of date", category=DeprecationWarning)
        return gym.make(env_id, **kwargs)


def make_vec(
    env_id: str,
    num_envs: int = 1,
    vectorization_mode: str | None = None,
    *,
    vector_kwargs: dict[str, Any] | None = None,
    wrappers: Any = None,
    **kwargs: Any,
) -> gym.vector.VectorEnv:
    """Create ``num_envs`` copies of a registered environment as one vector environment.

    Parameters
    ----------
    env_id:
        One of :func:`list_envs`.
    num_envs:
        Number of copies.
    vectorization_mode:
        ``None`` (default) selects the environment's native batched
        implementation when it has one (``EnvSpec.vector_entry_point``) and
        :class:`gymnasium.vector.SyncVectorEnv` otherwise. ``"sync"``,
        ``"async"`` and ``"vector_entry_point"`` force a mode.
    vector_kwargs:
        Arguments of the Sync/Async vector class (not allowed for the native
        implementation, which takes ``autoreset_mode`` through ``kwargs``).
    wrappers:
        Wrappers applied to every single environment (Sync/Async only).
    **kwargs:
        Environment constructor arguments.

    Examples
    --------
    >>> import env_lib
    >>> envs = env_lib.make_vec("Formation-v0", num_envs=256)   # native batch
    >>> obs, infos = envs.reset(seed=0)
    >>> obs.shape                                                   # (256, 8, 16)
    """
    register_envs()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*is out of date", category=DeprecationWarning)
        return gym.make_vec(
            env_id,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            vector_kwargs=vector_kwargs,
            wrappers=wrappers,
            **kwargs,
        )
