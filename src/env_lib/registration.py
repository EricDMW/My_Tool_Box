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

import re
import warnings
from dataclasses import dataclass, field, replace
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
    limit_kwarg:
        Name of the constructor argument that sets the episode length
        (``max_steps``, ``max_iter``, ...). :func:`make` and :func:`make_vec`
        translate ``max_episode_steps`` into it.
    per_agent_rewards:
        The environment returns per-agent reward and termination arrays
        instead of scalars; :func:`make_vec` then wraps the copies of
        Gymnasium's Sync/Async vector classes in
        :class:`~env_lib.wrappers.TeamReward`.
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
    limit_kwarg: str = "max_steps"
    per_agent_rewards: bool = False
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
        limit_kwarg="max_iter",
        **_DISCRETE,
    ),
    _Spec(
        "WirelessComm-v0",
        "env_lib.wireless_comm_env.wireless_comm_env:WirelessCommEnv",
        {},
        "6x6 wireless access grid",
        family="wireless_comm",
        limit_kwarg="max_iter",
        **_DISCRETE,
    ),
    _Spec(
        "WirelessComm-v1",
        "env_lib.wireless_comm_env.wireless_comm_env:WirelessCommEnv",
        {"grid_x": 4, "grid_y": 4},
        "4x4 wireless access grid",
        family="wireless_comm",
        limit_kwarg="max_iter",
        **_DISCRETE,
    ),
    _Spec(
        "Pistonball-v0",
        "env_lib.pistonball_env.pistonball_env:PistonballEnv",
        {},
        "cooperative physics game (requires pygame, pymunk)",
        family="pistonball",
        action_type="continuous|discrete",
        limit_kwarg="max_cycles",
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
        limit_kwarg="max_episode_steps",
        per_agent_rewards=True,
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


def _resolve(env_id: str, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Route ``max_episode_steps`` to the environment's own limit argument.

    Returns the id (or a Gymnasium spec carrying the limit) and the remaining
    keyword arguments. Without this, ``gymnasium.make`` would consume
    ``max_episode_steps`` and add a ``TimeLimit`` wrapper on top of the
    environment's own limit.
    """
    if "max_episode_steps" not in kwargs:
        return env_id, kwargs
    kwargs = dict(kwargs)
    limit = kwargs.pop("max_episode_steps")
    if limit is None:
        return env_id, kwargs
    try:
        target = get_spec(env_id).limit_kwarg
    except KeyError:  # not an env_lib id: let Gymnasium handle it
        kwargs["max_episode_steps"] = limit
        return env_id, kwargs
    if target in kwargs and kwargs[target] != limit:
        raise ValueError(
            f"max_episode_steps={limit!r} conflicts with {target}={kwargs[target]!r}; pass only one"
        )
    kwargs.pop(target, None)
    gym_spec = gym.spec(env_id)
    return replace(gym_spec, kwargs={**gym_spec.kwargs, target: limit}), kwargs


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
        for options such as ``render_mode``). ``max_episode_steps`` sets the
        environment's own episode limit (``max_steps``, ``max_iter``, ...; see
        :attr:`EnvSpec.limit_kwarg`) instead of adding a ``TimeLimit`` wrapper.
    """
    register_envs()
    target, kwargs = _resolve(env_id, kwargs)
    with warnings.catch_warnings():
        # Version suffixes of env_lib ids denote configurations, not revisions;
        # Gymnasium's "out of date" notice for -v0 ids is therefore misleading.
        warnings.filterwarnings("ignore", message=r".*is out of date", category=DeprecationWarning)
        try:
            return gym.make(target, **kwargs)
        except TypeError as exc:
            raise _argument_error(env_id, exc) from exc


def _argument_error(env_id: str, exc: TypeError) -> TypeError:
    """Name the environment and the argument instead of an internal function."""
    match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
    if match is None:
        return exc
    return TypeError(
        f"{env_id} does not accept the argument {match.group(1)!r}; "
        f"`env-lib describe {env_id}` lists its parameters"
    )


def make_vec(
    env_id: str,
    num_envs: int = 1,
    vectorization_mode: str | None = None,
    *,
    autoreset_mode: Any = None,
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
    autoreset_mode:
        ``"next_step"`` (Gymnasium's default), ``"same_step"`` or
        ``"disabled"`` (``gymnasium.vector.AutoresetMode`` members and the
        spellings ``"NextStep"``, ... are accepted too), for every
        vectorisation mode.
    vector_kwargs:
        Further arguments of the Sync/Async vector class (not allowed for the
        native implementation).
    wrappers:
        Wrappers applied to every single environment (Sync/Async only). For
        environments with per-agent reward arrays (``EnvSpec.per_agent_rewards``,
        AJLATT) :class:`~env_lib.wrappers.TeamReward` is applied when no
        wrappers are given, because Gymnasium's vector classes need scalar
        rewards; the per-agent rewards stay in ``infos["agent_rewards"]``.
    **kwargs:
        Environment constructor arguments (``max_episode_steps`` as in
        :func:`make`).

    Examples
    --------
    >>> import env_lib
    >>> envs = env_lib.make_vec("Formation-v0", num_envs=256)   # native batch
    >>> obs, infos = envs.reset(seed=0)
    >>> obs.shape                                                   # (256, 8, 16)
    """
    register_envs()
    if isinstance(num_envs, bool) or int(num_envs) != num_envs or num_envs < 1:
        raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
    target, kwargs = _resolve(env_id, kwargs)
    try:
        spec: EnvSpec | None = get_spec(env_id)
    except KeyError:
        spec = None
    mode = vectorization_mode
    if mode is None:
        has_native = gym.spec(env_id).vector_entry_point is not None
        mode = "vector_entry_point" if has_native else "sync"
    native = str(getattr(mode, "value", mode)) == "vector_entry_point"
    vector_kwargs = dict(vector_kwargs or {})
    if autoreset_mode is not None:
        if native:
            kwargs["autoreset_mode"] = autoreset_mode
        else:
            vector_kwargs["autoreset_mode"] = _gymnasium_autoreset_mode(autoreset_mode)
    if not native and wrappers is None and spec is not None and spec.per_agent_rewards:
        from env_lib.wrappers import TeamReward

        wrappers = [TeamReward]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*is out of date", category=DeprecationWarning)
        try:
            return gym.make_vec(
                target,
                num_envs=num_envs,
                vectorization_mode=mode,
                vector_kwargs=vector_kwargs,
                wrappers=wrappers,
                **kwargs,
            )
        except TypeError as exc:
            raise _argument_error(env_id, exc) from exc


def _gymnasium_autoreset_mode(value: Any) -> Any:
    """Convert an autoreset mode to ``gymnasium.vector.AutoresetMode``."""
    try:
        from gymnasium.vector import AutoresetMode
    except ImportError:  # Gymnasium 1.0: only next-step autoreset
        raw = str(getattr(value, "value", value)).replace("_", "").lower()
        if raw != "nextstep":
            raise ValueError(
                "Gymnasium 1.0 vector environments only support next-step autoreset; "
                "upgrade Gymnasium or use a native vector environment"
            ) from None
        return None
    if isinstance(value, AutoresetMode):
        return value
    raw = str(value).replace("_", "").lower()
    for member in AutoresetMode:
        if member.value.lower() == raw:
            return member
    raise ValueError(
        f"autoreset_mode must be one of 'next_step', 'same_step', 'disabled', got {value!r}"
    )
