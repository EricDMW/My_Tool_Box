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

__all__ = ["ENV_SPECS", "list_envs", "make", "register_envs"]


@dataclass(frozen=True)
class _Spec:
    id: str
    entry_point: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    disable_env_checker: bool = False


_KURAMOTO_NP = "env_lib.kos_env.kuramoto_env:KuramotoOscillatorEnv"
_KURAMOTO_TORCH = "env_lib.kos_env.kuramoto_env_torch:KuramotoOscillatorEnvTorch"

ENV_SPECS: list[_Spec] = [
    # Kuramoto oscillator networks (NumPy backend).
    _Spec("KuramotoOscillator-v0", _KURAMOTO_NP, {}, "10 oscillators, dynamic coupling"),
    _Spec(
        "KuramotoOscillator-v1",
        _KURAMOTO_NP,
        {"n_oscillators": 5},
        "5 oscillators, dynamic coupling",
    ),
    _Spec(
        "KuramotoOscillator-Constant-v0",
        _KURAMOTO_NP,
        {"n_oscillators": 6, "coupling_mode": "constant"},
        "6 oscillators, fixed coupling matrix",
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
    ),
    # Kuramoto oscillator networks (PyTorch backend, batched).
    _Spec("KuramotoOscillatorTorch-v0", _KURAMOTO_TORCH, {}, "PyTorch backend, 10 oscillators"),
    _Spec(
        "KuramotoOscillatorTorch-v1",
        _KURAMOTO_TORCH,
        {"n_oscillators": 8, "n_agents": 4, "device": "cpu"},
        "PyTorch backend, 4 parallel systems",
    ),
    _Spec(
        "KuramotoOscillatorTorch-v2",
        _KURAMOTO_TORCH,
        {"n_oscillators": 10, "n_agents": 1, "device": "auto", "integration_method": "rk4"},
        "PyTorch backend, RK4, GPU when available",
    ),
    _Spec(
        "KuramotoOscillatorTorch-Constant-v0",
        _KURAMOTO_TORCH,
        {"n_oscillators": 6, "coupling_mode": "constant"},
        "PyTorch backend, fixed coupling matrix",
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
    ),
    # Networked multi-agent benchmarks.
    _Spec(
        "LineMsg-v0", "env_lib.linemsg_env.linemsg_env:LineMsgEnv", {}, "message passing on a line"
    ),
    _Spec(
        "WirelessComm-v0",
        "env_lib.wireless_comm_env.wireless_comm_env:WirelessCommEnv",
        {},
        "6x6 wireless access grid",
    ),
    _Spec(
        "WirelessComm-v1",
        "env_lib.wireless_comm_env.wireless_comm_env:WirelessCommEnv",
        {"grid_x": 4, "grid_y": 4},
        "4x4 wireless access grid",
    ),
    _Spec(
        "Pistonball-v0",
        "env_lib.pistonball_env.pistonball_env:PistonballEnv",
        {},
        "cooperative physics game (requires pygame, pymunk)",
    ),
    _Spec(
        "Consensus-v0",
        "env_lib.consensus_env.consensus_env:ConsensusEnv",
        {},
        "networked consensus / rendezvous",
    ),
    _Spec(
        "Formation-v0",
        "env_lib.consensus_env.consensus_env:ConsensusEnv",
        {"task": "formation"},
        "networked formation control",
    ),
    # Multi-robot active localisation and target tracking. The environment
    # returns per-robot reward and termination arrays (joint multi-agent API),
    # which Gymnasium's single-agent passive checker would flag.
    _Spec(
        "AJLATT-v0",
        "env_lib.ajlatt_env.env:AJLATTEnv",
        {},
        "multi-robot target tracking",
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
            kwargs=dict(spec.kwargs),
            disable_env_checker=spec.disable_env_checker,
        )


def list_envs() -> list[str]:
    """Return the ids of all environments provided by ``env_lib``."""
    return [spec.id for spec in ENV_SPECS]


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
