"""Registry of the available algorithms and the one-call training entry point."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from marl_algorithms.core.base import Algorithm, Callback, TrainingLog
from marl_algorithms.core.runner import make_vector_env

__all__ = ["AlgorithmInfo", "get_algorithm", "list_algorithms", "make_algorithm", "train"]


@dataclass(frozen=True)
class AlgorithmInfo:
    """Description of a registered algorithm.

    Attributes
    ----------
    name:
        Registry name.
    entry_point:
        ``"module:Class"``.
    family:
        ``"on-policy"``, ``"off-policy actor-critic"`` or ``"value-based"``.
    action_kinds:
        Supported action kinds.
    summary:
        One-line description.
    reference:
        The paper that introduced the method.
    """

    name: str
    entry_point: str
    family: str
    action_kinds: tuple[str, ...]
    summary: str
    reference: str


_BOTH = ("continuous", "discrete")
_REGISTRY: dict[str, AlgorithmInfo] = {
    info.name: info
    for info in (
        AlgorithmInfo(
            "ippo",
            "marl_algorithms.algorithms.ppo.ippo:IPPO",
            "on-policy",
            _BOTH,
            "independent PPO: every agent learns from its own observation and reward",
            "de Witt et al., 2020, Is independent learning all you need in the StarCraft multi-agent challenge?",
        ),
        AlgorithmInfo(
            "mappo",
            "marl_algorithms.algorithms.ppo.mappo:MAPPO",
            "on-policy",
            _BOTH,
            "multi-agent PPO: decentralised actors, centralised critic on the global state",
            "Yu et al., 2022, The surprising effectiveness of PPO in cooperative multi-agent games",
        ),
        AlgorithmInfo(
            "maddpg",
            "marl_algorithms.algorithms.ddpg.maddpg:MADDPG",
            "off-policy actor-critic",
            ("continuous",),
            "multi-agent DDPG: deterministic actors, centralised Q-critics on joint actions",
            "Lowe et al., 2017, Multi-agent actor-critic for mixed cooperative-competitive environments",
        ),
        AlgorithmInfo(
            "matd3",
            "marl_algorithms.algorithms.ddpg.matd3:MATD3",
            "off-policy actor-critic",
            ("continuous",),
            "MADDPG with twin critics, delayed actor updates and target smoothing",
            "Ackermann et al., 2019, Reducing overestimation bias in multi-agent domains using double centralized critics",
        ),
        AlgorithmInfo(
            "iql",
            "marl_algorithms.algorithms.q_learning.iql:IQL",
            "value-based",
            ("discrete",),
            "independent Q-learning: one DQN per agent (shared weights)",
            "Tan, 1993, Multi-agent reinforcement learning: independent vs. cooperative agents",
        ),
        AlgorithmInfo(
            "vdn",
            "marl_algorithms.algorithms.q_learning.vdn:VDN",
            "value-based",
            ("discrete",),
            "value decomposition networks: team value as the sum of agent utilities",
            "Sunehag et al., 2018, Value-decomposition networks for cooperative multi-agent learning "
            "based on team reward",
        ),
        AlgorithmInfo(
            "qmix",
            "marl_algorithms.algorithms.q_learning.qmix:QMIX",
            "value-based",
            ("discrete",),
            "QMIX: monotonic, state-conditioned mixing of agent utilities",
            "Rashid et al., 2018, QMIX: monotonic value function factorisation for deep multi-agent "
            "reinforcement learning",
        ),
    )
}


def list_algorithms() -> list[AlgorithmInfo]:
    """Descriptions of all registered algorithms, in registry order."""
    return list(_REGISTRY.values())


def get_algorithm(name: str) -> type[Algorithm]:
    """Algorithm class registered under ``name`` (case-insensitive)."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown algorithm {name!r}; available: {sorted(_REGISTRY)}")
    module_name, class_name = _REGISTRY[key].entry_point.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def make_algorithm(name: str, env_or_spec: Any, **kwargs: Any) -> Algorithm:
    """Instantiate a registered algorithm for an environment or spec.

    ``kwargs`` are ``config``, ``device``, ``seed`` and configuration overrides.
    """
    return get_algorithm(name)(env_or_spec, **kwargs)


def train(
    algorithm: str,
    env_id: str,
    total_steps: int,
    *,
    num_envs: int = 16,
    seed: int | None = 0,
    env_kwargs: dict[str, Any] | None = None,
    device: str = "cpu",
    callback: Callback | None = None,
    **config: Any,
) -> tuple[Algorithm, TrainingLog]:
    """Train ``algorithm`` on ``num_envs`` batched copies of an ``env_lib`` environment.

    Parameters
    ----------
    algorithm:
        Registry name (see :func:`list_algorithms`).
    env_id:
        ``env_lib`` environment id.
    total_steps:
        Environment steps to collect (summed over copies).
    num_envs:
        Copies simulated in parallel (native batch when available).
    seed:
        Seed of the algorithm and the environment reset.
    env_kwargs:
        Environment constructor arguments.
    device:
        Torch device.
    callback:
        See :meth:`Algorithm.learn`.
    **config:
        Configuration overrides.

    Returns
    -------
    tuple
        ``(algorithm, training_log)``.

    Examples
    --------
    >>> from marl_algorithms import train
    >>> algo, log = train("mappo", "PowerGrid-v0", total_steps=200_000, num_envs=32)
    >>> print(algo.evaluate(env_lib.make_vec("PowerGrid-v0", 16), n_episodes=32))
    """
    envs = make_vector_env(env_id, num_envs, **(env_kwargs or {}))
    try:
        agent = make_algorithm(algorithm, envs, device=device, seed=seed, **config)
        log = agent.learn(envs, total_steps, seed=seed, callback=callback)
    finally:
        envs.close()
    log.env_id = env_id
    return agent, log
