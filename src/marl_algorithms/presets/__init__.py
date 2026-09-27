"""Tuned training settings ("presets") of the algorithms on the env_lib environments.

A preset fixes everything needed to reproduce one training run of an
algorithm on an environment. It is a dictionary with the keys

* ``"total_steps"`` -- environment steps to train for (summed over copies);
* ``"num_envs"`` -- batched copies;
* ``"config"`` -- configuration overrides of the algorithm;
* ``"env_kwargs"`` -- environment constructor arguments.

The tables live in one module per algorithm family, together with the notes
from tuning them: :mod:`~marl_algorithms.presets.ppo` (IPPO, MAPPO),
:mod:`~marl_algorithms.presets.ddpg` (MADDPG, MATD3) and
:mod:`~marl_algorithms.presets.q_learning` (IQL, VDN, QMIX). Every preset
trains in one to two minutes on one CPU core.
"""

from __future__ import annotations

import copy
from typing import Any

from marl_algorithms.core.base import Algorithm, Callback, TrainingLog
from marl_algorithms.presets import ddpg, ppo, q_learning
from marl_algorithms.registry import _REGISTRY, train

__all__ = ["PRESETS", "get_preset", "list_presets", "train_preset"]

#: Every preset, ``PRESETS[algorithm][env_id]``.
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    **ppo.PRESETS,
    **ddpg.PRESETS,
    **q_learning.PRESETS,
}


def _table(algorithm: str) -> dict[str, dict[str, Any]]:
    key = algorithm.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown algorithm {algorithm!r}; available: {sorted(_REGISTRY)}")
    return PRESETS.get(key, {})


def get_preset(algorithm: str, env_id: str) -> dict[str, Any] | None:
    """Preset of ``algorithm`` on ``env_id`` (a copy), or ``None``."""
    preset = _table(algorithm).get(env_id)
    return copy.deepcopy(preset) if preset is not None else None


def list_presets() -> list[tuple[str, str]]:
    """All ``(algorithm, env_id)`` pairs with a preset."""
    return [(name, env_id) for name in _REGISTRY for env_id in _table(name)]


def train_preset(
    algorithm: str,
    env_id: str,
    *,
    seed: int | None = 0,
    total_steps: int | None = None,
    num_envs: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
    device: str = "cpu",
    callback: Callback | None = None,
    **config: Any,
) -> tuple[Algorithm, TrainingLog]:
    """Train with the preset of ``algorithm`` on ``env_id``.

    ``total_steps``, ``num_envs`` and configuration overrides replace the
    preset's values; ``env_kwargs`` are merged into the preset's environment
    arguments.

    Raises
    ------
    KeyError
        If there is no preset for the pair.
    """
    preset = get_preset(algorithm, env_id)
    if preset is None:
        raise KeyError(f"no preset for {algorithm} on {env_id}; see list_presets()")
    return train(
        algorithm,
        env_id,
        total_steps if total_steps is not None else int(preset["total_steps"]),
        num_envs=num_envs if num_envs is not None else int(preset.get("num_envs", 16)),
        seed=seed,
        env_kwargs={**preset.get("env_kwargs", {}), **(env_kwargs or {})},
        device=device,
        callback=callback,
        **{**preset.get("config", {}), **config},
    )
