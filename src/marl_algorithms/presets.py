"""Tuned training settings ("presets") for the demonstration environments.

Every algorithm module defines a ``PRESETS`` table
``{algorithm: {env_id: preset}}`` where a preset is a dictionary with the keys

* ``"total_steps"`` -- environment steps to train for;
* ``"num_envs"`` -- batched copies;
* ``"config"`` -- configuration overrides;
* ``"env_kwargs"`` -- environment constructor arguments (optional).

The presets were tuned to train in one to two minutes on one CPU core.
"""

from __future__ import annotations

import copy
import importlib
from typing import Any

from marl_algorithms.core.base import Algorithm, Callback, TrainingLog
from marl_algorithms.registry import _REGISTRY, train

__all__ = ["get_preset", "list_presets", "train_preset"]


def _table(algorithm: str) -> dict[str, dict[str, Any]]:
    key = algorithm.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown algorithm {algorithm!r}; available: {sorted(_REGISTRY)}")
    module = importlib.import_module(_REGISTRY[key].entry_point.split(":")[0])
    presets = getattr(module, "PRESETS", {})
    return presets.get(key, {})


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
    device: str = "cpu",
    callback: Callback | None = None,
    **config: Any,
) -> tuple[Algorithm, TrainingLog]:
    """Train with the preset of ``algorithm`` on ``env_id``.

    ``total_steps``, ``num_envs`` and configuration overrides replace the
    preset's values.

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
        env_kwargs=preset.get("env_kwargs"),
        device=device,
        callback=callback,
        **{**preset.get("config", {}), **config},
    )
