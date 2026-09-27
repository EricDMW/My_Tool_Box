"""Tuned settings of MADDPG and MATD3 for the env_lib environments."""

from __future__ import annotations

import copy
from typing import Any

__all__ = ["PRESETS"]

# PowerGrid-v0 (16 buses): the team reward (the evaluation objective) avoids a
# free-rider effect of the per-agent rewards, where each bus sees only about
# 1/16 of the benefit of its frequency support but pays its full control cost.
# The Huber loss keeps the rare trip transitions (team reward about -1600) from
# dominating the critic updates. Consensus-v0 (8 agents on a ring): per-agent
# rewards (each agent's own neighbourhood disagreement), rewards scaled by 0.01
# to order one, observations normalised with warm-up statistics.
_POWER_GRID: dict[str, Any] = {
    "num_envs": 16,
    "total_steps": 64_000,
    "config": {
        "hidden_sizes": (64, 64),
        "critic_hidden_sizes": (128, 128),
        "batch_size": 64,
        "warmup_steps": 3_200,
        "gamma": 0.99,
        "reward_source": "team",
        "critic_loss": "huber",
    },
    "env_kwargs": {},
}
_CONSENSUS: dict[str, Any] = {
    "num_envs": 16,
    "total_steps": 96_000,
    "config": {
        "hidden_sizes": (64, 64),
        "critic_hidden_sizes": (128, 128),
        "batch_size": 64,
        "warmup_steps": 3_200,
        "gamma": 0.95,
        "reward_scale": 0.01,
        "normalize_observations": True,
        "exploration_noise": 0.2,
        "final_exploration_noise": 0.02,
        "noise_decay_steps": 96_000,
    },
    "env_kwargs": {},
}

#: Tuned settings, ``PRESETS[algorithm][env_id]``, for ``algorithm`` in
#: ``("maddpg", "matd3")``. Every entry holds ``num_envs`` (parallel copies),
#: ``total_steps`` (environment steps summed over copies), ``config``
#: (:class:`DDPGConfig` overrides) and ``env_kwargs`` (environment arguments), so
#: ``marl_algorithms.train(algorithm, env_id, preset["total_steps"],
#: num_envs=preset["num_envs"], env_kwargs=preset["env_kwargs"], **preset["config"])``
#: reproduces a demonstration run. Each trains in about 60-120 s on one CPU core.
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "maddpg": {
        "PowerGrid-v0": copy.deepcopy(_POWER_GRID),
        "Consensus-v0": copy.deepcopy(_CONSENSUS),
    },
    "matd3": {
        "PowerGrid-v0": {
            **copy.deepcopy(_POWER_GRID),
            "config": {
                **_POWER_GRID["config"],
                "exploration_noise": 0.3,
                "final_exploration_noise": 0.05,
                "noise_decay_steps": 64_000,
            },
        },
        "Consensus-v0": copy.deepcopy(_CONSENSUS),
    },
}
