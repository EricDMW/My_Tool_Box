"""Tuned settings of IPPO and MAPPO for the env_lib environments."""

from __future__ import annotations

from typing import Any

__all__ = ["PRESETS"]

#: Tuned settings, ``PRESETS[algorithm][env_id]``, for ``algorithm`` in
#: ``("ippo", "mappo")``. Every entry holds ``num_envs`` (parallel copies),
#: ``total_steps`` (environment steps summed over copies), ``config``
#: (:class:`PPOConfig` overrides) and ``env_kwargs`` (environment arguments), so
#: ``marl_algorithms.train(algorithm, env_id, preset["total_steps"],
#: num_envs=preset["num_envs"], env_kwargs=preset["env_kwargs"], **preset["config"])``
#: reproduces a demonstration run.
#:
#: Mean team return of the deterministic policy after training with seed 0
#: (``env_lib.evaluate`` on 64 copies, 64 episodes, seed 1;
#: ``benchmarks/benchmark_marl.py``); training time is CPU time on one core
#: (``torch.set_num_threads(1)``) and "baseline" is ``env_lib.baseline_policy``:
#:
#: ============  =====  ==========  ========  ========  =======  ========
#: environment   algo   env steps   CPU time  random    trained  baseline
#: ============  =====  ==========  ========  ========  =======  ========
#: PowerGrid-v0  IPPO   200,704     69 s      -629.3    -2.37    -0.79
#: PowerGrid-v0  MAPPO  251,904     106 s     -629.3    -0.89    -0.79
#: Platoon-v0    IPPO   450,560     88 s      -5,350    -84.8    -46.4
#: Platoon-v0    MAPPO  450,560     102 s     -5,350    -65.0    -46.4
#: Consensus-v0  IPPO   401,408     83 s      -18,041   -1,550   -1,518
#: Consensus-v0  MAPPO  401,408     96 s      -18,041   -1,586   -1,518
#: LineMsg-v0    IPPO   100,800     26 s      43.2      95.0     95.0
#: LineMsg-v0    MAPPO  100,800     26 s      43.2      95.0     95.0
#: ============  =====  ==========  ========  ========  =======  ========
#:
#: Lessons from tuning, reflected in the settings:
#:
#: * PowerGrid: a small initial exploration noise (``log_std_init=-1.5``) is
#:   essential -- with more noise the mean policy learns to compensate its own
#:   (observed) previous actions and performs poorly when executed without
#:   noise; a shorter horizon (``gamma=0.95``) and ``lr=1e-3`` speed learning.
#: * Platoon and Consensus: the critics learn each agent's own reward
#:   (``reward_source="agent"``), which isolates the effect of a follower's or
#:   agent's own command, and the observations are used unnormalised: they
#:   are already scaled to order one (Platoon) or in arena units (Consensus),
#:   and running statistics dominated by the large errors of early training
#:   hide the small errors that matter once the task is nearly solved. On
#:   Platoon, MAPPO with the team reward and normalised observations stalls
#:   at about -800 (the followers never learn the predecessor-acceleration
#:   feed-forward that makes the platoon string stable, and every episode
#:   ends in a collision).
#: * LineMsg: MAPPO learns the relay from the team reward in about 30,000
#:   steps, so the preset is shorter than the others.
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "ippo": {
        "PowerGrid-v0": {
            "num_envs": 32,
            "total_steps": 200_000,
            "config": {
                "rollout_length": 64,
                "n_epochs": 5,
                "lr": 1e-3,
                "gamma": 0.95,
                "log_std_init": -1.5,
            },
            "env_kwargs": {},
        },
        "Platoon-v0": {
            "num_envs": 32,
            "total_steps": 450_000,
            "config": {
                "rollout_length": 64,
                "n_epochs": 5,
                "lr": 3e-4,
                "gamma": 0.99,
                "log_std_init": -1.0,
                "normalize_observations": False,
                "value_clip_range": None,
            },
            "env_kwargs": {},
        },
        "Consensus-v0": {
            "num_envs": 32,
            "total_steps": 400_000,
            "config": {
                "rollout_length": 64,
                "n_epochs": 5,
                "lr": 1e-3,
                "gamma": 0.95,
                "normalize_observations": False,
            },
            "env_kwargs": {},
        },
        "LineMsg-v0": {
            "num_envs": 32,
            "total_steps": 100_000,
            "config": {"rollout_length": 50, "n_epochs": 5, "lr": 1e-3, "ent_coef": 0.01},
            "env_kwargs": {},
        },
    },
    "mappo": {
        "PowerGrid-v0": {
            "num_envs": 32,
            "total_steps": 250_000,
            "config": {
                "rollout_length": 64,
                "n_epochs": 5,
                "lr": 1e-3,
                "gamma": 0.95,
                "log_std_init": -1.5,
            },
            "env_kwargs": {},
        },
        "Platoon-v0": {
            "num_envs": 32,
            "total_steps": 450_000,
            "config": {
                "rollout_length": 64,
                "n_epochs": 5,
                "lr": 3e-4,
                "gamma": 0.99,
                "log_std_init": -1.0,
                "reward_source": "agent",
                "normalize_observations": False,
                "value_clip_range": None,
            },
            "env_kwargs": {},
        },
        "Consensus-v0": {
            "num_envs": 32,
            "total_steps": 400_000,
            "config": {
                "rollout_length": 64,
                "n_epochs": 5,
                "lr": 1e-3,
                "gamma": 0.9,
                "reward_source": "agent",
                "normalize_observations": False,
            },
            "env_kwargs": {},
        },
        "LineMsg-v0": {
            "num_envs": 32,
            "total_steps": 100_000,
            "config": {"rollout_length": 50, "n_epochs": 5, "lr": 1e-3, "ent_coef": 0.01},
            "env_kwargs": {},
        },
    },
}
