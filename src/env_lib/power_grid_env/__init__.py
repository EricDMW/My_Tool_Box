"""Frequency control of a networked power system.

Registered as ``PowerGrid-v0``::

    import env_lib
    from env_lib.power_grid_env import droop_policy

    env = env_lib.make("PowerGrid-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(droop_policy(obs))

    envs = env_lib.make_vec("PowerGrid-v0", num_envs=1024)   # native batch

The renderer (:mod:`env_lib.power_grid_env.rendering`) is imported lazily on
the first ``render()`` call.
"""

from __future__ import annotations

from env_lib.power_grid_env.power_grid_env import (
    DEFAULT_DROOP_GAIN,
    OBSERVATION_FEATURES,
    OBSERVATION_SCALE,
    TOPOLOGIES,
    PowerGridEnv,
    PowerGridVectorEnv,
    droop_policy,
)

__all__ = [
    "DEFAULT_DROOP_GAIN",
    "OBSERVATION_FEATURES",
    "OBSERVATION_SCALE",
    "TOPOLOGIES",
    "PowerGridEnv",
    "PowerGridVectorEnv",
    "droop_policy",
]
