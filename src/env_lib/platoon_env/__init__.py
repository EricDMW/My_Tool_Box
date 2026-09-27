"""Cooperative adaptive cruise control of a vehicle platoon.

Registered as ``Platoon-v0``; :func:`env_lib.make_vec` creates the native
batched :class:`PlatoonVectorEnv`::

    import env_lib
    from env_lib.platoon_env import cacc_policy

    env = env_lib.make("Platoon-v0", topology="predecessor", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(cacc_policy(obs))

    envs = env_lib.make_vec("Platoon-v0", num_envs=1024)
    obs, infos = envs.reset(seed=0)
    obs, rewards, terminated, truncated, infos = envs.step(cacc_policy(obs))

The renderer (:mod:`env_lib.platoon_env.rendering`) is imported lazily on the
first ``render()`` call.
"""

from __future__ import annotations

from env_lib.platoon_env.platoon_env import (
    CACC_GAINS,
    FEATURE_SCALES,
    OBSERVATION_FEATURES,
    SCENARIOS,
    TOPOLOGIES,
    PlatoonConfig,
    PlatoonEnv,
    PlatoonVectorEnv,
    cacc_policy,
    platoon_adjacency,
    string_stability_gain,
)

__all__ = [
    "CACC_GAINS",
    "FEATURE_SCALES",
    "OBSERVATION_FEATURES",
    "SCENARIOS",
    "TOPOLOGIES",
    "PlatoonConfig",
    "PlatoonEnv",
    "PlatoonVectorEnv",
    "cacc_policy",
    "platoon_adjacency",
    "string_stability_gain",
]
