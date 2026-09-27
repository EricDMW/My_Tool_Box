"""Wrappers and adapters that connect ``env_lib`` environments to RL libraries.

* :class:`FlattenJointSpaces` -- flat ``float32`` observation vector and flat
  action, for single-agent libraries (Stable-Baselines3, CleanRL, ...).
* :class:`TeamReward` -- scalar team reward and termination flag for
  environments with per-agent reward/termination arrays (AJLATT).
* :class:`ParallelEnvAdapter` / :func:`to_parallel` -- the PettingZoo Parallel
  API (per-agent dictionaries), for multi-agent libraries.

Example
-------
>>> import env_lib
>>> from env_lib.wrappers import FlattenJointSpaces, to_parallel
>>> env = FlattenJointSpaces(env_lib.make("Formation-v0"))
>>> env.observation_space.shape
(128,)
>>> par_env = to_parallel("Formation-v0")
>>> observations, infos = par_env.reset(seed=0)
>>> sorted(observations)[:2]
['agent_0', 'agent_1']
"""

from __future__ import annotations

from env_lib.wrappers.flatten import FlattenJointSpaces
from env_lib.wrappers.parallel import PETTINGZOO_AVAILABLE, ParallelEnvAdapter, to_parallel
from env_lib.wrappers.team_reward import TeamReward

__all__ = [
    "PETTINGZOO_AVAILABLE",
    "FlattenJointSpaces",
    "ParallelEnvAdapter",
    "TeamReward",
    "to_parallel",
]
