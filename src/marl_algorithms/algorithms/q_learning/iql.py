"""IQL: independent Q-learning with DQN agents."""

from __future__ import annotations

from typing import ClassVar

from marl_algorithms.algorithms.q_learning.common import _ValueBase

__all__ = ["IQL"]


class IQL(_ValueBase):
    """Independent Q-learning with DQN agents (Tan, 1993; Mnih et al., 2015).

    Every agent learns ``Q_i(o_i, a_i)`` from its own reward
    ``r_i`` (``info["agent_rewards"]``) with its own TD target::

        y_i = c r_i + gamma (1 - terminated) Q_i^-(o'_i, argmax_a Q_i(o'_i, a))
        L   = mean_batch mean_i  l(Q_i(o_i, a_i) - y_i)

    No mixing and no centralised information: the other agents are part of
    the environment. Environments that report no per-agent rewards give every
    agent the team reward; with a single agent IQL is (double) DQN.

    Agents are feed-forward networks trained on single transitions from a
    transition replay (see :mod:`marl_algorithms.algorithms.q_learning` for this
    simplification).

    Parameters
    ----------
    spec:
        Agent structure, or a discrete-action environment to read it from.
    config:
        :class:`QLearningConfig`; defaults to ``QLearningConfig()``.
    device:
        Torch device.
    seed:
        Seed of initialisation, exploration and replay sampling.
    **overrides:
        Configuration fields to override.

    Raises
    ------
    TypeError
        For continuous-action environments.

    Examples
    --------
    >>> from marl_algorithms import IQL, make_vector_env
    >>> envs = make_vector_env("LineMsg-v0", num_envs=16)
    >>> algo = IQL(envs, seed=0)
    >>> log = algo.learn(envs, total_steps=20_000, seed=0)
    """

    name: ClassVar[str] = "iql"
