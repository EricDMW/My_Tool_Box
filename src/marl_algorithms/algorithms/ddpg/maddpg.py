"""MADDPG: multi-agent deep deterministic policy gradient."""

from __future__ import annotations

from typing import ClassVar

from marl_algorithms.algorithms.ddpg.common import _MADDPGBase

__all__ = ["MADDPG"]


class MADDPG(_MADDPGBase):
    """Multi-agent deep deterministic policy gradient (Lowe et al., 2017).

    Agent ``i`` has a deterministic actor ``mu_i(o_i)`` and a centralised critic
    ``Q_i(s, a_1, ..., a_n)`` with target copies ``mu'_i`` and ``Q'_i``. On a
    replay batch ``(o, a, r, o', d)`` with global states ``s = (o_1..o_n)``:

    * critic loss (TD regression, ``c`` = ``reward_scale``)::

          y_i = c r_i + gamma (1 - d) Q'_i(s', mu'_1(o'_1), ..., mu'_n(o'_n))
          L(Q_i) = mean (Q_i(s, a_1, ..., a_n) - y_i)^2

      (or the Huber loss of the TD error with ``critic_loss="huber"``);

    * actor loss (deterministic policy gradient; the other agents' actions come
      from the replay batch)::

          L(mu_i) = -mean Q_i(s, a_1, ..., mu_i(o_i), ..., a_n)

    * target networks: ``theta' <- (1 - tau) theta' + tau theta`` after every
      update.

    ``d`` is the ``terminated`` flag: truncated episodes bootstrap from their
    final observations. ``r_i`` is agent ``i``'s own reward
    (``reward_source="agent"``) or the team reward (``"team"``). Exploration
    adds Gaussian noise to ``mu_i(o_i)``. With shared parameters (default) one
    critic, evaluated with agent ``i``'s one-hot identifier, plays all ``Q_i``;
    by default its input also repeats ``o_i`` and ``a_i``
    (``critic_local_inputs``). Losses are averaged over agents, and all agents
    are updated in one batched pass.

    Parameters
    ----------
    spec:
        Agent structure, or an environment (single or vector) to read it from.
        The actions must be continuous.
    config:
        :class:`DDPGConfig`; defaults to ``DDPGConfig()``.
    device:
        Torch device.
    seed:
        Seed of initialisation, exploration and replay sampling.
    **overrides:
        :class:`DDPGConfig` fields, for example ``actor_lr=3e-4``.

    Raises
    ------
    TypeError
        For discrete action spaces or unknown configuration fields.

    Examples
    --------
    >>> from marl_algorithms import MADDPG, make_vector_env
    >>> envs = make_vector_env("PowerGrid-v0", num_envs=16, n_buses=6)
    >>> algo = MADDPG(envs, seed=0, warmup_steps=1_000)
    >>> log = algo.learn(envs, total_steps=5_000, seed=0)
    >>> algo.act(envs.reset(seed=1)[0], deterministic=True).shape
    (16, 6, 1)
    """

    name: ClassVar[str] = "maddpg"
