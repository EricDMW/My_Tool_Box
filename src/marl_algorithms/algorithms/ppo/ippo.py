"""IPPO: independent PPO, with one decentralised critic per agent."""

from __future__ import annotations

from typing import ClassVar

import torch

from marl_algorithms.algorithms.ppo.common import _PPOBase

__all__ = ["IPPO"]


class IPPO(_PPOBase):
    """Independent PPO (de Witt et al., 2020).

    Every agent ``i`` has a decentralised actor ``pi_theta(a_i | o_i)`` and a
    decentralised critic ``V_phi(o_i)``, trained on the agent's own reward
    ``r_t^i`` (``reward_source="agent"``, the default; ``"team"`` trains every
    critic on the team reward). With ``share_parameters`` (default) all agents
    use the same actor and critic and the input is ``[o_i, onehot(i)]``.

    For every agent the advantages are generalised advantage estimates,

    ``delta_t^i = r_t^i + gamma (1 - term_t) V_phi(o_{t+1}^i) - V_phi(o_t^i)``,
    ``A_t^i = sum_l (gamma lambda)^l delta_{t+l}^i`` (cut at episode ends),

    with ``V(o_{t+1})`` of the final observation for truncated episodes, and the
    value targets are ``R_t^i = A_t^i + V_old(o_t^i)``. With the probability
    ratio ``rho_t^i = pi_theta(a_t^i | o_t^i) / pi_old(a_t^i | o_t^i)`` the actor
    minimises

    ``L_pi = -mean[min(rho A, clip(rho, 1 - eps, 1 + eps) A)] - c_H mean[H(pi_theta(. | o_t^i))]``

    and the critic minimises

    ``L_V = mean[max(l(V_phi - R), l(V_old + clip(V_phi - V_old, -eps_V, eps_V) - R))]``

    with ``l(x) = x^2 / 2`` or the Huber loss, the means running over steps,
    copies and agents. Actor and critic have separate Adam optimisers and
    gradient-norm clipping.

    Parameters
    ----------
    spec:
        Agent structure, or an environment (single or vector) to read it from.
    config:
        :class:`PPOConfig`; defaults to ``PPOConfig()``.
    device:
        Torch device.
    seed:
        Seed of the initialisation and of all sampling.
    **overrides:
        :class:`PPOConfig` fields, for example ``lr=1e-3``.

    Examples
    --------
    >>> from marl_algorithms import IPPO, make_vector_env
    >>> envs = make_vector_env("PowerGrid-v0", 16)
    >>> algo = IPPO(envs, seed=0)
    >>> log = algo.learn(envs, 20_000, seed=0)
    >>> actions = algo.act(envs.reset(seed=1)[0], deterministic=True)  # (16, 16, 1)
    """

    name: ClassVar[str] = "ippo"
    default_reward_source: ClassVar[str] = "agent"

    @property
    def critic_input_dim(self) -> int:
        return self.agent_input_dim

    def _critic_inputs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.agent_inputs(obs)
