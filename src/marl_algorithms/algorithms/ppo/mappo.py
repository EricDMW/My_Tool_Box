"""MAPPO: multi-agent PPO, with a centralised critic on the global state."""

from __future__ import annotations

from typing import ClassVar

import torch

from marl_algorithms.algorithms.ppo.common import _PPOBase
from marl_algorithms.core.networks import agent_one_hot

__all__ = ["MAPPO"]


class MAPPO(_PPOBase):
    """Multi-agent PPO with a centralised critic (Yu et al., 2022).

    The actors are decentralised, ``pi_theta(a_i | o_i)``, exactly as in
    :class:`IPPO`. The critic of agent ``i`` sees the global state
    ``s = [o_1, ..., o_n]`` (all normalised observations concatenated),
    ``V_phi(s, i)``: with shared parameters (default) one network takes
    ``[s, onehot(i)]`` -- effectively one value head per agent over the shared
    state -- and without sharing agent ``i`` has its own network on ``s``. The
    critics learn the value of the team reward (``reward_source="team"``,
    default) or of each agent's own reward (``"agent"``). The critic is used
    only in training; acting needs the local observation only.

    The losses are those of :class:`IPPO` with ``V_phi(o_t^i)`` replaced by
    ``V_phi(s_t, i)`` and ``r_t^i`` by the team reward ``r_t``:

    ``delta_t^i = r_t + gamma (1 - term_t) V_phi(s_{t+1}, i) - V_phi(s_t, i)``,
    ``A_t^i = sum_l (gamma lambda)^l delta_{t+l}^i``,

    ``L_pi = -mean[min(rho A, clip(rho, 1 - eps, 1 + eps) A)] - c_H mean[H]``,
    ``L_V = mean[max(l(V_phi - R), l(V_old + clip(V_phi - V_old, -eps_V, eps_V) - R))]``.

    The global state of every copy is built once by reshaping the ``(B, n, d)``
    observations to ``(B, n d)`` and repeating it for the ``n`` agents, so all
    critic values come from one batched forward pass.

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
    >>> from marl_algorithms import MAPPO, make_vector_env
    >>> envs = make_vector_env("LineMsg-v0", 16, num_agents=4)
    >>> algo = MAPPO(envs, seed=0, ent_coef=0.01)
    >>> log = algo.learn(envs, 10_000, seed=0)
    >>> actions = algo.act(envs.reset(seed=1)[0], deterministic=True)  # (16, 4) int64
    """

    name: ClassVar[str] = "mappo"
    default_reward_source: ClassVar[str] = "team"

    @property
    def critic_input_dim(self) -> int:
        spec = self.spec
        return spec.state_dim + (spec.n_agents if self.uses_agent_ids else 0)

    def _critic_inputs(self, obs: torch.Tensor) -> torch.Tensor:
        n = self.spec.n_agents
        batch = obs.shape[:-2]
        state = obs.reshape(*batch, 1, self.spec.state_dim).expand(*batch, n, self.spec.state_dim)
        if not self.uses_agent_ids:
            return state
        return torch.cat([state, agent_one_hot(batch, n, obs.device)], dim=-1)
