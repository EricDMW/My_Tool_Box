"""PPO family: IPPO and MAPPO (on-policy, continuous and discrete actions).

Both methods train decentralised stochastic actors ``pi(a_i | o_i)`` with the
clipped surrogate objective of PPO and differ only in the critic:

* **IPPO** (independent PPO) gives every agent a decentralised critic
  ``V(o_i)`` trained on the agent's own reward; each agent is an independent
  PPO learner that treats the others as part of the environment.
* **MAPPO** (multi-agent PPO) gives every agent a centralised critic
  ``V(s, i)`` on the global state ``s`` -- the concatenation of all agents'
  (normalised) observations -- trained on the team reward. The critic is only
  used during training, so execution stays decentralised (centralised training,
  decentralised execution).

The shared implementation follows the PPO recipe of the references below:
generalised advantage estimation per agent, advantage normalisation, several
epochs of shuffled minibatch SGD, clipped (and optionally Huber) value loss,
entropy bonus, gradient-norm clipping, optional linear learning-rate annealing,
running observation normalisation and reward scaling by the running standard
deviation of the return. With the default parameter sharing, one actor and one
critic serve all agents and a one-hot agent identifier is appended to their
inputs; every forward pass covers all copies and agents at once.

Standard simplifications, stated once: feed-forward networks instead of the
recurrent ones MAPPO uses for partially observed tasks (the ``env_lib``
observations carry the recent history that matters, such as previous actions
and rates of change), a state-independent Gaussian standard deviation for
continuous actions (sampled unbounded, clipped by the environment adapter, the
log-probability is that of the unclipped sample), and reward scaling instead of
MAPPO's value normalisation (PopArt / ValueNorm); both keep value targets of
order one. The global state of MAPPO is the concatenation of the local
observations (the paper's "CL" state), not an environment-specific state.

References
----------
* J. Schulman, F. Wolski, P. Dhariwal, A. Radford and O. Klimov, "Proximal
  policy optimization algorithms", arXiv:1707.06347, 2017.
* J. Schulman, P. Moritz, S. Levine, M. Jordan and P. Abbeel, "High-dimensional
  continuous control using generalized advantage estimation", ICLR 2016.
* C. S. de Witt, T. Gupta, D. Makoviichuk, V. Makoviychuk, P. H. S. Torr, M. Sun
  and S. Whiteson, "Is independent learning all you need in the StarCraft
  multi-agent challenge?", arXiv:2011.09533, 2020.
* C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen and Y. Wu, "The
  surprising effectiveness of PPO in cooperative multi-agent games", NeurIPS
  Datasets and Benchmarks 2022.
* M. Andrychowicz et al., "What matters in on-policy reinforcement learning? A
  large-scale empirical study", ICLR 2021.

Modules
-------
* :mod:`~marl_algorithms.algorithms.ppo.common` -- :class:`PPOConfig` and the
  shared implementation (rollouts, advantages, losses, normalisation);
* :mod:`~marl_algorithms.algorithms.ppo.ippo` -- :class:`IPPO`, decentralised
  critics;
* :mod:`~marl_algorithms.algorithms.ppo.mappo` -- :class:`MAPPO`, centralised
  critic.

Tuned settings are in :mod:`marl_algorithms.presets.ppo`.
"""

from marl_algorithms.algorithms.ppo.common import PPOConfig
from marl_algorithms.algorithms.ppo.ippo import IPPO
from marl_algorithms.algorithms.ppo.mappo import MAPPO

__all__ = ["IPPO", "MAPPO", "PPOConfig"]
