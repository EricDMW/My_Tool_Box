"""DDPG family: MADDPG and MATD3 (off-policy actor-critic, continuous actions).

Both methods train *decentralised deterministic actors* ``mu_i(o_i)`` with
*centralised critics* ``Q_i(s, a_1, ..., a_n)`` that see the global state ``s``
(here the concatenation of all agents' observations) and the joint action
(centralised training, decentralised execution). Experience comes from a
replay buffer, so every transition is reused many times.

* **MADDPG** (Lowe et al., 2017) extends DDPG (Lillicrap et al., 2016) to
  several agents. Critic ``i`` regresses on the one-step TD target computed with
  target actors and target critics; actor ``i`` follows the deterministic policy
  gradient (Silver et al., 2014) of ``Q_i`` with its own action replaced by
  ``mu_i(o_i)`` and the other agents' actions taken from the replay batch.
* **MATD3** (Ackermann et al., 2019) carries the three TD3 corrections
  (Fujimoto et al., 2018) over to centralised critics: twin critics with the
  minimum in the target (against overestimation), clipped Gaussian noise on the
  target actions (target policy smoothing), and actor and target-network
  updates only every ``policy_delay`` critic updates.

Implementation notes (standard choices, stated here once):

* Feed-forward networks; the global state is the concatenation of all agents'
  observations.
* Parameter sharing (default): one actor and one critic for all agents, with a
  one-hot agent identifier appended to the actor input ``o_i`` and to the critic
  input ``[s, a_1..a_n]``; the shared critic evaluated at agent ``i``'s
  identifier is ``Q_i``. With ``share_parameters=False`` every agent has its own
  actor and critic (:class:`~marl_algorithms.core.networks.PerAgent`).
* Agent-specific critic inputs (default, ``critic_local_inputs``): the input of
  ``Q_i`` also repeats agent ``i``'s own observation and action,
  ``[s, a_1..a_n, o_i, a_i, id_i]``. This is still a function of ``(s, a)``
  and ``i`` only (the "agent-specific global state" of Yu et al., 2022), but a
  shared critic no longer has to learn from the one-hot identifier which of
  the ``n`` blocks of ``s`` and ``a`` belong to agent ``i``. On Consensus-v0
  (8 agents) MADDPG does not learn without it and approaches the Laplacian
  baseline with it.
* Actors output normalised actions in ``[-1, 1]`` (``tanh``); the per-agent
  bounds of the environment map them to physical units. Critics, exploration
  noise and target smoothing all work in these normalised units, so one set of
  hyperparameters suits environments with very different action ranges.
* Optional observation normalisation uses statistics computed once from the
  warm-up data at the first gradient step and frozen afterwards, so the stored
  transitions and the learned values stay consistent.

References
----------
Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., Mordatch, I. (2017).
Multi-agent actor-critic for mixed cooperative-competitive environments. NeurIPS.

Ackermann, J., Gabler, V., Osa, T., Sugiyama, M. (2019). Reducing overestimation
bias in multi-agent domains using double centralized critics. NeurIPS Deep RL
Workshop, arXiv:1910.01465.

Fujimoto, S., van Hoof, H., Meger, D. (2018). Addressing function approximation
error in actor-critic methods. ICML.

Lillicrap, T. P., et al. (2016). Continuous control with deep reinforcement
learning. ICLR.

Silver, D., et al. (2014). Deterministic policy gradient algorithms. ICML.

Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., Wu, Y. (2022). The
surprising effectiveness of PPO in cooperative multi-agent games. NeurIPS.

Modules
-------
* :mod:`~marl_algorithms.algorithms.ddpg.common` -- :class:`DDPGConfig` and the
  shared implementation (actors, critics, targets, exploration, updates);
* :mod:`~marl_algorithms.algorithms.ddpg.maddpg` -- :class:`MADDPG`;
* :mod:`~marl_algorithms.algorithms.ddpg.matd3` -- :class:`MATD3`, twin critics,
  target smoothing and delayed actor updates.

Tuned settings are in :mod:`marl_algorithms.presets.ddpg`.
"""

from marl_algorithms.algorithms.ddpg.common import DDPGConfig
from marl_algorithms.algorithms.ddpg.maddpg import MADDPG
from marl_algorithms.algorithms.ddpg.matd3 import MATD3

__all__ = ["DDPGConfig", "MADDPG", "MATD3"]
