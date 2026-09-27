"""Q-learning family: IQL, VDN and QMIX (value-based, discrete actions).

All three methods learn per-agent *utilities* ``Q_i(o_i, a_i)`` with the DQN
machinery (replay buffer, target networks, epsilon-greedy exploration) and act
greedily and decentrally on them: agent ``i`` picks ``argmax_a Q_i(o_i, a)``
from its own observation. They differ only in how the utilities are trained.

* **IQL** -- independent Q-learning (Tan, 1993), here with DQN function
  approximation (Mnih et al., 2015). Every agent regresses its utility on its
  *own* reward with its *own* TD target; the other agents are part of a
  non-stationary environment.
* **VDN** -- value decomposition networks (Sunehag et al., 2018). The team
  value is the sum of the utilities, ``Q_tot = sum_i Q_i(o_i, a_i)``, trained
  end to end on the *team* reward with one TD target.
* **QMIX** (Rashid et al., 2018). The team value is a monotonic function of the
  utilities, ``Q_tot = f_mix(Q_1, ..., Q_n; s)`` with ``dQ_tot / dQ_i >= 0``,
  computed by a mixing network whose non-negative weights are produced by
  hypernetworks from the global state ``s``.

Because the VDN and QMIX mixing is monotonic in every utility, the joint greedy
action ``argmax_a Q_tot(s, a)`` is obtained by every agent maximising its own
utility (the individual-global-max property). The centralised TD target
``max_a' Q_tot^-(s', a')`` is therefore computed from per-agent maxima, and
execution stays decentralised.

Implementation choices (standard simplifications, stated explicitly):

* **Feed-forward agents on a transition replay.** The VDN and QMIX papers use
  recurrent (LSTM/GRU) agent networks trained on whole episodes sampled from an
  episode replay to cope with partial observability. Here the agent networks
  are MLPs trained on single transitions. The ``env_lib`` observations (local
  message/queue windows of LineMsg and WirelessComm, piston and ball features
  of Pistonball) are Markov enough for the demonstrations, and the
  feed-forward version is much cheaper on a CPU.
* **Parameter sharing.** By default one utility network serves all agents,
  with a one-hot agent identifier appended to the observation (as in the QMIX
  paper); ``share_parameters=False`` gives one network per agent.
* **Global state.** The ``env_lib`` environments do not expose a separate
  state, so the QMIX hypernetworks are conditioned on the concatenation of all
  agents' observations.
* One-step TD targets, double Q-learning (van Hasselt et al., 2016) by default,
  Adam instead of RMSprop, and Polyak-averaged target networks by default
  (hard copies every ``K`` gradient steps are available).

References
----------
Tan, M. (1993). Multi-agent reinforcement learning: independent vs.
cooperative agents. *ICML*.

Mnih, V. et al. (2015). Human-level control through deep reinforcement
learning. *Nature* 518, 529-533.

van Hasselt, H., Guez, A. and Silver, D. (2016). Deep reinforcement learning
with double Q-learning. *AAAI*.

Sunehag, P. et al. (2018). Value-decomposition networks for cooperative
multi-agent learning based on team reward. *AAMAS*.

Rashid, T. et al. (2018). QMIX: monotonic value function factorisation for
deep multi-agent reinforcement learning. *ICML*.

Modules
-------
* :mod:`~marl_algorithms.algorithms.q_learning.common` -- :class:`QLearningConfig`
  and the shared implementation (utilities, targets, exploration, updates);
* :mod:`~marl_algorithms.algorithms.q_learning.iql` -- :class:`IQL`, no mixing;
* :mod:`~marl_algorithms.algorithms.q_learning.vdn` -- :class:`VDN` and its
  additive :class:`VDNMixer`;
* :mod:`~marl_algorithms.algorithms.q_learning.qmix` -- :class:`QMIX`, with the
  monotonic mixer :class:`~marl_algorithms.core.networks.QMixer`.

Tuned settings are in :mod:`marl_algorithms.presets.q_learning`.
"""

from marl_algorithms.algorithms.q_learning.common import QLearningConfig
from marl_algorithms.algorithms.q_learning.iql import IQL
from marl_algorithms.algorithms.q_learning.qmix import QMIX
from marl_algorithms.algorithms.q_learning.vdn import VDN, VDNMixer

__all__ = ["IQL", "QMIX", "VDN", "QLearningConfig", "VDNMixer"]
