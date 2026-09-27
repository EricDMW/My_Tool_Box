"""Algorithm implementations, one module per algorithm, grouped by family.

* :mod:`~marl_algorithms.algorithms.ppo` -- IPPO and MAPPO (on-policy,
  continuous and discrete actions);
* :mod:`~marl_algorithms.algorithms.ddpg` -- MADDPG and MATD3 (off-policy
  actor-critic, continuous actions);
* :mod:`~marl_algorithms.algorithms.q_learning` -- IQL, VDN and QMIX
  (value-based, discrete actions).

Every family package holds a ``common`` module with its configuration class
and the implementation its algorithms share, and one module per algorithm with
what distinguishes it. Tuned settings for the ``env_lib`` environments are kept
apart, in :mod:`marl_algorithms.presets`.
"""
