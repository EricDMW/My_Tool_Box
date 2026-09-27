"""QMIX: monotonic value function factorisation with a state-conditioned mixer."""

from __future__ import annotations

from typing import ClassVar

from torch import nn

from marl_algorithms.algorithms.q_learning.common import _ValueBase
from marl_algorithms.core.networks import QMixer

__all__ = ["QMIX"]


class QMIX(_ValueBase):
    """QMIX: monotonic value function factorisation (Rashid et al., 2018).

    The team value mixes the agent utilities with a two-layer network whose
    weights are generated from the global state ``s`` by hypernetworks and made
    non-negative (absolute value), so ``dQ_tot / dQ_i >= 0``::

        Q_tot(o, a; s) = w_2(s)^T elu(W_1(s)^T q + b_1(s)) + b_2(s),   W_1, w_2 >= 0
        y              = c r + gamma (1 - terminated) Q_tot^-(q'^-; s')
        L              = mean_batch  l(Q_tot(o, a; s) - y)

    with ``q = (Q_1(o_1, a_1), ..., Q_n(o_n, a_n))``, ``q'^-`` the target
    utilities of the (double-Q) greedy next actions and ``Q_tot^-`` the target
    mixer. The state ``s`` is the concatenation of all agents' observations;
    the mixer (:class:`~marl_algorithms.core.networks.QMixer`) is only used in
    training, execution is decentralised. Agents are feed-forward networks
    trained on single transitions from a transition replay, not the recurrent
    agents and episode replay of the paper (see :mod:`marl_algorithms.algorithms.q_learning`).

    Parameters and exceptions as for :class:`IQL`.

    Examples
    --------
    >>> from marl_algorithms import train
    >>> algo, log = train("qmix", "LineMsg-v0", total_steps=50_000, num_envs=32)
    """

    name: ClassVar[str] = "qmix"

    def _make_mixer(self) -> nn.Module:
        cfg = self.config
        return QMixer(
            self.spec.n_agents, self.spec.state_dim, cfg.mixer_embed_dim, cfg.hypernet_hidden
        )
