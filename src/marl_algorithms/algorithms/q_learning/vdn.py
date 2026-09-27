"""VDN: value decomposition networks, with an additive team value."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn

from marl_algorithms.algorithms.q_learning.common import _ValueBase

__all__ = ["VDN", "VDNMixer"]


class VDNMixer(nn.Module):
    """Additive mixing of VDN: ``Q_tot = sum_i Q_i`` (no parameters).

    Takes the same arguments as :class:`~marl_algorithms.core.networks.QMixer`
    so that both mixers are interchangeable; the state is ignored.
    """

    def forward(self, q: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """Team value of shape ``q.shape[:-1]`` from utilities ``(..., n_agents)``."""
        del state
        return q.sum(dim=-1)


class VDN(_ValueBase):
    """Value decomposition networks (Sunehag et al., 2018).

    The team value is the sum of the agent utilities, trained on the team
    reward ``r``::

        Q_tot(o, a) = sum_i Q_i(o_i, a_i)
        y           = c r + gamma (1 - terminated) sum_i Q_i^-(o'_i, argmax_a Q_i(o'_i, a))
        L           = mean_batch  l(Q_tot(o, a) - y)

    Agents are feed-forward networks trained on single transitions from a
    transition replay, not the recurrent agents and episode replay of the
    paper (see :mod:`marl_algorithms.algorithms.q_learning`).

    Parameters and exceptions as for :class:`IQL`.

    Examples
    --------
    >>> from marl_algorithms import train
    >>> algo, log = train("vdn", "LineMsg-v0", total_steps=50_000, num_envs=32)
    """

    name: ClassVar[str] = "vdn"

    def _make_mixer(self) -> nn.Module:
        return VDNMixer()
