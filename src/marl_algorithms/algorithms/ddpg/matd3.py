"""MATD3: MADDPG with twin critics, target policy smoothing and delayed actor updates."""

from __future__ import annotations

from typing import ClassVar

from marl_algorithms.algorithms.ddpg.common import _MADDPGBase

__all__ = ["MATD3"]


class MATD3(_MADDPGBase):
    """Multi-agent TD3: MADDPG with double centralised critics (Ackermann et al., 2019).

    Every agent has twin critics ``Q_{i,1}, Q_{i,2}`` (and targets). With target
    actions smoothed by clipped noise,
    ``a'_j = clip(mu'_j(o'_j) + clip(eps_j, -c, c), -1, 1)``,
    ``eps_j ~ N(0, sigma^2)`` (``sigma = target_noise``, ``c =
    target_noise_clip``, normalised action units), the TD target takes the
    smaller estimate::

        y_i = c_r r_i + gamma (1 - d) min_k Q'_{i,k}(s', a'_1, ..., a'_n)
        L(Q_{i,1}, Q_{i,2}) = sum_k mean (Q_{i,k}(s, a_1, ..., a_n) - y_i)^2

    The actors maximise the first critic,
    ``L(mu_i) = -mean Q_{i,1}(s, a_1, ..., mu_i(o_i), ..., a_n)``, and they and
    all target networks are updated only every ``policy_delay`` critic
    updates. Everything else (parameter sharing, exploration, reward source,
    terminal masking) is as in :class:`MADDPG`.

    Parameters
    ----------
    spec, config, device, seed, **overrides:
        See :class:`MADDPG`.
    """

    name: ClassVar[str] = "matd3"
    n_critics: ClassVar[int] = 2
    smooth_targets: ClassVar[bool] = True
    delayed_policy: ClassVar[bool] = True
