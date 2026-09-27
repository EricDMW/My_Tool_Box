"""Shared implementation of IQL, VDN and QMIX.

:class:`QLearningConfig` holds the options of the three methods, and
:class:`_ValueBase` implements the agent utilities, epsilon-greedy
exploration, the double-Q targets, the target networks and the updates. The
methods differ only in the mixer that combines the utilities
(:mod:`~marl_algorithms.algorithms.q_learning.iql`,
:mod:`~marl_algorithms.algorithms.q_learning.vdn`,
:mod:`~marl_algorithms.algorithms.q_learning.qmix`); the methods, the
simplifications and the references are described in
:mod:`marl_algorithms.algorithms.q_learning`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from marl_algorithms.core.base import OffPolicyAlgorithm
from marl_algorithms.core.config import AlgorithmConfig, OffPolicyConfig
from marl_algorithms.core.networks import PerAgent, mlp, soft_update

__all__ = ["QLearningConfig"]

_LOSSES = ("huber", "mse")


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class QLearningConfig(OffPolicyConfig):
    """Options of the value-based methods (IQL, VDN, QMIX).

    Inherits the replay options of
    :class:`~marl_algorithms.core.config.OffPolicyConfig` (``buffer_size``,
    ``batch_size``, ``warmup_steps``, ``update_every``, ``gradient_steps``,
    ``tau``, ``reward_scale``) and the network options of
    :class:`~marl_algorithms.core.config.AlgorithmConfig`.

    Attributes
    ----------
    lr:
        Adam learning rate of the utility network (and the mixer).
    epsilon_start, epsilon_end:
        Exploration probability at the start of training and after the decay.
    epsilon_decay_steps:
        Environment steps (summed over copies) over which epsilon decreases
        linearly from ``epsilon_start`` to ``epsilon_end``; ``0`` uses
        ``epsilon_end`` from the start.
    double_q:
        Double Q-learning: the online network selects the next action and the
        target network evaluates it.
    target_update_interval:
        ``0`` updates the target networks by Polyak averaging with ``tau``
        after every gradient step; ``K > 0`` copies the online weights into
        the targets every ``K`` gradient steps (the original DQN scheme).
    loss:
        TD loss: ``"huber"`` (smooth L1 with threshold 1) or ``"mse"``.
    mixer_embed_dim:
        QMIX: width of the mixing layer.
    hypernet_hidden:
        QMIX: width of the hidden layer of the weight hypernetworks.
    """

    lr: float = 5e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    double_q: bool = True
    target_update_interval: int = 0
    loss: str = "huber"
    mixer_embed_dim: int = 32
    hypernet_hidden: int = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError(
                "epsilon must satisfy 0 <= epsilon_end <= epsilon_start <= 1, got "
                f"epsilon_start={self.epsilon_start}, epsilon_end={self.epsilon_end}"
            )
        if self.epsilon_decay_steps < 0:
            raise ValueError(
                f"epsilon_decay_steps must be non-negative, got {self.epsilon_decay_steps}"
            )
        if self.target_update_interval < 0:
            raise ValueError(
                "target_update_interval must be 0 (Polyak averaging) or a positive number of "
                f"gradient steps, got {self.target_update_interval}"
            )
        if self.loss not in _LOSSES:
            raise ValueError(f"loss must be one of {_LOSSES}, got {self.loss!r}")
        if self.mixer_embed_dim < 1 or self.hypernet_hidden < 1:
            raise ValueError(
                "mixer_embed_dim and hypernet_hidden must be positive, got "
                f"{self.mixer_embed_dim} and {self.hypernet_hidden}"
            )


# ---------------------------------------------------------------------------
# Shared implementation
# ---------------------------------------------------------------------------
class _ValueBase(OffPolicyAlgorithm):
    """Shared implementation of IQL, VDN and QMIX, parameterised by the mixer.

    Networks: the utility network ``Q(o_i, id_i) -> R^{n_actions}`` (shared by
    all agents with one-hot identifiers, or one network per agent), an
    optional mixer (``None`` for IQL) and target copies of both.

    With a transition ``(o, a, r, o', terminated)`` from the replay buffer
    (``o = (o_1, ..., o_n)``, state ``s = concat(o)``) and chosen utilities
    ``q_i = Q_i(o_i, a_i)``:

    * mixer (VDN, QMIX), team reward ``r``::

        a'_i  = argmax_a Q_i(o'_i, a)         (online net; target net if not double_q)
        y     = c r + gamma (1 - terminated) f_mix^-(Q_1^-(o'_1, a'_1), ..., Q_n^-(o'_n, a'_n); s')
        L     = mean_batch  l(f_mix(q_1, ..., q_n; s) - y)

    * no mixer (IQL), agent rewards ``r_i``::

        y_i   = c r_i + gamma (1 - terminated) Q_i^-(o'_i, a'_i)
        L     = mean_batch mean_i  l(q_i - y_i)

    where ``c`` is ``reward_scale``, ``l`` the Huber or squared loss and
    ``^-`` marks target networks. Truncated episodes bootstrap: the replay
    stores the final observation as ``o'`` and only ``terminated`` masks the
    bootstrap term.

    Every update records ``td_loss``, ``q_mean`` (mean of the trained value,
    ``Q_tot`` or ``Q_i``, in units of the scaled reward), ``target_mean``,
    ``grad_norm`` (before clipping) and ``epsilon``.
    """

    action_kinds: ClassVar[tuple[str, ...]] = ("discrete",)
    config_class: ClassVar[type[AlgorithmConfig]] = QLearningConfig

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _make_mixer(self) -> nn.Module | None:
        """The mixing network (``None``: independent learners)."""
        return None

    def _build(self) -> None:
        spec, cfg = self.spec, self.config
        in_dim = self.agent_input_dim

        def utility_network() -> nn.Module:
            return mlp(in_dim, spec.n_actions, cfg.hidden_sizes, cfg.activation)

        self.q_net = PerAgent(utility_network, spec.n_agents, cfg.share_parameters).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).requires_grad_(False)
        mixer = self._make_mixer()
        self.mixer = mixer.to(self.device) if mixer is not None else None
        self.target_mixer = (
            copy.deepcopy(self.mixer).requires_grad_(False) if self.mixer is not None else None
        )
        self._params = list(self.q_net.parameters())
        if self.mixer is not None:
            self._params += list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(self._params, lr=cfg.lr)

    def _modules(self) -> dict[str, nn.Module | torch.optim.Optimizer]:
        modules: dict[str, nn.Module | torch.optim.Optimizer] = {
            "q_net": self.q_net,
            "target_q_net": self.target_q_net,
            "optimizer": self.optimizer,
        }
        if self.mixer is not None:
            modules["mixer"] = self.mixer
            modules["target_mixer"] = self.target_mixer
        return modules

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def epsilon_at(self, env_steps: int) -> float:
        """Exploration probability after ``env_steps`` environment steps (linear schedule)."""
        cfg = self.config
        if cfg.epsilon_decay_steps <= 0:
            return float(cfg.epsilon_end)
        fraction = min(1.0, max(0, int(env_steps)) / cfg.epsilon_decay_steps)
        return float(cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start))

    @property
    def epsilon(self) -> float:
        """Current exploration probability (from :attr:`env_steps`)."""
        return self.epsilon_at(self.env_steps)

    def q_values(self, obs: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        """Agent utilities ``(*batch, n_agents, n_actions)`` for observations ``(*batch, n, d)``."""
        net = self.target_q_net if target else self.q_net
        return net(self.agent_inputs(obs))

    def mix(
        self, agent_q: torch.Tensor, obs: torch.Tensor, *, target: bool = False
    ) -> torch.Tensor:
        """Team value ``(*batch,)`` from chosen utilities ``(*batch, n)``.

        Without a mixer (IQL) the utilities are returned unchanged. The global
        state is the concatenation of all agents' observations ``obs``.
        """
        mixer = self.target_mixer if target else self.mixer
        if mixer is None:
            return agent_q
        return mixer(agent_q, obs.flatten(start_dim=-2))

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Greedy (``deterministic``) or epsilon-greedy per-agent actions.

        Parameters
        ----------
        obs:
            ``(*batch, n_agents, obs_dim)`` observations.
        deterministic:
            Greedy actions; otherwise every agent of every copy independently
            takes a uniformly random action with probability :attr:`epsilon`.

        Returns
        -------
        numpy.ndarray
            ``int64`` actions of shape ``(*batch, n_agents)``.
        """
        with torch.no_grad():
            greedy = self.q_values(self.tensor(obs)).argmax(dim=-1).cpu().numpy()
        if deterministic:
            return greedy
        return self._epsilon_greedy(greedy, self.epsilon)

    def _epsilon_greedy(self, greedy: np.ndarray, epsilon: float) -> np.ndarray:
        # Both arrays are always drawn so that the random stream does not depend on epsilon.
        explore = self.np_rng.random(greedy.shape) < epsilon
        random = self.np_rng.integers(0, self.spec.n_actions, size=greedy.shape)
        return np.where(explore, random, greedy)

    def _explore(self, obs: np.ndarray) -> np.ndarray:
        return self.act(obs, deterministic=False)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    @torch.no_grad()
    def td_target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """One-step TD targets: ``(N,)`` with a mixer, ``(N, n_agents)`` without.

        Parameters
        ----------
        batch:
            Tensors ``next_obs (N, n, d)``, ``reward (N,)`` (mixer) or
            ``agent_rewards (N, n)`` (no mixer) and ``terminated (N,)``.
        """
        cfg = self.config
        next_obs = batch["next_obs"]
        next_q_target = self.q_values(next_obs, target=True)
        selector = self.q_values(next_obs) if cfg.double_q else next_q_target
        next_actions = selector.argmax(dim=-1, keepdim=True)
        next_q = next_q_target.gather(-1, next_actions).squeeze(-1)
        next_value = self.mix(next_q, next_obs, target=True)
        reward = batch["reward"] if self.mixer is not None else batch["agent_rewards"]
        not_terminated = 1.0 - batch["terminated"].float()
        if next_value.dim() == 2:  # per-agent targets share the episode's termination flag
            not_terminated = not_terminated.unsqueeze(-1)
        return cfg.reward_scale * reward + cfg.gamma * not_terminated * next_value

    def td_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """TD loss of a batch and statistics (see the class docstring)."""
        obs = batch["obs"]
        actions = batch["actions"].long().unsqueeze(-1)
        chosen_q = self.q_values(obs).gather(-1, actions).squeeze(-1)  # (N, n)
        value = self.mix(chosen_q, obs)
        target = self.td_target(batch)
        if self.config.loss == "huber":
            loss = F.smooth_l1_loss(value, target)
        else:
            loss = F.mse_loss(value, target)
        stats = {
            "td_loss": float(loss.detach()),
            "q_mean": float(value.detach().mean()),
            "target_mean": float(target.mean()),
        }
        return loss, stats

    def _update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        data = {
            name: self.tensor(value, torch.long if name == "actions" else torch.float32)
            for name, value in batch.items()
        }
        loss, stats = self.td_loss(data)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self._params, self.config.max_grad_norm)
        self.optimizer.step()
        self._update_targets()
        stats["grad_norm"] = float(grad_norm)
        stats["epsilon"] = self.epsilon
        return stats

    def _update_targets(self) -> None:
        cfg = self.config
        pairs = [(self.target_q_net, self.q_net)]
        if self.mixer is not None:
            pairs.append((self.target_mixer, self.mixer))
        if cfg.target_update_interval > 0:
            # The base loop increments num_updates after this update returns.
            if (self.num_updates + 1) % cfg.target_update_interval == 0:
                for target, online in pairs:
                    target.load_state_dict(online.state_dict())
        else:
            for target, online in pairs:
                soft_update(target, online, cfg.tau)
