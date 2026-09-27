"""Shared implementation of MADDPG and MATD3.

:class:`DDPGConfig` holds the options of both methods, and
:class:`_MADDPGBase` implements the decentralised deterministic actors, the
centralised critics and their target copies, exploration and the updates.
MATD3 switches on its three TD3 corrections through class attributes
(:mod:`~marl_algorithms.algorithms.ddpg.matd3`); the methods, implementation
notes and references are described in :mod:`marl_algorithms.algorithms.ddpg`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from marl_algorithms.core.base import OffPolicyAlgorithm
from marl_algorithms.core.config import OffPolicyConfig
from marl_algorithms.core.networks import (
    DeterministicPolicy,
    PerAgent,
    agent_one_hot,
    mlp,
    soft_update,
)
from marl_algorithms.core.normalization import ObservationNormalizer

__all__ = ["DDPGConfig"]

_REWARD_SOURCES = ("agent", "team")
_CRITIC_LOSSES = ("mse", "huber")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DDPGConfig(OffPolicyConfig):
    """Options of MADDPG and MATD3.

    Noise scales are in units of half the action range (actions are
    normalised to ``[-1, 1]`` internally).

    Attributes
    ----------
    actor_lr, critic_lr:
        Adam learning rates of the actors and the critics.
    critic_hidden_sizes:
        Hidden layer widths of the critics; ``None`` uses ``hidden_sizes``
        (which always sets the actors).
    exploration_noise:
        Standard deviation of the Gaussian noise added to the actions while
        training.
    final_exploration_noise:
        When set, the noise decays linearly to this value over the first
        ``noise_decay_steps`` environment steps.
    noise_decay_steps:
        Length of the decay, in environment steps (summed over copies).
    reward_source:
        ``"agent"``: critic ``i`` learns from agent ``i``'s own reward
        (``info["agent_rewards"]``), as in MADDPG; ``"team"``: every critic
        learns from the environment's team reward (the sum of the agent
        rewards for PowerGrid-v0, their mean plus a success bonus for
        Consensus-v0).
    normalize_observations:
        Standardise observations with statistics of the warm-up data, frozen
        at the first gradient step.
    critic_local_inputs:
        Append agent ``i``'s own observation and action to the input of
        ``Q_i`` (agent-specific global state; see :mod:`marl_algorithms.algorithms.ddpg`).
    critic_loss:
        ``"mse"`` (squared TD error, as in the papers) or ``"huber"``
        (squared up to ``|TD error| = 1``, linear beyond), which bounds the
        gradient of rare, very large errors such as the trip penalty of
        PowerGrid-v0.
    policy_delay:
        MATD3 only: critic updates per actor and target-network update.
    target_noise, target_noise_clip:
        MATD3 only: standard deviation and clipping bound of the smoothing
        noise added to the target actions.
    """

    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    critic_hidden_sizes: tuple[int, ...] | None = None
    exploration_noise: float = 0.1
    final_exploration_noise: float | None = None
    noise_decay_steps: int = 0
    reward_source: str = "agent"
    normalize_observations: bool = False
    critic_local_inputs: bool = True
    critic_loss: str = "mse"
    policy_delay: int = 2
    target_noise: float = 0.2
    target_noise_clip: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.critic_hidden_sizes is not None:
            self.critic_hidden_sizes = tuple(int(h) for h in self.critic_hidden_sizes)
            if not self.critic_hidden_sizes or min(self.critic_hidden_sizes) < 1:
                raise ValueError(
                    f"critic_hidden_sizes must be positive, got {self.critic_hidden_sizes}"
                )
        for name in ("actor_lr", "critic_lr"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("exploration_noise", "target_noise", "target_noise_clip"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.final_exploration_noise is not None and self.final_exploration_noise < 0:
            raise ValueError(
                f"final_exploration_noise must be non-negative, got {self.final_exploration_noise}"
            )
        if self.noise_decay_steps < 0:
            raise ValueError(
                f"noise_decay_steps must be non-negative, got {self.noise_decay_steps}"
            )
        if self.reward_source not in _REWARD_SOURCES:
            raise ValueError(
                f"reward_source must be one of {_REWARD_SOURCES}, got {self.reward_source!r}"
            )
        if self.critic_loss not in _CRITIC_LOSSES:
            raise ValueError(
                f"critic_loss must be one of {_CRITIC_LOSSES}, got {self.critic_loss!r}"
            )
        if int(self.policy_delay) < 1:
            raise ValueError(f"policy_delay must be positive, got {self.policy_delay}")


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class _QNetworks(nn.Module):
    """``count`` independent Q-networks on the same input; output ``(..., count)``."""

    def __init__(
        self, in_dim: int, hidden_sizes: tuple[int, ...], activation: str, count: int
    ) -> None:
        super().__init__()
        self.nets = nn.ModuleList(mlp(in_dim, 1, hidden_sizes, activation) for _ in range(count))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([net(x) for net in self.nets], dim=-1)


class _MADDPGBase(OffPolicyAlgorithm):
    """Shared implementation of MADDPG and MATD3 (see :mod:`marl_algorithms.algorithms.ddpg`).

    Subclasses choose the variant with three class attributes: the number of
    critics per agent, whether target actions are smoothed with noise, and
    whether actor updates are delayed.
    """

    action_kinds: ClassVar[tuple[str, ...]] = ("continuous",)
    config_class: ClassVar[type[DDPGConfig]] = DDPGConfig
    #: Critics per agent (the target uses their minimum).
    n_critics: ClassVar[int] = 1
    #: Add clipped Gaussian noise to the target actions.
    smooth_targets: ClassVar[bool] = False
    #: Update actors and targets every ``config.policy_delay`` critic updates.
    delayed_policy: ClassVar[bool] = False

    config: DDPGConfig

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self) -> None:
        spec, cfg = self.spec, self.config
        n, action_dim = spec.n_agents, spec.action_dim
        shared = cfg.share_parameters
        ids = n if self.uses_agent_ids else 0
        critic_hidden = cfg.critic_hidden_sizes or cfg.hidden_sizes
        unit = np.ones(action_dim, dtype=np.float32)

        self.actor = PerAgent(
            lambda: DeterministicPolicy(
                self.agent_input_dim, action_dim, -unit, unit, cfg.hidden_sizes, cfg.activation
            ),
            n,
            shared,
        ).to(self.device)
        local = spec.obs_dim + action_dim if cfg.critic_local_inputs else 0
        critic_in = spec.state_dim + n * action_dim + local + ids
        self.critic = PerAgent(
            lambda: _QNetworks(critic_in, critic_hidden, cfg.activation, self.n_critics), n, shared
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # Per-agent affine map between normalised actions in [-1, 1] and env units.
        low = torch.as_tensor(spec.action_low, dtype=torch.float32, device=self.device)
        high = torch.as_tensor(spec.action_high, dtype=torch.float32, device=self.device)
        self._action_center = (high + low) / 2.0
        self._action_half_range = (high - low) / 2.0
        # own[i, j] is True when j == i: selects agent i's action in the joint action of row i.
        self._own = torch.eye(n, dtype=torch.bool, device=self.device).unsqueeze(-1)
        self.obs_normalizer = ObservationNormalizer(
            spec.obs_dim, enabled=cfg.normalize_observations
        )
        self._obs_stats_frozen = False
        self._critic_updates = 0
        self._last_actor_loss = float("nan")

    def _modules(self) -> dict[str, nn.Module | torch.optim.Optimizer]:
        return {
            "actor": self.actor,
            "critic": self.critic,
            "actor_target": self.actor_target,
            "critic_target": self.critic_target,
            "actor_optimizer": self.actor_optimizer,
            "critic_optimizer": self.critic_optimizer,
        }

    def _extra_state(self) -> dict[str, Any]:
        return {
            "obs_normalizer": self.obs_normalizer.state_dict(),
            "obs_stats_frozen": self._obs_stats_frozen,
            "critic_updates": self._critic_updates,
        }

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        if "obs_normalizer" in state:
            self.obs_normalizer.load_state_dict(state["obs_normalizer"])
        self._obs_stats_frozen = bool(state.get("obs_stats_frozen", False))
        self._critic_updates = int(state.get("critic_updates", 0))

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    @property
    def exploration_scale(self) -> float:
        """Current standard deviation of the exploration noise (normalised units)."""
        cfg = self.config
        if cfg.final_exploration_noise is None or cfg.noise_decay_steps == 0:
            return float(cfg.exploration_noise)
        progress = min(1.0, self.env_steps / cfg.noise_decay_steps)
        return float(
            cfg.exploration_noise + progress * (cfg.final_exploration_noise - cfg.exploration_noise)
        )

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Per-agent actions ``(*batch, n, action_dim)`` in environment units.

        Exploratory actions add Gaussian noise of standard deviation
        :attr:`exploration_scale` (in units of half the action range) and are
        clipped to the bounds.
        """
        obs_t = self.tensor(self.obs_normalizer(obs))
        with torch.inference_mode():
            action = self.actor(self.agent_inputs(obs_t))
            if not deterministic and self.exploration_scale > 0:
                noise = torch.randn(action.shape, generator=self.torch_rng, device=self.device)
                action = (action + self.exploration_scale * noise).clamp(-1.0, 1.0)
            action = self._action_center + self._action_half_range * action
        return action.cpu().numpy()

    def _explore(self, obs: np.ndarray) -> np.ndarray:
        return self.act(obs, deterministic=False)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def _critic_inputs(
        self, obs: torch.Tensor, actions: torch.Tensor, own: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-agent critic inputs ``(N, n, critic_input_dim)``.

        Row ``i`` is ``[s, a_1..a_n (, o_i, a_i) (, id_i)]``: the global state
        ``s`` (all observations), the joint action, optionally agent ``i``'s own
        observation and action, and its one-hot identifier when parameters are
        shared.

        Parameters
        ----------
        obs:
            ``(N, n, obs_dim)`` observations.
        actions:
            ``(N, n, action_dim)`` normalised joint actions.
        own:
            When given, row ``i`` uses ``own[:, i]`` as agent ``i``'s action
            and ``actions`` for the other agents (actor update).
        """
        batch, n = obs.shape[0], self.spec.n_agents
        state = obs.reshape(batch, 1, -1).expand(batch, n, -1)
        if own is None:
            own = actions
            joint = actions.reshape(batch, 1, -1).expand(batch, n, -1)
        else:
            joint = self._own_action_joint(own, actions)
        parts = [state, joint]
        if self.config.critic_local_inputs:
            parts += [obs, own]
        if self.uses_agent_ids:
            parts.append(agent_one_hot((batch,), n, obs.device))
        return torch.cat(parts, dim=-1)

    def _own_action_joint(self, own: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Joint actions ``(N, n, n * action_dim)`` whose row ``i`` has agent ``i``'s
        action replaced by ``own[:, i]`` and the other agents' actions from ``actions``."""
        joint = torch.where(self._own, own.unsqueeze(1), actions.unsqueeze(1))
        return joint.flatten(start_dim=-2)

    def _prepare_obs(self, batch: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalised observation tensors; freezes the statistics on first use."""
        if self.obs_normalizer.enabled and not self._obs_stats_frozen:
            size = len(self.replay) if hasattr(self, "replay") else 0
            source = self.replay.data["obs"][:size] if size else batch["obs"]
            self.obs_normalizer.update(source)
            self._obs_stats_frozen = True
        return (
            self.tensor(self.obs_normalizer(batch["obs"])),
            self.tensor(self.obs_normalizer(batch["next_obs"])),
        )

    def _update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        cfg = self.config
        obs, next_obs = self._prepare_obs(batch)
        actions = (self.tensor(batch["actions"]) - self._action_center) / self._action_half_range
        if cfg.reward_source == "agent":
            rewards = self.tensor(batch["agent_rewards"])
        else:
            rewards = self.tensor(batch["reward"]).unsqueeze(-1).expand_as(obs[..., 0])
        not_terminated = 1.0 - self.tensor(batch["terminated"]).unsqueeze(-1)

        # Critic: y_i = scale * r_i + gamma (1 - d) min_k Q'_{i,k}(s', mu'_1(o'_1), ..., mu'_n(o'_n)).
        with torch.no_grad():
            next_actions = self.actor_target(self.agent_inputs(next_obs))
            if self.smooth_targets:
                noise = torch.randn(
                    next_actions.shape, generator=self.torch_rng, device=self.device
                )
                noise = (cfg.target_noise * noise).clamp(
                    -cfg.target_noise_clip, cfg.target_noise_clip
                )
                next_actions = (next_actions + noise).clamp(-1.0, 1.0)
            next_q = self.critic_target(self._critic_inputs(next_obs, next_actions))
            targets = cfg.reward_scale * rewards + cfg.gamma * not_terminated * next_q.amin(-1)
        q = self.critic(self._critic_inputs(obs, actions))
        errors = q - targets.unsqueeze(-1)
        if cfg.critic_loss == "mse":
            critic_loss = errors.pow(2).mean(dim=(0, 1)).sum()
        else:  # Huber: quadratic up to |error| = 1, linear beyond (bounded outlier gradients)
            critic_loss = 2.0 * F.huber_loss(errors, torch.zeros_like(errors), reduction="none")
            critic_loss = critic_loss.mean(dim=(0, 1)).sum()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
        self.critic_optimizer.step()
        self._critic_updates += 1

        delay = cfg.policy_delay if self.delayed_policy else 1
        if self._critic_updates % delay == 0:
            self._last_actor_loss = self._update_actor(obs, actions)
            soft_update(self.actor_target, self.actor, cfg.tau)
            soft_update(self.critic_target, self.critic, cfg.tau)
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": self._last_actor_loss,
            "q_mean": q.detach()[..., 0].mean().item(),
            "target_mean": targets.mean().item(),
            "noise_scale": self.exploration_scale,
        }

    def _update_actor(self, obs: torch.Tensor, actions: torch.Tensor) -> float:
        """Deterministic policy gradient step: maximise ``Q_i(s, a_{-i}, mu_i(o_i))``."""
        cfg = self.config
        own = self.actor(self.agent_inputs(obs))
        critic_inputs = self._critic_inputs(obs, actions, own)
        # The critic is only differentiated with respect to its input here.
        self.critic.requires_grad_(False)
        actor_loss = -self.critic(critic_inputs)[..., 0].mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.critic.requires_grad_(True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
        self.actor_optimizer.step()
        return actor_loss.item()
