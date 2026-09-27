"""Shared implementation of IPPO and MAPPO.

:class:`PPOConfig` holds the options of both methods, and :class:`_PPOBase`
implements everything they have in common: rollouts on the vector
environment, generalised advantage estimation, the clipped policy and value
losses, observation normalisation, reward scaling and the optimisers. The two
algorithms only choose the inputs of the critic
(:mod:`~marl_algorithms.algorithms.ppo.ippo`,
:mod:`~marl_algorithms.algorithms.ppo.mappo`); the method and its references
are described in :mod:`marl_algorithms.algorithms.ppo`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from marl_algorithms.core.base import Callback, OnPolicyAlgorithm, TrainingLog
from marl_algorithms.core.buffers import RolloutBuffer, compute_gae
from marl_algorithms.core.config import OnPolicyConfig
from marl_algorithms.core.networks import (
    CategoricalPolicy,
    GaussianPolicy,
    PerAgent,
    mlp,
)
from marl_algorithms.core.normalization import ObservationNormalizer, RewardScaler

__all__ = ["PPOConfig"]

_REWARD_SOURCES = ("agent", "team")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PPOConfig(OnPolicyConfig):
    """Options of IPPO and MAPPO.

    Inherits ``rollout_length``, ``lr`` (actor learning rate), ``gae_lambda``,
    ``normalize_observations``, ``normalize_rewards`` and ``max_grad_norm``
    from :class:`~marl_algorithms.core.config.OnPolicyConfig` and the network
    options from :class:`~marl_algorithms.core.config.AlgorithmConfig`.

    Attributes
    ----------
    n_epochs:
        Passes over each rollout.
    num_minibatches:
        Minibatches per epoch. A sample is one copy at one step with all its
        agents (the centralised critic needs every agent's observation).
    clip_range:
        PPO ratio clipping ``epsilon``.
    value_clip_range:
        Clip the change of the value prediction to ``+-value_clip_range``
        around the pre-update value (PPO2 / MAPPO); ``None`` disables it.
    huber_delta:
        Use the Huber loss with this threshold for the critic; ``None`` uses
        the squared error ``0.5 (V - R)^2``.
    ent_coef:
        Weight ``c_H`` of the entropy bonus.
    normalize_advantages:
        Standardise the advantages over the whole rollout (all copies, steps
        and agents) before the epochs.
    critic_lr:
        Critic learning rate; ``None`` uses ``lr``.
    critic_hidden_sizes:
        Hidden layer widths of the critic; ``None`` uses ``hidden_sizes``.
    anneal_lr:
        Decay both learning rates linearly to zero over the ``total_steps``
        of :meth:`~marl_algorithms.core.base.OnPolicyAlgorithm.learn`.
    target_kl:
        Stop the epochs of an update early once the mean approximate KL
        divergence of an epoch exceeds ``1.5 * target_kl``; ``None`` disables it.
    log_std_init:
        Initial log standard deviation of the Gaussian policy, in units of half
        the action range (continuous actions only).
    reward_source:
        ``"team"``: every critic learns the value of the team reward
        (``Transition.reward``); ``"agent"``: critic ``i`` learns the value of
        agent ``i``'s own reward (``Transition.agent_rewards``). ``None``
        selects the default of the method: ``"agent"`` for IPPO and ``"team"``
        for MAPPO.
    """

    n_epochs: int = 10
    num_minibatches: int = 4
    clip_range: float = 0.2
    value_clip_range: float | None = 0.2
    huber_delta: float | None = None
    ent_coef: float = 0.0
    normalize_advantages: bool = True
    critic_lr: float | None = None
    critic_hidden_sizes: tuple[int, ...] | None = None
    anneal_lr: bool = False
    target_kl: float | None = None
    log_std_init: float = -0.5
    reward_source: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("n_epochs", "num_minibatches"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.clip_range <= 0:
            raise ValueError(f"clip_range must be positive, got {self.clip_range}")
        for name in ("value_clip_range", "huber_delta", "critic_lr", "target_kl"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None, got {value}")
        if self.ent_coef < 0:
            raise ValueError(f"ent_coef must be non-negative, got {self.ent_coef}")
        if self.critic_hidden_sizes is not None:
            self.critic_hidden_sizes = tuple(int(h) for h in self.critic_hidden_sizes)
            if not self.critic_hidden_sizes or min(self.critic_hidden_sizes) < 1:
                raise ValueError(
                    f"critic_hidden_sizes must be positive, got {self.critic_hidden_sizes}"
                )
        if self.reward_source is not None and self.reward_source not in _REWARD_SOURCES:
            raise ValueError(
                f"reward_source must be one of {_REWARD_SOURCES} or None, "
                f"got {self.reward_source!r}"
            )


# ---------------------------------------------------------------------------
# Shared implementation
# ---------------------------------------------------------------------------
class _PPOBase(OnPolicyAlgorithm):
    """Shared implementation of IPPO and MAPPO (see the subclasses).

    Subclasses set :attr:`default_reward_source` and implement
    :meth:`_critic_inputs` and :attr:`critic_input_dim`.
    """

    config_class: ClassVar[type[PPOConfig]] = PPOConfig
    #: Reward the critics learn when ``config.reward_source`` is ``None``.
    default_reward_source: ClassVar[str] = "agent"

    config: PPOConfig

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self) -> None:
        cfg, spec = self.config, self.spec
        self.reward_source = cfg.reward_source or self.default_reward_source
        shared = cfg.share_parameters
        # PerAgent calls the factory once (shared) or once per agent, in agent order.
        actor_index = iter(range(spec.n_agents))
        self.actor = PerAgent(
            lambda: self._make_actor(None if shared else next(actor_index)), spec.n_agents, shared
        )
        critic_hidden = cfg.critic_hidden_sizes or cfg.hidden_sizes
        self.critic = PerAgent(
            lambda: mlp(self.critic_input_dim, 1, critic_hidden, cfg.activation),
            spec.n_agents,
            shared,
        )
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, eps=1e-5
        )
        self.obs_normalizer = ObservationNormalizer(
            spec.obs_dim, enabled=cfg.normalize_observations
        )
        # Created by the first update, which knows the number of copies.
        self.reward_scaler: RewardScaler | None = None
        self._reward_scaler_envs = 0
        self._lr_schedule: tuple[int, int] | None = None

    def _make_actor(self, agent: int | None) -> nn.Module:
        """Policy network of one agent (``agent``) or of all agents (``None``, shared)."""
        cfg, spec = self.config, self.spec
        if not spec.continuous:
            return CategoricalPolicy(
                self.agent_input_dim, spec.n_actions, cfg.hidden_sizes, cfg.activation
            )
        # A shared policy keeps the (n_agents, action_dim) bounds, which broadcast
        # against its (..., n_agents, action_dim) output, so agents may differ.
        rows = slice(None) if agent is None else agent
        return GaussianPolicy(
            self.agent_input_dim,
            spec.action_dim,
            spec.action_low[rows],
            spec.action_high[rows],
            cfg.hidden_sizes,
            cfg.activation,
            log_std_init=cfg.log_std_init,
        )

    @property
    def critic_lr(self) -> float:
        """Critic learning rate (``config.critic_lr`` or ``config.lr``)."""
        return self.config.critic_lr if self.config.critic_lr is not None else self.config.lr

    @property
    def critic_input_dim(self) -> int:
        """Input size of the critic network."""
        raise NotImplementedError

    def _critic_inputs(self, obs: torch.Tensor) -> torch.Tensor:
        """Critic inputs ``(..., n, critic_input_dim)`` from normalised observations."""
        raise NotImplementedError

    def values(self, obs: torch.Tensor) -> torch.Tensor:
        """Critic values ``(..., n)`` of normalised observations ``(..., n, obs_dim)``."""
        return self.critic(self._critic_inputs(obs)).squeeze(-1)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Per-agent actions for per-agent observations.

        Parameters
        ----------
        obs:
            ``(*batch, n_agents, obs_dim)`` raw observations (normalised
            internally with the running statistics collected in training).
        deterministic:
            Return the mean (continuous) or the most likely action (discrete)
            instead of a sample.

        Returns
        -------
        numpy.ndarray
            ``(*batch, n_agents, action_dim)`` float32 actions clipped to the
            bounds, or ``(*batch, n_agents)`` int64 actions.

        Raises
        ------
        ValueError
            If the trailing shape of ``obs`` is not ``(n_agents, obs_dim)``.
        """
        spec = self.spec
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim < 2 or obs.shape[-2:] != (spec.n_agents, spec.obs_dim):
            raise ValueError(
                f"expected observations of shape (..., {spec.n_agents}, {spec.obs_dim}), "
                f"got {obs.shape}"
            )
        inputs = self.agent_inputs(self.tensor(self.obs_normalizer(obs)))
        with torch.inference_mode():
            if deterministic:
                output = self.actor(inputs)
                actions = output[0] if spec.continuous else output.argmax(dim=-1)
            else:
                actions, _ = self.actor.call("sample", inputs, self.torch_rng)
        actions = actions.cpu().numpy()
        if spec.continuous:
            return np.clip(actions, spec.action_low, spec.action_high).astype(np.float32)
        return actions.astype(np.int64)

    def _rollout_fields(self) -> dict[str, tuple[tuple[int, ...], Any]]:
        n, d = self.spec.n_agents, self.spec.obs_dim
        return {"norm_obs": ((n, d), np.float32), "log_prob": ((n,), np.float32)}

    def _rollout_step(self, obs: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        # The statistics include the observation before it is used, and the
        # normalised input is stored, so the update evaluates the policy on
        # exactly the input that produced the action.
        self.obs_normalizer.update(obs)
        norm_obs = self.obs_normalizer(obs)
        inputs = self.agent_inputs(self.tensor(norm_obs))
        with torch.inference_mode():
            actions, log_prob = self.actor.call("sample", inputs, self.torch_rng)
        # Continuous samples are stored unclipped: the log-probability is theirs.
        return actions.cpu().numpy(), {"norm_obs": norm_obs, "log_prob": log_prob.cpu().numpy()}

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def learn(
        self,
        envs: Any,
        total_steps: int,
        *,
        seed: int | None = None,
        log: TrainingLog | None = None,
        callback: Callback | None = None,
    ) -> TrainingLog:
        """Train for ``total_steps`` environment steps (summed over copies).

        See :meth:`~marl_algorithms.core.base.OnPolicyAlgorithm.learn`. With
        ``anneal_lr`` the learning rates decay linearly from their initial
        values at the current step count to zero at ``total_steps``.
        """
        self._lr_schedule = (self.env_steps, int(total_steps))
        return super().learn(envs, total_steps, seed=seed, log=log, callback=callback)

    def _learning_rate_fraction(self, batch_steps: int) -> float:
        """Remaining fraction of the learning rate before an update of ``batch_steps`` steps."""
        if not self.config.anneal_lr or self._lr_schedule is None:
            return 1.0
        start, total = self._lr_schedule
        done = self.env_steps - batch_steps - start
        return float(np.clip(1.0 - done / max(total - start, 1), 0.0, 1.0))

    def _training_rewards(self, rollout: RolloutBuffer) -> np.ndarray:
        """Rewards the critics learn, ``(T, B, n)`` (team or own), scaled when configured."""
        if self.reward_source == "team":
            rewards = rollout["reward"].astype(np.float64)[..., None]
        else:
            rewards = rollout["agent_rewards"].astype(np.float64)
        if self.config.normalize_rewards:
            if self.reward_scaler is None or self._reward_scaler_envs != rollout.num_envs:
                # One running return per copy; the statistics survive a change of copies.
                scaler = RewardScaler(rollout.num_envs, self.config.gamma)
                if self.reward_scaler is not None:
                    scaler.load_state_dict(self.reward_scaler.state_dict())
                self.reward_scaler, self._reward_scaler_envs = scaler, rollout.num_envs
            done = rollout["terminated"] | rollout["truncated"]
            # Step by step, as if scaled online: each step uses the statistics so far.
            rewards = np.stack([self.reward_scaler(r, d) for r, d in zip(rewards, done)])
        return np.broadcast_to(rewards, (*rewards.shape[:2], self.spec.n_agents))

    def _update(self, rollout: RolloutBuffer) -> dict[str, float]:
        cfg, spec = self.config, self.spec
        steps, copies = rollout.pos, rollout.num_envs
        fraction = self._learning_rate_fraction(steps * copies)
        for optimizer, lr in (
            (self.actor_optimizer, cfg.lr),
            (self.critic_optimizer, self.critic_lr),
        ):
            for group in optimizer.param_groups:
                group["lr"] = lr * fraction

        obs = self.tensor(rollout["norm_obs"])
        next_obs = self.tensor(self.obs_normalizer(rollout["next_obs"]))
        with torch.no_grad():
            values = self.values(obs)
            # For finished episodes next_obs holds the final observation, so
            # truncated episodes bootstrap from V(final) and terminated ones do not.
            next_values = self.values(next_obs)
        advantages, returns = compute_gae(
            self._training_rewards(rollout),
            values.cpu().numpy(),
            next_values.cpu().numpy(),
            rollout["terminated"],
            rollout["truncated"],
            cfg.gamma,
            cfg.gae_lambda,
        )
        explained_variance = _explained_variance(values.cpu().numpy(), returns)
        if cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Samples are (step, copy) pairs with all their agents: (N, n, ...).
        size = steps * copies
        batch = {
            "obs": obs.reshape(size, spec.n_agents, spec.obs_dim),
            "actions": self.tensor(
                rollout["actions"], torch.float32 if spec.continuous else torch.int64
            ).reshape(size, *rollout["actions"].shape[2:]),
            "log_prob": self.tensor(rollout["log_prob"]).reshape(size, spec.n_agents),
            "advantages": self.tensor(advantages).reshape(size, spec.n_agents),
            "returns": self.tensor(returns).reshape(size, spec.n_agents),
            "values": values.reshape(size, spec.n_agents),
        }
        totals: dict[str, float] = {}
        count = epochs = 0
        for _ in range(cfg.n_epochs):
            order = torch.randperm(size, generator=self.torch_rng, device=self.device)
            epoch_kl, epoch_batches = 0.0, 0
            for index in torch.tensor_split(order, min(cfg.num_minibatches, size)):
                stats = self._sgd_step({name: value[index] for name, value in batch.items()})
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0.0) + value
                epoch_kl += stats["approx_kl"]
                epoch_batches += 1
            count += epoch_batches
            epochs += 1
            if cfg.target_kl is not None and epoch_kl / epoch_batches > 1.5 * cfg.target_kl:
                break

        stats = {key: value / count for key, value in totals.items()}
        stats.update(
            explained_variance=explained_variance,
            epochs=float(epochs),
            learning_rate=cfg.lr * fraction,
            value_mean=float(values.mean()),
        )
        if spec.continuous:
            stats["action_std"] = float(
                torch.stack([net.log_std.detach().exp().mean() for net in self.actor.nets]).mean()
            )
        if self.reward_scaler is not None:
            stats["reward_std"] = float(self.reward_scaler.rms.std)
        return stats

    def _sgd_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """One gradient step of the actor and the critic on a minibatch."""
        cfg = self.config
        advantages = batch["advantages"]

        log_prob, entropy = self.actor.call(
            "log_prob", self.agent_inputs(batch["obs"]), batch["actions"]
        )
        log_ratio = log_prob - batch["log_prob"]
        ratio = log_ratio.exp()
        surrogate = torch.min(
            ratio * advantages,
            ratio.clamp(1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * advantages,
        )
        policy_loss = -surrogate.mean()
        entropy_mean = entropy.mean()
        _optimise(
            self.actor_optimizer,
            self.actor,
            policy_loss - cfg.ent_coef * entropy_mean,
            cfg.max_grad_norm,
        )

        value_loss = self._value_loss(self.values(batch["obs"]), batch["values"], batch["returns"])
        _optimise(self.critic_optimizer, self.critic, value_loss, cfg.max_grad_norm)

        with torch.no_grad():
            # Low-variance estimator of KL(old || new) (Schulman, 2020).
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > cfg.clip_range).float().mean()
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_mean.item(),
            "approx_kl": float(approx_kl),
            "clip_fraction": float(clip_fraction),
        }

    def _value_loss(
        self, values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        """Critic loss ``mean(max(l(V - R), l(V_clip - R)))``, or ``mean(l(V - R))`` unclipped."""
        cfg = self.config
        loss = self._error_loss(values - returns)
        if cfg.value_clip_range is not None:
            clipped = old_values + (values - old_values).clamp(
                -cfg.value_clip_range, cfg.value_clip_range
            )
            loss = torch.max(loss, self._error_loss(clipped - returns))
        return loss.mean()

    def _error_loss(self, error: torch.Tensor) -> torch.Tensor:
        """Element-wise ``0.5 error^2``, or the Huber loss with ``huber_delta``."""
        if self.config.huber_delta is None:
            return 0.5 * error.pow(2)
        return F.huber_loss(
            error, torch.zeros_like(error), reduction="none", delta=self.config.huber_delta
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _modules(self) -> dict[str, nn.Module | torch.optim.Optimizer]:
        return {
            "actor": self.actor,
            "critic": self.critic,
            "actor_optimizer": self.actor_optimizer,
            "critic_optimizer": self.critic_optimizer,
        }

    def _extra_state(self) -> dict[str, Any]:
        scaler = self.reward_scaler
        return {
            "obs_normalizer": self.obs_normalizer.state_dict(),
            "reward_scaler": None if scaler is None else scaler.state_dict(),
            "reward_scaler_envs": self._reward_scaler_envs,
        }

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        if "obs_normalizer" in state:
            self.obs_normalizer.load_state_dict(state["obs_normalizer"])
        if state.get("reward_scaler") is not None:
            self._reward_scaler_envs = int(state["reward_scaler_envs"])
            self.reward_scaler = RewardScaler(self._reward_scaler_envs, self.config.gamma)
            self.reward_scaler.load_state_dict(state["reward_scaler"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _optimise(
    optimizer: torch.optim.Optimizer, module: nn.Module, loss: torch.Tensor, max_norm: float
) -> None:
    """Gradient step with gradient-norm clipping."""
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(module.parameters(), max_norm)
    optimizer.step()


def _explained_variance(predictions: np.ndarray, targets: np.ndarray) -> float:
    """``1 - Var(targets - predictions) / Var(targets)`` (NaN when the targets are constant)."""
    variance = float(np.var(targets))
    if variance == 0.0:
        return float("nan")
    return 1.0 - float(np.var(targets - predictions)) / variance
