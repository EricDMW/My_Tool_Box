"""Off-policy multi-agent actor-critic methods: MADDPG and MATD3.

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

__all__ = ["DDPGConfig", "MADDPG", "MATD3", "PRESETS"]

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
        ``Q_i`` (agent-specific global state; see the module docstring).
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


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------
class _MADDPGBase(OffPolicyAlgorithm):
    """Shared implementation of MADDPG and MATD3 (see the module docstring).

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


class MADDPG(_MADDPGBase):
    """Multi-agent deep deterministic policy gradient (Lowe et al., 2017).

    Agent ``i`` has a deterministic actor ``mu_i(o_i)`` and a centralised critic
    ``Q_i(s, a_1, ..., a_n)`` with target copies ``mu'_i`` and ``Q'_i``. On a
    replay batch ``(o, a, r, o', d)`` with global states ``s = (o_1..o_n)``:

    * critic loss (TD regression, ``c`` = ``reward_scale``)::

          y_i = c r_i + gamma (1 - d) Q'_i(s', mu'_1(o'_1), ..., mu'_n(o'_n))
          L(Q_i) = mean (Q_i(s, a_1, ..., a_n) - y_i)^2

      (or the Huber loss of the TD error with ``critic_loss="huber"``);

    * actor loss (deterministic policy gradient; the other agents' actions come
      from the replay batch)::

          L(mu_i) = -mean Q_i(s, a_1, ..., mu_i(o_i), ..., a_n)

    * target networks: ``theta' <- (1 - tau) theta' + tau theta`` after every
      update.

    ``d`` is the ``terminated`` flag: truncated episodes bootstrap from their
    final observations. ``r_i`` is agent ``i``'s own reward
    (``reward_source="agent"``) or the team reward (``"team"``). Exploration
    adds Gaussian noise to ``mu_i(o_i)``. With shared parameters (default) one
    critic, evaluated with agent ``i``'s one-hot identifier, plays all ``Q_i``;
    by default its input also repeats ``o_i`` and ``a_i``
    (``critic_local_inputs``). Losses are averaged over agents, and all agents
    are updated in one batched pass.

    Parameters
    ----------
    spec:
        Agent structure, or an environment (single or vector) to read it from.
        The actions must be continuous.
    config:
        :class:`DDPGConfig`; defaults to ``DDPGConfig()``.
    device:
        Torch device.
    seed:
        Seed of initialisation, exploration and replay sampling.
    **overrides:
        :class:`DDPGConfig` fields, for example ``actor_lr=3e-4``.

    Raises
    ------
    TypeError
        For discrete action spaces or unknown configuration fields.

    Examples
    --------
    >>> from marl_algorithms import MADDPG, make_vector_env
    >>> envs = make_vector_env("PowerGrid-v0", num_envs=16, n_buses=6)
    >>> algo = MADDPG(envs, seed=0, warmup_steps=1_000)
    >>> log = algo.learn(envs, total_steps=5_000, seed=0)
    >>> algo.act(envs.reset(seed=1)[0], deterministic=True).shape
    (16, 6, 1)
    """

    name: ClassVar[str] = "maddpg"


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


# ---------------------------------------------------------------------------
# Tuned settings for the demonstrations
# ---------------------------------------------------------------------------
# PowerGrid-v0 (16 buses): the team reward (the evaluation objective) avoids a
# free-rider effect of the per-agent rewards, where each bus sees only about
# 1/16 of the benefit of its frequency support but pays its full control cost.
# The Huber loss keeps the rare trip transitions (team reward about -1600) from
# dominating the critic updates. Consensus-v0 (8 agents on a ring): per-agent
# rewards (each agent's own neighbourhood disagreement), rewards scaled by 0.01
# to order one, observations normalised with warm-up statistics.
_POWER_GRID: dict[str, Any] = {
    "num_envs": 16,
    "total_steps": 64_000,
    "config": {
        "hidden_sizes": (64, 64),
        "critic_hidden_sizes": (128, 128),
        "batch_size": 64,
        "warmup_steps": 3_200,
        "gamma": 0.99,
        "reward_source": "team",
        "critic_loss": "huber",
    },
    "env_kwargs": {},
}
_CONSENSUS: dict[str, Any] = {
    "num_envs": 16,
    "total_steps": 96_000,
    "config": {
        "hidden_sizes": (64, 64),
        "critic_hidden_sizes": (128, 128),
        "batch_size": 64,
        "warmup_steps": 3_200,
        "gamma": 0.95,
        "reward_scale": 0.01,
        "normalize_observations": True,
        "exploration_noise": 0.2,
        "final_exploration_noise": 0.02,
        "noise_decay_steps": 96_000,
    },
    "env_kwargs": {},
}

#: Tuned settings, ``PRESETS[algorithm][env_id]``, for ``algorithm`` in
#: ``("maddpg", "matd3")``. Every entry holds ``num_envs`` (parallel copies),
#: ``total_steps`` (environment steps summed over copies), ``config``
#: (:class:`DDPGConfig` overrides) and ``env_kwargs`` (environment arguments), so
#: ``marl_algorithms.train(algorithm, env_id, preset["total_steps"],
#: num_envs=preset["num_envs"], env_kwargs=preset["env_kwargs"], **preset["config"])``
#: reproduces a demonstration run. Each trains in about 60-120 s on one CPU core.
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "maddpg": {
        "PowerGrid-v0": copy.deepcopy(_POWER_GRID),
        "Consensus-v0": copy.deepcopy(_CONSENSUS),
    },
    "matd3": {
        "PowerGrid-v0": {
            **copy.deepcopy(_POWER_GRID),
            "config": {
                **_POWER_GRID["config"],
                "exploration_noise": 0.3,
                "final_exploration_noise": 0.05,
                "noise_decay_steps": 64_000,
            },
        },
        "Consensus-v0": copy.deepcopy(_CONSENSUS),
    },
}
