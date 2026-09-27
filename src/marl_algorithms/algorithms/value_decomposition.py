"""Value-based cooperative multi-agent Q-learning: IQL, VDN and QMIX.

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
from marl_algorithms.core.config import AlgorithmConfig, OffPolicyConfig
from marl_algorithms.core.networks import PerAgent, QMixer, mlp, soft_update

__all__ = ["IQL", "PRESETS", "QMIX", "VDN", "QLearningConfig", "VDNMixer"]

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


# ----------------------------------------------------------------------
# Mixers
# ----------------------------------------------------------------------
class VDNMixer(nn.Module):
    """Additive mixing of VDN: ``Q_tot = sum_i Q_i`` (no parameters).

    Takes the same arguments as :class:`~marl_algorithms.core.networks.QMixer`
    so that both mixers are interchangeable; the state is ignored.
    """

    def forward(self, q: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """Team value of shape ``q.shape[:-1]`` from utilities ``(..., n_agents)``."""
        del state
        return q.sum(dim=-1)


# ----------------------------------------------------------------------
# Algorithms
# ----------------------------------------------------------------------
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


class IQL(_ValueBase):
    """Independent Q-learning with DQN agents (Tan, 1993; Mnih et al., 2015).

    Every agent learns ``Q_i(o_i, a_i)`` from its own reward
    ``r_i`` (``info["agent_rewards"]``) with its own TD target::

        y_i = c r_i + gamma (1 - terminated) Q_i^-(o'_i, argmax_a Q_i(o'_i, a))
        L   = mean_batch mean_i  l(Q_i(o_i, a_i) - y_i)

    No mixing and no centralised information: the other agents are part of
    the environment. Environments that report no per-agent rewards give every
    agent the team reward; with a single agent IQL is (double) DQN.

    Agents are feed-forward networks trained on single transitions from a
    transition replay (see the module docstring for this simplification).

    Parameters
    ----------
    spec:
        Agent structure, or a discrete-action environment to read it from.
    config:
        :class:`QLearningConfig`; defaults to ``QLearningConfig()``.
    device:
        Torch device.
    seed:
        Seed of initialisation, exploration and replay sampling.
    **overrides:
        Configuration fields to override.

    Raises
    ------
    TypeError
        For continuous-action environments.

    Examples
    --------
    >>> from marl_algorithms import IQL, make_vector_env
    >>> envs = make_vector_env("LineMsg-v0", num_envs=16)
    >>> algo = IQL(envs, seed=0)
    >>> log = algo.learn(envs, total_steps=20_000, seed=0)
    """

    name: ClassVar[str] = "iql"


class VDN(_ValueBase):
    """Value decomposition networks (Sunehag et al., 2018).

    The team value is the sum of the agent utilities, trained on the team
    reward ``r``::

        Q_tot(o, a) = sum_i Q_i(o_i, a_i)
        y           = c r + gamma (1 - terminated) sum_i Q_i^-(o'_i, argmax_a Q_i(o'_i, a))
        L           = mean_batch  l(Q_tot(o, a) - y)

    Agents are feed-forward networks trained on single transitions from a
    transition replay, not the recurrent agents and episode replay of the
    paper (see the module docstring).

    Parameters and exceptions as for :class:`IQL`.

    Examples
    --------
    >>> from marl_algorithms import train
    >>> algo, log = train("vdn", "LineMsg-v0", total_steps=50_000, num_envs=32)
    """

    name: ClassVar[str] = "vdn"

    def _make_mixer(self) -> nn.Module:
        return VDNMixer()


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
    agents and episode replay of the paper (see the module docstring).

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


# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------
#: LineMsg-v0 (10 agents): every method learns "always relay" (return 95, the
#: optimum; random actions about 43) in about 10-20 s.
_LINEMSG: dict[str, Any] = {
    "num_envs": 16,
    "total_steps": 25_000,
    "config": {"batch_size": 128, "lr": 1e-3, "epsilon_decay_steps": 12_500},
    "env_kwargs": {},
}

#: WirelessComm-v1 (4x4 grid, 16 agents): packets are delivered in the step
#: they are sent, so a short horizon (gamma = 0.8) suffices and learns faster.
#: The collision-free schedule of env_lib.baseline_policy returns about 340,
#: random actions about 140. VDN and QMIX return 320-345 depending on the seed:
#: they learn either a collision-free "owners only" schedule (about 319: every
#: access point serves one fixed neighbouring agent) or, like the baseline,
#: also let the border agents borrow idle access points.
_WIRELESS_CONFIG: dict[str, Any] = {"batch_size": 128, "lr": 1e-3, "gamma": 0.8}

#: Tuned demonstration settings ``PRESETS[algorithm][env_id]`` for ``algorithm``
#: in ``"iql"``, ``"vdn"``, ``"qmix"``. A preset is a dictionary with the keys
#: ``"total_steps"`` (environment steps summed over copies), ``"num_envs"``
#: (batched copies), ``"config"`` (:class:`QLearningConfig` overrides) and
#: ``"env_kwargs"`` (environment arguments); see :mod:`marl_algorithms.presets`.
#: Every preset trains in at most about two minutes on one CPU core.
#:
#: IQL has no WirelessComm preset: with its per-agent rewards a collision costs
#: the transmitting agent nothing, so independent learners keep transmitting
#: and collide (hardly better than random actions), while VDN and QMIX, trained
#: on the team reward, learn to leave contested access points to one agent.
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "iql": {
        "LineMsg-v0": copy.deepcopy(_LINEMSG),
    },
    "vdn": {
        "LineMsg-v0": copy.deepcopy(_LINEMSG),
        "WirelessComm-v1": {
            "num_envs": 16,
            "total_steps": 120_000,
            "config": {**_WIRELESS_CONFIG, "epsilon_decay_steps": 30_000},
            "env_kwargs": {},
        },
    },
    "qmix": {
        "LineMsg-v0": copy.deepcopy(_LINEMSG),
        "WirelessComm-v1": {
            "num_envs": 16,
            "total_steps": 100_000,
            "config": {**_WIRELESS_CONFIG, "epsilon_decay_steps": 45_000},
            "env_kwargs": {},
        },
    },
}
