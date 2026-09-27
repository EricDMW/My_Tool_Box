"""Algorithm base classes and the training log.

:class:`Algorithm` fixes the interface every method implements: build networks
from a :class:`~marl_algorithms.core.spec.MultiAgentSpec`, act on per-agent
observations, learn from a vector environment, save and load, and evaluate with
``env_lib.evaluate``. Two subclasses provide the training loops:

* :class:`OnPolicyAlgorithm` collects rollouts of ``rollout_length`` vector
  steps and calls :meth:`OnPolicyAlgorithm._update` on each;
* :class:`OffPolicyAlgorithm` stores every transition in a replay buffer and
  performs gradient steps on sampled batches.
"""

from __future__ import annotations

import abc
import csv
import dataclasses
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

from marl_algorithms.core.buffers import ReplayBuffer, RolloutBuffer
from marl_algorithms.core.config import AlgorithmConfig, OffPolicyConfig, OnPolicyConfig
from marl_algorithms.core.networks import agent_one_hot
from marl_algorithms.core.runner import Transition, VectorRunner
from marl_algorithms.core.spec import MultiAgentSpec

__all__ = ["Algorithm", "OffPolicyAlgorithm", "OnPolicyAlgorithm", "TrainingLog"]

Callback = Callable[["Algorithm", "TrainingLog"], Any]


class TrainingLog:
    """Episode returns and update statistics recorded during training.

    Attributes
    ----------
    algorithm, env_id:
        What was trained.
    episodes:
        ``(env_steps, team_return, length)`` of every finished training episode.
    updates:
        ``(env_steps, statistics)`` after every update.
    env_steps:
        Environment steps taken when training last returned (the algorithm's
        counter, summed over copies).
    wall_time:
        Seconds from the creation of the log to its last record.
    """

    def __init__(self, algorithm: str, env_id: str | None = None) -> None:
        self.algorithm = algorithm
        self.env_id = env_id
        self.episodes: list[tuple[int, float, int]] = []
        self.updates: list[tuple[int, dict[str, float]]] = []
        self.env_steps = 0
        self._start = time.perf_counter()
        self.wall_time = 0.0

    def record_steps(self, env_steps: int) -> None:
        """Record the environment steps taken so far (called when training returns)."""
        self.env_steps = int(env_steps)
        self.wall_time = time.perf_counter() - self._start

    def record_episodes(self, env_steps: int, episodes: list[tuple[float, int]]) -> None:
        """Add finished episodes."""
        self.episodes.extend((int(env_steps), float(ret), int(length)) for ret, length in episodes)
        self.wall_time = time.perf_counter() - self._start

    def record_update(self, env_steps: int, stats: dict[str, float]) -> None:
        """Add the statistics of one update."""
        self.updates.append((int(env_steps), {k: float(v) for k, v in stats.items()}))
        self.wall_time = time.perf_counter() - self._start

    @property
    def episode_steps(self) -> np.ndarray:
        """Environment steps at which the episodes finished."""
        return np.array([e[0] for e in self.episodes], dtype=np.int64)

    @property
    def episode_returns(self) -> np.ndarray:
        """Team returns of the finished episodes."""
        return np.array([e[1] for e in self.episodes], dtype=np.float64)

    def mean_return(self, last: int = 100) -> float:
        """Mean team return of the last ``last`` episodes (NaN before the first)."""
        returns = self.episode_returns[-last:]
        return float(returns.mean()) if returns.size else float("nan")

    def curve(self, points: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Learning curve: mean return in ``points`` equal bins of environment steps."""
        steps, returns = self.episode_steps, self.episode_returns
        if steps.size == 0:
            return np.zeros(0), np.zeros(0)
        edges = np.linspace(0, steps.max(), int(points) + 1)
        index = np.clip(np.searchsorted(edges, steps, side="right") - 1, 0, int(points) - 1)
        sums = np.bincount(index, returns, minlength=int(points))
        counts = np.bincount(index, minlength=int(points))
        keep = counts > 0
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers[keep], sums[keep] / counts[keep]

    def summary(self) -> str:
        """One-line summary."""
        steps = max(
            self.env_steps,
            self.episodes[-1][0] if self.episodes else 0,
            self.updates[-1][0] if self.updates else 0,
        )
        return (
            f"{self.algorithm} on {self.env_id or 'environment'}: {len(self.episodes)} episodes, "
            f"{steps} env steps, mean return (last 100) {self.mean_return():.4g}, "
            f"{self.wall_time:.1f} s"
        )

    def to_csv(self, path: str | Path) -> Path:
        """Write the episodes as CSV (``env_steps, return, length``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["env_steps", "return", "length"])
            writer.writerows(self.episodes)
        return path


class Algorithm(abc.ABC):
    """Base class of all multi-agent algorithms.

    Parameters
    ----------
    spec:
        Agent structure, or an environment (single or vector) to read it from.
    config:
        Configuration; defaults to ``config_class()``.
    device:
        Torch device of the networks.
    seed:
        Seed of network initialisation and of all sampling done by the
        algorithm (exploration, minibatches, replay).
    **overrides:
        Configuration fields to override, for example ``lr=1e-3``.
    """

    #: Registry name, for example ``"mappo"``.
    name: ClassVar[str] = ""
    #: Supported action kinds.
    action_kinds: ClassVar[tuple[str, ...]] = ("continuous", "discrete")
    #: Configuration dataclass.
    config_class: ClassVar[type[AlgorithmConfig]] = AlgorithmConfig

    def __init__(
        self,
        spec: MultiAgentSpec | Any,
        config: AlgorithmConfig | None = None,
        *,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        **overrides: Any,
    ) -> None:
        if not isinstance(spec, MultiAgentSpec):
            spec = MultiAgentSpec.from_env(spec)
        if spec.action_kind not in self.action_kinds:
            raise TypeError(
                f"{type(self).__name__} supports {' and '.join(self.action_kinds)} actions; "
                f"this environment has {spec.action_kind} actions"
            )
        config = config if config is not None else self.config_class()
        if not isinstance(config, self.config_class):
            raise TypeError(
                f"config must be a {self.config_class.__name__}, got {type(config).__name__}"
            )
        if overrides:
            unknown = set(overrides) - set(config.field_names())
            if unknown:
                raise TypeError(f"unknown {self.config_class.__name__} fields: {sorted(unknown)}")
            config = dataclasses.replace(config, **overrides)
        self.spec = spec
        self.config = config
        self.device = torch.device(device)
        self.seed = seed
        seed_sequence = np.random.SeedSequence(seed)
        self.np_rng = np.random.default_rng(seed_sequence)
        torch_seed = int(seed_sequence.generate_state(1)[0])
        self.torch_rng = torch.Generator(device=self.device)
        self.torch_rng.manual_seed(torch_seed)
        self.env_steps = 0
        self.num_updates = 0
        # Initialise the networks from the seed without touching torch's global RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch_seed)
            self._build()

    # ------------------------------------------------------------------
    # To implement
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _build(self) -> None:
        """Create networks and optimisers."""

    @abc.abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Per-agent actions for per-agent observations.

        Parameters
        ----------
        obs:
            ``(*batch, n_agents, obs_dim)`` observations.
        deterministic:
            Greedy / mean actions instead of exploratory ones.

        Returns
        -------
        numpy.ndarray
            ``(*batch, n_agents, action_dim)`` floats or ``(*batch, n_agents)``
            integers.
        """

    @abc.abstractmethod
    def learn(
        self,
        envs: Any,
        total_steps: int,
        *,
        seed: int | None = None,
        log: TrainingLog | None = None,
        callback: Callback | None = None,
    ) -> TrainingLog:
        """Train on a vector environment until ``env_steps`` reaches ``total_steps``."""

    @abc.abstractmethod
    def _modules(self) -> dict[str, nn.Module | torch.optim.Optimizer]:
        """Networks and optimisers to save."""

    def _extra_state(self) -> dict[str, Any]:
        """Additional state to save (normalisers, schedules)."""
        return {}

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        """Restore :meth:`_extra_state` output (nothing by default)."""
        del state

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------
    @property
    def uses_agent_ids(self) -> bool:
        """Whether per-agent inputs carry a one-hot agent identifier."""
        cfg = self.config
        return bool(cfg.share_parameters and cfg.agent_ids and self.spec.n_agents > 1)

    @property
    def agent_input_dim(self) -> int:
        """Input size of per-agent networks."""
        return self.spec.obs_dim + (self.spec.n_agents if self.uses_agent_ids else 0)

    def agent_inputs(self, obs: torch.Tensor) -> torch.Tensor:
        """Per-agent network inputs ``(..., n, obs_dim [+ n])`` from observations."""
        if not self.uses_agent_ids:
            return obs
        ids = agent_one_hot(obs.shape[:-2], self.spec.n_agents, obs.device)
        return torch.cat([obs, ids], dim=-1)

    def tensor(self, array: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """``array`` as a tensor on the algorithm's device."""
        return torch.as_tensor(np.asarray(array), dtype=dtype, device=self.device)

    def random_actions(self, batch: int) -> np.ndarray:
        """Uniformly random per-agent actions for ``batch`` copies."""
        spec = self.spec
        if spec.continuous:
            u = self.np_rng.random((batch, spec.n_agents, spec.action_dim))
            return (spec.action_low + u * (spec.action_high - spec.action_low)).astype(np.float32)
        return self.np_rng.integers(0, spec.n_actions, size=(batch, spec.n_agents))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def policy(self, deterministic: bool = True) -> Callable[[Any], np.ndarray]:
        """Callable mapping environment observations to environment actions.

        Works for single environments (one observation) and vector
        environments (batched observations), so it can be passed to
        ``env_lib.evaluate`` and ``env_lib.utils.record_episode``.
        """
        spec = self.spec

        def policy(observation: Any) -> np.ndarray:
            obs = spec.agent_obs(observation)
            single = obs.ndim == 2
            actions = self.act(obs[None] if single else obs, deterministic=deterministic)
            return spec.env_action(actions[0] if single else actions)

        return policy

    def evaluate(
        self, env: Any, n_episodes: int = 10, *, seed: int | None = None, deterministic: bool = True
    ):
        """Evaluate with ``env_lib.evaluate`` (single or vector environment)."""
        from env_lib.utils.evaluation import evaluate

        return evaluate(env, self.policy(deterministic), n_episodes=n_episodes, seed=seed)

    def state_dict(self) -> dict[str, Any]:
        """Networks, optimisers, counters and extra state."""
        return {
            "modules": {name: module.state_dict() for name, module in self._modules().items()},
            "extra": self._extra_state(),
            "env_steps": self.env_steps,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore :meth:`state_dict` output."""
        modules = self._modules()
        for name, module_state in state["modules"].items():
            modules[name].load_state_dict(module_state)
        self._load_extra_state(state.get("extra", {}))
        self.env_steps = int(state.get("env_steps", 0))
        self.num_updates = int(state.get("num_updates", 0))

    def save(self, path: str | Path) -> Path:
        """Save the algorithm (spec, configuration and state) with ``torch.save``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": self.name,
                "spec": dataclasses.asdict(self.spec),
                "config": self.config.to_dict(),
                "seed": self.seed,
                "state": self.state_dict(),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> Algorithm:
        """Load an algorithm saved with :meth:`save`."""
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        spec_fields = dict(payload["spec"])
        for key in ("obs_shape", "action_shape"):
            spec_fields[key] = tuple(spec_fields[key])
        spec = MultiAgentSpec(**spec_fields)
        algorithm_cls = cls
        if cls is Algorithm or cls.name != payload["algorithm"]:
            from marl_algorithms.registry import get_algorithm

            algorithm_cls = get_algorithm(payload["algorithm"])
        config = algorithm_cls.config_class(**payload["config"])
        algorithm = algorithm_cls(spec, config, device=device, seed=payload.get("seed"))
        algorithm.load_state_dict(payload["state"])
        return algorithm

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.spec.describe()}, env_steps={self.env_steps})"


class OnPolicyAlgorithm(Algorithm):
    """Base class of on-policy actor-critic methods.

    Subclasses implement :meth:`_rollout_fields`, :meth:`_rollout_step` and
    :meth:`_update`; this class runs the loop: collect ``rollout_length`` vector
    steps into a :class:`~marl_algorithms.core.buffers.RolloutBuffer`, update,
    repeat.
    """

    config_class: ClassVar[type[AlgorithmConfig]] = OnPolicyConfig

    def _rollout_fields(self) -> dict[str, tuple[tuple[int, ...], Any]]:
        """Extra per-step fields stored by :meth:`_rollout_step` (besides the transition)."""
        return {}

    @abc.abstractmethod
    def _rollout_step(self, obs: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Exploratory actions for ``(B, n, obs_dim)`` observations and extra fields."""

    @abc.abstractmethod
    def _update(self, rollout: RolloutBuffer) -> dict[str, float]:
        """Update from a full rollout; returns statistics."""

    def _observe_transition(self, transition: Transition) -> None:
        """Hook called for every collected transition (for running statistics)."""

    def learn(
        self,
        envs: Any,
        total_steps: int,
        *,
        seed: int | None = None,
        log: TrainingLog | None = None,
        callback: Callback | None = None,
    ) -> TrainingLog:
        """Train until :attr:`env_steps` (summed over copies) reaches ``total_steps``.

        Parameters
        ----------
        envs:
            Vector environment with ``"same_step"`` autoreset
            (:func:`~marl_algorithms.core.runner.make_vector_env`).
        total_steps:
            Cumulative target for :attr:`env_steps`: a second call continues
            from the steps already taken, so ``learn(envs, 2 * n)`` after
            ``learn(envs, n)`` collects ``n`` more steps. Every call resets the
            environment copies first.
        seed:
            Seed of the environment reset.
        log:
            Log to append to (a new one by default).
        callback:
            Called as ``callback(algorithm, log)`` after every update; a return
            value of ``False`` stops training.
        """
        runner = VectorRunner(envs, self.spec)
        log = log if log is not None else TrainingLog(self.name, _env_id(envs))
        obs = runner.reset(seed=seed)
        spec, n_envs = self.spec, runner.num_envs
        action_shape = (spec.n_agents, spec.action_dim) if spec.continuous else (spec.n_agents,)
        fields = {
            "obs": ((spec.n_agents, spec.obs_dim), np.float32),
            "actions": (action_shape, np.float32 if spec.continuous else np.int64),
            "reward": ((), np.float32),
            "agent_rewards": ((spec.n_agents,), np.float32),
            "next_obs": ((spec.n_agents, spec.obs_dim), np.float32),
            "terminated": ((), bool),
            "truncated": ((), bool),
            **self._rollout_fields(),
        }
        buffer = RolloutBuffer(self.config.rollout_length, n_envs, fields)
        while self.env_steps < total_steps:
            buffer.reset()
            while not buffer.full:
                actions, extras = self._rollout_step(obs)
                transition = runner.step(actions)
                self._observe_transition(transition)
                buffer.add(
                    obs=transition.obs,
                    actions=transition.actions,
                    reward=transition.reward,
                    agent_rewards=transition.agent_rewards,
                    next_obs=transition.next_obs,
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                    **extras,
                )
                obs = runner.obs
                self.env_steps += n_envs
            stats = self._update(buffer)
            self.num_updates += 1
            log.record_episodes(self.env_steps, runner.pop_episodes())
            log.record_update(self.env_steps, stats)
            if callback is not None and callback(self, log) is False:
                break
        log.record_steps(self.env_steps)
        return log


class OffPolicyAlgorithm(Algorithm):
    """Base class of replay-based methods.

    Subclasses implement :meth:`_explore` and :meth:`_update`; this class runs
    the loop: act (uniformly at random during the warm-up), store the
    transition, and every ``update_every`` vector steps perform
    ``gradient_steps`` updates on batches sampled from the replay buffer.
    """

    config_class: ClassVar[type[AlgorithmConfig]] = OffPolicyConfig

    @abc.abstractmethod
    def _explore(self, obs: np.ndarray) -> np.ndarray:
        """Exploratory actions for ``(B, n, obs_dim)`` observations."""

    @abc.abstractmethod
    def _update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """One gradient step on a sampled batch; returns statistics."""

    def learn(
        self,
        envs: Any,
        total_steps: int,
        *,
        seed: int | None = None,
        log: TrainingLog | None = None,
        callback: Callback | None = None,
    ) -> TrainingLog:
        """Train until :attr:`env_steps` (summed over copies) reaches ``total_steps``.

        See :meth:`OnPolicyAlgorithm.learn` for the parameters; ``callback`` is
        called after every round of gradient steps.
        """
        cfg: OffPolicyConfig = self.config  # type: ignore[assignment]
        runner = VectorRunner(envs, self.spec)
        log = log if log is not None else TrainingLog(self.name, _env_id(envs))
        if not hasattr(self, "replay"):
            self.replay = ReplayBuffer(cfg.buffer_size, self.spec)
        obs = runner.reset(seed=seed)
        n_envs = runner.num_envs
        vector_steps = 0
        while self.env_steps < total_steps:
            if self.env_steps < cfg.warmup_steps:
                actions = self.random_actions(n_envs)
            else:
                actions = self._explore(obs)
            transition = runner.step(actions)
            self.replay.add(transition)
            obs = runner.obs
            self.env_steps += n_envs
            vector_steps += 1
            log.record_episodes(self.env_steps, runner.pop_episodes())
            ready = self.env_steps >= cfg.warmup_steps and len(self.replay) >= cfg.batch_size
            if ready and vector_steps % cfg.update_every == 0:
                stats: dict[str, float] = {}
                for _ in range(cfg.gradient_steps):
                    stats = self._update(self.replay.sample(cfg.batch_size, self.np_rng))
                    self.num_updates += 1
                log.record_update(self.env_steps, stats)
                if callback is not None and callback(self, log) is False:
                    break
        log.record_steps(self.env_steps)
        return log


def _env_id(envs: Any) -> str | None:
    spec = getattr(envs, "spec", None)
    return getattr(spec, "id", None)
