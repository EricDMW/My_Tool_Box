"""Experience storage: rollout buffers for on-policy and replay buffers for off-policy methods."""

from __future__ import annotations

from typing import Any

import numpy as np

from marl_algorithms.core.runner import Transition
from marl_algorithms.core.spec import MultiAgentSpec

__all__ = ["ReplayBuffer", "RolloutBuffer", "compute_gae"]


class RolloutBuffer:
    """Fixed-length storage of named arrays with shape ``(T, B, ...)``.

    Parameters
    ----------
    length:
        Number of vector steps ``T`` per rollout.
    num_envs:
        Number of copies ``B``.
    fields:
        ``{name: (trailing_shape, dtype)}``.

    Examples
    --------
    >>> buf = RolloutBuffer(4, 2, {"reward": ((), np.float32), "obs": ((3, 5), np.float32)})
    >>> buf.add(reward=np.zeros(2), obs=np.zeros((2, 3, 5)))
    >>> buf["obs"].shape
    (4, 2, 3, 5)
    """

    def __init__(
        self, length: int, num_envs: int, fields: dict[str, tuple[tuple[int, ...], Any]]
    ) -> None:
        self.length = int(length)
        self.num_envs = int(num_envs)
        self.data = {
            name: np.zeros((self.length, self.num_envs, *shape), dtype=dtype)
            for name, (shape, dtype) in fields.items()
        }
        self.pos = 0

    def add(self, **values: Any) -> None:
        """Store one vector step (every field must be given)."""
        if self.pos >= self.length:
            raise RuntimeError("rollout buffer is full; call reset()")
        missing = set(self.data) - set(values)
        if missing:
            raise KeyError(f"missing rollout fields: {sorted(missing)}")
        for name, value in values.items():
            self.data[name][self.pos] = value
        self.pos += 1

    @property
    def full(self) -> bool:
        """Whether ``length`` steps have been stored."""
        return self.pos == self.length

    def reset(self) -> None:
        """Start a new rollout (the arrays are reused)."""
        self.pos = 0

    def __getitem__(self, name: str) -> np.ndarray:
        return self.data[name][: self.pos]

    def __contains__(self, name: str) -> bool:
        return name in self.data


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation over a rollout.

    All arrays have shape ``(T, B, ...)`` (trailing dimensions such as agents
    are broadcast against the ``(T, B)`` flags). ``next_values`` are the values
    of the successor observations, which for finished episodes are the final
    observations (``"same_step"`` autoreset), so truncated episodes bootstrap
    and terminated ones do not.

    Returns
    -------
    tuple
        ``(advantages, returns)`` with the shape of ``rewards``.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    extra = (1,) * (rewards.ndim - 2)
    not_terminated = (~np.asarray(terminated, dtype=bool)).reshape(*terminated.shape, *extra)
    not_done = (~(np.asarray(terminated, bool) | np.asarray(truncated, bool))).reshape(
        *terminated.shape, *extra
    )
    deltas = rewards + gamma * not_terminated * next_values - values
    advantages = np.zeros_like(rewards)
    running = np.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = deltas[t] + gamma * gae_lambda * not_done[t] * running
        advantages[t] = running
    return advantages, advantages + values


class ReplayBuffer:
    """Circular buffer of joint transitions for off-policy methods.

    Parameters
    ----------
    capacity:
        Maximum number of stored transitions (one per copy and step).
    spec:
        Agent structure (defines the array shapes).
    """

    def __init__(self, capacity: int, spec: MultiAgentSpec) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}")
        n, d = spec.n_agents, spec.obs_dim
        self.capacity = int(capacity)
        self.spec = spec
        if spec.continuous:
            actions = np.zeros((self.capacity, n, spec.action_dim), dtype=np.float32)
        else:
            actions = np.zeros((self.capacity, n), dtype=np.int64)
        self.data = {
            "obs": np.zeros((self.capacity, n, d), dtype=np.float32),
            "actions": actions,
            "reward": np.zeros(self.capacity, dtype=np.float32),
            "agent_rewards": np.zeros((self.capacity, n), dtype=np.float32),
            "next_obs": np.zeros((self.capacity, n, d), dtype=np.float32),
            "terminated": np.zeros(self.capacity, dtype=np.float32),
        }
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, transition: Transition) -> None:
        """Store every copy of a vector transition."""
        actions = np.asarray(transition.actions)
        if self.spec.continuous and actions.ndim == 2:
            actions = actions[..., None]
        batch = {
            "obs": transition.obs,
            "actions": actions,
            "reward": transition.reward,
            "agent_rewards": transition.agent_rewards,
            "next_obs": transition.next_obs,
            "terminated": transition.terminated,
        }
        count = len(transition.reward)
        index = (self.pos + np.arange(count)) % self.capacity
        for name, value in batch.items():
            self.data[name][index] = value
        self.pos = int((self.pos + count) % self.capacity)
        self.size = int(min(self.size + count, self.capacity))

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Uniformly sample ``batch_size`` stored transitions."""
        if self.size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        index = rng.integers(0, self.size, size=int(batch_size))
        return {name: value[index] for name, value in self.data.items()}
