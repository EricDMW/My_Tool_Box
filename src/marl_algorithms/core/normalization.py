"""Running statistics for observation normalisation and reward scaling."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["ObservationNormalizer", "RewardScaler", "RunningMeanStd"]


class RunningMeanStd:
    """Running mean and variance over the leading axes (parallel update rule).

    Parameters
    ----------
    shape:
        Shape of one sample (the trailing axes of the arrays passed to
        :meth:`update`).
    """

    def __init__(self, shape: tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        """Include a batch of samples of shape ``(..., *shape)``."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, *self.mean.shape)
        if x.shape[0] == 0:
            return
        batch_mean, batch_var, batch_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = (
            self.var * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / total
        )
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        """Standard deviation (at least ``1e-8``)."""
        return np.sqrt(np.maximum(self.var, 1e-16))

    def state_dict(self) -> dict[str, Any]:
        """Serialisable state."""
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": float(self.count)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore :meth:`state_dict` output."""
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class ObservationNormalizer:
    """Standardise per-agent observations with statistics shared by all agents.

    Parameters
    ----------
    obs_dim:
        Size of one agent's observation.
    clip:
        Normalised values are clipped to ``[-clip, clip]``.
    enabled:
        When false, :meth:`__call__` returns its input as ``float32``.
    """

    def __init__(self, obs_dim: int, clip: float = 10.0, enabled: bool = True) -> None:
        self.rms = RunningMeanStd((int(obs_dim),))
        self.clip = float(clip)
        self.enabled = bool(enabled)

    def update(self, obs: np.ndarray) -> None:
        """Include observations of shape ``(..., obs_dim)``."""
        if self.enabled:
            self.rms.update(obs)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if not self.enabled:
            return obs
        normalized = (obs - self.rms.mean) / self.rms.std
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        """Serialisable state."""
        return {"rms": self.rms.state_dict(), "clip": self.clip, "enabled": self.enabled}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore :meth:`state_dict` output."""
        self.rms.load_state_dict(state["rms"])
        self.clip = float(state["clip"])
        self.enabled = bool(state["enabled"])


class RewardScaler:
    """Divide rewards by the running standard deviation of the discounted return.

    This keeps value targets of order one whatever the reward scale of the
    environment (PowerGrid rewards are about 1e-3 per step, Consensus rewards
    about 1e2), without changing the optimal policy.

    Parameters
    ----------
    num_envs:
        Number of copies (one running return each).
    gamma:
        Discount factor.
    enabled:
        When false, rewards are returned unchanged.
    """

    def __init__(self, num_envs: int, gamma: float, enabled: bool = True) -> None:
        self.rms = RunningMeanStd(())
        self.gamma = float(gamma)
        self.enabled = bool(enabled)
        self._returns = np.zeros(int(num_envs))

    def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        """Scale rewards of shape ``(B, ...)`` and advance the running returns."""
        reward = np.asarray(reward, dtype=np.float64)
        if not self.enabled:
            return reward
        team = reward.reshape(reward.shape[0], -1).mean(axis=1)
        self._returns = self._returns * self.gamma + team
        self.rms.update(self._returns)
        self._returns[np.asarray(done, dtype=bool)] = 0.0
        return reward / self.rms.std

    def state_dict(self) -> dict[str, Any]:
        """Serialisable state."""
        return {"rms": self.rms.state_dict(), "gamma": self.gamma, "enabled": self.enabled}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore :meth:`state_dict` output."""
        self.rms.load_state_dict(state["rms"])
        self.gamma = float(state["gamma"])
        self.enabled = bool(state["enabled"])
