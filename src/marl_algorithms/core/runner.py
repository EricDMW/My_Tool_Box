"""Collect experience from a vector environment in per-agent form.

:class:`VectorRunner` steps a Gymnasium vector environment (normally a native
``env_lib`` batched environment created by :func:`make_vector_env`) and returns
:class:`Transition` batches of per-agent arrays. It relies on the
``"same_step"`` autoreset mode: every transition is a real one, and the final
observation of a finished episode is available for bootstrapping truncated
episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marl_algorithms.core.spec import MultiAgentSpec

__all__ = ["Transition", "VectorRunner", "make_vector_env"]


def make_vector_env(env_id: str, num_envs: int = 16, **env_kwargs: Any) -> Any:
    """Create ``num_envs`` copies of an ``env_lib`` environment for training.

    The copies use the native batched implementation when the environment has
    one, and ``"same_step"`` autoreset. LineMsg defaults to per-agent binary
    actions (``action_space_type="multibinary"``).

    Parameters
    ----------
    env_id:
        A registered ``env_lib`` id.
    num_envs:
        Number of copies simulated in parallel.
    **env_kwargs:
        Environment constructor arguments.
    """
    import env_lib

    if env_id.startswith("LineMsg"):
        env_kwargs.setdefault("action_space_type", "multibinary")
    return env_lib.make_vec(env_id, num_envs, autoreset_mode="same_step", **env_kwargs)


@dataclass
class Transition:
    """One vector step in per-agent form (``B`` copies, ``n`` agents).

    Attributes
    ----------
    obs:
        ``(B, n, obs_dim)`` observations the actions were taken in.
    actions:
        ``(B, n, action_dim)`` floats or ``(B, n)`` integers.
    reward:
        ``(B,)`` team rewards.
    agent_rewards:
        ``(B, n)`` per-agent rewards (the team reward when the environment
        reports none).
    next_obs:
        ``(B, n, obs_dim)`` successor observations; for a finished episode the
        final observation, not the first one of the next episode.
    terminated, truncated:
        ``(B,)`` episode-end flags.
    """

    obs: np.ndarray
    actions: np.ndarray
    reward: np.ndarray
    agent_rewards: np.ndarray
    next_obs: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    @property
    def done(self) -> np.ndarray:
        """``terminated | truncated``."""
        return self.terminated | self.truncated


class VectorRunner:
    """Step a vector environment and keep per-copy episode statistics.

    Parameters
    ----------
    envs:
        A Gymnasium vector environment with ``autoreset_mode="same_step"``
        (see :func:`make_vector_env`).
    spec:
        Agent structure; read from ``envs`` when omitted.
    """

    def __init__(self, envs: Any, spec: MultiAgentSpec | None = None) -> None:
        mode = envs.metadata.get("autoreset_mode")
        mode_name = str(getattr(mode, "value", mode)).replace("_", "").lower()
        if mode is not None and mode_name != "samestep":
            raise ValueError(
                "VectorRunner needs autoreset_mode='same_step'; create the environment with "
                "marl_algorithms.make_vector_env() or env_lib.make_vec(..., autoreset_mode='same_step')"
            )
        self.envs = envs
        self.spec = spec if spec is not None else MultiAgentSpec.from_env(envs)
        self.num_envs = int(envs.num_envs)
        self.obs: np.ndarray | None = None
        self._returns = np.zeros(self.num_envs)
        self._lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._finished: list[tuple[float, int]] = []

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset every copy; returns ``(B, n, obs_dim)`` observations."""
        obs, _ = self.envs.reset(seed=seed)
        self.obs = self.spec.agent_obs(obs)
        self._returns[:] = 0.0
        self._lengths[:] = 0
        return self.obs

    def step(self, actions: np.ndarray) -> Transition:
        """Apply per-agent ``actions`` to every copy.

        Returns
        -------
        Transition
            The transition; :attr:`obs` of the runner now holds the observations
            to act on next (first observations of new episodes where copies
            finished).
        """
        if self.obs is None:
            raise RuntimeError("call reset() before step()")
        spec = self.spec
        obs, reward, terminated, truncated, infos = self.envs.step(spec.env_action(actions))
        reward = np.asarray(reward, dtype=np.float64).reshape(self.num_envs)
        terminated = np.asarray(terminated, dtype=bool).reshape(self.num_envs)
        truncated = np.asarray(truncated, dtype=bool).reshape(self.num_envs)
        done = terminated | truncated

        agent_obs = spec.agent_obs(obs)
        next_obs = agent_obs.copy()
        agent_rewards = np.repeat(reward[:, None], spec.n_agents, axis=1)
        _fill_rows(agent_rewards, infos, "agent_rewards", ~done)
        if done.any():
            final_obs = infos.get("final_obs")
            if final_obs is not None:
                for index in np.flatnonzero(done):
                    if final_obs[index] is not None:
                        next_obs[index] = spec.agent_obs(final_obs[index])
            final_info = infos.get("final_info")
            if isinstance(final_info, dict):
                _fill_rows(agent_rewards, final_info, "agent_rewards", done)

        transition = Transition(
            obs=self.obs,
            actions=np.asarray(actions),
            reward=reward,
            agent_rewards=agent_rewards,
            next_obs=next_obs,
            terminated=terminated,
            truncated=truncated,
        )
        self._returns += reward
        self._lengths += 1
        for index in np.flatnonzero(done):
            self._finished.append((float(self._returns[index]), int(self._lengths[index])))
        self._returns[done] = 0.0
        self._lengths[done] = 0
        self.obs = agent_obs
        return transition

    def pop_episodes(self) -> list[tuple[float, int]]:
        """Return and clear the ``(return, length)`` of episodes finished so far."""
        finished, self._finished = self._finished, []
        return finished


def _fill_rows(target: np.ndarray, infos: dict[str, Any], key: str, rows: np.ndarray) -> None:
    """Copy ``infos[key]`` into ``target`` for the masked rows (per-agent values)."""
    value = infos.get(key)
    if value is None or not rows.any():
        return
    mask = infos.get(f"_{key}")
    use = rows if mask is None else rows & np.asarray(mask, dtype=bool)
    if not use.any():
        return
    array = np.stack(
        [np.asarray(value[i], dtype=np.float64).reshape(-1) for i in np.flatnonzero(use)]
    )
    if array.shape[1] == target.shape[1]:
        target[use] = array
