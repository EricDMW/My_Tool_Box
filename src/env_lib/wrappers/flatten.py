"""Flatten joint multi-agent spaces for single-agent RL libraries."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

__all__ = ["FlattenJointSpaces"]


def _flat_observation_space(space: spaces.Space) -> spaces.Box:
    if isinstance(space, spaces.Box):
        low = np.asarray(space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(space.high, dtype=np.float32).reshape(-1)
        return spaces.Box(low, high, dtype=np.float32)
    if isinstance(space, spaces.MultiBinary):
        size = int(np.prod(space.shape))
        return spaces.Box(0.0, 1.0, (size,), dtype=np.float32)
    if isinstance(space, spaces.MultiDiscrete):
        start = np.asarray(space.start, dtype=np.float32).reshape(-1)
        high = start + np.asarray(space.nvec, dtype=np.float32).reshape(-1) - 1.0
        return spaces.Box(start, high, dtype=np.float32)
    raise TypeError(
        f"FlattenJointSpaces supports Box, MultiBinary and MultiDiscrete observation spaces, "
        f"got {space}"
    )


def _flat_action_space(space: spaces.Space) -> spaces.Space:
    if isinstance(space, spaces.Box):
        return spaces.Box(space.low.reshape(-1), space.high.reshape(-1), dtype=space.dtype)
    if isinstance(space, spaces.MultiBinary):
        return spaces.MultiBinary(int(np.prod(space.shape)))
    if isinstance(space, spaces.MultiDiscrete):
        return spaces.MultiDiscrete(
            space.nvec.reshape(-1), dtype=space.dtype, start=space.start.reshape(-1)
        )
    if isinstance(space, spaces.Discrete):
        return space  # already a single flat action
    raise TypeError(
        f"FlattenJointSpaces supports Box, MultiBinary, MultiDiscrete and Discrete action "
        f"spaces, got {space}"
    )


class FlattenJointSpaces(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Present a joint multi-agent environment as a flat single-agent one.

    The joint observation (for example ``(n_agents, obs_dim)``) becomes a 1-D
    ``float32`` vector, and the wrapper accepts a flat action that it reshapes
    to the joint action shape. This is the form expected by single-agent
    libraries such as Stable-Baselines3 or CleanRL, which then learn one
    centralised policy for the whole team.

    Supported spaces: ``Box``, ``MultiBinary`` and ``MultiDiscrete``
    observations (``MultiBinary``/``MultiDiscrete`` values become ``float32``
    entries with the matching bounds) and ``Box``, ``MultiBinary``,
    ``MultiDiscrete`` and ``Discrete`` actions (``Discrete`` is passed through
    unchanged). Rewards and flags are passed through: combine with
    :class:`~env_lib.wrappers.TeamReward` for environments that return
    per-agent reward arrays (AJLATT).

    Parameters
    ----------
    env:
        The environment to wrap.

    Raises
    ------
    TypeError
        For other space types (``Dict``, ``Tuple``, ...).

    Examples
    --------
    >>> import env_lib
    >>> env = FlattenJointSpaces(env_lib.make("Consensus-v0"))
    >>> env.observation_space.shape, env.action_space.shape
    ((128,), (16,))
    >>> obs, info = env.reset(seed=0)
    >>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """

    def __init__(self, env: gym.Env) -> None:
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        self._joint_observation_space = env.observation_space
        self._joint_action_space = env.action_space
        self.observation_space = _flat_observation_space(env.observation_space)
        self.action_space = _flat_action_space(env.action_space)

    def observation(self, observation: Any) -> np.ndarray:
        """Flatten a joint observation to a 1-D ``float32`` array."""
        if hasattr(observation, "detach") and hasattr(observation, "cpu"):
            observation = observation.detach().cpu().numpy()
        return np.asarray(observation, dtype=np.float32).reshape(-1)

    def action(self, action: Any) -> Any:
        """Reshape a flat action to the joint action shape."""
        space = self._joint_action_space
        if isinstance(space, spaces.Discrete):
            return action
        dtype = space.dtype if isinstance(space, spaces.Box) else None
        array = np.asarray(action, dtype=dtype)
        if array.size != int(np.prod(space.shape)):
            raise ValueError(
                f"flat action must have {int(np.prod(space.shape))} entries, got shape "
                f"{array.shape}"
            )
        return array.reshape(space.shape)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        return self.observation(observation), info

    def step(self, action: Any) -> tuple[np.ndarray, Any, Any, Any, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(self.action(action))
        return self.observation(observation), reward, terminated, truncated, info
