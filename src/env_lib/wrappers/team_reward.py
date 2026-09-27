"""Scalar team reward and termination for joint multi-agent environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

__all__ = ["TeamReward"]

_REDUCTIONS = ("sum", "mean")
_SOURCES = ("reward", "agent_rewards")


class TeamReward(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Return a scalar team reward and scalar ``terminated``/``truncated`` flags.

    Most ``env_lib`` environments already return a scalar team reward; some
    (AJLATT) return per-agent reward and termination arrays, which
    single-agent libraries and Gymnasium's vector environments cannot handle.
    This wrapper reduces them:

    * reward: with ``source="reward"`` (default) an array reward is reduced
      with ``reduce`` and a scalar reward is passed through unchanged (it
      already is the environment's team reward, which may include bonuses).
      With ``source="agent_rewards"`` the team reward is always
      ``reduce(info["agent_rewards"])``.
    * ``terminated`` is ``True`` when any agent terminated, ``truncated`` when
      any agent was truncated.
    * The per-agent values stay available: ``info["agent_rewards"]`` (added
      from the array reward if the environment does not report it) and, for
      array flags, ``info["agent_terminated"]``.

    It generalises :class:`env_lib.ajlatt_env.TeamRewardWrapper` (which is
    ``TeamReward(env, reduce="sum")``).

    Parameters
    ----------
    env:
        Environment to wrap.
    reduce:
        ``"sum"`` or ``"mean"`` over agents.
    source:
        ``"reward"`` or ``"agent_rewards"`` (see above).

    Examples
    --------
    >>> import env_lib
    >>> env = TeamReward(env_lib.make("AJLATT-v0"), reduce="mean")
    >>> obs, info = env.reset(seed=0)
    >>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    >>> type(reward), type(terminated), info["agent_rewards"].shape
    (<class 'float'>, <class 'bool'>, (4,))
    """

    def __init__(self, env: gym.Env, reduce: str = "sum", source: str = "reward") -> None:
        gym.utils.RecordConstructorArgs.__init__(self, reduce=reduce, source=source)
        gym.Wrapper.__init__(self, env)
        if reduce not in _REDUCTIONS:
            raise ValueError(f"reduce must be one of {_REDUCTIONS}, got {reduce!r}")
        if source not in _SOURCES:
            raise ValueError(f"source must be one of {_SOURCES}, got {source!r}")
        self.reduce = reduce
        self.source = source

    def _reduce(self, values: np.ndarray) -> float:
        return float(values.sum() if self.reduce == "sum" else values.mean())

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        if hasattr(reward, "detach") and hasattr(reward, "cpu"):
            reward = reward.detach().cpu().numpy()
        rewards = np.asarray(reward, dtype=np.float64)
        if rewards.ndim > 0 and "agent_rewards" not in info:
            info["agent_rewards"] = rewards.reshape(-1).copy()
        if self.source == "agent_rewards":
            if "agent_rewards" not in info:
                raise KeyError(
                    "TeamReward(source='agent_rewards') needs info['agent_rewards'], which this "
                    "environment does not report"
                )
            team = self._reduce(np.asarray(info["agent_rewards"], dtype=np.float64))
        elif rewards.ndim > 0:
            team = self._reduce(rewards)
        else:
            team = float(rewards)
        flags = np.asarray(terminated, dtype=bool)
        if flags.ndim > 0:
            info["agent_terminated"] = flags.reshape(-1).copy()
        return observation, team, bool(flags.any()), bool(np.any(truncated)), info
