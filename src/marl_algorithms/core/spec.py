"""Per-agent view of an ``env_lib`` environment.

The ``env_lib`` environments expose joint spaces: observations stacked per agent
as ``(n_agents, obs_dim)`` and a joint action (``Box(n_agents,)``,
``Box(n_agents, d)``, ``MultiDiscrete``, ``MultiBinary``). The algorithms in
this package work on per-agent arrays instead. :class:`MultiAgentSpec` reads the
agent structure from the spaces once and converts in both directions:

* :meth:`MultiAgentSpec.agent_obs` maps an environment observation with any
  leading batch shape to ``(*batch, n_agents, obs_dim)``;
* :meth:`MultiAgentSpec.env_action` maps per-agent actions
  (``(*batch, n_agents, action_dim)`` floats or ``(*batch, n_agents)`` integers)
  back to the joint action the environment expects.

Environments whose spaces are not stacked per agent (the Kuramoto environments
observe and act on the whole network) are treated as a single agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

__all__ = ["MultiAgentSpec"]


@dataclass(frozen=True)
class MultiAgentSpec:
    """Agent structure of a multi-agent environment.

    Attributes
    ----------
    n_agents:
        Number of agents ``n``.
    obs_dim:
        Size of one agent's (flattened) observation.
    action_kind:
        ``"continuous"`` or ``"discrete"``.
    action_dim:
        Continuous: size of one agent's action. Discrete: ``1``.
    n_actions:
        Discrete: number of choices of every agent. Continuous: ``None``.
    action_low, action_high:
        Continuous: per-agent bounds of shape ``(n_agents, action_dim)``.
    obs_shape, action_shape:
        Shapes of one environment's observation and joint action.
    action_dtype:
        dtype of the joint action.
    """

    n_agents: int
    obs_dim: int
    action_kind: str
    action_dim: int
    n_actions: int | None
    action_low: np.ndarray | None
    action_high: np.ndarray | None
    obs_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    action_dtype: Any

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env: Any) -> MultiAgentSpec:
        """Read the agent structure of a single or vector environment."""
        obs_space = getattr(env, "single_observation_space", None) or env.observation_space
        act_space = getattr(env, "single_action_space", None) or env.action_space
        return cls.from_spaces(obs_space, act_space)

    @classmethod
    def from_spaces(cls, obs_space: spaces.Space, act_space: spaces.Space) -> MultiAgentSpec:
        """Build the spec from one environment's observation and action spaces.

        Raises
        ------
        TypeError
            For unsupported spaces (for example a joint ``Discrete`` action over
            several agents, or ``MultiDiscrete`` with unequal choice counts).
        """
        if not isinstance(obs_space, spaces.Box):
            raise TypeError(f"observation space must be a Box, got {obs_space}")
        obs_shape = tuple(int(s) for s in obs_space.shape)
        act_shape = tuple(int(s) for s in (act_space.shape or ()))

        if isinstance(act_space, spaces.Box):
            if len(act_shape) == 2 and len(obs_shape) == 2 and act_shape[0] == obs_shape[0]:
                n, action_dim = act_shape
            elif len(act_shape) == 1 and len(obs_shape) == 2 and act_shape[0] == obs_shape[0]:
                n, action_dim = act_shape[0], 1
            else:  # whole-system action: one agent
                n, action_dim = 1, int(np.prod(act_shape))
            low = np.broadcast_to(act_space.low, act_shape).reshape(n, action_dim)
            high = np.broadcast_to(act_space.high, act_shape).reshape(n, action_dim)
            if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
                raise TypeError("continuous actions must have finite bounds")
            kind, n_actions = "continuous", None
            low, high = low.astype(np.float32), high.astype(np.float32)
        elif isinstance(act_space, (spaces.MultiDiscrete, spaces.MultiBinary)):
            if isinstance(act_space, spaces.MultiBinary):
                nvec = np.full(act_shape, 2)
            else:
                nvec = np.asarray(act_space.nvec)
                if np.any(np.asarray(act_space.start) != 0):
                    raise TypeError("MultiDiscrete spaces must start at 0")
            if nvec.ndim != 1 or np.unique(nvec).size != 1:
                raise TypeError(f"every agent must have the same number of actions, got {nvec}")
            n, action_dim, n_actions = int(nvec.size), 1, int(nvec[0])
            kind, low, high = "discrete", None, None
        elif isinstance(act_space, spaces.Discrete):
            if len(obs_shape) == 2 and obs_shape[0] > 1:
                raise TypeError(
                    "a joint Discrete action over several agents is not supported; "
                    "create the environment with per-agent actions (for LineMsg: "
                    'action_space_type="multibinary")'
                )
            if int(act_space.start) != 0:
                raise TypeError("Discrete spaces must start at 0")
            n, action_dim, n_actions = 1, 1, int(act_space.n)
            kind, low, high = "discrete", None, None
        else:
            raise TypeError(f"unsupported action space {act_space}")

        if n > 1:
            if len(obs_shape) != 2 or obs_shape[0] != n:
                raise TypeError(
                    f"observations must be stacked per agent as ({n}, d), got {obs_shape}"
                )
            obs_dim = obs_shape[1]
        else:
            obs_dim = int(np.prod(obs_shape))
        return cls(
            n_agents=int(n),
            obs_dim=int(obs_dim),
            action_kind=kind,
            action_dim=int(action_dim),
            n_actions=n_actions,
            action_low=low,
            action_high=high,
            obs_shape=obs_shape,
            action_shape=act_shape,
            action_dtype=act_space.dtype,
        )

    # ------------------------------------------------------------------
    @property
    def continuous(self) -> bool:
        """Whether actions are continuous."""
        return self.action_kind == "continuous"

    @property
    def state_dim(self) -> int:
        """Size of the global state used by centralised critics (all observations)."""
        return self.n_agents * self.obs_dim

    def agent_obs(self, obs: Any) -> np.ndarray:
        """Environment observation(s) as ``(*batch, n_agents, obs_dim)`` float32."""
        array = np.asarray(obs, dtype=np.float32)
        batch = array.shape[: array.ndim - len(self.obs_shape)]
        if array.shape[len(batch) :] != self.obs_shape:
            raise ValueError(
                f"expected observations of shape (..., {self.obs_shape}), got {array.shape}"
            )
        return array.reshape(*batch, self.n_agents, self.obs_dim)

    def env_action(self, actions: Any) -> np.ndarray:
        """Per-agent actions as the environment's joint action (any batch shape).

        Continuous actions have shape ``(*batch, n_agents, action_dim)`` (or
        ``(*batch, n_agents)`` when ``action_dim == 1``) and are clipped to the
        bounds; discrete actions have shape ``(*batch, n_agents)``.
        """
        array = np.asarray(actions)
        if self.continuous:
            if self.action_dim == 1 and array.shape[-1:] == (self.n_agents,):
                array = array[..., None]
            batch = array.shape[:-2]
            array = np.clip(array, self.action_low, self.action_high)
            return array.reshape(*batch, *self.action_shape).astype(self.action_dtype, copy=False)
        batch = array.shape[:-1]
        if self.action_shape == ():
            return array.reshape(batch).astype(self.action_dtype, copy=False)
        return array.reshape(*batch, *self.action_shape).astype(self.action_dtype, copy=False)

    def describe(self) -> str:
        """One-line summary, for logs."""
        if self.continuous:
            action = f"continuous {self.action_dim}-d"
        else:
            action = f"{self.n_actions} discrete choices"
        agents = "1 agent" if self.n_agents == 1 else f"{self.n_agents} agents"
        return f"{agents}, {self.obs_dim}-d observations, {action} per agent"
