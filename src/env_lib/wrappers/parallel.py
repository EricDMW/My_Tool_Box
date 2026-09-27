"""PettingZoo Parallel API adapter for joint multi-agent environments.

``env_lib`` environments expose the whole team through one Gymnasium
interface: stacked observations ``(n_agents, obs_dim)``, a joint action and a
team reward with the per-agent rewards in ``info["agent_rewards"]``. Many
multi-agent libraries instead expect the PettingZoo *Parallel* API, where every
quantity is a dictionary keyed by agent name. :class:`ParallelEnvAdapter`
converts between the two.

PettingZoo is not a dependency of ``env_lib``: when it is installed the
adapter subclasses :class:`pettingzoo.utils.env.ParallelEnv` (so
``isinstance`` checks and PettingZoo's utilities work), otherwise it is a plain
class with the same protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

__all__ = ["PETTINGZOO_AVAILABLE", "ParallelEnvAdapter", "to_parallel"]

try:  # pragma: no cover - depends on the installation
    from pettingzoo.utils.env import ParallelEnv as _ParallelBase

    PETTINGZOO_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the installation
    PETTINGZOO_AVAILABLE = False

    class _ParallelBase:  # type: ignore[no-redef]
        """Minimal stand-in for :class:`pettingzoo.utils.env.ParallelEnv`."""

        metadata: dict[str, Any] = {}
        possible_agents: list[str]
        agents: list[str]

        @property
        def num_agents(self) -> int:
            return len(self.agents)

        @property
        def max_num_agents(self) -> int:
            return len(self.possible_agents)

        @property
        def unwrapped(self) -> Any:
            return self


def _num_agents(observation_space: spaces.Space) -> int:
    """Number of agents of a joint observation space (leading axis of a 2-D Box)."""
    shape = observation_space.shape
    if isinstance(observation_space, spaces.Box) and shape is not None and len(shape) >= 2:
        return int(shape[0])
    return 1


class _ActionCodec:
    """Split a joint action space per agent and join per-agent actions again."""

    def __init__(self, space: spaces.Space, n_agents: int) -> None:
        self.space = space
        self.n = n_agents
        self.kind = self._kind(space, n_agents)
        self.agent_spaces = [self._agent_space(i) for i in range(n_agents)]

    @staticmethod
    def _kind(space: spaces.Space, n: int) -> str:
        if n == 1:
            return "single"
        shape = space.shape
        if isinstance(space, spaces.Discrete):
            if int(space.n) == 2**n and int(space.start) == 0:
                return "bits"
        elif (
            shape
            and shape[0] == n
            and isinstance(space, (spaces.Box, spaces.MultiDiscrete, spaces.MultiBinary))
        ):
            return "rows" if len(shape) >= 2 else "entries"
        raise TypeError(
            f"cannot split the joint action space {space} among {n} agents; expected a leading "
            f"agent axis of size {n} (Box, MultiDiscrete, MultiBinary) or Discrete(2 ** {n})"
        )

    def _agent_space(self, i: int) -> spaces.Space:
        space, kind = self.space, self.kind
        if kind == "single":
            return space
        if kind == "bits":
            return spaces.Discrete(2)
        if isinstance(space, spaces.Box):
            if kind == "rows":
                return spaces.Box(space.low[i], space.high[i], dtype=space.dtype)
            return spaces.Box(space.low[i : i + 1], space.high[i : i + 1], dtype=space.dtype)
        if isinstance(space, spaces.MultiDiscrete):
            if kind == "rows":
                return spaces.MultiDiscrete(space.nvec[i], dtype=space.dtype, start=space.start[i])
            return spaces.Discrete(int(space.nvec[i]), start=int(space.start[i]))
        # MultiBinary
        if kind == "rows":
            return spaces.MultiBinary(list(space.shape[1:]))
        return spaces.Discrete(2)

    def join(self, actions: Sequence[Any]) -> Any:
        kind, space = self.kind, self.space
        if kind == "single":
            return actions[0]
        if kind == "bits":
            return sum(int(np.asarray(a).reshape(-1)[0]) << i for i, a in enumerate(actions))
        if isinstance(space, spaces.Box):
            rows = [np.asarray(a, dtype=space.dtype) for a in actions]
            if kind == "rows":
                return np.stack([r.reshape(space.shape[1:]) for r in rows])
            return np.concatenate([r.reshape(1) for r in rows])
        rows = [np.asarray(a) for a in actions]
        if kind == "rows":
            return np.stack([r.reshape(space.shape[1:]) for r in rows]).astype(space.dtype)
        return np.array([int(r.reshape(-1)[0]) for r in rows], dtype=space.dtype)

    def split(self, joint: Any) -> list[Any]:
        kind = self.kind
        if kind == "single":
            return [joint]
        if kind == "bits":
            value = int(np.asarray(joint).reshape(-1)[0])
            return [np.int64((value >> i) & 1) for i in range(self.n)]
        array = np.asarray(joint)
        if kind == "rows":
            return [array[i] for i in range(self.n)]
        if isinstance(self.space, spaces.Box):
            return [array[i : i + 1] for i in range(self.n)]
        return [array.dtype.type(array[i]) for i in range(self.n)]


class ParallelEnvAdapter(_ParallelBase):
    """Expose a joint multi-agent Gymnasium environment through the PettingZoo Parallel API.

    Agents are named ``"agent_0"``, ``"agent_1"``, ... in the order of the
    leading axis of the joint observation. Environments with a flat
    observation (Kuramoto) have a single agent ``"agent_0"``.

    * ``observation_space(agent)`` / ``action_space(agent)``: slices of the
      joint spaces (a ``Box (n, d)`` gives ``Box (d,)``, a ``Box (n,)`` gives
      ``Box (1,)``, ``MultiDiscrete``/``MultiBinary`` give ``Discrete``, and the
      joint ``Discrete(2 ** n)`` of LineMsg gives ``Discrete(2)`` per agent).
    * ``reset(seed, options)`` returns ``(observations, infos)`` dictionaries;
      ``options`` are forwarded to the environment unchanged.
    * ``step(actions)`` takes one action per live agent and returns
      ``(observations, rewards, terminations, truncations, infos)``. Rewards
      come from ``info["agent_rewards"]`` (or a per-agent reward array; a scalar
      team reward is given to every agent). Termination and truncation flags
      are per agent when the environment reports arrays (AJLATT) and shared
      otherwise. When the joint episode ends, agents that did not terminate are
      marked truncated and :attr:`agents` becomes empty.
    * Array entries of the info dictionary whose leading dimension is
      ``n_agents`` are sliced per agent; all other entries are shared.
    * ``state()`` returns the joint observation (``state_space`` is the joint
      observation space); ``render()`` and ``close()`` are forwarded.

    Parameters
    ----------
    env:
        A Gymnasium environment with joint spaces (for example from
        :func:`env_lib.make`).
    agent_prefix:
        Prefix of the agent names.

    Raises
    ------
    TypeError
        For a vector environment, or if the joint action space cannot be split
        among the agents.

    Examples
    --------
    >>> import env_lib
    >>> from env_lib.wrappers import to_parallel
    >>> env = to_parallel("Consensus-v0")
    >>> observations, infos = env.reset(seed=0)
    >>> actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    >>> observations, rewards, terminations, truncations, infos = env.step(actions)
    """

    def __init__(self, env: gym.Env, *, agent_prefix: str = "agent") -> None:
        if isinstance(env, gym.vector.VectorEnv):
            raise TypeError(
                "ParallelEnvAdapter adapts a single environment; create it with env_lib.make() "
                "(PettingZoo libraries vectorise parallel environments themselves)"
            )
        self.env = env
        joint_obs = env.observation_space
        self.n_agents = _num_agents(joint_obs)
        n = self.n_agents
        self.possible_agents = [f"{agent_prefix}_{i}" for i in range(n)]
        self.agents = list(self.possible_agents)
        self._index = {agent: i for i, agent in enumerate(self.possible_agents)}
        self._codec = _ActionCodec(env.action_space, n)
        if n == 1:
            agent_obs = [joint_obs]
        else:
            agent_obs = [
                spaces.Box(joint_obs.low[i], joint_obs.high[i], dtype=joint_obs.dtype)
                for i in range(n)
            ]
        self.observation_spaces = dict(zip(self.possible_agents, agent_obs))
        self.action_spaces = dict(zip(self.possible_agents, self._codec.agent_spaces))
        self.state_space = joint_obs
        spec = getattr(env, "spec", None)
        name = getattr(spec, "id", None) or type(env.unwrapped).__name__
        self.metadata = {
            "render_modes": list(env.metadata.get("render_modes", [])),
            "render_fps": env.metadata.get("render_fps"),
            "name": name,
            "is_parallelizable": True,
        }
        self.render_mode = getattr(env, "render_mode", None)
        self._last_observation: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------
    def observation_space(self, agent: str) -> spaces.Space:
        """Observation space of ``agent`` (always the same object)."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """Action space of ``agent`` (always the same object)."""
        return self.action_spaces[agent]

    # ------------------------------------------------------------------
    # Conversion helpers (also used by env_lib.baseline_policy)
    # ------------------------------------------------------------------
    def stack_observations(self, observations: Mapping[str, Any]) -> np.ndarray:
        """Joint observation from a dictionary holding every agent's observation."""
        if self.n_agents == 1:
            return np.asarray(observations[self.possible_agents[0]])
        return np.stack([np.asarray(observations[agent]) for agent in self.possible_agents])

    def split_actions(self, joint_action: Any, agents: Sequence[str] | None = None) -> dict:
        """Per-agent action dictionary from a joint action."""
        parts = self._codec.split(joint_action)
        names = self.possible_agents if agents is None else list(agents)
        return {agent: parts[self._index[agent]] for agent in names}

    def _split_observation(self, observation: Any) -> dict[str, Any]:
        observation = np.asarray(observation)
        self._last_observation = observation
        if self.n_agents == 1:
            return {self.possible_agents[0]: observation}
        return {agent: observation[i] for i, agent in enumerate(self.possible_agents)}

    def _split_info(self, info: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        n = self.n_agents
        if n == 1:
            return {self.possible_agents[0]: dict(info)}
        out: dict[str, dict[str, Any]] = {agent: {} for agent in self.possible_agents}
        for key, value in info.items():
            per_agent = isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n
            for i, agent in enumerate(self.possible_agents):
                out[agent][key] = value[i] if per_agent else value
        return out

    def _per_agent(self, value: Any, dtype: Any) -> np.ndarray:
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=dtype).reshape(-1)
        if array.size == 1:
            return np.full(self.n_agents, array[0], dtype=dtype)
        if array.size != self.n_agents:
            raise ValueError(f"expected {self.n_agents} per-agent values, got shape {array.shape}")
        return array

    # ------------------------------------------------------------------
    # Parallel API
    # ------------------------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Reset the environment; returns ``(observations, infos)`` dictionaries."""
        observation, info = self.env.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        return self._split_observation(observation), self._split_info(info)

    def step(self, actions: Mapping[str, Any]) -> tuple[dict, dict, dict, dict, dict]:
        """Step all agents; returns per-agent dictionaries.

        Raises
        ------
        RuntimeError
            If the episode has ended (call :meth:`reset`).
        ValueError
            If an action is missing for a live agent.
        """
        if not self.agents:
            raise RuntimeError("the episode has ended; call reset() before step()")
        missing = [agent for agent in self.agents if agent not in actions]
        if missing:
            raise ValueError(f"missing actions for agents {missing}")
        joint = self._codec.join([actions[agent] for agent in self.possible_agents])
        observation, reward, terminated, truncated, info = self.env.step(joint)

        if isinstance(info, Mapping) and info.get("agent_rewards") is not None:
            rewards = self._per_agent(info["agent_rewards"], np.float64)
        else:
            rewards = self._per_agent(reward, np.float64)
        terminations = self._per_agent(terminated, bool)
        truncations = self._per_agent(truncated, bool)
        if terminations.any() or truncations.any():
            # The joint episode is over: agents that did not terminate are cut off.
            truncations = truncations | ~terminations

        agents = self.possible_agents
        result = (
            self._split_observation(observation),
            {agent: float(rewards[i]) for i, agent in enumerate(agents)},
            {agent: bool(terminations[i]) for i, agent in enumerate(agents)},
            {agent: bool(truncations[i]) for i, agent in enumerate(agents)},
            self._split_info(info),
        )
        if terminations.any() or truncations.any():
            self.agents = []
        return result

    def state(self) -> np.ndarray:
        """Global state: the current joint observation."""
        if self._last_observation is None:
            raise RuntimeError("call reset() before state()")
        return self._last_observation.copy()

    def render(self) -> Any:
        """Render the wrapped environment."""
        return self.env.render()

    def close(self) -> None:
        """Close the wrapped environment."""
        self.env.close()

    def __repr__(self) -> str:
        return f"ParallelEnvAdapter({self.metadata['name']}, {self.n_agents} agents)"


def to_parallel(env_or_id: gym.Env | str, **kwargs: Any) -> ParallelEnvAdapter:
    """Adapt an environment (or create one by id) to the PettingZoo Parallel API.

    Parameters
    ----------
    env_or_id:
        A Gymnasium environment or a registered ``env_lib`` id.
    **kwargs:
        Constructor arguments when an id is given (for example
        ``render_mode="rgb_array"`` or ``n_agents=16``).

    Returns
    -------
    ParallelEnvAdapter

    Raises
    ------
    TypeError
        If keyword arguments are given together with an environment instance.
    """
    if isinstance(env_or_id, str):
        from env_lib.registration import make

        return ParallelEnvAdapter(make(env_or_id, **kwargs))
    if kwargs:
        raise TypeError(
            "keyword arguments are only accepted together with an environment id, got "
            f"{sorted(kwargs)}"
        )
    return ParallelEnvAdapter(env_or_id)
