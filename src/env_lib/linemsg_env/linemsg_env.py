"""Line message-passing environment (``LineMsg-v0``).

``num_agents`` agents sit on a line and relay a message from the right end
(agent ``num_agents - 1``, the source) towards the left end (agent ``0``, the
sink). Every agent holds a binary state, ``1`` when it holds the message and
``0`` otherwise. At every step each agent chooses a binary action and the
states evolve as follows (``s_i`` state, ``a_i`` action, all updated
synchronously from the previous states)::

    s_0     <- s_1                                  (the sink copies agent 1)
    s_i     <- a_i and (s_{i+1} or u_i < 0.8)       (0 < i < num_agents - 1)
    s_{N-1} <- a_{N-1}                              (the source acts on its own)

where ``u_i ~ U(0, 1)`` are drawn once per step for the ``num_agents - 2``
middle agents. Action ``1`` of a middle agent therefore activates its link to
the right neighbour: the agent receives the neighbour's message for sure, and
otherwise obtains one with probability 0.8. Action ``0`` clears the state.

The per-agent reward is ``0.1`` for holding the message, ``1.0`` for the sink;
the team reward is their sum.

The joint action is either one integer in ``Discrete(2 ** num_agents)`` whose
bit ``i`` is the action of agent ``i`` (default, backwards compatible), or an
array of ``num_agents`` binary actions (always accepted, and the declared
action space when ``action_space_type="multibinary"``).
"""

from __future__ import annotations

import warnings
from numbers import Integral
from typing import Any

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete, MultiBinary
from gymnasium.utils import EzPickle, seeding
from numpy.lib.stride_tricks import sliding_window_view

from env_lib.errors import ResetNeededError
from env_lib.utils.rendering import validate_render_mode

__all__ = ["LineMsgEnv"]

#: Largest line length for which ``Discrete(2 ** num_agents)`` fits in int64.
MAX_DISCRETE_AGENTS = 62

#: Probability that an active middle agent obtains a message from an empty neighbour.
RELAY_PROBABILITY = 0.8

#: State value of the boundary padding cells in :attr:`LineMsgEnv.state`.
BOUNDARY = 2

_ACTION_SPACE_TYPES = ("discrete", "multibinary")
_HISTORY_CAPACITY = 512


def _check_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


class LineMsgEnv(Env, EzPickle):
    """Cooperative message relay along a line of agents.

    Parameters
    ----------
    num_agents : int, default 10
        Number of agents on the line (at least 2: a sink and a source; the
        agents in between relay).
    n_obs_neighbors : int, default 1
        Each agent observes the states of the ``n_obs_neighbors`` agents on
        either side of it. ``0`` is treated as ``1`` (the dynamics need one
        boundary cell on each side), as in earlier versions.
    max_iter : int, default 50
        Episode length; ``truncated`` becomes ``True`` after ``max_iter`` steps.
    render_mode : {"human", "rgb_array", "ansi"} or None, default None
        ``"rgb_array"`` returns an RGB frame of the dashboard, ``"human"``
        shows it in a matplotlib window and ``"ansi"`` returns a text summary.
    action_space_type : {"discrete", "multibinary"}, default "discrete"
        Declared action space: ``Discrete(2 ** num_agents)`` (only possible for
        ``num_agents <= 62``) or ``MultiBinary(num_agents)``. Both action
        formats are accepted by :meth:`step` in either case.

    Attributes
    ----------
    state : numpy.ndarray of int, shape (num_agents + 2 * n_obs_nghbr,)
        Agent states padded with ``n_obs_nghbr`` boundary cells (value 2) on
        each side; agent ``i`` is ``state[n_obs_nghbr + i]``.
    actions : numpy.ndarray of int, shape (num_agents + 2 * n_obs_nghbr,)
        Last joint action with the same padding (``2`` before the first step).

    Notes
    -----
    Observation: ``Box(0, 3, (num_agents, 2 * n_obs_nghbr + 1), float32)``; row
    ``i`` is the window ``state[i : i + 2 * n_obs_nghbr + 1]`` centred on agent
    ``i`` (values 0 = no message, 1 = message, 2 = boundary).

    ``info["agent_rewards"]`` holds the per-agent rewards (sum = team reward).

    Examples
    --------
    >>> from env_lib.linemsg_env import LineMsgEnv
    >>> env = LineMsgEnv(num_agents=5)
    >>> obs, info = env.reset(seed=0)
    >>> obs.shape
    (5, 3)
    >>> obs, reward, terminated, truncated, info = env.step([0, 1, 1, 1, 1])
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "name": "LineMsg-v0",
        "render_fps": 10,
    }

    def __init__(
        self,
        num_agents: int = 10,
        n_obs_neighbors: int = 1,
        max_iter: int = 50,
        render_mode: str | None = None,
        action_space_type: str = "discrete",
    ):
        EzPickle.__init__(
            self,
            num_agents=num_agents,
            n_obs_neighbors=n_obs_neighbors,
            max_iter=max_iter,
            render_mode=render_mode,
            action_space_type=action_space_type,
        )
        self.num_agents = _check_int("num_agents", num_agents, 2)
        n_obs_neighbors = _check_int("n_obs_neighbors", n_obs_neighbors, 0)
        if n_obs_neighbors == 0:
            warnings.warn(
                "LineMsgEnv: n_obs_neighbors=0 is treated as 1 (minimum neighbourhood size).",
                UserWarning,
                stacklevel=2,
            )
        self.n_obs_nghbr = max(1, n_obs_neighbors)
        self.max_iter = _check_int("max_iter", max_iter, 1)
        validate_render_mode(render_mode, self.metadata["render_modes"])
        self.render_mode = render_mode
        if action_space_type not in _ACTION_SPACE_TYPES:
            raise ValueError(
                f"action_space_type must be one of {_ACTION_SPACE_TYPES}, got {action_space_type!r}"
            )
        if action_space_type == "discrete" and self.num_agents > MAX_DISCRETE_AGENTS:
            raise ValueError(
                f"action_space_type='discrete' encodes the joint action as one integer in "
                f"Discrete(2 ** num_agents), which only fits for num_agents <= "
                f"{MAX_DISCRETE_AGENTS} (got {self.num_agents}); use "
                f"action_space_type='multibinary'."
            )
        self.action_space_type = action_space_type

        self.agents: list[str] = [f"agent_{i}" for i in range(self.num_agents)]
        self.agent_name_mapping: dict[str, int] = {name: i for i, name in enumerate(self.agents)}

        if action_space_type == "discrete":
            self.action_space = Discrete(2**self.num_agents)
        else:
            self.action_space = MultiBinary(self.num_agents)
        self.window = 2 * self.n_obs_nghbr + 1
        self.observation_space = Box(
            low=0, high=3, shape=(self.num_agents, self.window), dtype=np.float32
        )

        # Precomputed constants for the hot path.
        n, big_n = self.n_obs_nghbr, self.num_agents
        self._agents = slice(n, n + big_n)
        self._middle = slice(n + 1, n + big_n - 1)
        self._right_of_middle = slice(n + 2, n + big_n)
        self._bit_shifts = np.arange(big_n, dtype=np.int64)
        self._int_actions = big_n <= MAX_DISCRETE_AGENTS
        self._reward_weights = np.full(big_n, 0.1)
        self._reward_weights[0] = 1.0

        self.state: np.ndarray | None = None
        self.actions: np.ndarray | None = None
        self.num_moves = 0
        self.closed = False

        self._window_view: np.ndarray | None = None
        self._window_base: np.ndarray | None = None
        self._last_reward = 0.0
        self._return = 0.0
        self._renderer = None
        self._warned_no_render = False
        self._history: dict[str, np.ndarray] | None = None
        self._history_len = 0

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    def seed(self, seed: int | None = None) -> list[int]:
        """Reseed the environment's random generator.

        .. deprecated::
            Use ``reset(seed=...)`` instead.
        """
        warnings.warn(
            "LineMsgEnv.seed() is deprecated; use reset(seed=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._np_random, seed = seeding.np_random(seed)
        self._np_random_seed = seed
        return [seed]

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset all agents to hold the message.

        Parameters
        ----------
        seed : int, optional
            Seed for the environment's random generator.
        options : dict, optional
            Unused.

        Returns
        -------
        observation : numpy.ndarray
            ``(num_agents, 2 * n_obs_nghbr + 1)`` float32 array.
        info : dict
            ``{"agent_rewards": zeros(num_agents)}``.
        """
        super().reset(seed=seed)
        n = self.n_obs_nghbr
        size = self.num_agents + 2 * n
        self.state = np.ones(size, dtype=int)
        self.state[:n] = BOUNDARY
        self.state[-n:] = BOUNDARY
        self.actions = np.full(size, 2, dtype=int)
        self.num_moves = 0
        self._last_reward = 0.0
        self._return = 0.0

        if self.render_mode in ("human", "rgb_array"):
            self._reset_history()
            if self._renderer is not None:
                self._renderer.reset()
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, {"agent_rewards": np.zeros(self.num_agents)}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply a joint action.

        Parameters
        ----------
        action : int or array-like
            Integer in ``[0, 2 ** num_agents)`` (bit ``i`` = action of agent
            ``i``) or an array of ``num_agents`` values in ``{0, 1}``.

        Returns
        -------
        observation, reward, terminated, truncated, info
            ``reward`` is the team reward (a float), ``terminated`` is always
            ``False`` and ``info["agent_rewards"]`` the per-agent rewards.

        Raises
        ------
        RuntimeError
            If :meth:`reset` has not been called.
        ValueError
            If ``action`` is out of range or has the wrong shape.
        """
        state, actions = self.state, self.actions
        if state is None or actions is None:
            raise ResetNeededError("LineMsgEnv.step() called before reset(); call reset() first.")
        bits = self._decode_action(action)
        actions[self._agents] = bits

        # Synchronous update from the previous states.
        s_right = state[self._right_of_middle]
        a_mid = bits[1:-1]
        new_mid = (s_right + a_mid) == 2
        relay = ((1 - s_right) + a_mid) == 2
        u = self.np_random.random(self.num_agents - 2)
        new_mid[relay & (u < RELAY_PROBABILITY)] = True
        n = self.n_obs_nghbr
        new_sink = state[n + 1]
        state[self._middle] = new_mid
        state[n] = new_sink
        state[n + self.num_agents - 1] = bits[-1]

        agent_rewards = (state[self._agents] == 1) * self._reward_weights
        reward = float(agent_rewards.sum())
        self.num_moves += 1
        self._last_reward = reward
        self._return += reward
        truncated = self.num_moves >= self.max_iter

        if self._history is not None:
            self._push_history(reward)
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, reward, False, truncated, {"agent_rewards": agent_rewards}

    def render(self) -> Any:
        """Render the current state.

        Returns
        -------
        numpy.ndarray or str or None
            ``(H, W, 3)`` uint8 frame for ``"rgb_array"``, the text summary for
            ``"ansi"`` and ``None`` for ``"human"`` (a window is updated) or
            when no render mode was set.

        Raises
        ------
        RuntimeError
            If :meth:`reset` has not been called.
        """
        if self.render_mode is None:
            if not self._warned_no_render:
                warnings.warn(
                    "LineMsgEnv.render() called without a render_mode; pass "
                    "render_mode='rgb_array', 'human' or 'ansi' to the constructor.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_no_render = True
            return None
        if self.state is None or self.actions is None:
            raise ResetNeededError("LineMsgEnv.render() called before reset(); call reset() first.")
        if self.render_mode == "ansi":
            return self._render_text()
        if self._history is None:
            self._reset_history()
        if self._renderer is None:
            from env_lib.linemsg_env.rendering import LineMsgRenderer

            self._renderer = LineMsgRenderer(
                self.render_mode,
                num_agents=self.num_agents,
                n_obs_neighbors=self.n_obs_nghbr,
                window=self._history["rewards"].shape[0],
                fps=self.metadata["render_fps"],
            )
        states, rewards, first_step = self._ordered_history()
        return self._renderer.render(
            state=self.state,
            actions=self.actions,
            state_history=states,
            reward_history=rewards,
            first_step=first_step,
            step=self.num_moves,
            max_iter=self.max_iter,
            reward=self._last_reward,
            episode_return=self._return,
        )

    def close(self) -> None:
        """Close the render window (idempotent)."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.closed = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _decode_action(self, action: Any) -> np.ndarray:
        """Return the ``(num_agents,)`` int64 vector of binary agent actions."""
        if isinstance(action, (int, np.integer)) and not isinstance(action, bool):
            return self._decode_int(int(action))
        arr = np.asarray(action)
        if arr.ndim == 0:
            if not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(
                    f"integer joint actions must have an integer dtype, got {arr.dtype} ({action!r})"
                )
            return self._decode_int(int(arr))
        if arr.shape != (self.num_agents,):
            raise ValueError(
                f"action must be an integer in [0, 2**{self.num_agents}) or an array of shape "
                f"({self.num_agents},) with entries in {{0, 1}}; got shape {arr.shape}"
            )
        bits = arr.astype(np.int64)
        if arr.dtype.kind == "b":
            return bits
        if arr.dtype.kind not in "iu" and not np.array_equal(bits, arr):
            raise ValueError(f"per-agent actions must be 0 or 1, got {arr!r}")
        if (bits >> 1).any():
            raise ValueError(f"per-agent actions must be 0 or 1, got {arr!r}")
        return bits

    def _decode_int(self, value: int) -> np.ndarray:
        if not self._int_actions:
            raise ValueError(
                f"integer joint actions are only supported for num_agents <= "
                f"{MAX_DISCRETE_AGENTS}; pass an array of {self.num_agents} binary actions"
            )
        if not 0 <= value < (1 << self.num_agents):
            raise ValueError(f"integer action must lie in [0, 2**{self.num_agents}), got {value}")
        return (value >> self._bit_shifts) & 1

    def _get_obs(self) -> np.ndarray:
        """Stack the observation windows of all agents as float32."""
        if self.state is None:
            raise ResetNeededError("LineMsgEnv has no state; call reset() first.")
        if self._window_base is not self.state:
            self._window_view = sliding_window_view(self.state, self.window)
            self._window_base = self.state
        return self._window_view.astype(np.float32)

    def _render_text(self) -> str:
        assert self.state is not None and self.actions is not None
        n = self.n_obs_nghbr
        lines = ["Current state: "]
        for i in range(self.num_agents):
            lines.append(f"Agent {i}: action = {self.actions[i + n]}, state = {self.state[i + n]}")
        return "\n".join(lines) + "\n"

    # Episode history for the space-time raster (only kept when rendering).
    def _reset_history(self) -> None:
        capacity = min(self.max_iter + 1, _HISTORY_CAPACITY)
        self._history = {
            "states": np.zeros((capacity, self.num_agents), dtype=np.int8),
            "rewards": np.zeros(capacity),
        }
        self._history_len = 0
        self._push_history(0.0, initial=True)

    def _push_history(self, reward: float, initial: bool = False) -> None:
        assert self._history is not None and self.state is not None
        if initial:
            self._history_len = 0
        capacity = self._history["rewards"].shape[0]
        row = self._history_len % capacity
        self._history["states"][row] = self.state[self._agents]
        self._history["rewards"][row] = reward
        self._history_len += 1

    def _ordered_history(self) -> tuple[np.ndarray, np.ndarray, int]:
        assert self._history is not None
        capacity = self._history["rewards"].shape[0]
        count = self._history_len
        if count <= capacity:
            return self._history["states"][:count], self._history["rewards"][:count], 0
        shift = count % capacity
        states = np.roll(self._history["states"], -shift, axis=0)
        rewards = np.roll(self._history["rewards"], -shift)
        return states, rewards, count - capacity
