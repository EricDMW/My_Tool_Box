"""Wireless multiple-access environment (``WirelessComm-v0`` / ``WirelessComm-v1``).

``grid_x * grid_y`` agents sit on a grid; agent ``(i, j)`` (row ``i``, column
``j``, id ``i * grid_y + j``) holds a packet queue with ``ddl`` deadline slots
(slot 0 expires first). Access points sit at the ``(grid_x - 1) * (grid_y - 1)``
interior corners of the grid; access point ``(a, b)`` lies between agents
``(a, b)``, ``(a + 1, b)``, ``(a, b + 1)`` and ``(a + 1, b + 1)``.

Each agent chooses one of five actions per step:

====== ============================== ==========================
action meaning                         access point
====== ============================== ==========================
0      stay idle                      --
1      transmit up-left               ``(i - 1, j - 1)``
2      transmit down-left             ``(i, j - 1)``
3      transmit up-right              ``(i - 1, j)``
4      transmit down-right            ``(i, j)``
====== ============================== ==========================

A transmission is *eligible* when the agent holds at least one packet, its
access point exists and no other agent transmits to that access point in the
same step (the load of the access point is exactly 1; every transmitting agent
counts, with or without a packet). An eligible transmission succeeds with
probability ``success_transmission_probability``; the earliest-deadline packet
is then removed and the agent earns reward 1. Afterwards all queues shift one
slot towards the deadline (slot 0 is dropped) and a new packet arrives in the
last slot with probability ``packet_arrival_probability``.
"""

from __future__ import annotations

import warnings
from numbers import Integral, Real
from typing import Any

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete
from gymnasium.utils import EzPickle, seeding
from numpy.lib.stride_tricks import sliding_window_view

from env_lib.utils.rendering import validate_render_mode

__all__ = [
    "OUTCOME_COLLISION",
    "OUTCOME_IDLE",
    "OUTCOME_LABELS",
    "OUTCOME_LOST",
    "OUTCOME_SUCCESS",
    "WirelessCommEnv",
]

#: Per-agent outcome codes reported in ``info["outcomes"]``.
OUTCOME_IDLE = 0  #: action 0
OUTCOME_SUCCESS = 1  #: packet delivered (reward 1)
OUTCOME_COLLISION = 2  #: access point used by two or more agents
OUTCOME_LOST = 3  #: other failed attempt: no packet, no access point, or channel loss
OUTCOME_LABELS: tuple[str, ...] = ("idle", "success", "collision", "lost")

#: Access-point offsets (row, column) for actions 0..4 (action 0 is unused).
ACTION_OFFSETS = np.array([[0, 0], [-1, -1], [0, -1], [-1, 0], [0, 0]], dtype=np.int64)

_HISTORY_CAPACITY = 512


def _check_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _check_probability(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}")
    return float(value)


class WirelessCommEnv(Env, EzPickle):
    """Cooperative random access to shared wireless access points.

    Parameters
    ----------
    grid_x, grid_y : int, default 6
        Number of agent rows and columns (each at least 2, so that at least one
        access point exists).
    ddl : int, default 2
        Deadline horizon: number of queue slots per agent.
    packet_arrival_probability : float, default 0.8
        Probability ``p`` that a new packet arrives at an agent in a step.
    success_transmission_probability : float, default 0.8
        Probability ``q`` that an eligible (collision-free) transmission
        succeeds.
    n_obs_neighbors : int, default 1
        Each agent observes the queues of the agents within Chebyshev
        distance ``n_obs_neighbors``.
    max_iter : int, default 50
        Episode length; ``truncated`` becomes ``True`` after ``max_iter`` steps.
    render_mode : {"human", "rgb_array", "ansi"} or None, default None
        ``"rgb_array"`` returns an RGB frame of the dashboard, ``"human"``
        shows it in a matplotlib window and ``"ansi"`` returns a text summary.

    Attributes
    ----------
    state : numpy.ndarray of float32, shape (ddl, grid_x + 2 * n_obs_nghbr, grid_y + 2 * n_obs_nghbr)
        Packet queues padded with ``n_obs_nghbr`` cells of value 2; the queue of
        agent ``(i, j)`` is ``state[:, n_obs_nghbr + i, n_obs_nghbr + j]``.
    actions : numpy.ndarray of int, same padded 2-D layout as ``state[0]``
        Last joint action (zero padding).
    last_outcomes : numpy.ndarray of int8, shape (n_agents,)
        Outcome code of every agent in the last step (see ``OUTCOME_LABELS``).
    last_ap_load : numpy.ndarray of int64, shape (grid_x - 1, grid_y - 1)
        Number of agents that transmitted to each access point in the last step.

    Notes
    -----
    Action space: ``MultiDiscrete([5] * n_agents)``. Observation space:
    ``Box(0, 2, (n_agents, ddl * (2 * n_obs_nghbr + 1) ** 2), float32)``; row
    ``k`` is the ``(ddl, 2 * n_obs_nghbr + 1, 2 * n_obs_nghbr + 1)`` window of
    ``state`` centred on agent ``k``, flattened in C order (values 0 = empty
    slot, 1 = packet, 2 = outside the grid).

    The team reward is the number of packets delivered in the step;
    ``info["agent_rewards"]`` holds the per-agent 0/1 rewards,
    ``info["outcomes"]`` the per-agent outcome codes and ``info["ap_load"]``
    the access-point loads.

    Examples
    --------
    >>> from env_lib.wireless_comm_env import WirelessCommEnv
    >>> env = WirelessCommEnv(grid_x=4, grid_y=4)
    >>> obs, info = env.reset(seed=0)
    >>> obs.shape
    (16, 18)
    >>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "name": "WirelessComm-v0",
        "render_fps": 10,
    }

    def __init__(
        self,
        grid_x: int = 6,
        grid_y: int = 6,
        ddl: int = 2,
        packet_arrival_probability: float = 0.8,
        success_transmission_probability: float = 0.8,
        n_obs_neighbors: int = 1,
        max_iter: int = 50,
        render_mode: str | None = None,
    ):
        EzPickle.__init__(
            self,
            grid_x=grid_x,
            grid_y=grid_y,
            ddl=ddl,
            packet_arrival_probability=packet_arrival_probability,
            success_transmission_probability=success_transmission_probability,
            n_obs_neighbors=n_obs_neighbors,
            max_iter=max_iter,
            render_mode=render_mode,
        )
        self.grid_x = _check_int("grid_x", grid_x, 2)
        self.grid_y = _check_int("grid_y", grid_y, 2)
        self.ddl = _check_int("ddl", ddl, 1)
        self.p = _check_probability("packet_arrival_probability", packet_arrival_probability)
        self.q = _check_probability(
            "success_transmission_probability", success_transmission_probability
        )
        self.n_obs_nghbr = _check_int("n_obs_neighbors", n_obs_neighbors, 0)
        self.max_iter = _check_int("max_iter", max_iter, 1)
        validate_render_mode(render_mode, self.metadata["render_modes"])
        self.render_mode = render_mode

        self.n_agents = self.grid_x * self.grid_y
        self.agents: list[str] = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_name_mapping: dict[str, int] = {name: i for i, name in enumerate(self.agents)}
        self.n_access_points = (self.grid_x - 1) * (self.grid_y - 1)

        self.action_space = MultiDiscrete([5] * self.n_agents)
        self.window = 2 * self.n_obs_nghbr + 1
        obs_dim = self.ddl * self.window**2
        self.observation_space = Box(
            low=0, high=2, shape=(self.n_agents, obs_dim), dtype=np.float32
        )

        # Precomputed access-point table: _ap_table[a, k] is the flat index
        # (row * (grid_y - 1) + column) of the access point that agent k reaches
        # with action a, or -1 (idle or outside the grid).
        rows, cols = np.divmod(np.arange(self.n_agents), self.grid_y)
        ap_rows = rows[None, :] + ACTION_OFFSETS[:, :1]
        ap_cols = cols[None, :] + ACTION_OFFSETS[:, 1:]
        valid = (
            (ap_rows >= 0)
            & (ap_rows < self.grid_x - 1)
            & (ap_cols >= 0)
            & (ap_cols < self.grid_y - 1)
        )
        valid[0] = False
        self._ap_table = np.where(valid, ap_rows * (self.grid_y - 1) + ap_cols, -1)
        self._ap_table_flat = self._ap_table.ravel()
        self._agent_ids = np.arange(self.n_agents)
        n = self.n_obs_nghbr
        self._interior = (slice(None), slice(n, n + self.grid_x), slice(n, n + self.grid_y))

        self.state: np.ndarray | None = None
        self.actions: np.ndarray | None = None
        self.num_moves = 0
        self.closed = False
        self.last_action = np.zeros(self.n_agents, dtype=np.int64)
        self.last_outcomes = np.zeros(self.n_agents, dtype=np.int8)
        self.last_ap_load = np.zeros((self.grid_x - 1, self.grid_y - 1), dtype=np.int64)

        self._window_view: np.ndarray | None = None
        self._window_base: np.ndarray | None = None
        self._last_reward = 0.0
        self._return = 0.0
        self._renderer = None
        self._warned_no_render = False
        self._history: np.ndarray | None = None
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
            "WirelessCommEnv.seed() is deprecated; use reset(seed=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._np_random, seed = seeding.np_random(seed)
        self._np_random_seed = seed
        return [seed]

    # ------------------------------------------------------------------
    # Public helpers (kept for backwards compatibility)
    # ------------------------------------------------------------------
    def access_point_mapping(
        self, i: int, j: int, agent_action: int
    ) -> tuple[int | None, int | None]:
        """Return the access point ``(row, column)`` reached by agent ``(i, j)``.

        Parameters
        ----------
        i, j : int
            Row and column of the agent.
        agent_action : int
            Action in ``0..4``.

        Returns
        -------
        tuple
            ``(row, column)`` of the access point, or ``(None, None)`` for the
            idle action and for access points outside the grid.

        Raises
        ------
        ValueError
            If the action is not in ``0..4`` or ``(i, j)`` is not on the grid.
        """
        if isinstance(agent_action, bool) or not isinstance(agent_action, Integral):
            raise ValueError(f"agent_action = {agent_action!r} is not defined!")
        if not 0 <= agent_action <= 4:
            raise ValueError(f"agent_action = {agent_action} is not defined!")
        if not (0 <= i < self.grid_x and 0 <= j < self.grid_y):
            raise ValueError(f"agent ({i}, {j}) is outside the {self.grid_x}x{self.grid_y} grid")
        ap = int(self._ap_table[agent_action, i * self.grid_y + j])
        if ap < 0:
            return None, None
        return divmod(ap, self.grid_y - 1)

    def check_transmission_fail(
        self,
        agent_access_point_x: int | None,
        agent_access_point_y: int | None,
        access_point_profile: np.ndarray,
    ) -> bool:
        """Return ``True`` unless exactly one agent uses the given access point.

        ``access_point_profile`` is a ``(grid_x - 1, grid_y - 1)`` array of
        access-point loads (for example :attr:`last_ap_load`); a ``None``
        coordinate (no access point) always fails.
        """
        if agent_access_point_x is None or agent_access_point_y is None:
            return True
        return bool(access_point_profile[agent_access_point_x, agent_access_point_y] != 1)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the queues with uniformly random packets.

        Parameters
        ----------
        seed : int, optional
            Seed for the environment's random generator.
        options : dict, optional
            Unused.

        Returns
        -------
        observation : numpy.ndarray
            ``(n_agents, ddl * (2 * n_obs_nghbr + 1) ** 2)`` float32 array.
        info : dict
            Same keys as in :meth:`step`, all zero.
        """
        super().reset(seed=seed)
        n = self.n_obs_nghbr
        shape = (self.ddl, self.grid_x + 2 * n, self.grid_y + 2 * n)
        self.state = np.full(shape, fill_value=2, dtype=np.float32)
        self.state[self._interior] = self.np_random.choice(
            2, size=(self.ddl, self.grid_x, self.grid_y)
        )
        self.actions = np.zeros(shape[1:], dtype=int)
        self.num_moves = 0
        self.last_action = np.zeros(self.n_agents, dtype=np.int64)
        self.last_outcomes = np.zeros(self.n_agents, dtype=np.int8)
        self.last_ap_load = np.zeros((self.grid_x - 1, self.grid_y - 1), dtype=np.int64)
        self._last_reward = 0.0
        self._return = 0.0

        if self.render_mode in ("human", "rgb_array"):
            self._reset_history()
            if self._renderer is not None:
                self._renderer.reset()
        obs = self._get_obs()
        info = {
            "agent_rewards": np.zeros(self.n_agents),
            "outcomes": self.last_outcomes.copy(),
            "ap_load": self.last_ap_load.copy(),
        }
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply a joint action.

        Parameters
        ----------
        action : array-like of int
            ``n_agents`` actions in ``0..4``, shape ``(n_agents,)`` or
            ``(grid_x, grid_y)``.

        Returns
        -------
        observation, reward, terminated, truncated, info
            ``reward`` is the number of delivered packets (a float),
            ``terminated`` is always ``False``. ``info`` holds
            ``"agent_rewards"`` (float64, ``(n_agents,)``), ``"outcomes"``
            (int8 outcome codes) and ``"ap_load"`` (access-point loads).

        Raises
        ------
        RuntimeError
            If :meth:`reset` has not been called.
        ValueError
            If ``action`` has the wrong shape or values outside ``0..4``.
        """
        state = self.state
        if state is None:
            raise RuntimeError("WirelessCommEnv.step() called before reset(); call reset() first.")
        act = self._validate_action(action)
        n_agents = self.n_agents

        # Access-point load (bin 0 collects idle agents and missing access points).
        ap = self._ap_table_flat[act * n_agents + self._agent_ids]
        counts = np.bincount(ap + 1, minlength=self.n_access_points + 1)
        load = counts[ap + 1]

        interior = state[self._interior]  # view (ddl, grid_x, grid_y)
        queues = np.array(interior).reshape(self.ddl, n_agents)  # private copy
        has_packet = queues.any(axis=0)
        transmit = act != 0
        eligible = transmit & has_packet & (ap >= 0) & (load == 1)

        # One uniform draw per eligible agent, in agent order.
        candidates = np.flatnonzero(eligible)
        winners = candidates[self.np_random.random(candidates.size) <= self.q]
        if winners.size:
            first_packet = queues[:, winners].argmax(axis=0)
            queues[first_packet, winners] = 0.0
        agent_rewards = np.zeros(n_agents)
        agent_rewards[winners] = 1.0

        # Deadline shift and new arrivals.
        interior[:-1] = queues[1:].reshape(self.ddl - 1, self.grid_x, self.grid_y)
        interior[-1] = self.np_random.random((self.grid_x, self.grid_y)) <= self.p

        outcomes = np.where(transmit, OUTCOME_LOST, OUTCOME_IDLE).astype(np.int8)
        outcomes[transmit & (ap >= 0) & (load >= 2)] = OUTCOME_COLLISION
        outcomes[winners] = OUTCOME_SUCCESS
        self.last_action = act
        self.last_outcomes = outcomes
        self.last_ap_load = counts[1:].reshape(self.grid_x - 1, self.grid_y - 1)
        n = self.n_obs_nghbr
        self.actions[n : n + self.grid_x, n : n + self.grid_y] = act.reshape(
            self.grid_x, self.grid_y
        )

        reward = float(winners.size)
        self.num_moves += 1
        self._last_reward = reward
        self._return += reward
        truncated = self.num_moves >= self.max_iter
        if self._history is not None:
            self._push_history(reward, outcomes)

        obs = self._get_obs()
        info = {
            "agent_rewards": agent_rewards,
            "outcomes": outcomes,
            "ap_load": self.last_ap_load,
        }
        if self.render_mode == "human":
            self.render()
        return obs, reward, False, truncated, info

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
                    "WirelessCommEnv.render() called without a render_mode; pass "
                    "render_mode='rgb_array', 'human' or 'ansi' to the constructor.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_no_render = True
            return None
        if self.state is None or self.actions is None:
            raise RuntimeError(
                "WirelessCommEnv.render() called before reset(); call reset() first."
            )
        if self.render_mode == "ansi":
            return self._render_text()
        if self._history is None:
            self._reset_history()
        if self._renderer is None:
            from env_lib.wireless_comm_env.rendering import WirelessCommRenderer

            self._renderer = WirelessCommRenderer(
                self.render_mode,
                grid_x=self.grid_x,
                grid_y=self.grid_y,
                ddl=self.ddl,
                window=self._history.shape[0],
                packet_arrival_probability=self.p,
                success_transmission_probability=self.q,
                fps=self.metadata["render_fps"],
            )
        history, first_step = self._ordered_history()
        return self._renderer.render(
            queues=self.state[self._interior],
            actions=self.last_action,
            outcomes=self.last_outcomes,
            ap_load=self.last_ap_load,
            history=history,
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
    def _validate_action(self, action: Any) -> np.ndarray:
        act = np.array(action, dtype=np.int64)  # private copy (kept as last_action)
        if act.shape != (self.n_agents,):
            if act.shape != (self.grid_x, self.grid_y):
                raise ValueError(
                    f"action must have shape ({self.n_agents},) or ({self.grid_x}, "
                    f"{self.grid_y}), got {np.shape(action)}"
                )
            act = act.reshape(-1)
        if act.min() < 0 or act.max() > 4:
            raise ValueError(f"agent actions must lie in 0..4, got {np.asarray(action)!r}")
        return act

    def _get_obs(self) -> np.ndarray:
        """Stack the flattened ``(ddl, w, w)`` windows of all agents as float32."""
        if self.state is None:
            raise RuntimeError("WirelessCommEnv has no state; call reset() first.")
        if self._window_base is not self.state:
            view = sliding_window_view(self.state, (self.window, self.window), axis=(1, 2))
            # (ddl, grid_x, grid_y, w, w) -> (grid_x, grid_y, ddl, w, w)
            self._window_view = view.transpose(1, 2, 0, 3, 4)
            self._window_base = self.state
        obs = np.empty(self.observation_space.shape, dtype=np.float32)
        np.copyto(obs.reshape(self._window_view.shape), self._window_view)
        return obs

    def _render_text(self) -> str:
        assert self.state is not None and self.actions is not None
        n = self.n_obs_nghbr
        lines = ["Current state: "]
        for k in range(self.n_agents):
            x, y = divmod(k, self.grid_y)
            lines.append(
                f"Agent {k}: action = {self.actions[x + n, y + n]}, "
                f"state = {self.state[:, x + n, y + n]}"
            )
        return "\n".join(lines) + "\n"

    # Per-step counts for the throughput panel (only kept when rendering).
    # Columns: reward, successes, collisions, lost, transmissions.
    def _reset_history(self) -> None:
        capacity = min(self.max_iter + 1, _HISTORY_CAPACITY)
        self._history = np.zeros((capacity, 5))
        self._history_len = 1

    def _push_history(self, reward: float, outcomes: np.ndarray) -> None:
        assert self._history is not None
        capacity = self._history.shape[0]
        counts = np.bincount(outcomes, minlength=4)
        row = self._history[self._history_len % capacity]
        row[0] = reward
        row[1] = counts[OUTCOME_SUCCESS]
        row[2] = counts[OUTCOME_COLLISION]
        row[3] = counts[OUTCOME_LOST]
        row[4] = self.n_agents - counts[OUTCOME_IDLE]
        self._history_len += 1

    def _ordered_history(self) -> tuple[np.ndarray, int]:
        assert self._history is not None
        capacity = self._history.shape[0]
        count = self._history_len
        if count <= capacity:
            return self._history[:count], 0
        return np.roll(self._history, -(count % capacity), axis=0), count - capacity
