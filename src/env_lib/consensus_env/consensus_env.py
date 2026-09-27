"""Networked consensus (rendezvous) and formation control environment.

``N`` agents move in a bounded 2-D arena and must either agree on a common
position (``task="consensus"``, also called rendezvous) or reach a geometric
formation up to translation (``task="formation"``). Each agent only observes
the agents it is connected to in a communication graph, which is either fixed
(ring, line, star, complete, Erdos-Renyi) or rebuilt every step from the
inter-agent distances (proximity / disk graph).

The module also exposes the small graph and formation helpers used by the
environment (:func:`make_topology`, :func:`proximity_adjacency`,
:func:`make_formation`, :func:`algebraic_connectivity`, :func:`is_connected`),
which are handy for analysis and for writing reference controllers.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env_lib.errors import ResetNeededError
from env_lib.utils.rendering import validate_render_mode

__all__ = [
    "DYNAMICS",
    "FORMATION_SHAPES",
    "TASKS",
    "TOPOLOGIES",
    "ConsensusEnv",
    "algebraic_connectivity",
    "is_connected",
    "make_formation",
    "make_topology",
    "proximity_adjacency",
]

logger = logging.getLogger(__name__)

TASKS: tuple[str, ...] = ("consensus", "formation")
TOPOLOGIES: tuple[str, ...] = ("ring", "line", "star", "complete", "erdos_renyi", "proximity")
DYNAMICS: tuple[str, ...] = ("single", "double")
FORMATION_SHAPES: tuple[str, ...] = ("circle", "line", "grid", "wedge")

#: Half opening angle of the ``"wedge"`` formation, in radians (35 degrees).
WEDGE_HALF_ANGLE: float = math.radians(35.0)

_ER_MAX_ATTEMPTS = 1000
_INIT_BATCH = 256
_INIT_MAX_BATCHES = 32
_INIT_SPREAD = 0.8
_PROXIMITY_DEFAULT_SLOTS = 6


# ---------------------------------------------------------------------------
# Graph and formation helpers
# ---------------------------------------------------------------------------
def _batched_connected(adjacency: np.ndarray) -> np.ndarray:
    """Connectivity test for a batch of undirected graphs.

    Parameters
    ----------
    adjacency:
        Boolean array of shape ``(B, n, n)``.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(B,)``.
    """
    n = adjacency.shape[-1]
    reach = (adjacency | np.eye(n, dtype=bool)).astype(np.float32)
    # After k squarings, reach[i, j] > 0 iff j is reachable from i in <= 2**k hops.
    for _ in range(max(1, math.ceil(math.log2(max(n - 1, 1))))):
        reach = (np.matmul(reach, reach) > 0).astype(np.float32)
    return reach[:, 0, :].all(axis=-1)


def is_connected(adjacency: np.ndarray) -> bool:
    """Return ``True`` if the undirected graph with the given adjacency matrix is connected.

    Parameters
    ----------
    adjacency:
        Square boolean (or 0/1) matrix of shape ``(n, n)``.

    Returns
    -------
    bool
    """
    adj = np.asarray(adjacency).astype(bool)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency must be a square matrix, got shape {adj.shape}")
    return bool(_batched_connected(adj[None])[0])


def algebraic_connectivity(adjacency: np.ndarray) -> float:
    """Fiedler value (second smallest eigenvalue) of the graph Laplacian ``L = D - A``.

    It is positive exactly when the graph is connected and bounds the
    convergence rate of linear consensus (``exp(-lambda_2 t)`` for the
    single-integrator protocol with unit gain).

    Parameters
    ----------
    adjacency:
        Symmetric boolean (or weighted) matrix of shape ``(n, n)``, ``n >= 2``.

    Returns
    -------
    float
        ``lambda_2(L) >= 0``.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1] or adj.shape[0] < 2:
        raise ValueError(f"adjacency must be a square matrix with n >= 2, got shape {adj.shape}")
    laplacian = np.diag(adj.sum(axis=1)) - adj
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return float(max(eigenvalues[1], 0.0))


def proximity_adjacency(positions: np.ndarray, comm_radius: float) -> np.ndarray:
    """Disk-graph adjacency: agents closer than ``comm_radius`` are connected.

    Parameters
    ----------
    positions:
        Array of shape ``(n, 2)``.
    comm_radius:
        Communication range (inclusive).

    Returns
    -------
    numpy.ndarray
        Boolean ``(n, n)`` matrix with a ``False`` diagonal.
    """
    pos = np.asarray(positions, dtype=np.float64)
    delta = pos[None, :, :] - pos[:, None, :]
    sq_dist = np.einsum("ijk,ijk->ij", delta, delta)
    adj = sq_dist <= comm_radius * comm_radius
    np.fill_diagonal(adj, False)
    return adj


def make_topology(
    topology: str,
    n_agents: int,
    *,
    edge_probability: float = 0.3,
    rng: np.random.Generator | None = None,
    max_attempts: int = _ER_MAX_ATTEMPTS,
) -> np.ndarray:
    """Build the adjacency matrix of a static communication graph.

    Parameters
    ----------
    topology:
        ``"ring"``, ``"line"``, ``"star"`` (agent 0 is the hub), ``"complete"``
        or ``"erdos_renyi"``. ``"proximity"`` graphs depend on the positions;
        use :func:`proximity_adjacency` for them.
    n_agents:
        Number of nodes (``>= 2``).
    edge_probability:
        Independent edge probability of the Erdos-Renyi model.
    rng:
        Generator for the Erdos-Renyi model (a fresh unseeded one if omitted).
    max_attempts:
        Erdos-Renyi graphs are resampled until connected, at most this often.

    Returns
    -------
    numpy.ndarray
        Symmetric boolean ``(n_agents, n_agents)`` matrix with a ``False`` diagonal.

    Raises
    ------
    ValueError
        For an unknown topology, or if no connected Erdos-Renyi graph was found.
    """
    n = int(n_agents)
    if n < 2:
        raise ValueError(f"n_agents must be >= 2, got {n_agents}")
    idx = np.arange(n)
    adj = np.zeros((n, n), dtype=bool)
    if topology == "ring":
        adj[idx, (idx + 1) % n] = True
    elif topology == "line":
        adj[idx[:-1], idx[1:]] = True
    elif topology == "star":
        adj[0, 1:] = True
    elif topology == "complete":
        adj[:] = True
    elif topology == "erdos_renyi":
        generator = rng if rng is not None else np.random.default_rng()
        rows, cols = np.triu_indices(n, k=1)
        for attempt in range(1, max_attempts + 1):
            adj[:] = False
            adj[rows, cols] = generator.random(rows.size) < edge_probability
            adj |= adj.T
            if is_connected(adj):
                logger.debug("connected Erdos-Renyi graph found after %d attempt(s)", attempt)
                return adj
        raise ValueError(
            f"Could not sample a connected Erdos-Renyi graph with n_agents={n} and "
            f"edge_probability={edge_probability} in {max_attempts} attempts; increase "
            f"edge_probability (connectivity threshold is about ln(n)/n = {math.log(n) / n:.3f})."
        )
    elif topology == "proximity":
        raise ValueError("proximity graphs depend on positions; use proximity_adjacency()")
    else:
        raise ValueError(f"topology must be one of {TOPOLOGIES}, got {topology!r}")
    adj |= adj.T
    np.fill_diagonal(adj, False)
    return adj


def make_formation(shape: str, n_agents: int, radius: float = 1.0) -> np.ndarray:
    """Formation offsets ``d_i``, centred at the origin.

    All shapes have a characteristic extent of ``2 * radius``:

    * ``"circle"`` -- regular polygon of radius ``radius``, agent 0 at the top.
    * ``"line"`` -- evenly spaced on a horizontal segment of length ``2 * radius``.
    * ``"grid"`` -- row-major grid with ``ceil(sqrt(n))`` columns whose longer side
      spans ``2 * radius``.
    * ``"wedge"`` -- V shape pointing up (+y), agent 0 at the apex, the others
      alternating between the two arms (half opening angle 35 degrees, arm length
      ``2 * radius``).

    Parameters
    ----------
    shape:
        One of :data:`FORMATION_SHAPES`.
    n_agents:
        Number of agents (``>= 2``).
    radius:
        Scale of the formation.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_agents, 2)`` whose rows sum to zero.
    """
    n = int(n_agents)
    if n < 2:
        raise ValueError(f"n_agents must be >= 2, got {n_agents}")
    k = np.arange(n)
    if shape == "circle":
        angle = 0.5 * np.pi + 2.0 * np.pi * k / n
        points = radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    elif shape == "line":
        points = np.stack([np.linspace(-radius, radius, n), np.zeros(n)], axis=1)
    elif shape == "grid":
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        spacing = 2.0 * radius / (max(cols, rows) - 1)
        points = np.stack([(k % cols) * spacing, -(k // cols) * spacing], axis=1)
    elif shape == "wedge":
        rank = (k + 1) // 2
        side = np.where(k % 2 == 1, -1.0, 1.0)
        side[0] = 0.0
        spacing = 2.0 * radius / rank.max()
        points = np.stack(
            [
                side * rank * spacing * math.sin(WEDGE_HALF_ANGLE),
                -rank * spacing * math.cos(WEDGE_HALF_ANGLE),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"formation_shape must be one of {FORMATION_SHAPES}, got {shape!r}")
    points = points.astype(np.float64)
    return points - points.mean(axis=0)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
def _check_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _check_float(
    name: str, value: Any, *, minimum: float = -math.inf, strict: bool = False
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < minimum or (strict and value == minimum):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be {relation} {minimum}, got {value}")
    return value


def _check_choice(name: str, value: Any, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return str(value)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class ConsensusEnv(gym.Env):
    """Networked multi-agent consensus (rendezvous) and formation control.

    Agent ``i`` has position ``x_i`` and velocity ``v_i`` in the arena
    ``[-arena_size, arena_size]^2`` and a formation offset ``d_i`` (zero for
    ``task="consensus"``; see :func:`make_formation`). The task is solved when
    the shifted positions ``y_i = x_i - d_i`` agree, i.e. when the task error

    ``e = (1/N) * sum_i ||y_i - mean_j y_j||^2``

    drops below ``tolerance``. Agents communicate over an undirected graph with
    neighbour sets ``N_i``.

    **Dynamics** (``u_i`` is the action of agent ``i``, clipped to
    ``[-max_control, max_control]^2``, ``w`` standard Gaussian noise):

    * ``"single"``: ``x_i <- clip(x_i + dt * u_i + noise_std * sqrt(dt) * w)``.
      The reported velocity is the realised one, ``(x_i(t+1) - x_i(t)) / dt``.
    * ``"double"``: ``v_i <- v_i + dt * (u_i - damping * v_i) + noise_std * sqrt(dt) * w``,
      then ``x_i <- clip(x_i + dt * v_i)`` (semi-implicit Euler). A velocity
      component is set to zero when the arena wall stops the agent.

    **Observation** -- array of shape ``(n_agents, obs_dim)``, ``float32``, with
    ``obs_dim = 6 + 5 * max_neighbors``. Row ``i`` contains

    =========================  ==========  =======================================
    columns                    layout key  content
    =========================  ==========  =======================================
    ``0:2``                    position    ``x_i``
    ``2:4``                    velocity    ``v_i``
    ``4:6``                    offset      ``d_i`` (zeros for consensus)
    ``6+5k : 6+5k+2``          neighbors   slot ``k``: ``(x_j - d_j) - (x_i - d_i)``
    ``6+5k+2 : 6+5k+4``        neighbors   slot ``k``: ``v_j - v_i``
    ``6+5k+4``                 neighbors   slot ``k``: valid mask (1 or 0)
    =========================  ==========  =======================================

    Slots ``k = 0 .. max_neighbors - 1`` hold the neighbours ``j in N_i``
    sorted by physical distance ``||x_j - x_i||`` (ties broken by index).
    Unused slots are all zeros (mask 0). If an agent has more neighbours than
    slots, the farthest ones are left out of the observation (they still count
    in the reward and in :meth:`laplacian_policy`). ``obs[:, layout["neighbors"]]
    .reshape(n_agents, max_neighbors, 5)`` gives the slots, whose columns are
    described by :attr:`neighbor_slot_layout`. The observation space is an
    unbounded ``Box``; all entries are finite.

    **Action** -- ``Box(-max_control, max_control, shape=(n_agents, 2), float32)``:
    velocity command (``"single"``) or acceleration (``"double"``).

    **Reward** -- per agent,

    ``r_i = -(1/|N_i|) * sum_{j in N_i} ||y_i - y_j||^2 - control_cost * ||u_i||^2``

    (isolated agents only pay the control cost). The team reward returned by
    :meth:`step` is ``mean_i r_i``, plus ``success_bonus`` on the step on which
    the task error first drops below ``tolerance``.

    **Episode end** -- ``terminated`` is ``True`` when ``e < tolerance``;
    ``truncated`` is ``True`` once ``max_steps`` steps were taken (both can be
    ``True`` on the last step, as with Gymnasium's ``TimeLimit``).

    **Info** -- ``agent_rewards`` (``(n_agents,)`` float64), ``error`` (float),
    ``algebraic_connectivity`` (Fiedler value of the current graph Laplacian),
    ``adjacency`` (``(n_agents, n_agents)`` bool), ``success`` (bool, sticky
    within an episode) and ``step`` (int).

    **Initial state** -- positions uniform in ``[-0.8 a, 0.8 a]^2``
    (``a = arena_size``), velocities zero, drawn from ``self.np_random``.
    For ``topology="proximity"`` the uniform draw is conditioned on the
    initial disk graph being connected (rejection sampling in vectorised
    batches). If ``comm_radius`` is so small that rejection sampling fails,
    agents are placed sequentially, each within ``0.95 * comm_radius`` of a
    randomly chosen, previously placed agent, and a warning is issued once.
    During the episode a proximity graph may disconnect (``lambda_2 = 0``).

    Parameters
    ----------
    n_agents:
        Number of agents (``>= 2``).
    task:
        ``"consensus"`` (rendezvous) or ``"formation"`` (formation up to translation).
    topology:
        ``"ring"``, ``"line"``, ``"star"`` (agent 0 is the hub), ``"complete"``,
        ``"erdos_renyi"`` (static, resampled until connected) or ``"proximity"``
        (disk graph rebuilt every step from the distances, range ``comm_radius``).
    dynamics:
        ``"single"`` (action = velocity) or ``"double"`` (action = acceleration,
        with linear damping).
    dt:
        Integration time step (``> 0``).
    max_steps:
        Episode length before truncation (``>= 1``).
    arena_size:
        Positions are clipped to ``[-arena_size, arena_size]^2`` (``> 0``).
    max_control:
        Bound of every action component (``> 0``).
    control_cost:
        Weight of the quadratic control penalty (``>= 0``).
    noise_std:
        Intensity of the Gaussian process noise (``>= 0``); each step adds
        ``noise_std * sqrt(dt)`` standard-deviation noise to the position
        (``"single"``) or velocity (``"double"``).
    tolerance:
        Success threshold on the task error (``> 0``, squared distance units).
    success_bonus:
        Team-reward bonus paid once, on the step the task is solved.
    formation_shape:
        ``"circle"``, ``"line"``, ``"grid"`` or ``"wedge"`` (``task="formation"`` only).
    formation_radius:
        Scale of the formation (``> 0``); see :func:`make_formation`.
    comm_radius:
        Communication range of the ``"proximity"`` topology (``> 0``).
    edge_probability:
        Edge probability of the ``"erdos_renyi"`` topology, in ``(0, 1]``.
    graph_seed:
        Seed of the local generator used for the ``"erdos_renyi"`` graph, which
        is sampled once at construction (``None`` draws fresh OS entropy, so two
        instances may then have different graphs and observation sizes).
    max_neighbors:
        Number of neighbour slots in the observation (``>= 1``). Defaults to
        the maximum degree of a static graph, and to ``min(6, n_agents - 1)``
        for ``"proximity"``.
    damping:
        Linear velocity damping of the double integrator (``>= 0``).
    render_mode:
        ``None``, ``"human"`` or ``"rgb_array"``.

    Raises
    ------
    ValueError
        For out-of-range or unknown argument values, or a formation that does
        not fit in the arena.
    TypeError
        For arguments of the wrong type.

    Examples
    --------
    >>> env = ConsensusEnv(task="formation", formation_shape="wedge")
    >>> obs, info = env.reset(seed=0)
    >>> obs.shape
    (8, 16)
    >>> for _ in range(200):
    ...     obs, reward, terminated, truncated, info = env.step(env.laplacian_policy())
    ...     if terminated or truncated:
    ...         break
    >>> bool(info["success"])
    True
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    #: Width of one neighbour slot in the observation.
    SLOT_DIM: int = 5
    #: Columns of one neighbour slot.
    neighbor_slot_layout: dict[str, slice] = {
        "rel_position": slice(0, 2),
        "rel_velocity": slice(2, 4),
        "mask": slice(4, 5),
    }

    def __init__(
        self,
        *,
        n_agents: int = 8,
        task: str = "consensus",
        topology: str = "ring",
        dynamics: str = "single",
        dt: float = 0.1,
        max_steps: int = 200,
        arena_size: float = 10.0,
        max_control: float = 1.0,
        control_cost: float = 0.01,
        noise_std: float = 0.0,
        tolerance: float = 0.05,
        success_bonus: float = 10.0,
        formation_shape: str = "circle",
        formation_radius: float = 3.0,
        comm_radius: float = 4.0,
        edge_probability: float = 0.3,
        graph_seed: int | None = None,
        max_neighbors: int | None = None,
        damping: float = 0.5,
        render_mode: str | None = None,
    ):
        super().__init__()
        validate_render_mode(render_mode, self.metadata["render_modes"])
        self.n_agents = _check_int("n_agents", n_agents, 2)
        self.task = _check_choice("task", task, TASKS)
        self.topology = _check_choice("topology", topology, TOPOLOGIES)
        self.dynamics = _check_choice("dynamics", dynamics, DYNAMICS)
        self.dt = _check_float("dt", dt, minimum=0.0, strict=True)
        self.max_steps = _check_int("max_steps", max_steps, 1)
        self.arena_size = _check_float("arena_size", arena_size, minimum=0.0, strict=True)
        self.max_control = _check_float("max_control", max_control, minimum=0.0, strict=True)
        self.control_cost = _check_float("control_cost", control_cost, minimum=0.0)
        self.noise_std = _check_float("noise_std", noise_std, minimum=0.0)
        self.tolerance = _check_float("tolerance", tolerance, minimum=0.0, strict=True)
        self.success_bonus = _check_float("success_bonus", success_bonus)
        self.formation_shape = _check_choice("formation_shape", formation_shape, FORMATION_SHAPES)
        self.formation_radius = _check_float(
            "formation_radius", formation_radius, minimum=0.0, strict=True
        )
        self.comm_radius = _check_float("comm_radius", comm_radius, minimum=0.0, strict=True)
        self.edge_probability = _check_float(
            "edge_probability", edge_probability, minimum=0.0, strict=True
        )
        if self.edge_probability > 1.0:
            raise ValueError(f"edge_probability must be in (0, 1], got {edge_probability}")
        if graph_seed is not None:
            graph_seed = _check_int("graph_seed", graph_seed, 0)
        self.graph_seed = graph_seed
        self.damping = _check_float("damping", damping, minimum=0.0)
        self.render_mode = render_mode

        n = self.n_agents
        # Formation offsets d_i (zero for consensus) and their pairwise differences d_j - d_i.
        if self.task == "formation":
            self._offsets = make_formation(self.formation_shape, n, self.formation_radius)
            span = self._offsets.max(axis=0) - self._offsets.min(axis=0)
            if np.any(span > 2.0 * self.arena_size):
                raise ValueError(
                    f"formation {self.formation_shape!r} with formation_radius="
                    f"{self.formation_radius} spans {span.max():.2f}, which does not fit in the "
                    f"arena of width {2.0 * self.arena_size}; reduce formation_radius or "
                    "increase arena_size"
                )
        else:
            self._offsets = np.zeros((n, 2), dtype=np.float64)
        self._offset_delta = self._offsets[None, :, :] - self._offsets[:, None, :]

        # Communication graph.
        self._dynamic_graph = self.topology == "proximity"
        if self._dynamic_graph:
            self._static_adjacency: np.ndarray | None = None
            default_slots = min(_PROXIMITY_DEFAULT_SLOTS, n - 1)
        else:
            rng = np.random.default_rng(self.graph_seed)
            self._static_adjacency = make_topology(
                self.topology, n, edge_probability=self.edge_probability, rng=rng
            )
            self._static_adjacency.setflags(write=False)
            self._static_lambda2 = algebraic_connectivity(self._static_adjacency)
            default_slots = int(self._static_adjacency.sum(axis=1).max())
        if max_neighbors is None:
            self.max_neighbors = default_slots
        else:
            self.max_neighbors = _check_int("max_neighbors", max_neighbors, 1)
        self._visible_slots = min(self.max_neighbors, n)
        self._rows = np.arange(n)[:, None]

        self.obs_dim = 6 + self.SLOT_DIM * self.max_neighbors
        self._layout: dict[str, slice] = {
            "position": slice(0, 2),
            "velocity": slice(2, 4),
            "offset": slice(4, 6),
            "neighbors": slice(6, self.obs_dim),
        }
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n, self.obs_dim), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-self.max_control, high=self.max_control, shape=(n, 2), dtype=np.float32
        )

        # Episode state (float64 internally; observations are cast to float32 once).
        self._pos: np.ndarray | None = None
        self._vel = np.zeros((n, 2), dtype=np.float64)
        self._adj = np.zeros((n, n), dtype=bool)
        self._adj_f = np.zeros((n, n), dtype=np.float64)
        self._deg = np.zeros(n, dtype=np.float64)
        self._lambda2 = 0.0
        self._error = math.inf
        self._step_count = 0
        self._success = False
        self._noise_scale = self.noise_std * math.sqrt(self.dt)

        self._renderer = None
        self._warned_no_render_mode = False
        self._warned_sequential_init = False
        self._warned_policy_gain = False

    # ------------------------------------------------------------------
    # Public read-only views
    # ------------------------------------------------------------------
    @property
    def observation_layout(self) -> dict[str, slice]:
        """Column slices of one observation row (``position``, ``velocity``, ``offset``, ``neighbors``)."""
        return dict(self._layout)

    @property
    def positions(self) -> np.ndarray:
        """Current positions, shape ``(n_agents, 2)`` (a copy)."""
        self._require_reset()
        return self._pos.copy()

    @property
    def velocities(self) -> np.ndarray:
        """Current velocities, shape ``(n_agents, 2)`` (a copy)."""
        self._require_reset()
        return self._vel.copy()

    @property
    def formation_offsets(self) -> np.ndarray:
        """Formation offsets ``d_i``, shape ``(n_agents, 2)`` (zeros for consensus)."""
        return self._offsets.copy()

    @property
    def adjacency(self) -> np.ndarray:
        """Adjacency matrix of the current communication graph (bool, a copy)."""
        if self._dynamic_graph:
            self._require_reset()
            return self._adj.copy()
        return self._static_adjacency.copy()

    @property
    def laplacian(self) -> np.ndarray:
        """Graph Laplacian ``L = D - A`` of the current communication graph."""
        adj = self.adjacency.astype(np.float64)
        return np.diag(adj.sum(axis=1)) - adj

    @property
    def algebraic_connectivity(self) -> float:
        """Fiedler value of the current graph Laplacian."""
        if self._dynamic_graph:
            self._require_reset()
            return self._lambda2
        return self._static_lambda2

    @property
    def task_error(self) -> float:
        """Current task error ``e`` (see the class docstring)."""
        self._require_reset()
        return self._error

    @property
    def step_count(self) -> int:
        """Number of steps taken in the current episode."""
        return self._step_count

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        Parameters
        ----------
        seed:
            Seed for ``self.np_random`` (initial positions and process noise).
        options:
            Optional ``{"positions": array (n_agents, 2), "velocities": array
            (n_agents, 2)}`` to start from a given state (positions are clipped
            to the arena, velocities default to zero).

        Returns
        -------
        observation, info
        """
        super().reset(seed=seed)
        options = dict(options or {})
        unknown = set(options) - {"positions", "velocities"}
        if unknown:
            raise ValueError(
                f"Unknown reset options {sorted(unknown)}; use 'positions', 'velocities'"
            )

        n = self.n_agents
        if options.get("positions") is not None:
            positions = self._state_option("positions", options["positions"])
            positions = np.clip(positions, -self.arena_size, self.arena_size)
        else:
            positions = self._sample_positions()
        if options.get("velocities") is not None:
            velocities = self._state_option("velocities", options["velocities"])
        else:
            velocities = np.zeros((n, 2), dtype=np.float64)

        self._pos = positions
        self._vel = velocities
        self._step_count = 0
        self._success = False

        delta_x, sq_x, delta_y, sq_y = self._pairwise()
        self._update_graph(sq_x)
        self._error = self._compute_error()
        observation = self._observation(delta_y, sq_x)
        info = self._info(np.zeros(n, dtype=np.float64))

        if self._renderer is not None:
            self._renderer.reset()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the system by one time step.

        Parameters
        ----------
        action:
            Array of shape ``(n_agents, 2)`` (anything reshapeable to it);
            clipped to the action bounds.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        self._require_reset()
        control = self._validate_action(action)

        pos, vel = self._pos, self._vel
        if self.dynamics == "single":
            moved = pos + self.dt * control
            if self._noise_scale > 0.0:
                moved += self._noise_scale * self.np_random.standard_normal(pos.shape)
            new_pos = np.clip(moved, -self.arena_size, self.arena_size)
            new_vel = (new_pos - pos) / self.dt
        else:
            new_vel = vel + self.dt * (control - self.damping * vel)
            if self._noise_scale > 0.0:
                new_vel += self._noise_scale * self.np_random.standard_normal(vel.shape)
            moved = pos + self.dt * new_vel
            new_pos = np.clip(moved, -self.arena_size, self.arena_size)
            new_vel[moved != new_pos] = 0.0
        self._pos, self._vel = new_pos, new_vel
        self._step_count += 1

        delta_x, sq_x, delta_y, sq_y = self._pairwise()
        self._update_graph(sq_x)

        disagreement = (self._adj_f * sq_y).sum(axis=1) / np.maximum(self._deg, 1.0)
        agent_rewards = -disagreement - self.control_cost * np.einsum("ij,ij->i", control, control)

        self._error = self._compute_error()
        solved = self._error < self.tolerance
        reward = float(agent_rewards.mean())
        if solved and not self._success:
            reward += self.success_bonus
        self._success = self._success or solved
        terminated = bool(solved)
        truncated = bool(self._step_count >= self.max_steps)

        observation = self._observation(delta_y, sq_x)
        info = self._info(agent_rewards)
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the current state.

        Returns
        -------
        numpy.ndarray or None
            ``(H, W, 3)`` ``uint8`` frame in ``"rgb_array"`` mode, else ``None``.
        """
        if self.render_mode is None:
            if not self._warned_no_render_mode:
                warnings.warn(
                    "render() called without a render_mode; create the environment with "
                    "render_mode='rgb_array' or 'human'",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_no_render_mode = True
            return None
        self._require_reset()
        if self._renderer is None:
            from env_lib.consensus_env.rendering import ConsensusRenderer

            self._renderer = ConsensusRenderer(
                self.render_mode,
                n_agents=self.n_agents,
                arena_size=self.arena_size,
                task=self.task,
                formation_shape=self.formation_shape,
                topology=self.topology,
                dynamics=self.dynamics,
                max_steps=self.max_steps,
                tolerance=self.tolerance,
                comm_radius=self.comm_radius if self._dynamic_graph else None,
                fps=self.metadata["render_fps"],
            )
        centroid = (self._pos - self._offsets).mean(axis=0)
        return self._renderer.render(
            positions=self._pos,
            adjacency=self._adj,
            targets=centroid + self._offsets,
            centroid=centroid,
            step=self._step_count,
            error=self._error,
            lambda2=self._lambda2,
            success=self._success,
        )

    def close(self) -> None:
        """Release the rendering resources (idempotent)."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Baseline controller
    # ------------------------------------------------------------------
    def laplacian_policy(self, gain: float = 1.0, velocity_gain: float | None = None) -> np.ndarray:
        """Classic distributed consensus / formation feedback.

        ``u_i = -gain * sum_{j in N_i} ((x_i - d_i) - (x_j - d_j))``, i.e.
        ``u = -gain * L (x - d)``. For the double integrator the velocity feedback
        ``-velocity_gain * v_i`` is added, which (together with the physical
        ``damping``) brings the agents to rest. The result is clipped to the
        action bounds. The feedback uses the full neighbour sets, even if the
        observation is truncated by ``max_neighbors``.

        The discrete-time single-integrator loop is stable when
        ``gain * dt * lambda_max(L) < 2``; a warning is issued once otherwise.

        Parameters
        ----------
        gain:
            Position feedback gain (``> 0``).
        velocity_gain:
            Velocity feedback gain of the double integrator (``>= 0``). Defaults
            to ``max(0, sqrt(gain) - damping)``, so that the total damping
            ``damping + velocity_gain`` is ``sqrt(gain)`` (critical damping for
            the Laplacian eigenvalue 1/4; lighter damping than that of the
            faster modes keeps the saturated transient short). Ignored for the
            single integrator.

        Returns
        -------
        numpy.ndarray
            Action of shape ``(n_agents, 2)``, dtype ``float32``.
        """
        self._require_reset()
        gain = _check_float("gain", gain, minimum=0.0, strict=True)
        shifted = self._pos - self._offsets
        feedback = self._deg[:, None] * shifted - self._adj_f @ shifted
        control = -gain * feedback
        if self.dynamics == "double":
            if velocity_gain is None:
                velocity_gain = max(0.0, math.sqrt(gain) - self.damping)
            velocity_gain = _check_float("velocity_gain", velocity_gain, minimum=0.0)
            control -= velocity_gain * self._vel
        elif not self._warned_policy_gain:
            # Gershgorin bound lambda_max(L) <= 2 * max degree.
            lambda_bound = 2.0 * float(self._deg.max())
            if gain * self.dt * lambda_bound >= 2.0:
                lambda_max = float(np.linalg.eigvalsh(np.diag(self._deg) - self._adj_f)[-1])
                if gain * self.dt * lambda_max >= 2.0:
                    warnings.warn(
                        f"laplacian_policy(gain={gain}) is unstable for dt={self.dt} and "
                        f"lambda_max(L)={lambda_max:.2f} (needs gain * dt * lambda_max < 2); "
                        "lower the gain",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_policy_gain = True
        return np.clip(control, -self.max_control, self.max_control).astype(np.float32)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_reset(self) -> None:
        if self._pos is None:
            raise ResetNeededError("Call reset() before using the environment")

    def _state_option(self, name: str, value: Any) -> np.ndarray:
        array = np.array(value, dtype=np.float64)
        if array.shape != (self.n_agents, 2):
            raise ValueError(
                f"options[{name!r}] must have shape {(self.n_agents, 2)}, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"options[{name!r}] must be finite")
        return array

    def _validate_action(self, action: Any) -> np.ndarray:
        control = np.asarray(action, dtype=np.float64)
        if control.size != 2 * self.n_agents:
            raise ValueError(f"action must have shape {(self.n_agents, 2)}, got {np.shape(action)}")
        control = control.reshape(self.n_agents, 2)
        if not np.all(np.isfinite(control)):
            raise ValueError("action contains NaN or inf")
        return np.clip(control, -self.max_control, self.max_control)

    def _sample_positions(self) -> np.ndarray:
        n, limit = self.n_agents, _INIT_SPREAD * self.arena_size
        rng = self.np_random
        if not self._dynamic_graph:
            return rng.uniform(-limit, limit, size=(n, 2))
        # Uniform positions conditioned on a connected initial disk graph.
        radius_sq = self.comm_radius * self.comm_radius
        for _ in range(_INIT_MAX_BATCHES):
            candidates = rng.uniform(-limit, limit, size=(_INIT_BATCH, n, 2))
            delta = candidates[:, None, :, :] - candidates[:, :, None, :]
            connected = _batched_connected(np.einsum("bijk,bijk->bij", delta, delta) <= radius_sq)
            if connected.any():
                return candidates[int(np.argmax(connected))]
        if not self._warned_sequential_init:
            warnings.warn(
                f"comm_radius={self.comm_radius} is small for arena_size={self.arena_size} and "
                f"n_agents={n}: no connected uniform placement found, falling back to sequential "
                "placement within communication range",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warned_sequential_init = True
        return self._sequential_positions(limit)

    def _sequential_positions(self, limit: float) -> np.ndarray:
        rng, n = self.np_random, self.n_agents
        positions = np.empty((n, 2), dtype=np.float64)
        positions[0] = rng.uniform(-limit, limit, size=2)
        for i in range(1, n):  # rare fallback at reset time only
            parent = positions[rng.integers(i)]
            radius = 0.95 * self.comm_radius * math.sqrt(rng.random())
            angle = 2.0 * math.pi * rng.random()
            step = radius * np.array([math.cos(angle), math.sin(angle)])
            # Projection onto the box is non-expansive, so the parent stays in range.
            positions[i] = np.clip(parent + step, -limit, limit)
        return positions[rng.permutation(n)]

    def _pairwise(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pairwise differences ``[i, j] -> x_j - x_i`` and ``y_j - y_i`` with squared norms."""
        delta_x = self._pos[None, :, :] - self._pos[:, None, :]
        sq_x = np.einsum("ijk,ijk->ij", delta_x, delta_x)
        if self.task == "consensus":
            return delta_x, sq_x, delta_x, sq_x
        delta_y = delta_x - self._offset_delta
        sq_y = np.einsum("ijk,ijk->ij", delta_y, delta_y)
        return delta_x, sq_x, delta_y, sq_y

    def _update_graph(self, sq_x: np.ndarray) -> None:
        if self._dynamic_graph:
            adj = sq_x <= self.comm_radius * self.comm_radius
            np.fill_diagonal(adj, False)
            if self._step_count > 0 and np.array_equal(adj, self._adj):
                return  # unchanged graph: keep the Laplacian quantities
            self._adj = adj
            self._adj_f = adj.astype(np.float64)
            self._deg = self._adj_f.sum(axis=1)
            laplacian = np.diag(self._deg) - self._adj_f
            self._lambda2 = float(max(np.linalg.eigvalsh(laplacian)[1], 0.0))
        elif self._step_count == 0:
            self._adj = self._static_adjacency
            self._adj_f = self._adj.astype(np.float64)
            self._deg = self._adj_f.sum(axis=1)
            self._lambda2 = self._static_lambda2

    def _compute_error(self) -> float:
        shifted = self._pos - self._offsets
        centred = shifted - shifted.mean(axis=0)
        return float(np.einsum("ij,ij->", centred, centred) / self.n_agents)

    def _observation(self, delta_y: np.ndarray, sq_x: np.ndarray) -> np.ndarray:
        n, slots = self.n_agents, self._visible_slots
        obs = np.zeros((n, self.obs_dim), dtype=np.float32)
        obs[:, 0:2] = self._pos
        obs[:, 2:4] = self._vel
        obs[:, 4:6] = self._offsets
        # Neighbours first (sorted by distance), non-neighbours (inf) last.
        keys = np.where(self._adj, sq_x, np.inf)
        order = np.argsort(keys, axis=1, kind="stable")[:, :slots]
        mask = self._adj[self._rows, order]
        weight = mask[..., None]
        block = np.zeros((n, self.max_neighbors, self.SLOT_DIM), dtype=np.float32)
        block[:, :slots, 0:2] = np.where(weight, delta_y[self._rows, order], 0.0)
        block[:, :slots, 2:4] = np.where(weight, self._vel[order] - self._vel[:, None, :], 0.0)
        block[:, :slots, 4] = mask
        obs[:, 6:] = block.reshape(n, -1)
        return obs

    def _info(self, agent_rewards: np.ndarray) -> dict[str, Any]:
        return {
            "agent_rewards": np.asarray(agent_rewards, dtype=np.float64),
            "error": self._error,
            "algebraic_connectivity": self._lambda2,
            "adjacency": self._adj.copy(),
            "success": bool(self._success),
            "step": self._step_count,
        }
