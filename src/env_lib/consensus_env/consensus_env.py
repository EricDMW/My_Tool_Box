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

from env_lib.consensus_env._core import ConsensusKernel
from env_lib.consensus_env._core import batched_connected as _batched_connected
from env_lib.errors import ResetNeededError
from env_lib.utils.rendering import state_without_renderer, validate_render_mode

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
_PROXIMITY_DEFAULT_SLOTS = 6


# ---------------------------------------------------------------------------
# Graph and formation helpers
# ---------------------------------------------------------------------------
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


def _parse_config(
    *,
    n_agents: Any,
    task: Any,
    topology: Any,
    dynamics: Any,
    dt: Any,
    max_steps: Any,
    arena_size: Any,
    max_control: Any,
    control_cost: Any,
    noise_std: Any,
    tolerance: Any,
    success_bonus: Any,
    formation_shape: Any,
    formation_radius: Any,
    comm_radius: Any,
    edge_probability: Any,
    graph_seed: Any,
    max_neighbors: Any,
    damping: Any,
) -> dict[str, Any]:
    """Validate the constructor arguments shared by the single and vector environments.

    Returns
    -------
    dict
        Validated values (``max_neighbors`` stays ``None`` when not given) plus
        the formation ``offsets`` of shape ``(n_agents, 2)``.
    """
    cfg: dict[str, Any] = {
        "n_agents": _check_int("n_agents", n_agents, 2),
        "task": _check_choice("task", task, TASKS),
        "topology": _check_choice("topology", topology, TOPOLOGIES),
        "dynamics": _check_choice("dynamics", dynamics, DYNAMICS),
        "dt": _check_float("dt", dt, minimum=0.0, strict=True),
        "max_steps": _check_int("max_steps", max_steps, 1),
        "arena_size": _check_float("arena_size", arena_size, minimum=0.0, strict=True),
        "max_control": _check_float("max_control", max_control, minimum=0.0, strict=True),
        "control_cost": _check_float("control_cost", control_cost, minimum=0.0),
        "noise_std": _check_float("noise_std", noise_std, minimum=0.0),
        "tolerance": _check_float("tolerance", tolerance, minimum=0.0, strict=True),
        "success_bonus": _check_float("success_bonus", success_bonus),
        "formation_shape": _check_choice("formation_shape", formation_shape, FORMATION_SHAPES),
        "formation_radius": _check_float(
            "formation_radius", formation_radius, minimum=0.0, strict=True
        ),
        "comm_radius": _check_float("comm_radius", comm_radius, minimum=0.0, strict=True),
        "edge_probability": _check_float(
            "edge_probability", edge_probability, minimum=0.0, strict=True
        ),
    }
    if cfg["edge_probability"] > 1.0:
        raise ValueError(f"edge_probability must be in (0, 1], got {edge_probability}")
    if graph_seed is not None:
        graph_seed = _check_int("graph_seed", graph_seed, 0)
    cfg["graph_seed"] = graph_seed
    cfg["damping"] = _check_float("damping", damping, minimum=0.0)
    if max_neighbors is not None:
        max_neighbors = _check_int("max_neighbors", max_neighbors, 1)
    cfg["max_neighbors"] = max_neighbors

    n = cfg["n_agents"]
    # Formation offsets d_i (zero for consensus).
    if cfg["task"] == "formation":
        offsets = make_formation(cfg["formation_shape"], n, cfg["formation_radius"])
        span = offsets.max(axis=0) - offsets.min(axis=0)
        if np.any(span > 2.0 * cfg["arena_size"]):
            raise ValueError(
                f"formation {cfg['formation_shape']!r} with formation_radius="
                f"{cfg['formation_radius']} spans {span.max():.2f}, which does not fit in the "
                f"arena of width {2.0 * cfg['arena_size']}; reduce formation_radius or "
                "increase arena_size"
            )
    else:
        offsets = np.zeros((n, 2), dtype=np.float64)
    cfg["offsets"] = offsets
    return cfg


def _make_kernel(cfg: dict[str, Any], max_neighbors: int) -> ConsensusKernel:
    """Array kernel for a validated configuration (see :func:`_parse_config`)."""
    return ConsensusKernel(
        n_agents=cfg["n_agents"],
        dynamics=cfg["dynamics"],
        dt=cfg["dt"],
        arena_size=cfg["arena_size"],
        max_control=cfg["max_control"],
        control_cost=cfg["control_cost"],
        damping=cfg["damping"],
        noise_std=cfg["noise_std"],
        offsets=cfg["offsets"],
        formation=cfg["task"] == "formation",
        max_neighbors=max_neighbors,
        comm_radius=cfg["comm_radius"],
    )


#: Keys of ``reset(options=...)`` understood by the consensus environments.
RESET_OPTION_KEYS: tuple[str, ...] = ("positions", "velocities")


def _known_reset_options(owner: str, options: Any, stacklevel: int) -> dict[str, Any]:
    """Known ``reset`` options; unknown keys are ignored with a ``UserWarning``.

    Unknown keys are tolerated (not an error) because generic tooling, e.g.
    PettingZoo's API test, passes arbitrary options. ``stacklevel`` is that
    of the warning as seen from the caller of this helper.
    """
    options = dict(options or {})
    unknown = sorted(set(options) - set(RESET_OPTION_KEYS))
    if unknown:
        warnings.warn(
            f"{owner}.reset(): ignoring unknown reset option(s) {unknown}; supported: "
            f"{list(RESET_OPTION_KEYS)}",
            UserWarning,
            stacklevel=stacklevel + 1,
        )
        for key in unknown:
            del options[key]
    return options


def _check_state_option(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Validate a ``reset(options=...)`` state array (float64 copy)."""
    array = np.array(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"options[{name!r}] must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"options[{name!r}] must be finite")
    return array


def _sequential_init_warning(cfg: dict[str, Any], stacklevel: int) -> None:
    warnings.warn(
        f"comm_radius={cfg['comm_radius']} is small for arena_size={cfg['arena_size']} and "
        f"n_agents={cfg['n_agents']}: no connected uniform placement found, falling back to "
        "sequential placement within communication range",
        RuntimeWarning,
        stacklevel=stacklevel,
    )


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
    SLOT_DIM: int = ConsensusKernel.SLOT_DIM
    #: Columns of one neighbour slot.
    neighbor_slot_layout: dict[str, slice] = {
        "rel_position": slice(0, 2),
        "rel_velocity": slice(2, 4),
        "mask": slice(4, 5),
    }

    def __getstate__(self) -> dict[str, Any]:
        # Pickle and deep-copy without the renderer (rebuilt on the next render()).
        return state_without_renderer(self)

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
        cfg = _parse_config(
            n_agents=n_agents,
            task=task,
            topology=topology,
            dynamics=dynamics,
            dt=dt,
            max_steps=max_steps,
            arena_size=arena_size,
            max_control=max_control,
            control_cost=control_cost,
            noise_std=noise_std,
            tolerance=tolerance,
            success_bonus=success_bonus,
            formation_shape=formation_shape,
            formation_radius=formation_radius,
            comm_radius=comm_radius,
            edge_probability=edge_probability,
            graph_seed=graph_seed,
            max_neighbors=max_neighbors,
            damping=damping,
        )
        self._cfg = cfg
        self.n_agents: int = cfg["n_agents"]
        self.task: str = cfg["task"]
        self.topology: str = cfg["topology"]
        self.dynamics: str = cfg["dynamics"]
        self.dt: float = cfg["dt"]
        self.max_steps: int = cfg["max_steps"]
        self.arena_size: float = cfg["arena_size"]
        self.max_control: float = cfg["max_control"]
        self.control_cost: float = cfg["control_cost"]
        self.noise_std: float = cfg["noise_std"]
        self.tolerance: float = cfg["tolerance"]
        self.success_bonus: float = cfg["success_bonus"]
        self.formation_shape: str = cfg["formation_shape"]
        self.formation_radius: float = cfg["formation_radius"]
        self.comm_radius: float = cfg["comm_radius"]
        self.edge_probability: float = cfg["edge_probability"]
        self.graph_seed: int | None = cfg["graph_seed"]
        self.damping: float = cfg["damping"]
        self.render_mode = render_mode

        n = self.n_agents
        self._offsets: np.ndarray = cfg["offsets"]

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
        self.max_neighbors: int = (
            default_slots if cfg["max_neighbors"] is None else cfg["max_neighbors"]
        )
        # Batch-first array kernels, shared with ConsensusVectorEnv (called with B = 1).
        self._kernel = _make_kernel(cfg, self.max_neighbors)

        self.obs_dim = self._kernel.obs_dim
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
            to the arena, velocities default to zero). This is also how a copy
            of :class:`~env_lib.consensus_env.vector.ConsensusVectorEnv` is
            reproduced by a single environment. Unknown keys are ignored with
            a ``UserWarning``.

        Returns
        -------
        observation, info
        """
        super().reset(seed=seed)
        options = _known_reset_options(type(self).__name__, options, stacklevel=2)

        n = self.n_agents
        if options.get("positions") is not None:
            positions = self._state_option("positions", options["positions"])
            positions = np.clip(positions, -self.arena_size, self.arena_size)
        else:
            positions = self._kernel.sample_positions(
                self.np_random, 1, self._dynamic_graph, self._on_sequential_init
            )[0]
        if options.get("velocities") is not None:
            velocities = self._state_option("velocities", options["velocities"])
        else:
            velocities = np.zeros((n, 2), dtype=np.float64)

        self._pos = positions
        self._vel = velocities
        self._step_count = 0
        self._success = False

        rel, sq_x, _ = self._kernel.pairwise(positions[None])
        self._update_graph(sq_x)
        self._error = float(self._kernel.task_error(positions[None])[0])
        observation = self._observation(rel, sq_x)
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
        control = self._validate_action(action)[None]
        kernel = self._kernel
        noise = None
        if kernel.noise_scale > 0.0:
            noise = self.np_random.standard_normal(control.shape)
        pos, vel = kernel.integrate(self._pos[None], self._vel[None], control, noise)
        self._pos, self._vel = pos[0], vel[0]
        self._step_count += 1

        rel, sq_x, sq_y = kernel.pairwise(pos)
        self._update_graph(sq_x)
        agent_rewards = kernel.agent_rewards(self._adj_f[None], self._deg[None], sq_y, control)[0]

        self._error = float(kernel.task_error(pos)[0])
        solved = self._error < self.tolerance
        reward = float(np.add.reduce(agent_rewards) / self.n_agents)  # the mean
        if solved and not self._success:
            reward += self.success_bonus
        self._success = self._success or solved
        terminated = bool(solved)
        truncated = bool(self._step_count >= self.max_steps)

        observation = self._observation(rel, sq_x)
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
            self._renderer = _make_renderer(self._cfg, self.render_mode, self.metadata)
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
        gain, velocity_gain = _policy_gains(self._cfg, gain, velocity_gain)
        if self.dynamics == "single" and not self._warned_policy_gain:
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
        return self._kernel.feedback(
            self._pos[None],
            self._vel[None],
            self._adj_f[None],
            self._deg[None],
            gain,
            velocity_gain,
        )[0]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_reset(self) -> None:
        if self._pos is None:
            raise ResetNeededError("Call reset() before using the environment")

    def _state_option(self, name: str, value: Any) -> np.ndarray:
        return _check_state_option(name, value, (self.n_agents, 2))

    def _validate_action(self, action: Any) -> np.ndarray:
        control = np.asarray(action, dtype=np.float64)
        if control.size != 2 * self.n_agents:
            raise ValueError(f"action must have shape {(self.n_agents, 2)}, got {np.shape(action)}")
        control = control.reshape(self.n_agents, 2)
        if not np.all(np.isfinite(control)):
            raise ValueError("action contains NaN or inf")
        return np.clip(control, -self.max_control, self.max_control)

    def _on_sequential_init(self) -> None:
        if not self._warned_sequential_init:
            _sequential_init_warning(self._cfg, stacklevel=5)
            self._warned_sequential_init = True

    def _update_graph(self, sq_x: np.ndarray) -> None:
        """Refresh the graph quantities from the squared distances ``(1, n, n)``."""
        if self._dynamic_graph:
            adj = self._kernel.proximity(sq_x)[0]
            if self._step_count > 0 and np.array_equal(adj, self._adj):
                return  # unchanged graph: keep the Laplacian quantities
            adj_f, deg, lambda2 = self._kernel.graph_quantities(adj[None])
            self._adj, self._adj_f, self._deg = adj, adj_f[0], deg[0]
            self._lambda2 = float(lambda2[0])
        elif self._step_count == 0:
            self._adj = self._static_adjacency
            self._adj_f = self._adj.astype(np.float64)
            self._deg = self._adj_f.sum(axis=1)
            self._lambda2 = self._static_lambda2

    def _observation(self, rel: tuple[np.ndarray, np.ndarray], sq_x: np.ndarray) -> np.ndarray:
        return self._kernel.observe(
            self._pos[None], self._vel[None], self._adj[None], sq_x, rel, self._deg[None]
        )[0]

    def _info(self, agent_rewards: np.ndarray) -> dict[str, Any]:
        return {
            "agent_rewards": np.asarray(agent_rewards, dtype=np.float64),
            "error": self._error,
            "algebraic_connectivity": self._lambda2,
            "adjacency": self._adj.copy(),
            "success": bool(self._success),
            "step": self._step_count,
        }


def _policy_gains(cfg: dict[str, Any], gain: Any, velocity_gain: Any) -> tuple[float, float | None]:
    """Validate the gains of the Laplacian baseline (``velocity_gain`` defaulted)."""
    gain = _check_float("gain", gain, minimum=0.0, strict=True)
    if cfg["dynamics"] != "double":
        return gain, None
    if velocity_gain is None:
        velocity_gain = max(0.0, math.sqrt(gain) - cfg["damping"])
    return gain, _check_float("velocity_gain", velocity_gain, minimum=0.0)


def _make_renderer(cfg: dict[str, Any], render_mode: str, metadata: dict[str, Any]):
    """Create the dashboard renderer (imported lazily)."""
    from env_lib.consensus_env.rendering import ConsensusRenderer

    return ConsensusRenderer(
        render_mode,
        n_agents=cfg["n_agents"],
        arena_size=cfg["arena_size"],
        task=cfg["task"],
        formation_shape=cfg["formation_shape"],
        topology=cfg["topology"],
        dynamics=cfg["dynamics"],
        max_steps=cfg["max_steps"],
        tolerance=cfg["tolerance"],
        comm_radius=cfg["comm_radius"] if cfg["topology"] == "proximity" else None,
        fps=metadata["render_fps"],
    )
