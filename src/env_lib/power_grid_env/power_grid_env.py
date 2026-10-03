"""Frequency control of a networked power system (``PowerGrid-v0``).

``n`` buses -- synchronous generators or inverter-based resources, one agent
each -- are coupled through the lossless transmission lines of a Kron-reduced
network. Their rotor (or voltage) angles and frequencies follow the classical
swing equations. Load disturbances push the frequencies away from nominal and
excite electro-mechanical oscillations; every agent injects a bounded fast
frequency response (storage, inverter headroom) to arrest and damp them.

The module provides

* :class:`PowerGridEnv` -- the single Gymnasium environment,
* :class:`PowerGridVectorEnv` -- a natively batched vector environment that
  shares the same batch-first simulation core,
* :func:`droop_policy` -- the decentralised droop-control baseline, a pure
  function of the observation.

All dynamics are written once for a leading batch dimension ``B`` (``B = 1``
for the single environment), so a seeded single environment reproduces copy 0
of the vector environment when both start from the same state.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import scipy.sparse as sp
from gymnasium import spaces

from env_lib.errors import ResetNeededError
from env_lib.utils import graphs
from env_lib.utils.render_modes import state_without_renderer, validate_render_mode
from env_lib.utils.vector import BatchedVectorEnv

__all__ = [
    "DEFAULT_DROOP_GAIN",
    "OBSERVATION_FEATURES",
    "OBSERVATION_SCALE",
    "TOPOLOGIES",
    "PowerGridEnv",
    "PowerGridVectorEnv",
    "droop_policy",
]

logger = logging.getLogger(__name__)

#: Network topologies accepted by the ``topology`` argument (see :func:`env_lib.utils.graphs.make_graph`).
TOPOLOGIES: tuple[str, ...] = graphs.TOPOLOGIES

#: Per-agent observation features, in column order.
OBSERVATION_FEATURES: tuple[str, ...] = (
    "omega",
    "rocof",
    "disturbance",
    "power_flow",
    "action",
    "neighbour_omega_mean",
    "neighbour_omega_max",
    "neighbour_action_mean",
    "capacity",
    "inertia",
)

#: Fixed factor applied to each feature (observation = physical value * scale).
#: Frequencies are in rad/s, RoCoF in rad/s^2, powers in per unit, inertia
#: constants in seconds. The factors do not depend on constructor arguments.
OBSERVATION_SCALE: dict[str, float] = {
    "omega": 1.0,
    "rocof": 0.1,
    "disturbance": 10.0,
    "power_flow": 10.0,
    "action": 10.0,
    "neighbour_omega_mean": 1.0,
    "neighbour_omega_max": 1.0,
    "neighbour_action_mean": 10.0,
    "capacity": 10.0,
    "inertia": 0.1,
}

#: Default gain of :func:`droop_policy`, in per unit power per rad/s.
DEFAULT_DROOP_GAIN: float = 0.25

OBS_DIM: int = len(OBSERVATION_FEATURES)
_COL = {name: index for index, name in enumerate(OBSERVATION_FEATURES)}
_N_DYNAMIC = 8  # measured features first, then the static local parameters
_DYNAMIC_SCALE = np.array([OBSERVATION_SCALE[k] for k in OBSERVATION_FEATURES[:_N_DYNAMIC]])
_RESET_OPTIONS = frozenset({"theta", "omega", "disturbance"})
_TOPOLOGY_KWARGS = frozenset({"edge_probability", "neighbors", "rewire_probability", "radius"})
_RK4_WARN = 2.5  # h * omega_bound above which RK4 is close to its stability limit (2.83)
_DENSE_SOLVE_MAX = 256
_DENSE_WORK_MAX = 65536  # 2 B n^2 below which the coupling product uses a dense matrix


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
    name: str,
    value: Any,
    *,
    minimum: float = -math.inf,
    strict: bool = False,
    maximum: float = math.inf,
    strict_max: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{name} must be {'>' if strict else '>='} {minimum}, got {value}")
    if value > maximum or (strict_max and value == maximum):
        raise ValueError(f"{name} must be {'<' if strict_max else '<='} {maximum}, got {value}")
    return value


def _check_range(name: str, value: Any, *, minimum: float, strict: bool) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError(f"{name} must be a (low, high) pair, got {value!r}")
    low = _check_float(f"{name}[0]", value[0], minimum=minimum, strict=strict)
    high = _check_float(f"{name}[1]", value[1], minimum=minimum, strict=strict)
    if high < low:
        raise ValueError(f"{name} must satisfy low <= high, got {value!r}")
    return low, high


@dataclass(frozen=True)
class _Config:
    """Validated constructor arguments shared by the single and the vector environment."""

    n_buses: int
    topology: str
    topology_kwargs: dict[str, Any]
    network_seed: int | None
    randomize_network: bool
    inertia_range: tuple[float, float]
    damping_range: tuple[float, float]
    susceptance_range: tuple[float, float]
    line_loading: float
    nominal_frequency: float
    u_max: np.ndarray
    dt: float
    substeps: int
    max_steps: int
    initial_disturbances: int
    disturbance_rate: float
    disturbance_magnitude: tuple[float, float]
    load_increase_prob: float
    noise_std: float
    noise_tau: float
    frequency_limit: float
    w_freq: float
    w_rocof: float
    w_u: float
    trip_penalty: float
    neighbourhood: int
    omega_limit: float = field(init=False)
    omega_nominal: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "omega_limit", 2.0 * math.pi * self.frequency_limit)
        object.__setattr__(self, "omega_nominal", 2.0 * math.pi * self.nominal_frequency)


def _make_config(
    *,
    n_buses: int = 16,
    topology: str = "small_world",
    topology_kwargs: Mapping[str, Any] | None = None,
    network_seed: int | None = 0,
    randomize_network: bool = False,
    inertia_range: tuple[float, float] = (2.0, 8.0),
    damping_range: tuple[float, float] = (2.0, 6.0),
    susceptance_range: tuple[float, float] = (0.3, 0.8),
    line_loading: float = 0.5,
    nominal_frequency: float = 50.0,
    u_max: float | Sequence[float] | np.ndarray = 0.1,
    dt: float = 0.05,
    substeps: int = 2,
    max_steps: int = 200,
    initial_disturbances: int = 1,
    disturbance_rate: float = 0.1,
    disturbance_magnitude: tuple[float, float] = (0.05, 0.2),
    load_increase_prob: float = 0.5,
    noise_std: float = 0.0,
    noise_tau: float = 1.0,
    frequency_limit: float = 0.5,
    w_freq: float = 1.0,
    w_rocof: float = 0.1,
    w_u: float = 10.0,
    trip_penalty: float = 100.0,
    neighbourhood: int = 1,
) -> _Config:
    n = _check_int("n_buses", n_buses, 2)
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology must be one of {TOPOLOGIES}, got {topology!r}")
    kwargs = dict(topology_kwargs or {})
    unknown = set(kwargs) - _TOPOLOGY_KWARGS
    if unknown:
        raise ValueError(
            f"Unknown topology_kwargs {sorted(unknown)}; allowed: {sorted(_TOPOLOGY_KWARGS)}"
        )
    if topology == "erdos_renyi" and "edge_probability" not in kwargs:
        # Mean degree grows like 2 ln(n): connected with high probability, sparse for large n.
        kwargs["edge_probability"] = min(1.0, 2.0 * math.log(n) / n)
    if network_seed is not None:
        network_seed = _check_int("network_seed", network_seed, 0)
    if not isinstance(randomize_network, (bool, np.bool_)):
        raise TypeError(f"randomize_network must be a bool, got {type(randomize_network).__name__}")

    u_max_array = np.array(u_max, dtype=np.float64, copy=True)
    if u_max_array.ndim == 0:
        u_max_array = np.full(n, float(u_max_array))
    if u_max_array.shape != (n,):
        raise ValueError(f"u_max must be a scalar or have shape ({n},), got {u_max_array.shape}")
    if not np.all(np.isfinite(u_max_array)) or np.any(u_max_array <= 0.0):
        raise ValueError("u_max must be finite and > 0")
    u_max_array.setflags(write=False)

    initial = _check_int("initial_disturbances", initial_disturbances, 0)
    if initial > n:
        raise ValueError(f"initial_disturbances must be <= n_buses={n}, got {initial}")

    config = _Config(
        n_buses=n,
        topology=topology,
        topology_kwargs=kwargs,
        network_seed=network_seed,
        randomize_network=bool(randomize_network),
        inertia_range=_check_range("inertia_range", inertia_range, minimum=0.0, strict=True),
        damping_range=_check_range("damping_range", damping_range, minimum=0.0, strict=False),
        susceptance_range=_check_range(
            "susceptance_range", susceptance_range, minimum=0.0, strict=True
        ),
        line_loading=_check_float(
            "line_loading", line_loading, minimum=0.0, maximum=1.0, strict_max=True
        ),
        nominal_frequency=_check_float(
            "nominal_frequency", nominal_frequency, minimum=0.0, strict=True
        ),
        u_max=u_max_array,
        dt=_check_float("dt", dt, minimum=0.0, strict=True),
        substeps=_check_int("substeps", substeps, 1),
        max_steps=_check_int("max_steps", max_steps, 1),
        initial_disturbances=initial,
        disturbance_rate=_check_float("disturbance_rate", disturbance_rate, minimum=0.0),
        disturbance_magnitude=_check_range(
            "disturbance_magnitude", disturbance_magnitude, minimum=0.0, strict=False
        ),
        load_increase_prob=_check_float(
            "load_increase_prob", load_increase_prob, minimum=0.0, maximum=1.0
        ),
        noise_std=_check_float("noise_std", noise_std, minimum=0.0),
        noise_tau=_check_float("noise_tau", noise_tau, minimum=0.0, strict=True),
        frequency_limit=_check_float("frequency_limit", frequency_limit, minimum=0.0, strict=True),
        w_freq=_check_float("w_freq", w_freq, minimum=0.0),
        w_rocof=_check_float("w_rocof", w_rocof, minimum=0.0),
        w_u=_check_float("w_u", w_u, minimum=0.0),
        trip_penalty=_check_float("trip_penalty", trip_penalty, minimum=0.0),
        neighbourhood=_check_int("neighbourhood", neighbourhood, 1),
    )
    return config


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
@dataclass
class _Network:
    """One sampled power network and its synchronous operating point.

    Attributes
    ----------
    adjacency:
        Boolean ``(n, n)`` line graph.
    edges, susceptance:
        Lines ``(i, j)`` with ``i < j`` as an ``(E, 2)`` array and their
        susceptances ``B_ij`` (per unit), shape ``(E,)``.
    inertia_constant, inertia, damping:
        ``H_i`` (s), ``M_i = 2 H_i / omega_0`` and ``D_i = d_i / omega_0``.
    injection, theta_star:
        Nominal net injections ``P_i^0`` (sum zero) and the equilibrium angles.
    omega_bound:
        Gershgorin bound on the fastest electro-mechanical mode (rad/s).
    native_positions:
        Node coordinates from :func:`graphs.make_graph` (used for plotting).
    weights:
        Weighted adjacency ``W = [B_ij]`` as a CSR matrix.
    neighbour_index, neighbour_mask:
        Padded ``kappa``-hop neighbourhood table ``(n, K)`` and its 0/1 mask.
    """

    adjacency: np.ndarray
    edges: np.ndarray
    susceptance: np.ndarray
    inertia_constant: np.ndarray
    inertia: np.ndarray
    damping: np.ndarray
    injection: np.ndarray
    theta_star: np.ndarray
    omega_bound: float
    native_positions: np.ndarray
    weights: sp.csr_matrix
    neighbour_index: np.ndarray
    neighbour_mask: np.ndarray


def _weighted_adjacency(n: int, edges: np.ndarray, susceptance: np.ndarray) -> sp.csr_matrix:
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([susceptance, susceptance])
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    matrix.sort_indices()
    return matrix


def _dc_angles(weights: sp.csr_matrix, injection: np.ndarray) -> np.ndarray:
    """Solve the linearised (DC) flow equations ``L_B theta = P`` with ``mean(theta) = 0``.

    Node 0 is grounded; the reduced Laplacian of a connected graph is
    non-singular and ``sum(P) = 0`` makes the grounded solution exact.
    """
    n = weights.shape[0]
    theta = np.zeros(n)
    if n <= _DENSE_SOLVE_MAX:
        laplacian = graphs.laplacian(weights.toarray())
        theta[1:] = np.linalg.solve(laplacian[1:, 1:], injection[1:])
    else:
        from scipy.sparse.linalg import spsolve

        laplacian = sp.diags(np.asarray(weights.sum(axis=1)).ravel()) - weights
        theta[1:] = spsolve(sp.csc_matrix(laplacian[1:, 1:]), injection[1:])
    return theta - theta.mean()


def _neighbour_table(adjacency: np.ndarray, hops: int) -> tuple[np.ndarray, np.ndarray]:
    """Padded neighbour indices ``(n, K)`` (own index as padding) and a 0/1 mask."""
    reach = graphs.k_hop_adjacency(adjacency, hops)
    n = reach.shape[0]
    width = max(int(reach.sum(axis=1).max()), 1)
    order = np.argsort(~reach, axis=1, kind="stable")[:, :width]
    mask = np.take_along_axis(reach, order, axis=1)
    index = np.where(mask, order, np.arange(n)[:, None])
    return index, mask.astype(np.float64)


def _sample_network(config: _Config, rng: np.random.Generator) -> _Network:
    """Sample topology, physical parameters and the synchronous operating point.

    The draws are, in order: the graph (:func:`graphs.make_graph`), ``H_i``,
    ``d_i``, the line susceptances and the raw injection pattern ``z``.
    """
    n = config.n_buses
    adjacency, native_positions = graphs.make_graph(
        config.topology, n, rng=rng, return_positions=True, **config.topology_kwargs
    )
    edges = graphs.edge_list(adjacency)
    inertia_constant = rng.uniform(*config.inertia_range, size=n)
    damping_pu = rng.uniform(*config.damping_range, size=n)
    susceptance = rng.uniform(*config.susceptance_range, size=edges.shape[0])
    pattern = rng.uniform(-1.0, 1.0, size=n)
    pattern -= pattern.mean()

    # Operating point: DC angles of the raw pattern, scaled so that the most
    # loaded line has |sin(theta_i - theta_j)| = line_loading. The nominal
    # injections are then the exact AC flows at these angles, so theta_star is
    # an exact (and, since all |theta_i - theta_j| < pi/2, stable) equilibrium.
    weights = _weighted_adjacency(n, edges, susceptance)
    theta = _dc_angles(weights, pattern)
    spread = np.abs(theta[edges[:, 0]] - theta[edges[:, 1]]).max()
    if spread > 0.0 and config.line_loading > 0.0:
        theta_star = theta * (math.asin(config.line_loading) / spread)
    else:
        theta_star = np.zeros(n)
    injection = _flows(weights, theta_star[:, None])[0][:, 0]

    omega0 = config.omega_nominal
    inertia = 2.0 * inertia_constant / omega0
    degree = np.asarray(weights.sum(axis=1)).ravel()
    omega_bound = float(np.sqrt(np.max(2.0 * degree / inertia)))
    index, mask = _neighbour_table(adjacency, config.neighbourhood)
    logger.debug(
        "sampled %s network: n=%d, lines=%d, fastest-mode bound %.3g rad/s",
        config.topology,
        n,
        edges.shape[0],
        omega_bound,
    )
    return _Network(
        adjacency=adjacency,
        edges=edges,
        susceptance=susceptance,
        inertia_constant=inertia_constant,
        inertia=inertia,
        damping=damping_pu / omega0,
        injection=injection,
        theta_star=theta_star,
        omega_bound=omega_bound,
        native_positions=np.asarray(native_positions, dtype=np.float64),
        weights=weights,
        neighbour_index=index,
        neighbour_mask=mask,
    )


def _flows(
    coupling: np.ndarray | sp.csr_matrix, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Electrical power leaving every bus, ``P^e_i = sum_j B_ij sin(theta_i - theta_j)``.

    Uses ``sin(a - b) = sin a cos b - cos a sin b``, so only ``2 n B``
    trigonometric evaluations and one (sparse) matrix product are needed:
    ``P^e = s * (W c) - c * (W s)`` with ``s = sin(theta)``, ``c = cos(theta)``.

    Parameters
    ----------
    coupling:
        Symmetric ``(n, n)`` weighted adjacency ``W = [B_ij]`` shared by all
        copies (dense or CSR), or the ``(n B, n B)`` CSR matrix of per-copy
        networks in node-major order (row ``i B + b`` is bus ``i`` of copy ``b``).
    theta:
        Angles in node-major layout, shape ``(n, B)``.

    Returns
    -------
    tuple
        ``(P^e, sin(theta), cos(theta))``, each of shape ``(n, B)``.
    """
    n, batch = theta.shape
    if coupling.shape[0] == n:  # one network shared by every copy
        stacked = np.empty((n, 2 * batch))
        cos = np.cos(theta, out=stacked[:, :batch])
        sin = np.sin(theta, out=stacked[:, batch:])
        product = coupling @ stacked  # (n, 2B), contiguous operand
        w_cos, w_sin = product[:, :batch], product[:, batch:]
    else:  # per-copy networks
        sin, cos = np.sin(theta), np.cos(theta)
        w_cos = (coupling @ cos.ravel()).reshape(n, batch)
        w_sin = (coupling @ sin.ravel()).reshape(n, batch)
    power = sin * w_cos
    power -= cos * w_sin
    return power, sin, cos


def _layout(topology: str, network: _Network) -> np.ndarray:
    """Stable plotting positions in ``[-1, 1]^2``."""
    n = network.adjacency.shape[0]
    if topology in ("grid", "random_geometric"):
        positions = network.native_positions.copy()
    elif topology in ("ring", "small_world", "complete"):
        positions = graphs.circular_layout(n)
    else:
        positions = graphs.spring_layout(network.adjacency, seed=0)
    positions = positions - positions.mean(axis=0)
    scale = np.abs(positions).max()
    return positions / scale if scale > 0 else positions


# ---------------------------------------------------------------------------
# Batch-first simulation core
# ---------------------------------------------------------------------------
class _GridBatch:
    """State and dynamics of ``B`` power grids advanced as one batch.

    The single environment uses ``B = 1``. State arrays are stored node-major,
    with shape ``(n, B)`` (column ``b`` is copy ``b``): the network coupling is
    then a product with contiguous operands, and neighbour gathers copy whole
    rows. Parameters have shape ``(n, 1)`` when every copy shares one network
    (``randomize_network=False``) and ``(n, B)`` otherwise. The public methods
    take and return batch-first arrays where noted.
    """

    def __init__(self, config: _Config, num_envs: int, network: _Network):
        self.config = config
        self.num_envs = batch = int(num_envs)
        self.shared = not config.randomize_network
        n = config.n_buses
        self.networks: list[_Network] = [network] * (1 if self.shared else batch)
        self._u_max = config.u_max[:, None]
        self._rebuild()

        self.theta = np.zeros((n, batch))
        self.omega = np.zeros((n, batch))
        self.rocof = np.zeros((n, batch))
        self.step_disturbance = np.zeros((n, batch))
        self.noise = np.zeros((n, batch))
        self.disturbance = np.zeros((n, batch))
        self.action = np.zeros((n, batch))
        self.power_out = np.zeros((n, batch))
        self.agent_rewards = np.zeros((n, batch))
        self._sin = np.zeros((n, batch))
        self._cos = np.ones((n, batch))
        self.steps = np.zeros(batch, dtype=np.int64)
        self.nadir = np.zeros(batch)
        self.peak = np.zeros(batch)
        self.max_abs = np.zeros(batch)
        self.tripped = np.zeros(batch, dtype=bool)

        decay = math.exp(-config.dt / config.noise_tau)
        self._noise_decay = decay
        self._noise_gain = config.noise_std * math.sqrt(max(0.0, 1.0 - decay * decay))
        self._event_probability = -math.expm1(-config.disturbance_rate * config.dt)
        self._copies = np.arange(batch)
        self._warned_step_size = False
        self._check_step_size()

    # -- network structures ------------------------------------------------
    def _rebuild(self) -> None:
        """Derive the stacked parameter arrays and coupling structures from ``self.networks``."""
        nets, n, batch = self.networks, self.config.n_buses, self.num_envs
        self.inertia = np.stack([net.inertia for net in nets], axis=1)
        self.inertia_constant = np.stack([net.inertia_constant for net in nets], axis=1)
        self.damping = np.stack([net.damping for net in nets], axis=1)
        self.injection = np.stack([net.injection for net in nets], axis=1)
        self.theta_star = np.stack([net.theta_star for net in nets], axis=1)
        self.inv_inertia = 1.0 / self.inertia
        self.total_inertia = self.inertia.sum(axis=0)
        capacity = np.broadcast_to(self.config.u_max[:, None], self.inertia.shape)
        self._static_features = np.stack(
            (
                capacity.T * OBSERVATION_SCALE["capacity"],
                self.inertia_constant.T * OBSERVATION_SCALE["inertia"],
            ),
            axis=-1,
        ).astype(np.float32)  # (1 or B, n, 2), batch-first like the observation

        # Neighbourhood tables for the observation aggregates, as K "slots":
        # slot k holds the k-th neighbour of every bus, or the index n of an
        # always-zero padding row when a bus has fewer than k + 1 neighbours.
        width = max(net.neighbour_index.shape[1] for net in nets)
        index = np.full((len(nets), n, width), n, dtype=np.int64)
        count = np.zeros((n, len(nets)))
        for k, net in enumerate(nets):  # reset-time only
            used = net.neighbour_index.shape[1]
            index[k, :, :used] = np.where(net.neighbour_mask > 0, net.neighbour_index, n)
            count[:, k] = net.neighbour_mask.sum(axis=1)
        if len(nets) == 1:
            self._slots = [index[0, :, k] for k in range(width)]  # (n,) row indices
        else:  # flat indices into a node-major (n + 1, B) array
            copies = np.arange(batch)[None, :]
            self._slots = [index[:, :, k].T * batch + copies for k in range(width)]
        self._neighbour_inv_count = 1.0 / np.maximum(count, 1.0)

        if len(nets) == 1:
            weights = nets[0].weights
            # Dense products are faster for tiny workloads only (BLAS threading
            # overhead dominates beyond); sparse products scale with the edges.
            if 2 * batch * n * n <= _DENSE_WORK_MAX:
                weights = weights.toarray()
            self.weights = weights
        else:
            blocks = [net.weights.tocoo() for net in nets]
            rows = np.concatenate([b.row * batch + k for k, b in enumerate(blocks)])
            cols = np.concatenate([b.col * batch + k for k, b in enumerate(blocks)])
            data = np.concatenate([b.data for b in blocks])
            size = n * batch
            self.weights = sp.csr_matrix((data, (rows, cols)), shape=(size, size))
            self.weights.sort_indices()

    def _check_step_size(self) -> None:
        h = self.config.dt / self.config.substeps
        bound = max(net.omega_bound for net in self.networks)
        if h * bound > _RK4_WARN and not self._warned_step_size:
            warnings.warn(
                f"The RK4 step dt/substeps = {h:.4g} s is large for the fastest network mode "
                f"(bound {bound:.3g} rad/s, h * omega = {h * bound:.2f} > {_RK4_WARN}); the "
                "integration may be inaccurate or unstable. Increase substeps or reduce "
                "susceptance_range.",
                RuntimeWarning,
                stacklevel=4,
            )
            self._warned_step_size = True

    def flows(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(P^e, sin theta, cos theta)`` for a node-major ``(n, B)`` angle array."""
        return _flows(self.weights, theta)

    def network(self, index: int) -> _Network:
        """Network of copy ``index``."""
        return self.networks[0 if self.shared else index]

    # -- reset ---------------------------------------------------------------
    def reset(
        self,
        mask: np.ndarray,
        rng: np.random.Generator,
        options: Mapping[str, Any] | None,
    ) -> None:
        """Start new episodes in the copies selected by the boolean ``mask`` (shape ``(B,)``)."""
        config, n = self.config, self.config.n_buses
        options = dict(options or {})
        unknown = set(options) - _RESET_OPTIONS
        if unknown:
            raise ValueError(
                f"Unknown reset options {sorted(unknown)}; use {sorted(_RESET_OPTIONS)}"
            )
        rows = np.flatnonzero(mask)
        count = rows.size
        if count == 0:
            return
        theta = self._state_option(options, "theta", rows)
        omega = self._state_option(options, "omega", rows)
        steps = self._state_option(options, "disturbance", rows)

        if config.randomize_network:
            for row in rows:  # reset-time only: graph sampling is per network
                self.networks[row] = _sample_network(config, rng)
            self._rebuild()
            self._check_step_size()

        if theta is None:
            theta = np.broadcast_to(self.theta_star, self.theta.shape)[:, rows]
        self.theta[:, rows] = theta
        self.omega[:, rows] = 0.0 if omega is None else omega
        if steps is None:
            steps = np.zeros((count, n))
            k = config.initial_disturbances
            if k > 0:
                # k distinct buses per copy: the k smallest of n uniform keys.
                buses = rng.random((count, n)).argsort(axis=1)[:, :k]
                values = self._event_values(rng.random((count, k, 2)))
                np.put_along_axis(steps, buses, values, axis=1)
            steps = steps.T
        self.step_disturbance[:, rows] = steps
        if config.noise_std > 0.0:
            self.noise[:, rows] = config.noise_std * rng.standard_normal((n, count))
        else:
            self.noise[:, rows] = 0.0
        self.disturbance[:, rows] = self.step_disturbance[:, rows] + self.noise[:, rows]

        self.rocof[:, rows] = 0.0
        self.action[:, rows] = 0.0
        self.agent_rewards[:, rows] = 0.0
        self.steps[rows] = 0
        self.tripped[rows] = False
        self.max_abs = np.abs(self.omega).max(axis=0)
        self.nadir[rows] = self.omega[:, rows].min(axis=0)
        self.peak[rows] = self.max_abs[rows]
        self.power_out, self._sin, self._cos = self.flows(self.theta)

    def _state_option(
        self, options: Mapping[str, Any], name: str, rows: np.ndarray
    ) -> np.ndarray | None:
        """Option ``name`` as a node-major ``(n, len(rows))`` array, or ``None``."""
        value = options.get(name)
        if value is None:
            return None
        n = self.config.n_buses
        array = np.array(value, dtype=np.float64)
        if array.shape == (n,):
            array = np.repeat(array[:, None], rows.size, axis=1)
        elif array.shape == (self.num_envs, n):
            array = array[rows].T
        else:
            expected = f"({n},)" if self.num_envs == 1 else f"({n},) or ({self.num_envs}, {n})"
            raise ValueError(f"options[{name!r}] must have shape {expected}, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"options[{name!r}] must be finite")
        return array

    def _event_values(self, uniforms: np.ndarray) -> np.ndarray:
        """Signed step sizes from two uniform variates in the last axis.

        ``uniforms[..., 0]`` sets the size ``|Delta P|`` (uniform in
        ``disturbance_magnitude``) and ``uniforms[..., 1] < load_increase_prob``
        makes the step a load increase (negative injection change).
        """
        low, high = self.config.disturbance_magnitude
        magnitude = low + (high - low) * uniforms[..., 0]
        increase = uniforms[..., 1] < self.config.load_increase_prob
        return np.where(increase, -magnitude, magnitude)

    # -- step ----------------------------------------------------------------
    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """Batch-first ``(B, n)`` action -> clipped node-major ``(n, B)`` float64 control."""
        control = np.array(action.T, dtype=np.float64, order="C")
        return np.clip(control, -self._u_max, self._u_max, out=control)

    def step(self, control: np.ndarray, rng: np.random.Generator) -> None:
        """Advance all copies by ``dt`` with the clipped node-major control held constant."""
        config = self.config
        omega_old = self.omega
        self._integrate(control)
        self.steps += 1
        self.rocof = (self.omega - omega_old) * (1.0 / config.dt)
        self.action = control

        omega, rocof = self.omega, self.rocof
        cost = config.w_freq * (omega * omega)
        cost += config.w_rocof * (rocof * rocof)
        cost += config.w_u * (control * control)
        cost *= -config.dt
        self.max_abs = np.abs(omega).max(axis=0)
        self.tripped = self.max_abs > config.omega_limit
        if self.tripped.any():
            cost[:, self.tripped] -= config.trip_penalty
        self.agent_rewards = cost
        self.nadir = np.minimum(self.nadir, omega.min(axis=0))
        self.peak = np.maximum(self.peak, self.max_abs)
        self._advance_disturbances(rng)

    def _integrate(self, control: np.ndarray) -> None:
        """Classical fourth-order Runge-Kutta over ``substeps`` sub-intervals (zero-order hold)."""
        config = self.config
        h = config.dt / config.substeps
        half, sixth = 0.5 * h, h / 6.0
        drive = self.injection + self.disturbance + control
        damping, inv_m = self.damping, self.inv_inertia
        theta, omega, power = self.theta, self.omega, self.power_out
        for _ in range(config.substeps):
            a1 = (drive - damping * omega - power) * inv_m
            w2 = omega + half * a1
            a2 = (drive - damping * w2 - self.flows(theta + half * omega)[0]) * inv_m
            w3 = omega + half * a2
            a3 = (drive - damping * w3 - self.flows(theta + half * w2)[0]) * inv_m
            w4 = omega + h * a3
            a4 = (drive - damping * w4 - self.flows(theta + h * w3)[0]) * inv_m
            theta = theta + sixth * (omega + 2.0 * (w2 + w3) + w4)
            omega = omega + sixth * (a1 + 2.0 * (a2 + a3) + a4)
            power, self._sin, self._cos = self.flows(theta)
        self.theta, self.omega, self.power_out = theta, omega, power

    def _advance_disturbances(self, rng: np.random.Generator) -> None:
        """Random step events (Poisson, at most one per copy and step) and OU fluctuations."""
        config, batch = self.config, self.num_envs
        if config.disturbance_rate > 0.0:
            # One (B, 4) block of uniforms per step: event?, bus, size, sign.
            uniforms = rng.random((batch, 4))
            occurs = uniforms[:, 0] < self._event_probability
            if occurs.any():
                n = config.n_buses
                buses = np.minimum((uniforms[:, 1] * n).astype(np.int64), n - 1)
                values = self._event_values(uniforms[:, 2:])
                self.step_disturbance[buses[occurs], self._copies[occurs]] += values[occurs]
        if config.noise_std > 0.0:
            noise = self._noise_gain * rng.standard_normal(self.noise.shape)
            noise += self._noise_decay * self.noise
            self.noise = noise
        self.disturbance = self.step_disturbance + self.noise

    # -- observation and diagnostics ---------------------------------------
    def neighbour_aggregates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mean ``omega``, max ``|omega|`` and mean previous action over each neighbourhood.

        Returns node-major ``(n, B)`` arrays.
        """
        n, batch = self.omega.shape
        padded = np.zeros((2, n + 1, batch))  # omega and action with a zero padding row
        padded[0, :n] = self.omega
        padded[1, :n] = self.action
        magnitude = np.abs(padded[0])
        slots = self._slots
        if slots[0].ndim == 1:  # one network: gather rows
            total = np.take(padded, slots[0], axis=1)
            peak = np.take(magnitude, slots[0], axis=0)
            for slot in slots[1:]:
                total += np.take(padded, slot, axis=1)
                np.maximum(peak, np.take(magnitude, slot, axis=0), out=peak)
        else:  # per-copy networks: flat indices
            flat, flat_magnitude = padded.reshape(2, -1), magnitude.ravel()
            total = np.take(flat, slots[0], axis=1)
            peak = np.take(flat_magnitude, slots[0])
            for slot in slots[1:]:
                total += np.take(flat, slot, axis=1)
                np.maximum(peak, np.take(flat_magnitude, slot), out=peak)
        total *= self._neighbour_inv_count
        return total[0], peak, total[1]

    def observe(self) -> np.ndarray:
        """Joint observation of all copies, batch-first ``(B, n, OBS_DIM)``, float32."""
        omega_mean, omega_max, action_mean = self.neighbour_aggregates()
        dynamic = (
            self.omega,
            self.rocof,
            self.disturbance,
            self.power_out - self.injection,
            self.action,
            omega_mean,
            omega_max,
            action_mean,
        )
        n, batch = self.omega.shape
        node_major = np.empty((n, batch, OBS_DIM), dtype=np.float32)
        for column, (values, scale) in enumerate(zip(dynamic, _DYNAMIC_SCALE)):
            np.multiply(values, scale, out=node_major[..., column], casting="same_kind")
        node_major[..., _N_DYNAMIC:] = self._static_features.transpose(1, 0, 2)
        return np.ascontiguousarray(node_major.transpose(1, 0, 2))

    def infos(self) -> dict[str, np.ndarray]:
        """Batched diagnostics, batch-first (leading dimension ``B``)."""
        omega = self.omega
        inv_n = 1.0 / self.config.n_buses
        centred = omega - omega.sum(axis=0) * inv_n
        return {
            "agent_rewards": self.agent_rewards.T.copy(),
            "omega": omega.T.copy(),
            "rocof": self.rocof.T.copy(),
            "max_abs_omega": self.max_abs.copy(),
            "frequency_nadir": self.nadir.copy(),
            "peak_abs_omega": self.peak.copy(),
            "coi_omega": (self.inertia * omega).sum(axis=0) / self.total_inertia,
            "frequency_spread": np.sqrt((centred * centred).sum(axis=0) * inv_n),
            "control_effort": np.abs(self.action).sum(axis=0),
            "order_parameter": np.hypot(self._cos.sum(axis=0), self._sin.sum(axis=0)) * inv_n,
            "total_disturbance": self.disturbance.sum(axis=0),
            "tripped": self.tripped.copy(),
            "step": self.steps.copy(),
            "time": self.steps * self.config.dt,
        }


def _default_network(config: _Config) -> _Network:
    return _sample_network(config, np.random.default_rng(config.network_seed))


def _validate_action(action: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Cast to float32 (the action-space dtype), check size and finiteness."""
    array = np.asarray(action, dtype=np.float32)
    if array.shape != shape:
        if array.size != int(np.prod(shape)):
            raise ValueError(f"action must have shape {shape}, got {array.shape}")
        array = array.reshape(shape)
    if not np.all(np.isfinite(array)):
        raise ValueError("action contains NaN or infinite values")
    return array


def _known_options(
    options: Mapping[str, Any] | None, allowed: frozenset[str], owner: str
) -> dict[str, Any] | None:
    """Drop unknown reset options with a warning (generic tools pass their own keys).

    The warning points at the caller of ``owner.reset``.
    """
    if options is None:
        return None
    options = dict(options)
    unknown = sorted(set(options) - allowed, key=str)
    if unknown:
        warnings.warn(
            f"{owner}.reset: ignoring unknown reset option(s) {unknown}; supported options "
            f"are {sorted(_RESET_OPTIONS)}",
            UserWarning,
            stacklevel=3,
        )
        for key in unknown:
            del options[key]
    return options


def _spaces(config: _Config) -> tuple[spaces.Box, spaces.Box]:
    n = config.n_buses
    observation_space = spaces.Box(-np.inf, np.inf, shape=(n, OBS_DIM), dtype=np.float32)
    bound = config.u_max.astype(np.float32)
    action_space = spaces.Box(-bound, bound, shape=(n,), dtype=np.float32)
    return observation_space, action_space


def _observation_layout() -> dict[str, slice]:
    return {name: slice(k, k + 1) for k, name in enumerate(OBSERVATION_FEATURES)}


# ---------------------------------------------------------------------------
# Baseline controller
# ---------------------------------------------------------------------------
def droop_policy(observation: Any, gain: float = DEFAULT_DROOP_GAIN) -> np.ndarray:
    """Decentralised droop control ``u_i = clip(-gain * omega_i, -u_max_i, u_max_i)``.

    Every agent only uses its own observation row: the frequency deviation
    ``omega_i`` (column ``"omega"``) and its capacity ``u_max_i`` (column
    ``"capacity"``), both un-scaled with :data:`OBSERVATION_SCALE`. The function
    works for any leading batch shape, so it drives both :class:`PowerGridEnv`
    (``(n, obs_dim) -> (n,)``) and :class:`PowerGridVectorEnv`
    (``(B, n, obs_dim) -> (B, n)``).

    The action is held for one step of length ``dt``. The sampled-data loop at
    bus ``i`` is non-oscillatory when ``gain * dt < M_i`` and unstable when
    ``gain * dt > 2 M_i`` (``M_i = 2 H_i / omega_0``). The default gain 0.25
    satisfies the first condition for every inertia constant ``H_i >= 2 s`` at
    50 Hz with ``dt = 0.05 s``.

    Parameters
    ----------
    observation:
        Array of shape ``(..., n_buses, obs_dim)``.
    gain:
        Droop gain ``k >= 0`` in per unit power per rad/s.

    Returns
    -------
    numpy.ndarray
        Actions of shape ``(..., n_buses)``, dtype float32.

    Examples
    --------
    >>> env = PowerGridEnv()
    >>> obs, info = env.reset(seed=0)
    >>> obs, reward, terminated, truncated, info = env.step(droop_policy(obs))
    """
    obs = np.asarray(observation)
    if obs.ndim < 2 or obs.shape[-1] != OBS_DIM:
        raise ValueError(f"observation must have shape (..., n_buses, {OBS_DIM}), got {obs.shape}")
    gain = _check_float("gain", gain, minimum=0.0)
    omega = obs[..., _COL["omega"]].astype(np.float64) / OBSERVATION_SCALE["omega"]
    capacity = obs[..., _COL["capacity"]].astype(np.float64) / OBSERVATION_SCALE["capacity"]
    return np.clip(-gain * omega, -capacity, capacity).astype(np.float32)


# ---------------------------------------------------------------------------
# Single environment
# ---------------------------------------------------------------------------
class PowerGridEnv(gym.Env):
    """Frequency control of a networked power system (swing equations).

    **Model.** Bus ``i`` has the voltage/rotor angle ``theta_i`` (rad) and the
    frequency deviation ``omega_i`` (rad/s) from the nominal ``omega_0 = 2 pi f_0``.
    On the Kron-reduced, lossless network with line susceptances ``B_ij``
    (per unit),

    ``d theta_i / dt = omega_i``

    ``M_i d omega_i / dt = P_i(t) - D_i omega_i - sum_j B_ij sin(theta_i - theta_j) + u_i``

    with inertia ``M_i = 2 H_i / omega_0``, damping ``D_i = d_i / omega_0``, net
    injection ``P_i(t) = P_i^0 + Delta P_i(t)`` and control ``u_i``, clipped to
    ``[-u_max_i, u_max_i]``. The actions are held constant over the control
    interval ``dt``; the equations are integrated with the classical RK4 scheme
    and ``substeps`` sub-steps.

    **Network and operating point.** The line graph comes from
    :func:`env_lib.utils.graphs.make_graph`; ``H_i``, ``d_i`` and ``B_ij`` are
    uniform in their ranges. A raw injection pattern ``z_i ~ U(-1, 1)`` (minus
    its mean) is solved with the linearised flow equations ``L_B theta = z``;
    the angles are scaled so that the most loaded line has
    ``|sin(theta_i* - theta_j*)| = line_loading``, and the nominal injections
    are the exact flows ``P_i^0 = sum_j B_ij sin(theta_i* - theta_j*)`` (they sum
    to zero). Hence ``(theta*, 0)`` is an exact synchronous equilibrium, and a
    stable one since every line angle is below ``pi/2``. Every episode starts
    from it, so only the disturbances ``Delta P_i`` drive the system away.

    **Disturbances.** ``Delta P_i = S_i + X_i``: step load changes ``S_i``
    (``initial_disturbances`` distinct buses at reset, then Poisson events with
    rate ``disturbance_rate``, sizes uniform in ``disturbance_magnitude``, load
    increases -- negative ``Delta P`` -- with probability ``load_increase_prob``)
    plus an optional Ornstein-Uhlenbeck fluctuation ``X_i`` with stationary
    standard deviation ``noise_std`` and correlation time ``noise_tau``, updated
    exactly at every step boundary. All randomness comes from ``self.np_random``.

    **Observation** -- ``Box(-inf, inf, (n_buses, 10), float32)``; row ``i``
    holds local measurements of bus ``i`` and aggregates over its
    ``kappa``-hop neighbourhood ``N_i`` (``kappa = neighbourhood``), each
    multiplied by the fixed factor in :data:`OBSERVATION_SCALE`:

    ======  ========================  =================================================
    column  key                       content (physical unit, scale)
    ======  ========================  =================================================
    0       ``omega``                 ``omega_i`` (rad/s, x1)
    1       ``rocof``                 ``(omega_i(t) - omega_i(t - dt)) / dt`` (rad/s^2, x0.1)
    2       ``disturbance``           ``Delta P_i`` for the next interval (pu, x10)
    3       ``power_flow``            ``P^e_i - P^0_i``, change of exported power (pu, x10)
    4       ``action``                previous applied ``u_i`` (pu, x10)
    5       ``neighbour_omega_mean``  mean of ``omega_j`` over ``N_i`` (rad/s, x1)
    6       ``neighbour_omega_max``   max of ``|omega_j|`` over ``N_i`` (rad/s, x1)
    7       ``neighbour_action_mean`` mean previous ``u_j`` over ``N_i`` (pu, x10)
    8       ``capacity``              ``u_max_i`` (pu, x10)
    9       ``inertia``               ``H_i`` (s, x0.1)
    ======  ========================  =================================================

    The RoCoF is zero on the first observation of an episode.

    **Action** -- ``Box(-u_max, u_max, (n_buses,), float32)``: power injected by
    each bus controller (per unit). Values are clipped; NaN raises ``ValueError``.

    **Reward.** Per agent,
    ``r_i = -(w_freq omega_i^2 + w_rocof rocof_i^2 + w_u u_i^2) dt``, evaluated at
    the end of the step; the team reward is ``sum_i r_i``.

    **Episode end.** ``terminated`` when any ``|omega_i|`` exceeds
    ``2 pi frequency_limit`` (loss of stability / protection trip); every agent
    then receives an extra ``-trip_penalty``. ``truncated`` after ``max_steps``.

    **Info.** ``agent_rewards`` (float64 ``(n,)``), ``omega`` and ``rocof``
    (``(n,)``), ``max_abs_omega``, ``frequency_nadir`` (lowest ``omega_i`` so far),
    ``peak_abs_omega`` (largest ``|omega_i|`` so far), ``coi_omega`` (centre-of-
    inertia frequency ``sum M_i omega_i / sum M_i``), ``frequency_spread`` (std of
    ``omega_i``), ``control_effort`` (``sum |u_i|``), ``order_parameter`` (phase
    order ``|mean exp(j theta_i)|``), ``total_disturbance`` (``sum Delta P_i``),
    ``tripped``, ``step`` and ``time`` (s).

    Parameters
    ----------
    n_buses:
        Number of buses (agents), ``>= 2``.
    topology:
        Line graph, one of :data:`TOPOLOGIES` (``"small_world"``, ``"ring"``,
        ``"line"``, ``"star"``, ``"complete"``, ``"grid"``, ``"erdos_renyi"``,
        ``"random_geometric"``); built with :func:`env_lib.utils.graphs.make_graph`.
    topology_kwargs:
        Extra :func:`~env_lib.utils.graphs.make_graph` arguments
        (``edge_probability``, ``neighbors``, ``rewire_probability``, ``radius``).
        ``"erdos_renyi"`` defaults to ``edge_probability = min(1, 2 ln(n) / n)``.
    network_seed:
        Seed of the benchmark network (topology and physical parameters),
        sampled once at construction. ``None`` draws fresh entropy.
    randomize_network:
        Resample the network (topology and all physical parameters) from
        ``self.np_random`` on every reset (per copy in the vector environment).
    inertia_range:
        Range of the inertia constants ``H_i`` in seconds; ``M_i = 2 H_i / omega_0``.
    damping_range:
        Range of the per-unit damping ``d_i`` (load frequency sensitivity plus
        mechanical damping); ``D_i = d_i / omega_0``.
    susceptance_range:
        Range of the line susceptances ``B_ij`` (per unit).
    line_loading:
        Loading ``max |sin(theta_i* - theta_j*)|`` of the most loaded line at
        the operating point, in ``[0, 1)``.
    nominal_frequency:
        ``f_0`` in Hz; ``omega_0 = 2 pi f_0``.
    u_max:
        Control bound ``u_max_i`` (per unit), scalar or shape ``(n_buses,)``.
    dt:
        Control interval in seconds (actions are held constant over it).
    substeps:
        RK4 sub-steps per control interval.
    max_steps:
        Episode length before truncation.
    initial_disturbances:
        Number of distinct buses that receive a step load change at reset.
    disturbance_rate:
        Rate (1/s) of further step load changes during the episode (Poisson,
        at most one per step, at a uniformly random bus).
    disturbance_magnitude:
        Range of the step sizes ``|Delta P|`` (per unit).
    load_increase_prob:
        Probability that a step is a load increase (negative injection change).
    noise_std:
        Stationary standard deviation (per unit) of the Ornstein-Uhlenbeck load
        fluctuation at every bus (``0`` disables it).
    noise_tau:
        Correlation time of the fluctuation in seconds.
    frequency_limit:
        Admissible frequency deviation in Hz; ``|omega_i| > 2 pi frequency_limit``
        at any bus trips the system (termination).
    w_freq, w_rocof, w_u:
        Weights of the frequency, RoCoF and control terms of the reward.
    trip_penalty:
        Penalty added to every agent's reward on the step the system trips.
    neighbourhood:
        Number of hops ``kappa`` of the neighbourhood aggregated in the observation.
    render_mode:
        ``None``, ``"human"`` or ``"rgb_array"``.

    Raises
    ------
    ValueError
        For out-of-range values or an invalid topology.
    TypeError
        For arguments of the wrong type.

    Examples
    --------
    >>> env = PowerGridEnv()
    >>> obs, info = env.reset(seed=0)
    >>> obs.shape
    (16, 10)
    >>> total = 0.0
    >>> while True:
    ...     obs, reward, terminated, truncated, info = env.step(droop_policy(obs))
    ...     total += reward
    ...     if terminated or truncated:
    ...         break
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __getstate__(self) -> dict[str, Any]:
        # Pickle and deep-copy without the renderer (rebuilt on the next render()).
        return state_without_renderer(self)

    def __init__(
        self,
        *,
        n_buses: int = 16,
        topology: str = "small_world",
        topology_kwargs: Mapping[str, Any] | None = None,
        network_seed: int | None = 0,
        randomize_network: bool = False,
        inertia_range: tuple[float, float] = (2.0, 8.0),
        damping_range: tuple[float, float] = (2.0, 6.0),
        susceptance_range: tuple[float, float] = (0.3, 0.8),
        line_loading: float = 0.5,
        nominal_frequency: float = 50.0,
        u_max: float | Sequence[float] | np.ndarray = 0.1,
        dt: float = 0.05,
        substeps: int = 2,
        max_steps: int = 200,
        initial_disturbances: int = 1,
        disturbance_rate: float = 0.1,
        disturbance_magnitude: tuple[float, float] = (0.05, 0.2),
        load_increase_prob: float = 0.5,
        noise_std: float = 0.0,
        noise_tau: float = 1.0,
        frequency_limit: float = 0.5,
        w_freq: float = 1.0,
        w_rocof: float = 0.1,
        w_u: float = 10.0,
        trip_penalty: float = 100.0,
        neighbourhood: int = 1,
        render_mode: str | None = None,
    ):
        super().__init__()
        validate_render_mode(render_mode, self.metadata["render_modes"])
        self.config = _make_config(
            n_buses=n_buses,
            topology=topology,
            topology_kwargs=topology_kwargs,
            network_seed=network_seed,
            randomize_network=randomize_network,
            inertia_range=inertia_range,
            damping_range=damping_range,
            susceptance_range=susceptance_range,
            line_loading=line_loading,
            nominal_frequency=nominal_frequency,
            u_max=u_max,
            dt=dt,
            substeps=substeps,
            max_steps=max_steps,
            initial_disturbances=initial_disturbances,
            disturbance_rate=disturbance_rate,
            disturbance_magnitude=disturbance_magnitude,
            load_increase_prob=load_increase_prob,
            noise_std=noise_std,
            noise_tau=noise_tau,
            frequency_limit=frequency_limit,
            w_freq=w_freq,
            w_rocof=w_rocof,
            w_u=w_u,
            trip_penalty=trip_penalty,
            neighbourhood=neighbourhood,
        )
        self.render_mode = render_mode
        self.n_buses = self.n_agents = self.config.n_buses
        self.obs_dim = OBS_DIM
        self.max_steps = self.config.max_steps
        self.dt = self.config.dt
        self.observation_space, self.action_space = _spaces(self.config)
        self._core = _GridBatch(self.config, 1, _default_network(self.config))
        self._started = False
        self._renderer = None
        self._layout_cache: tuple[_Network, np.ndarray] | None = None
        self._warned_no_render_mode = False

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------
    @property
    def observation_layout(self) -> dict[str, slice]:
        """Column slice of every feature of an observation row (see :data:`OBSERVATION_FEATURES`)."""
        return _observation_layout()

    @property
    def observation_scale(self) -> dict[str, float]:
        """Factor applied to every feature (observation = physical value * scale)."""
        return dict(OBSERVATION_SCALE)

    @property
    def adjacency(self) -> np.ndarray:
        """Boolean ``(n, n)`` adjacency of the transmission network (a copy)."""
        return self._network.adjacency.copy()

    @property
    def edges(self) -> np.ndarray:
        """Lines ``(i, j)``, ``i < j``, shape ``(E, 2)`` (a copy)."""
        return self._network.edges.copy()

    @property
    def susceptance(self) -> np.ndarray:
        """Weighted adjacency ``B_ij`` in per unit, shape ``(n, n)`` (a copy)."""
        return self._network.weights.toarray()

    @property
    def inertia(self) -> np.ndarray:
        """Inertia coefficients ``M_i = 2 H_i / omega_0``, shape ``(n,)``."""
        return self._network.inertia.copy()

    @property
    def inertia_constants(self) -> np.ndarray:
        """Inertia constants ``H_i`` in seconds, shape ``(n,)``."""
        return self._network.inertia_constant.copy()

    @property
    def damping(self) -> np.ndarray:
        """Damping coefficients ``D_i = d_i / omega_0``, shape ``(n,)``."""
        return self._network.damping.copy()

    @property
    def nominal_injections(self) -> np.ndarray:
        """Nominal net injections ``P_i^0`` (per unit, sum zero), shape ``(n,)``."""
        return self._network.injection.copy()

    @property
    def equilibrium(self) -> np.ndarray:
        """Synchronous equilibrium angles ``theta*`` (mean zero), shape ``(n,)``."""
        return self._network.theta_star.copy()

    @property
    def u_max(self) -> np.ndarray:
        """Control bounds ``u_max_i`` (per unit), shape ``(n,)``."""
        return self.config.u_max.copy()

    @property
    def omega_limit(self) -> float:
        """Trip threshold ``2 pi frequency_limit`` in rad/s."""
        return self.config.omega_limit

    @property
    def theta(self) -> np.ndarray:
        """Current angles ``theta_i`` (rad), shape ``(n,)``."""
        self._require_reset()
        return self._core.theta[:, 0].copy()

    @property
    def omega(self) -> np.ndarray:
        """Current frequency deviations ``omega_i`` (rad/s), shape ``(n,)``."""
        self._require_reset()
        return self._core.omega[:, 0].copy()

    @property
    def disturbance(self) -> np.ndarray:
        """Current injection disturbance ``Delta P_i`` (per unit), shape ``(n,)``."""
        self._require_reset()
        return self._core.disturbance[:, 0].copy()

    @property
    def line_flows(self) -> np.ndarray:
        """Line flows ``B_ij sin(theta_i - theta_j)`` in the order of :attr:`edges`, shape ``(E,)``."""
        self._require_reset()
        net, theta = self._network, self._core.theta[:, 0]
        return net.susceptance * np.sin(theta[net.edges[:, 0]] - theta[net.edges[:, 1]])

    @property
    def time(self) -> float:
        """Simulated time of the current episode in seconds."""
        return float(self._core.steps[0] * self.dt)

    @property
    def step_count(self) -> int:
        """Number of steps taken in the current episode."""
        return int(self._core.steps[0])

    @property
    def positions(self) -> np.ndarray:
        """Plotting positions of the buses in ``[-1, 1]^2``, shape ``(n, 2)``."""
        net = self._network
        if self._layout_cache is None or self._layout_cache[0] is not net:
            self._layout_cache = (net, _layout(self.config.topology, net))
        return self._layout_cache[1].copy()

    @property
    def _network(self) -> _Network:
        return self._core.network(0)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode from the synchronous equilibrium.

        Parameters
        ----------
        seed:
            Seed for ``self.np_random`` (disturbances and, with
            ``randomize_network=True``, the network).
        options:
            Optional initial state ``{"theta": (n,), "omega": (n,),
            "disturbance": (n,)}``. Missing entries default to the equilibrium
            angles, zero frequency deviation and random initial steps; a given
            ``"disturbance"`` replaces the random initial steps. Values of these
            keys are validated (``ValueError`` for a wrong shape or non-finite
            entries); other keys are ignored with a ``UserWarning``.

        Returns
        -------
        observation, info
        """
        options = _known_options(options, _RESET_OPTIONS, "PowerGridEnv")
        super().reset(seed=seed)
        self._core.reset(np.ones(1, dtype=bool), self.np_random, options)
        self._started = True
        if self._renderer is not None:
            self._renderer.reset()
        if self.render_mode == "human":
            self.render()
        return self._core.observe()[0], self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the grid by ``dt`` seconds.

        Parameters
        ----------
        action:
            Control injections ``u_i``, shape ``(n_buses,)`` (any array with
            ``n_buses`` entries); clipped to ``[-u_max_i, u_max_i]``.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        self._require_reset()
        array = _validate_action(action, (self.n_buses,))
        control = self._core.clip_action(array[None, :])
        self._core.step(control, self.np_random)
        info = self._info()
        reward = float(info["agent_rewards"].sum())
        terminated = bool(self._core.tripped[0])
        truncated = bool(self._core.steps[0] >= self.max_steps)
        observation = self._core.observe()[0]
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the dashboard of the current state.

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
            from env_lib.power_grid_env.rendering import PowerGridRenderer

            self._renderer = PowerGridRenderer(
                self.render_mode,
                n_buses=self.n_buses,
                max_steps=self.max_steps,
                dt=self.dt,
                frequency_limit=self.config.frequency_limit,
                topology=self.config.topology,
                nominal_frequency=self.config.nominal_frequency,
                fps=self.metadata["render_fps"],
            )
        return self._renderer.render(**_render_state(self._core, 0, self.positions))

    def close(self) -> None:
        """Release the rendering resources (idempotent)."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_reset(self) -> None:
        if not self._started:
            raise ResetNeededError("Call reset() before using the environment")

    def _info(self) -> dict[str, Any]:
        batched = self._core.infos()
        info: dict[str, Any] = {}
        for key, value in batched.items():
            item = value[0]
            if isinstance(item, np.ndarray):
                info[key] = item
            elif isinstance(item, np.bool_):
                info[key] = bool(item)
            elif isinstance(item, np.integer):
                info[key] = int(item)
            else:
                info[key] = float(item)
        return info


def _render_state(core: _GridBatch, index: int, positions: np.ndarray) -> dict[str, Any]:
    """Everything the renderer needs about copy ``index``."""
    net = core.network(index)
    theta = core.theta[:, index]
    loading = np.abs(np.sin(theta[net.edges[:, 0]] - theta[net.edges[:, 1]]))
    omega = core.omega[:, index]
    return {
        "positions": positions,
        "edges": net.edges,
        "line_loading": loading,
        "omega": omega,
        "action": core.action[:, index].copy(),
        "u_max": core.config.u_max,
        "step_disturbance": core.step_disturbance[:, index].copy(),
        "total_disturbance": float(core.disturbance[:, index].sum()),
        "coi_omega": float((net.inertia * omega).sum() / net.inertia.sum()),
        "step": int(core.steps[index]),
        "nadir": float(core.nadir[index]),
        "tripped": bool(core.tripped[index]),
    }


# ---------------------------------------------------------------------------
# Vector environment
# ---------------------------------------------------------------------------
class PowerGridVectorEnv(BatchedVectorEnv):
    """``num_envs`` power grids simulated as one batch (native vector environment).

    Accepts every keyword argument of :class:`PowerGridEnv` and shares its
    batch-first simulation core, so copy ``b`` evolves exactly like a single
    environment started from the same state (inject it with
    ``reset(options=...)``) and driven by the same actions, as long as no
    random disturbance is drawn during the episode. With the default fixed
    network all copies share it; with ``randomize_network=True`` each copy
    samples its own network on every reset.

    Observations have shape ``(num_envs, n_buses, 10)``, actions
    ``(num_envs, n_buses)``. The reward of a copy is its team reward; the
    per-agent rewards and all other diagnostics of :class:`PowerGridEnv` are
    returned as batched arrays in ``infos`` (``infos["agent_rewards"]`` has
    shape ``(num_envs, n_buses)``). ``render()`` draws copy 0.

    Parameters
    ----------
    num_envs:
        Number of copies ``B``.
    autoreset_mode:
        ``"next_step"`` (default), ``"same_step"`` or ``"disabled"``; see
        :class:`env_lib.utils.vector.BatchedVectorEnv`.
    render_mode:
        ``None``, ``"human"`` or ``"rgb_array"`` (renders copy 0).
    **kwargs:
        Keyword arguments of :class:`PowerGridEnv`.

    Examples
    --------
    >>> envs = PowerGridVectorEnv(256)
    >>> obs, infos = envs.reset(seed=0)
    >>> obs.shape
    (256, 16, 10)
    >>> obs, rewards, terminated, truncated, infos = envs.step(droop_policy(obs))
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __getstate__(self) -> dict[str, Any]:
        # Pickle and deep-copy without the renderer (rebuilt on the next render()).
        return state_without_renderer(self)

    def __init__(
        self,
        num_envs: int = 1,
        *,
        autoreset_mode: str = "next_step",
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        validate_render_mode(render_mode, self.metadata["render_modes"])
        config = _make_config(**kwargs)
        observation_space, action_space = _spaces(config)
        super().__init__(
            num_envs,
            observation_space,
            action_space,
            autoreset_mode=autoreset_mode,
            render_mode=render_mode,
        )
        self.config = config
        self.n_buses = self.n_agents = config.n_buses
        self.obs_dim = OBS_DIM
        self.max_steps = config.max_steps
        self.dt = config.dt
        self._core = _GridBatch(config, self.num_envs, _default_network(config))
        self._renderer = None
        self._layout_cache: tuple[_Network, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Read-only views (batched over copies)
    # ------------------------------------------------------------------
    @property
    def observation_layout(self) -> dict[str, slice]:
        """Column slice of every feature of an observation row."""
        return _observation_layout()

    @property
    def observation_scale(self) -> dict[str, float]:
        """Factor applied to every feature (observation = physical value * scale)."""
        return dict(OBSERVATION_SCALE)

    @property
    def adjacency(self) -> np.ndarray:
        """Boolean adjacency of every copy, shape ``(num_envs, n, n)``."""
        return np.stack([self._core.network(b).adjacency for b in range(self.num_envs)])

    @property
    def inertia(self) -> np.ndarray:
        """``M_i`` of every copy, shape ``(num_envs, n)``."""
        return np.broadcast_to(self._core.inertia.T, (self.num_envs, self.n_buses)).copy()

    @property
    def damping(self) -> np.ndarray:
        """``D_i`` of every copy, shape ``(num_envs, n)``."""
        return np.broadcast_to(self._core.damping.T, (self.num_envs, self.n_buses)).copy()

    @property
    def nominal_injections(self) -> np.ndarray:
        """``P_i^0`` of every copy, shape ``(num_envs, n)``."""
        return np.broadcast_to(self._core.injection.T, (self.num_envs, self.n_buses)).copy()

    @property
    def equilibrium(self) -> np.ndarray:
        """``theta*`` of every copy, shape ``(num_envs, n)``."""
        return np.broadcast_to(self._core.theta_star.T, (self.num_envs, self.n_buses)).copy()

    @property
    def u_max(self) -> np.ndarray:
        """Control bounds ``u_max_i`` (per unit), shape ``(n,)``."""
        return self.config.u_max.copy()

    @property
    def theta(self) -> np.ndarray:
        """Angles, shape ``(num_envs, n)``."""
        self._require_reset()
        return self._core.theta.T.copy()

    @property
    def omega(self) -> np.ndarray:
        """Frequency deviations (rad/s), shape ``(num_envs, n)``."""
        self._require_reset()
        return self._core.omega.T.copy()

    @property
    def disturbance(self) -> np.ndarray:
        """Injection disturbances ``Delta P`` (per unit), shape ``(num_envs, n)``."""
        self._require_reset()
        return self._core.disturbance.T.copy()

    # ------------------------------------------------------------------
    # BatchedVectorEnv hooks
    # ------------------------------------------------------------------
    def _reset_envs(self, mask: np.ndarray, options: dict[str, Any] | None) -> None:
        self._core.reset(mask, self.np_random, options)
        if self._renderer is not None and mask[0]:
            self._renderer.reset()

    def _step_envs(
        self, actions: np.ndarray, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        control = self._core.clip_action(actions)
        self._core.step(control, self.np_random)
        infos = self._core.infos()
        rewards = infos["agent_rewards"].sum(axis=1)
        terminated = self._core.tripped.copy()
        truncated = self._core.steps >= self.max_steps
        return rewards, terminated, truncated, infos

    def _observe(self) -> np.ndarray:
        return self._core.observe()

    def _reset_infos(self, mask: np.ndarray) -> dict[str, Any]:
        return self._core.infos()

    def step(self, actions: Any):
        """Advance every copy by ``dt``; see :meth:`BatchedVectorEnv.step`.

        Copies reset automatically within this call report the info of their
        new episode, as in :class:`gymnasium.vector.SyncVectorEnv`.
        """
        result = super().step(actions)
        if self.render_mode == "human":
            self.render()
        return result

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset all copies (or those in ``options["reset_mask"]``).

        ``options`` may also hold ``"theta"``, ``"omega"`` and ``"disturbance"``
        arrays of shape ``(n,)`` (broadcast) or ``(num_envs, n)``; see
        :meth:`PowerGridEnv.reset`. Other keys are ignored with a ``UserWarning``.
        """
        options = _known_options(options, _RESET_OPTIONS | {"reset_mask"}, type(self).__name__)
        result = super().reset(seed=seed, options=options)
        if self.render_mode == "human":
            self.render()
        return result

    def render(self) -> np.ndarray | None:
        """Render the dashboard of copy 0 (``None`` without a render mode)."""
        if self.render_mode is None:
            return None
        self._require_reset()
        if self._renderer is None:
            from env_lib.power_grid_env.rendering import PowerGridRenderer

            self._renderer = PowerGridRenderer(
                self.render_mode,
                n_buses=self.n_buses,
                max_steps=self.max_steps,
                dt=self.dt,
                frequency_limit=self.config.frequency_limit,
                topology=self.config.topology,
                nominal_frequency=self.config.nominal_frequency,
                fps=self.metadata["render_fps"],
            )
        net = self._core.network(0)
        if self._layout_cache is None or self._layout_cache[0] is not net:
            self._layout_cache = (net, _layout(self.config.topology, net))
        return self._renderer.render(**_render_state(self._core, 0, self._layout_cache[1]))

    def close(self, **kwargs: Any) -> None:
        """Release the rendering resources (idempotent)."""
        if getattr(self, "_renderer", None) is not None:  # absent if __init__ failed early
            self._renderer.close()
            self._renderer = None
        super().close(**kwargs)

    def _require_reset(self) -> None:
        if self._needs_reset:
            raise ResetNeededError("Call reset() before using the environment")
