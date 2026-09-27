"""Graph helpers for networked environments.

Networked environments describe who interacts with whom by a symmetric
adjacency matrix: agent ``i`` can observe, communicate with or is physically
coupled to agent ``j`` exactly when ``adjacency[i, j]`` is non-zero. This
module builds such matrices for the common topologies, computes the graph
quantities that appear in networked control (Laplacian, algebraic
connectivity, ``k``-hop neighbourhoods) and places nodes for plotting.

Every random topology takes a :class:`numpy.random.Generator`, so graphs are
reproducible when the generator is seeded (environments pass their own
``np_random``).

Example
-------
>>> import numpy as np
>>> from env_lib.utils import graphs
>>> adj = graphs.make_graph("small_world", 12, rng=np.random.default_rng(0))
>>> graphs.is_connected(adj)
True
>>> hops = graphs.k_hop_adjacency(adj, 2)   # who is within two hops of whom
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "TOPOLOGIES",
    "algebraic_connectivity",
    "circular_layout",
    "degree",
    "edge_list",
    "grid_shape",
    "is_connected",
    "k_hop_adjacency",
    "laplacian",
    "make_graph",
    "neighbor_lists",
    "spring_layout",
]

#: Topologies understood by :func:`make_graph`.
TOPOLOGIES: tuple[str, ...] = (
    "ring",
    "line",
    "star",
    "complete",
    "grid",
    "erdos_renyi",
    "small_world",
    "random_geometric",
)

_MAX_ATTEMPTS = 1000


def _as_square(adjacency: Any, *, dtype: Any = bool) -> np.ndarray:
    adj = np.asarray(adjacency)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency must be a square matrix, got shape {adj.shape}")
    return adj.astype(dtype, copy=False)


def grid_shape(n: int) -> tuple[int, int]:
    """Rows and columns of the most square grid with exactly ``n`` nodes.

    For a prime ``n`` this is a single row.
    """
    rows = int(math.isqrt(n))
    while n % rows:
        rows -= 1
    return rows, n // rows


def _reachability(adjacency: np.ndarray, hops: int) -> np.ndarray:
    """Boolean matrix of node pairs connected by a path of at most ``hops`` edges."""
    n = adjacency.shape[-1]
    step = (adjacency != 0) | np.eye(n, dtype=bool)
    reach = step.copy()
    # Repeated squaring: after k squarings, paths of length <= 2**k are covered.
    covered = 1
    while covered < hops:
        if 2 * covered <= hops:
            reach = (reach.astype(np.float32) @ reach.astype(np.float32)) > 0
            covered *= 2
        else:
            reach = (reach.astype(np.float32) @ step.astype(np.float32)) > 0
            covered += 1
    return reach


def is_connected(adjacency: Any) -> bool:
    """Return ``True`` if the undirected graph is connected.

    Parameters
    ----------
    adjacency:
        Square matrix; non-zero entries are edges.
    """
    adj = _as_square(adjacency)
    n = adj.shape[0]
    if n <= 1:
        return True
    return bool(_reachability(adj | adj.T, n - 1)[0].all())


def k_hop_adjacency(adjacency: Any, k: int, *, include_self: bool = False) -> np.ndarray:
    """Pairs of nodes within ``k`` hops of each other.

    This is the neighbourhood used by ``kappa``-hop (truncated) policies and
    critics in networked multi-agent reinforcement learning.

    Parameters
    ----------
    adjacency:
        Square matrix; non-zero entries are edges.
    k:
        Number of hops (``k >= 0``; ``k = 0`` gives only the node itself when
        ``include_self`` is true).
    include_self:
        Whether a node belongs to its own neighbourhood.

    Returns
    -------
    numpy.ndarray
        Boolean ``(n, n)`` matrix.
    """
    if int(k) != k or k < 0:
        raise ValueError(f"k must be a non-negative integer, got {k!r}")
    adj = _as_square(adjacency)
    n = adj.shape[0]
    if k == 0:
        reach = np.eye(n, dtype=bool)
    else:
        reach = _reachability(adj, int(k))
    if not include_self:
        reach = reach & ~np.eye(n, dtype=bool)
    return reach


def degree(adjacency: Any) -> np.ndarray:
    """Weighted degree of every node (row sums of the adjacency matrix)."""
    return _as_square(adjacency, dtype=np.float64).sum(axis=1)


def laplacian(adjacency: Any) -> np.ndarray:
    """Graph Laplacian ``L = D - A`` (weighted when ``adjacency`` is weighted)."""
    adj = _as_square(adjacency, dtype=np.float64)
    return np.diag(adj.sum(axis=1)) - adj


def algebraic_connectivity(adjacency: Any) -> float:
    """Fiedler value ``lambda_2`` of the Laplacian of an undirected graph.

    It is positive exactly when the graph is connected, and it sets the
    convergence rate of linear consensus.
    """
    adj = _as_square(adjacency, dtype=np.float64)
    if adj.shape[0] < 2:
        raise ValueError("algebraic connectivity needs at least two nodes")
    eigenvalues = np.linalg.eigvalsh(laplacian(0.5 * (adj + adj.T)))
    return float(max(eigenvalues[1], 0.0))


def neighbor_lists(adjacency: Any) -> list[np.ndarray]:
    """Indices of the neighbours of every node (self-loops excluded)."""
    adj = _as_square(adjacency) != 0
    np.fill_diagonal(adj, False)
    return [np.flatnonzero(row) for row in adj]


def edge_list(adjacency: Any) -> np.ndarray:
    """Undirected edges ``(i, j)`` with ``i < j`` as an ``(n_edges, 2)`` array."""
    adj = _as_square(adjacency) != 0
    rows, cols = np.nonzero(np.triu(adj | adj.T, k=1))
    return np.stack([rows, cols], axis=1)


def make_graph(
    topology: str,
    n: int,
    *,
    rng: np.random.Generator | None = None,
    edge_probability: float = 0.3,
    neighbors: int = 4,
    rewire_probability: float = 0.1,
    radius: float | None = None,
    connected: bool = True,
    max_attempts: int = _MAX_ATTEMPTS,
    return_positions: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Adjacency matrix of a standard topology.

    Parameters
    ----------
    topology:
        One of :data:`TOPOLOGIES`:

        * ``"ring"``, ``"line"``, ``"star"`` (node 0 is the hub), ``"complete"``;
        * ``"grid"``: 4-neighbour lattice of :func:`grid_shape` ``(n)``;
        * ``"erdos_renyi"``: every edge independently with ``edge_probability``;
        * ``"small_world"``: Watts-Strogatz ring lattice where every node links
          to its ``neighbors`` nearest nodes and each edge is rewired with
          ``rewire_probability``;
        * ``"random_geometric"``: nodes uniform in the unit square, linked when
          closer than ``radius`` (default: about twice the connectivity
          threshold ``sqrt(log(n) / (pi n))``).
    n:
        Number of nodes (``>= 2``).
    rng:
        Generator for the random topologies (a fresh unseeded one if omitted).
    connected:
        Resample random topologies until the graph is connected.
    max_attempts:
        Maximum number of samples when ``connected`` is true.
    return_positions:
        Also return node positions of shape ``(n, 2)``: the sampled points of
        ``"random_geometric"``, lattice coordinates of ``"grid"`` and a
        circular layout otherwise.

    Returns
    -------
    numpy.ndarray or tuple
        Symmetric boolean ``(n, n)`` matrix with a ``False`` diagonal, and the
        positions if requested.

    Raises
    ------
    ValueError
        For an unknown topology, invalid parameters, or when no connected
        random graph was found in ``max_attempts`` samples.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"a graph needs at least two nodes, got n={n}")
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology must be one of {TOPOLOGIES}, got {topology!r}")
    generator = rng if rng is not None else np.random.default_rng()
    idx = np.arange(n)
    positions = circular_layout(n)

    def finish(adj: np.ndarray):
        adj = adj | adj.T
        np.fill_diagonal(adj, False)
        return (adj, positions) if return_positions else adj

    if topology in ("ring", "line", "star", "complete", "grid"):
        adj = np.zeros((n, n), dtype=bool)
        if topology == "ring":
            adj[idx, (idx + 1) % n] = True
        elif topology == "line":
            adj[idx[:-1], idx[1:]] = True
        elif topology == "star":
            adj[0, 1:] = True
        elif topology == "complete":
            adj[:] = True
        else:
            rows, cols = grid_shape(n)
            r, c = np.divmod(idx, cols)
            right = c < cols - 1
            adj[idx[right], idx[right] + 1] = True
            down = r < rows - 1
            adj[idx[down], idx[down] + cols] = True
            positions = np.stack([c, rows - 1 - r], axis=1).astype(np.float64)
        return finish(adj)

    for _ in range(max(1, int(max_attempts))):
        adj = np.zeros((n, n), dtype=bool)
        if topology == "erdos_renyi":
            if not 0.0 < edge_probability <= 1.0:
                raise ValueError(f"edge_probability must be in (0, 1], got {edge_probability}")
            upper_rows, upper_cols = np.triu_indices(n, k=1)
            keep = generator.random(upper_rows.size) < edge_probability
            adj[upper_rows[keep], upper_cols[keep]] = True
        elif topology == "small_world":
            k = int(neighbors)
            if k < 2 or k % 2 or k >= n:
                raise ValueError(f"neighbors must be even and in [2, n), got {neighbors}")
            for shift in range(1, k // 2 + 1):
                targets = (idx + shift) % n
                rewire = generator.random(n) < rewire_probability
                adj[idx[~rewire], targets[~rewire]] = True
                for i in idx[rewire]:
                    taken = adj[i] | adj[:, i]
                    taken[i] = True
                    choices = np.flatnonzero(~taken)
                    target = generator.choice(choices) if choices.size else targets[i]
                    adj[i, target] = True
        else:
            r = radius if radius is not None else 2.0 * math.sqrt(math.log(n) / (math.pi * n))
            if r <= 0:
                raise ValueError(f"radius must be positive, got {radius}")
            positions = generator.random((n, 2))
            delta = positions[:, None, :] - positions[None, :, :]
            adj = np.einsum("ijk,ijk->ij", delta, delta) <= r * r
        adj = adj | adj.T
        np.fill_diagonal(adj, False)
        if not connected or is_connected(adj):
            return finish(adj)
    raise ValueError(
        f"no connected {topology!r} graph with n={n} found in {max_attempts} attempts; "
        "increase the edge probability, neighbour count or radius"
    )


def circular_layout(n: int, radius: float = 1.0) -> np.ndarray:
    """Positions of ``n`` nodes evenly spaced on a circle, shape ``(n, 2)``."""
    angles = 2.0 * np.pi * np.arange(n) / max(n, 1) + np.pi / 2.0
    return radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def spring_layout(
    adjacency: Any,
    *,
    iterations: int = 200,
    seed: int = 0,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Force-directed (Fruchterman-Reingold) node positions for plotting.

    Parameters
    ----------
    adjacency:
        Square matrix; non-zero entries are edges.
    iterations:
        Number of layout iterations.
    seed:
        Seed of the initial positions (the layout is deterministic).
    initial:
        Optional initial positions of shape ``(n, 2)``.

    Returns
    -------
    numpy.ndarray
        Positions of shape ``(n, 2)`` scaled to ``[-1, 1]``.
    """
    adj = _as_square(adjacency, dtype=np.float64)
    adj = ((adj + adj.T) != 0).astype(np.float64)
    n = adj.shape[0]
    if n == 1:
        return np.zeros((1, 2))
    if initial is not None:
        pos = np.array(initial, dtype=np.float64, copy=True)
    else:
        pos = circular_layout(n) + 0.05 * np.random.default_rng(seed).standard_normal((n, 2))
    k = 1.0 / math.sqrt(n)
    temperature = 0.1
    cooling = temperature / (iterations + 1)
    for _ in range(int(iterations)):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.maximum(np.linalg.norm(delta, axis=-1), 1e-6)
        force = (k * k / dist**2 - adj * dist / k)[:, :, None] * delta
        displacement = force.sum(axis=1)
        length = np.maximum(np.linalg.norm(displacement, axis=1, keepdims=True), 1e-9)
        pos += displacement / length * np.minimum(length, temperature)
        temperature -= cooling
    pos -= pos.mean(axis=0)
    scale = np.abs(pos).max()
    return pos / scale if scale > 0 else pos
