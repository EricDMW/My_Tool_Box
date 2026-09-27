"""Batch-first array kernels shared by the single and the vector consensus environments.

Every function here works on arrays with a leading batch dimension ``B``:
positions and velocities ``(B, n, 2)``, adjacency matrices ``(B, n, n)``,
per-agent quantities ``(B, n)`` and per-copy scalars ``(B,)``.
:class:`~env_lib.consensus_env.ConsensusEnv` calls them with ``B = 1`` and
:class:`~env_lib.consensus_env.vector.ConsensusVectorEnv` with
``B = num_envs``, so both share one implementation of the dynamics, graph,
reward and observation. The formulations are chosen so that a copy's result
does not depend on ``B`` (bit for bit): the vector environment reproduces the
single environment exactly from the same initial state.

This module is private; its API may change without notice.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

__all__ = ["ConsensusKernel", "batched_connected", "sequential_positions"]

#: Candidate placements per rejection-sampling batch and the number of batches.
INIT_BATCH = 256
INIT_MAX_BATCHES = 32
#: Initial positions are uniform in ``[-INIT_SPREAD * a, INIT_SPREAD * a]^2``.
INIT_SPREAD = 0.8
#: Upper bound on the elements of one connectivity test array while sampling.
_SAMPLE_CHUNK_ELEMENTS = 1 << 21
#: Up to this many pairs (B * n * n), pairwise differences are formed in one array.
_SMALL_PAIRWISE = 4096


def batched_connected(adjacency: np.ndarray) -> np.ndarray:
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


def sequential_positions(
    rng: np.random.Generator, n_agents: int, limit: float, comm_radius: float
) -> np.ndarray:
    """Place agents one by one within ``0.95 * comm_radius`` of a placed agent.

    Fallback of the connected initial placement when rejection sampling fails
    (a rare, reset-time-only path). Returns an array of shape ``(n_agents, 2)``.
    """
    positions = np.empty((n_agents, 2), dtype=np.float64)
    positions[0] = rng.uniform(-limit, limit, size=2)
    for i in range(1, n_agents):
        parent = positions[rng.integers(i)]
        radius = 0.95 * comm_radius * math.sqrt(rng.random())
        angle = 2.0 * math.pi * rng.random()
        step = radius * np.array([math.cos(angle), math.sin(angle)])
        # Projection onto the box is non-expansive, so the parent stays in range.
        positions[i] = np.clip(parent + step, -limit, limit)
    return positions[rng.permutation(n_agents)]


class ConsensusKernel:
    """Dynamics, graph, reward and observation kernels of the consensus model.

    Parameters
    ----------
    n_agents, dynamics, dt, arena_size, max_control, control_cost, damping, noise_std:
        Model parameters (already validated), see
        :class:`~env_lib.consensus_env.ConsensusEnv`.
    offsets:
        Formation offsets ``d_i``, shape ``(n_agents, 2)`` (zeros for consensus).
    formation:
        ``True`` for ``task="formation"``.
    max_neighbors:
        Number of neighbour slots in the observation.
    comm_radius:
        Range of the proximity graph.
    """

    #: Width of one neighbour slot in the observation.
    SLOT_DIM = 5

    def __init__(
        self,
        *,
        n_agents: int,
        dynamics: str,
        dt: float,
        arena_size: float,
        max_control: float,
        control_cost: float,
        damping: float,
        noise_std: float,
        offsets: np.ndarray,
        formation: bool,
        max_neighbors: int,
        comm_radius: float,
    ) -> None:
        self.n_agents = n = int(n_agents)
        self.double = dynamics == "double"
        self.dt = float(dt)
        self.arena_size = float(arena_size)
        self.max_control = float(max_control)
        self.control_cost = float(control_cost)
        self.damping = float(damping)
        self.noise_scale = float(noise_std) * math.sqrt(self.dt)
        self.offsets = np.asarray(offsets, dtype=np.float64)
        self.offset_delta = self.offsets[None, :, :] - self.offsets[:, None, :]
        self._offset_delta_x = np.ascontiguousarray(self.offset_delta[..., 0])
        self._offset_delta_y = np.ascontiguousarray(self.offset_delta[..., 1])
        self.formation = bool(formation)
        self.max_neighbors = int(max_neighbors)
        self.visible_slots = min(self.max_neighbors, n)
        self.obs_dim = 6 + self.SLOT_DIM * self.max_neighbors
        self.comm_radius_sq = float(comm_radius) * float(comm_radius)
        self.comm_radius = float(comm_radius)
        self._slot_index = np.arange(self.visible_slots)
        self._bases: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def integrate(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        control: np.ndarray,
        noise: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One Euler step of the single or double integrator.

        Parameters
        ----------
        pos, vel:
            Current state, ``(B, n, 2)``.
        control:
            Clipped action, ``(B, n, 2)`` float64.
        noise:
            Standard-normal samples ``(B, n, 2)`` or ``None`` (no process noise).

        Returns
        -------
        tuple
            New positions and velocities (fresh arrays).
        """
        if not self.double:
            moved = pos + self.dt * control
            if noise is not None:
                moved += self.noise_scale * noise
            new_pos = np.clip(moved, -self.arena_size, self.arena_size)
            new_vel = (new_pos - pos) / self.dt
        else:
            new_vel = vel + self.dt * (control - self.damping * vel)
            if noise is not None:
                new_vel += self.noise_scale * noise
            moved = pos + self.dt * new_vel
            new_pos = np.clip(moved, -self.arena_size, self.arena_size)
            new_vel[moved != new_pos] = 0.0
        return new_pos, new_vel

    # ------------------------------------------------------------------
    # Geometry and graphs
    # ------------------------------------------------------------------
    def pairwise(
        self, pos: np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
        """Pairwise quantities ``[b, i, j]``.

        Returns
        -------
        tuple
            ``((rel_x, rel_y), sq_x, sq_y)``: the components of ``y_j - y_i``
            (``y = x - d``, computed as ``(x_j - x_i) - (d_j - d_i)``), and the
            squared norms of ``x_j - x_i`` and ``y_j - y_i``; all ``(B, n, n)``.
        """
        if pos.shape[0] * self.n_agents * self.n_agents <= _SMALL_PAIRWISE:
            # Few elements: fewer (per-call dominated) array operations.
            delta = pos[:, None, :, :] - pos[:, :, None, :]
            sq_x = np.einsum("bijk,bijk->bij", delta, delta)
            if not self.formation:
                return (delta[..., 0], delta[..., 1]), sq_x, sq_x
            delta = delta - self.offset_delta
            sq_y = np.einsum("bijk,bijk->bij", delta, delta)
            return (delta[..., 0], delta[..., 1]), sq_x, sq_y
        # Many elements: component-wise arrays keep the inner loops long.
        px, py = pos[..., 0], pos[..., 1]
        dx = px[:, None, :] - px[:, :, None]
        dy = py[:, None, :] - py[:, :, None]
        sq_x = dx * dx + dy * dy
        if not self.formation:
            return (dx, dy), sq_x, sq_x
        dx -= self._offset_delta_x
        dy -= self._offset_delta_y
        return (dx, dy), sq_x, dx * dx + dy * dy

    def proximity(self, sq_x: np.ndarray) -> np.ndarray:
        """Disk-graph adjacency ``(B, n, n)`` from squared distances (``False`` diagonal)."""
        adj = sq_x <= self.comm_radius_sq
        n = self.n_agents
        adj.reshape(-1, n * n)[:, :: n + 1] = False  # diagonal of every copy
        return adj

    @staticmethod
    def graph_quantities(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Float adjacency, degrees and Fiedler values of a batch of graphs.

        Returns
        -------
        tuple
            ``(adj_f (B, n, n), degree (B, n), lambda2 (B,))``.
        """
        adj_f = adj.astype(np.float64)
        deg = adj_f.sum(axis=-1)
        if adj.shape[0] == 0:
            return adj_f, deg, np.zeros(0)
        n = adj.shape[-1]
        laplacian = np.zeros_like(adj_f)
        laplacian[:, np.arange(n), np.arange(n)] = deg
        laplacian -= adj_f
        lambda2 = np.maximum(np.linalg.eigvalsh(laplacian)[:, 1], 0.0)
        return adj_f, deg, lambda2

    # ------------------------------------------------------------------
    # Reward, error and observation
    # ------------------------------------------------------------------
    def agent_rewards(
        self, adj_f: np.ndarray, deg: np.ndarray, sq_y: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        """Per-agent rewards ``(B, n)`` (disagreement plus control cost)."""
        disagreement = (adj_f * sq_y).sum(axis=-1) / np.maximum(deg, 1.0)
        return -disagreement - self.control_cost * np.einsum("bij,bij->bi", control, control)

    def task_error(self, pos: np.ndarray) -> np.ndarray:
        """Task error ``e = mean_i ||y_i - mean_j y_j||^2`` per copy, shape ``(B,)``."""
        shifted = pos - self.offsets
        # np.add.reduce(...) / n is exactly ndarray.mean, without its Python overhead.
        centred = shifted - np.add.reduce(shifted, axis=1, keepdims=True) / self.n_agents
        return np.einsum("bij,bij->b", centred, centred) / self.n_agents

    def observe(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        adj: np.ndarray,
        sq_x: np.ndarray,
        rel: tuple[np.ndarray, np.ndarray],
        deg: np.ndarray,
    ) -> np.ndarray:
        """Joint observations ``(B, n, obs_dim)`` float32 (see the environment docstring).

        ``rel`` and ``sq_x`` are the outputs of :meth:`pairwise` for ``pos``;
        ``deg`` are the node degrees of ``adj``, shape ``(B, n)``.
        """
        batch, n, slots = pos.shape[0], self.n_agents, self.visible_slots
        obs = np.zeros((batch, n, self.obs_dim), dtype=np.float32)
        obs[..., 0:2] = pos
        obs[..., 2:4] = vel
        obs[..., 4:6] = self.offsets
        # Neighbours first (sorted by distance, ties by index), non-neighbours (inf) last,
        # so slot k holds a neighbour exactly when k < degree.
        order = np.argsort(np.where(adj, sq_x, np.inf), axis=-1, kind="stable")[..., :slots]
        mask = self._slot_index < deg[..., None]
        pair_base, agent_base = self._index_bases(batch)
        pair = order + pair_base  # flat index of (b, i, order) in (B, n, n)
        features = np.empty((batch, n, slots, 4))
        features[..., 0] = rel[0].take(pair)
        features[..., 1] = rel[1].take(pair)
        np.subtract(
            vel.reshape(-1, 2).take(order + agent_base, axis=0),
            vel[:, :, None, :],
            out=features[..., 2:4],
        )
        block = obs[..., 6:].reshape(batch, n, self.max_neighbors, self.SLOT_DIM)
        block[:, :, :slots, 0:4] = np.where(mask[..., None], features, 0.0)
        block[:, :, :slots, 4] = mask
        return obs

    def _index_bases(self, batch: int) -> tuple[np.ndarray, np.ndarray]:
        """Offsets turning per-copy neighbour indices into flat indices (cached per batch)."""
        bases = self._bases.get(batch)
        if bases is None:
            n = self.n_agents
            bases = (
                (np.arange(batch * n) * n).reshape(batch, n, 1),
                (np.arange(batch) * n).reshape(batch, 1, 1),
            )
            self._bases[batch] = bases
        return bases

    def feedback(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        adj_f: np.ndarray,
        deg: np.ndarray,
        gain: float,
        velocity_gain: float | None,
    ) -> np.ndarray:
        """Laplacian feedback ``-gain * L (x - d) - velocity_gain * v``, clipped, float32."""
        shifted = pos - self.offsets
        control = -gain * (deg[..., None] * shifted - np.matmul(adj_f, shifted))
        if self.double and velocity_gain is not None:
            control -= velocity_gain * vel
        return np.clip(control, -self.max_control, self.max_control).astype(np.float32)

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------
    def sample_positions(
        self,
        rng: np.random.Generator,
        count: int,
        connected: bool,
        on_fallback: Callable[[], None] | None = None,
    ) -> np.ndarray:
        """Draw initial positions for ``count`` copies, shape ``(count, n, 2)``.

        Positions are uniform in ``[-0.8 a, 0.8 a]^2``. With ``connected=True``
        (proximity graphs) each copy is conditioned on a connected initial disk
        graph: rejection sampling in rounds of :data:`INIT_BATCH` candidates
        per copy, then sequential placement for copies that still failed
        (``on_fallback`` is called once in that case). For ``count = 1`` the
        random stream is consumed exactly as by earlier releases.
        """
        n = self.n_agents
        limit = INIT_SPREAD * self.arena_size
        if not connected:
            return rng.uniform(-limit, limit, size=(count, n, 2))
        out = np.empty((count, n, 2), dtype=np.float64)
        pending = np.arange(count)
        chunk = max(1, _SAMPLE_CHUNK_ELEMENTS // (INIT_BATCH * n * n))
        for _ in range(INIT_MAX_BATCHES):
            if pending.size == 0:
                return out
            still = []
            for start in range(0, pending.size, chunk):
                ids = pending[start : start + chunk]
                candidates = rng.uniform(-limit, limit, size=(ids.size, INIT_BATCH, n, 2))
                flat = candidates.reshape(ids.size * INIT_BATCH, n, 2)
                delta = flat[:, None, :, :] - flat[:, :, None, :]
                sq = np.einsum("bijk,bijk->bij", delta, delta)
                ok = batched_connected(sq <= self.comm_radius_sq).reshape(ids.size, INIT_BATCH)
                found = ok.any(axis=1)
                first = np.argmax(ok, axis=1)
                out[ids[found]] = candidates[found, first[found]]
                still.append(ids[~found])
            pending = np.concatenate(still)
        if pending.size:
            if on_fallback is not None:
                on_fallback()
            for index in pending:  # rare fallback, reset time only
                out[index] = sequential_positions(rng, n, limit, self.comm_radius)
        return out
