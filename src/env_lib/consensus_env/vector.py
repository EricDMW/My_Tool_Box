"""Natively batched consensus / formation environment.

:class:`ConsensusVectorEnv` simulates ``num_envs`` copies of
:class:`~env_lib.consensus_env.ConsensusEnv` as one batch: positions,
velocities and graphs are stored in arrays with a leading ``num_envs``
dimension and every step is a fixed number of array operations, independent
of ``num_envs``. Both environments use the same array kernels
(:mod:`env_lib.consensus_env._core`), so copy ``b`` reproduces a single
environment started from the same state exactly (bit for bit, without process
noise).

It is the ``vector_entry_point`` of ``Consensus-v0`` and ``Formation-v0``::

    import env_lib

    envs = env_lib.make_vec("Formation-v0", num_envs=256)
    obs, infos = envs.reset(seed=0)            # obs: (256, 8, 16) float32
    obs, rewards, terminated, truncated, infos = envs.step(envs.laplacian_policy())
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from gymnasium import spaces

from env_lib.consensus_env.consensus_env import (
    _PROXIMITY_DEFAULT_SLOTS,
    ConsensusEnv,
    _check_state_option,
    _known_reset_options,
    _make_kernel,
    _make_renderer,
    _parse_config,
    _policy_gains,
    _sequential_init_warning,
    make_topology,
)
from env_lib.errors import ResetNeededError
from env_lib.utils.rendering import validate_render_mode
from env_lib.utils.vector import BatchedVectorEnv

__all__ = ["ConsensusVectorEnv"]

_CONFIG_ATTRIBUTES = (
    "n_agents",
    "task",
    "topology",
    "dynamics",
    "dt",
    "max_steps",
    "arena_size",
    "max_control",
    "control_cost",
    "noise_std",
    "tolerance",
    "success_bonus",
    "formation_shape",
    "formation_radius",
    "comm_radius",
    "edge_probability",
    "graph_seed",
    "damping",
)


class ConsensusVectorEnv(BatchedVectorEnv):
    """``num_envs`` copies of :class:`~env_lib.consensus_env.ConsensusEnv` simulated as one batch.

    Every copy follows exactly the model, observation, reward, termination
    and info definitions of :class:`~env_lib.consensus_env.ConsensusEnv`;
    see its documentation. Batched shapes (``B = num_envs``, ``n = n_agents``):

    * observations ``(B, n, obs_dim)`` float32, actions ``(B, n, 2)``
      (converted to float32 by the base class, then clipped to
      ``[-max_control, max_control]``);
    * rewards ``(B,)``: the team reward of each copy (mean per-agent reward
      plus the one-off success bonus);
    * infos: ``agent_rewards`` ``(B, n)``, ``error`` ``(B,)``,
      ``algebraic_connectivity`` ``(B,)``, ``adjacency`` ``(B, n, n)``,
      ``success`` ``(B,)`` and ``step`` ``(B,)``, each with the Gymnasium
      ``"_key"`` mask.

    Graphs: static topologies (ring, line, star, complete) are shared by all
    copies; proximity graphs are rebuilt every step for every copy; for
    ``topology="erdos_renyi"`` every copy gets its own connected random graph,
    sampled once at construction from ``numpy.random.default_rng(graph_seed)``
    (copy 0 has the graph of a single environment with the same
    ``graph_seed``). The default ``max_neighbors`` is then the maximum degree
    over all copies, so that no neighbour is left out of an observation.

    Initial states are drawn from the shared generator ``self.np_random`` as
    in the single environment (proximity: conditioned on a connected disk
    graph). ``reset(options={"positions": ..., "velocities": ...})`` starts
    from given states: arrays of shape ``(n, 2)`` (the same state for every
    reset copy) or ``(num_envs, n, 2)`` (row ``b`` for copy ``b``). Combine
    with ``"reset_mask"`` to reset only some copies.

    Parameters
    ----------
    num_envs:
        Number of copies ``B`` (``>= 1``).
    autoreset_mode:
        ``"next_step"`` (default), ``"same_step"`` or ``"disabled"``; see
        :class:`~env_lib.utils.vector.BatchedVectorEnv`.
    render_mode:
        ``None``, ``"rgb_array"`` or ``"human"``; :meth:`render` draws copy 0.
    **kwargs:
        All keyword arguments of :class:`~env_lib.consensus_env.ConsensusEnv`
        with the same defaults and validation.

    Examples
    --------
    >>> envs = ConsensusVectorEnv(num_envs=64, task="formation", formation_shape="wedge")
    >>> obs, infos = envs.reset(seed=0)
    >>> obs.shape
    (64, 8, 16)
    >>> obs, rewards, terminated, truncated, infos = envs.step(envs.laplacian_policy())
    >>> rewards.shape, infos["agent_rewards"].shape
    ((64,), (64, 8))
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    #: Width of one neighbour slot in the observation.
    SLOT_DIM: int = ConsensusEnv.SLOT_DIM
    #: Columns of one neighbour slot.
    neighbor_slot_layout: dict[str, slice] = ConsensusEnv.neighbor_slot_layout

    def __init__(
        self,
        num_envs: int = 1,
        *,
        autoreset_mode: str = "next_step",
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
        if isinstance(num_envs, bool) or int(num_envs) != num_envs or num_envs < 1:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        batch = int(num_envs)
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
        for name in _CONFIG_ATTRIBUTES:
            setattr(self, name, cfg[name])
        n = self.n_agents
        self._offsets: np.ndarray = cfg["offsets"]

        # Communication graphs: (B, n, n) arrays (broadcast views when shared).
        self._dynamic_graph = self.topology == "proximity"
        if self._dynamic_graph:
            default_slots = min(_PROXIMITY_DEFAULT_SLOTS, n - 1)
        else:
            if self.topology == "erdos_renyi":
                rng = np.random.default_rng(self.graph_seed)
                graphs = np.stack(
                    [
                        make_topology(
                            "erdos_renyi", n, edge_probability=self.edge_probability, rng=rng
                        )
                        for _ in range(batch)
                    ]
                )
            else:
                graphs = make_topology(self.topology, n)[None]
            default_slots = int(graphs.sum(axis=-1).max())
        self.max_neighbors: int = (
            default_slots if cfg["max_neighbors"] is None else cfg["max_neighbors"]
        )
        self._kernel = _make_kernel(cfg, self.max_neighbors)
        self.obs_dim = self._kernel.obs_dim
        self._layout: dict[str, slice] = {
            "position": slice(0, 2),
            "velocity": slice(2, 4),
            "offset": slice(4, 6),
            "neighbors": slice(6, self.obs_dim),
        }
        super().__init__(
            batch,
            spaces.Box(low=-np.inf, high=np.inf, shape=(n, self.obs_dim), dtype=np.float32),
            spaces.Box(
                low=-self.max_control, high=self.max_control, shape=(n, 2), dtype=np.float32
            ),
            autoreset_mode=autoreset_mode,
            render_mode=render_mode,
        )

        # Batched episode state.
        self._pos = np.zeros((batch, n, 2), dtype=np.float64)
        self._vel = np.zeros((batch, n, 2), dtype=np.float64)
        self._error = np.full(batch, math.inf)
        self._step = np.zeros(batch, dtype=np.int64)
        self._success = np.zeros(batch, dtype=bool)
        self._rel = (np.zeros((batch, n, n)), np.zeros((batch, n, n)))
        self._sq_x = np.zeros((batch, n, n), dtype=np.float64)
        if self._dynamic_graph:
            self._adj = np.zeros((batch, n, n), dtype=bool)
            self._adj_f = np.zeros((batch, n, n), dtype=np.float64)
            self._deg = np.zeros((batch, n), dtype=np.float64)
            self._lambda2 = np.zeros(batch, dtype=np.float64)
        else:
            adj_f, deg, lambda2 = self._kernel.graph_quantities(graphs)
            shape = (batch, n, n)
            self._adj = np.broadcast_to(graphs, shape)
            self._adj_f = np.broadcast_to(adj_f, shape)
            self._deg = np.broadcast_to(deg, (batch, n))
            self._lambda2 = np.broadcast_to(lambda2, (batch,))

        self._renderer = None
        self._warned_sequential_init = False
        self._warned_policy_gain = False

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------
    @property
    def observation_layout(self) -> dict[str, slice]:
        """Column slices of one observation row (as in the single environment)."""
        return dict(self._layout)

    @property
    def positions(self) -> np.ndarray:
        """Current positions, shape ``(num_envs, n_agents, 2)`` (a copy)."""
        self._require_reset()
        return self._pos.copy()

    @property
    def velocities(self) -> np.ndarray:
        """Current velocities, shape ``(num_envs, n_agents, 2)`` (a copy)."""
        self._require_reset()
        return self._vel.copy()

    @property
    def formation_offsets(self) -> np.ndarray:
        """Formation offsets ``d_i``, shape ``(n_agents, 2)`` (zeros for consensus)."""
        return self._offsets.copy()

    @property
    def adjacency(self) -> np.ndarray:
        """Current communication graphs, bool ``(num_envs, n_agents, n_agents)`` (a copy)."""
        if self._dynamic_graph:
            self._require_reset()
        return np.array(self._adj, copy=True)

    @property
    def laplacian(self) -> np.ndarray:
        """Graph Laplacians ``L = D - A``, shape ``(num_envs, n_agents, n_agents)``."""
        adj = self.adjacency.astype(np.float64)
        laplacian = np.zeros_like(adj)
        index = np.arange(self.n_agents)
        laplacian[:, index, index] = adj.sum(axis=-1)
        laplacian -= adj
        return laplacian

    @property
    def algebraic_connectivity(self) -> np.ndarray:
        """Fiedler values of the current graph Laplacians, shape ``(num_envs,)``."""
        if self._dynamic_graph:
            self._require_reset()
        return np.array(self._lambda2, copy=True)

    @property
    def task_error(self) -> np.ndarray:
        """Current task errors, shape ``(num_envs,)``."""
        self._require_reset()
        return self._error.copy()

    @property
    def step_count(self) -> np.ndarray:
        """Steps taken in the current episode of every copy, shape ``(num_envs,)``."""
        return self._step.copy()

    # ------------------------------------------------------------------
    # Gymnasium vector API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset all copies (or those in ``options["reset_mask"]``).

        Parameters
        ----------
        seed:
            Seed for ``self.np_random``.
        options:
            Optional ``"reset_mask"`` (bool ``(num_envs,)``), ``"positions"``
            and ``"velocities"`` (``(n_agents, 2)`` or
            ``(num_envs, n_agents, 2)``). Unknown keys are ignored with a
            ``UserWarning``.

        Returns
        -------
        observations, infos
        """
        result = super().reset(seed=seed, options=options)
        if self.render_mode == "human":
            self.render()
        return result

    def step(
        self, actions: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Advance every copy by one step (see :class:`BatchedVectorEnv`)."""
        result = super().step(actions)
        if self.render_mode == "human":
            self.render()
        return result

    def render(self) -> np.ndarray | None:
        """Render copy 0 (``(H, W, 3)`` uint8 frame in ``"rgb_array"`` mode, else ``None``)."""
        if self.render_mode is None:
            return None
        self._require_reset()
        if self._renderer is None:
            self._renderer = _make_renderer(self._cfg, self.render_mode, self.metadata)
        pos = self._pos[0]
        centroid = (pos - self._offsets).mean(axis=0)
        return self._renderer.render(
            positions=pos,
            adjacency=np.asarray(self._adj[0]),
            targets=centroid + self._offsets,
            centroid=centroid,
            step=int(self._step[0]),
            error=float(self._error[0]),
            lambda2=float(self._lambda2[0]),
            success=bool(self._success[0]),
        )

    def close_extras(self, **kwargs: Any) -> None:
        """Release the rendering resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Baseline controller
    # ------------------------------------------------------------------
    def laplacian_policy(self, gain: float = 1.0, velocity_gain: float | None = None) -> np.ndarray:
        """Batched :meth:`ConsensusEnv.laplacian_policy <env_lib.consensus_env.ConsensusEnv.laplacian_policy>`.

        Returns
        -------
        numpy.ndarray
            Actions of shape ``(num_envs, n_agents, 2)``, dtype ``float32``.
        """
        self._require_reset()
        gain, velocity_gain = _policy_gains(self._cfg, gain, velocity_gain)
        if self.dynamics == "single" and not self._warned_policy_gain:
            # Gershgorin bound lambda_max(L) <= 2 * max degree.
            if gain * self.dt * 2.0 * float(self._deg.max()) >= 2.0:
                lambda_max = float(np.linalg.eigvalsh(self.laplacian)[:, -1].max())
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
            self._pos, self._vel, self._adj_f, self._deg, gain, velocity_gain
        )

    # ------------------------------------------------------------------
    # BatchedVectorEnv hooks
    # ------------------------------------------------------------------
    def _reset_envs(self, mask: np.ndarray, options: dict[str, Any] | None) -> None:
        index = np.flatnonzero(mask)
        count = index.size
        if count == 0:
            return
        # Called by reset() -> BatchedVectorEnv.reset() -> here: warn at the caller.
        options = _known_reset_options(type(self).__name__, options, stacklevel=4)
        kernel = self._kernel
        if options.get("positions") is not None:
            positions = self._state_option("positions", options["positions"], index)
            positions = np.clip(positions, -self.arena_size, self.arena_size)
        else:
            positions = kernel.sample_positions(
                self.np_random, count, self._dynamic_graph, self._on_sequential_init
            )
        if options.get("velocities") is not None:
            velocities = self._state_option("velocities", options["velocities"], index)
        else:
            velocities = np.zeros((count, self.n_agents, 2), dtype=np.float64)

        self._pos[index] = positions
        self._vel[index] = velocities
        self._step[index] = 0
        self._success[index] = False
        rel, sq_x, _ = kernel.pairwise(positions)
        self._rel[0][index] = rel[0]
        self._rel[1][index] = rel[1]
        self._sq_x[index] = sq_x
        if self._dynamic_graph:
            adj = kernel.proximity(sq_x)
            adj_f, deg, lambda2 = kernel.graph_quantities(adj)
            self._adj[index] = adj
            self._adj_f[index] = adj_f
            self._deg[index] = deg
            self._lambda2[index] = lambda2
        self._error[index] = kernel.task_error(positions)
        if self._renderer is not None and mask[0]:
            self._renderer.reset()

    def _step_envs(
        self, actions: np.ndarray, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        kernel = self._kernel
        control = np.clip(
            np.asarray(actions, dtype=np.float64), -self.max_control, self.max_control
        )
        noise = None
        if kernel.noise_scale > 0.0:
            noise = self.np_random.standard_normal(control.shape)
        self._pos, self._vel = kernel.integrate(self._pos, self._vel, control, noise)
        self._step += 1

        rel, sq_x, sq_y = kernel.pairwise(self._pos)
        self._rel, self._sq_x = rel, sq_x
        if self._dynamic_graph:
            self._update_proximity(sq_x)
        agent_rewards = kernel.agent_rewards(self._adj_f, self._deg, sq_y, control)

        self._error = kernel.task_error(self._pos)
        solved = self._error < self.tolerance
        rewards = np.add.reduce(agent_rewards, axis=-1) / self.n_agents  # the mean
        rewards = np.where(solved & ~self._success, rewards + self.success_bonus, rewards)
        self._success |= solved
        truncated = self._step >= self.max_steps
        return rewards, solved, truncated, self._infos(agent_rewards)

    def _observe(self) -> np.ndarray:
        return self._kernel.observe(
            self._pos, self._vel, self._adj, self._sq_x, self._rel, self._deg
        )

    def _reset_infos(self, mask: np.ndarray) -> dict[str, Any]:
        infos = self._infos(np.zeros((self.num_envs, self.n_agents), dtype=np.float64))
        if not mask.all():
            for key in list(infos):
                infos[f"_{key}"] = mask.copy()
        return infos

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_reset(self) -> None:
        if self._needs_reset:
            raise ResetNeededError("Call reset() before using the environment")

    def _state_option(self, name: str, value: Any, index: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 3:
            return _check_state_option(name, array, (self.num_envs, self.n_agents, 2))[index]
        array = _check_state_option(name, array, (self.n_agents, 2))
        return np.broadcast_to(array, (index.size, self.n_agents, 2)).copy()

    def _on_sequential_init(self) -> None:
        if not self._warned_sequential_init:
            _sequential_init_warning(self._cfg, stacklevel=6)
            self._warned_sequential_init = True

    def _update_proximity(self, sq_x: np.ndarray) -> None:
        """Rebuild the disk graphs; Laplacian quantities only for copies whose graph changed."""
        adj = self._kernel.proximity(sq_x)
        changed = np.flatnonzero((adj != self._adj).any(axis=(1, 2)))
        if changed.size:
            adj_f, deg, lambda2 = self._kernel.graph_quantities(adj[changed])
            self._adj_f[changed] = adj_f
            self._deg[changed] = deg
            self._lambda2[changed] = lambda2
        self._adj = adj

    def _infos(self, agent_rewards: np.ndarray) -> dict[str, Any]:
        return {
            "agent_rewards": np.asarray(agent_rewards, dtype=np.float64),
            "error": self._error.copy(),
            "algebraic_connectivity": np.array(self._lambda2, copy=True),
            "adjacency": np.array(self._adj, copy=True),
            "success": self._success.copy(),
            "step": self._step.copy(),
        }
