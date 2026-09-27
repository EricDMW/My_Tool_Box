"""Kuramoto oscillator synchronisation environment (NumPy backend).

The environment simulates ``N`` coupled phase oscillators

.. code-block:: text

    dtheta_i/dt = omega_i + a_i(t) + c * sum_j K_ij(t) sin(theta_j - theta_i)

with natural frequencies ``omega_i``, control inputs ``a_i``, a coupling matrix
``K`` restricted to a network topology and ``c = 1`` (or ``1/N`` with
``normalize_coupling=True``). Optional Gaussian phase noise with standard
deviation ``noise_std`` is added after every integration step. The agent is
rewarded for synchronising the oscillators.

See :class:`env_lib.kos_env.kuramoto_env_torch.KuramotoOscillatorEnvTorch` for
a batched PyTorch implementation of the same model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from env_lib.errors import ResetNeededError
from env_lib.kos_env._common import (
    KuramotoEnvBase,
    combine_reward,
    order_parameter,
    phase_coherence,
    wrap_phases,
)

__all__ = ["KuramotoOscillatorEnv"]


class KuramotoOscillatorEnv(KuramotoEnvBase):
    """Kuramoto oscillator synchronisation environment (NumPy, single system).

    Observation (``float32``, unbounded ``Box``)
        ``[phases (N), natural_frequencies (N), coupling_strengths (M), control_inputs (N)]``
        in dynamic coupling mode and ``[phases, natural_frequencies, control_inputs]``
        (``3N``) in constant mode. Phases are wrapped to ``[-pi, pi)``; ``M`` is the
        number of undirected edges of the topology (``N(N-1)/2`` when fully connected).

    Action (``float32`` ``Box``)
        ``[control_inputs (N), coupling_strengths (M)]`` in dynamic mode, bounded by
        ``control_input_range`` and ``coupling_range``; only the ``N`` control inputs in
        constant mode. Actions are clipped to the bounds. Coupling strength ``k`` of
        edge ``(i, j)`` sets ``K_ij = K_ji = k``; the edge order is
        :attr:`coupling_indices` (row-major upper triangle).

    Reward
        ``"order_parameter"``: ``r = |mean_j exp(i theta_j)|``;
        ``"phase_coherence"``: ``exp(-Var(theta))``; ``"combined"``: their sum;
        ``"frequency_synchronization"``: ``-mean_i |dtheta_i/dt - target_frequency|``
        (derivative at the start of the step). When ``r > sync_threshold`` the episode
        terminates and ``sync_bonus`` is added, except for
        ``"frequency_synchronization"``.

    Parameters
    ----------
    n_oscillators:
        Number of oscillators ``N`` (>= 2).
    n_agents:
        Compatibility argument: the NumPy backend simulates one system. It sets the
        length of ``info["agent_rewards"]`` and is reported as ``info["n_agents"]``.
    dt:
        Integration time step.
    max_steps:
        Episode length; reaching it sets ``truncated=True``.
    coupling_range:
        Bounds of the coupling-strength actions (dynamic mode) and of the random
        initial coupling strengths.
    control_input_range:
        Bounds of the control inputs ``a_i``.
    natural_freq_range:
        Natural frequencies are drawn uniformly from this range at every reset.
    device:
        Ignored; accepted for compatibility with the PyTorch backend.
    render_mode:
        ``None``, ``"rgb_array"`` or ``"human"``.
    integration_method:
        ``"euler"`` or ``"rk4"``.
    reward_type:
        ``"order_parameter"``, ``"phase_coherence"``, ``"combined"`` or
        ``"frequency_synchronization"``.
    noise_std:
        Standard deviation of the Gaussian phase noise added after each step.
    topology:
        ``"fully_connected"``, ``"ring"``, ``"star"``, ``"random"`` or ``"custom"``.
    adj_matrix:
        Custom ``(N, N)`` adjacency matrix (non-zero entries are edges); overrides
        ``topology``, which is then reported as ``"custom"``.
    target_frequency:
        Target of the ``"frequency_synchronization"`` reward.
    coupling_mode:
        ``"dynamic"`` (coupling strengths are part of the action) or ``"constant"``.
    constant_coupling_matrix:
        Fixed ``(N, N)`` coupling matrix for constant mode. When omitted, constant mode
        uses ``coupling_strength * topology_matrix``.
    coupling_strength:
        Uniform coupling strength of constant mode without an explicit matrix.
    normalize_coupling:
        Divide the coupling sum by ``N`` (classical ``K/N`` scaling).
    topology_seed:
        Seed of the local random generator that builds the ``"random"`` topology.
    sync_threshold:
        The episode terminates when the order parameter exceeds this value
        (values above 1 disable synchronisation termination).
    sync_bonus:
        Reward bonus on synchronisation (not applied to ``"frequency_synchronization"``).

    Raises
    ------
    ValueError, TypeError
        If an argument is invalid.

    Examples
    --------
    >>> env = KuramotoOscillatorEnv(n_oscillators=6, coupling_mode="constant")
    >>> obs, info = env.reset(seed=0)
    >>> obs.shape
    (18,)
    >>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """

    _backend_name = "numpy"

    def __init__(
        self,
        n_oscillators: int = 10,
        n_agents: int = 1,
        dt: float = 0.01,
        max_steps: int = 1000,
        coupling_range: tuple[float, float] = (0.0, 5.0),
        control_input_range: tuple[float, float] = (-1.0, 1.0),
        natural_freq_range: tuple[float, float] = (0.5, 2.0),
        device: str = "cpu",
        render_mode: str | None = None,
        integration_method: str = "euler",
        reward_type: str = "order_parameter",
        noise_std: float = 0.0,
        topology: str = "fully_connected",
        adj_matrix: np.ndarray | None = None,
        target_frequency: float = 1.0,
        coupling_mode: str = "dynamic",
        constant_coupling_matrix: np.ndarray | None = None,
        coupling_strength: float = 1.0,
        normalize_coupling: bool = False,
        topology_seed: int = 42,
        sync_threshold: float = 0.99,
        sync_bonus: float = 10.0,
    ):
        super().__init__()
        self._init_common(
            n_oscillators=n_oscillators,
            n_agents=n_agents,
            dt=dt,
            max_steps=max_steps,
            coupling_range=coupling_range,
            control_input_range=control_input_range,
            natural_freq_range=natural_freq_range,
            render_mode=render_mode,
            integration_method=integration_method,
            reward_type=reward_type,
            noise_std=noise_std,
            topology=topology,
            adj_matrix=adj_matrix,
            target_frequency=target_frequency,
            coupling_mode=coupling_mode,
            constant_coupling_matrix=constant_coupling_matrix,
            coupling_strength=coupling_strength,
            normalize_coupling=normalize_coupling,
            topology_seed=topology_seed,
            sync_threshold=sync_threshold,
            sync_bonus=sync_bonus,
        )
        self.device = device  # ignored; kept for API compatibility with the torch backend
        self.topology_matrix: np.ndarray = self._topology_np
        self.coupling_matrix: np.ndarray | None = self._constant_np

        self.phases: np.ndarray | None = None
        self.natural_frequencies: np.ndarray | None = None
        self.coupling_strengths: np.ndarray | None = None
        self.control_inputs: np.ndarray | None = None
        self._current_coupling: np.ndarray | None = None
        self._last_reward: float | None = None

    # ------------------------------------------------------------------ model
    def _coupling_matrix_from_actions(self, actions: np.ndarray) -> np.ndarray:
        """Scatter per-edge coupling strengths into a symmetric ``(N, N)`` matrix."""
        n = self.n_oscillators
        matrix = np.zeros((n, n))
        matrix[self._edge_rows, self._edge_cols] = actions
        matrix[self._edge_cols, self._edge_rows] = actions
        return matrix

    def _kuramoto_dynamics(
        self,
        phases: np.ndarray,
        natural_frequencies: np.ndarray,
        coupling_matrix: np.ndarray,
        control_inputs: np.ndarray,
    ) -> np.ndarray:
        """Phase velocities ``omega_i + a_i + c * sum_j K_ij sin(theta_j - theta_i)``.

        Uses ``sin(theta_j - theta_i) = sin(theta_j) cos(theta_i) - cos(theta_j) sin(theta_i)``,
        i.e. two matrix-vector products instead of ``N^2`` sine evaluations.
        """
        sin, cos = np.sin(phases), np.cos(phases)
        coupling = cos * (coupling_matrix @ sin) - sin * (coupling_matrix @ cos)
        if self.normalize_coupling:
            coupling *= self._coupling_factor
        return natural_frequencies + control_inputs + coupling

    def _integrate_rk4(
        self,
        phases: np.ndarray,
        natural_frequencies: np.ndarray,
        coupling_matrix: np.ndarray,
        control_inputs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classical fourth-order Runge-Kutta step. Returns ``(new_phases, k1)``."""
        dt = self.dt
        args = (natural_frequencies, coupling_matrix, control_inputs)
        k1 = self._kuramoto_dynamics(phases, *args)
        k2 = self._kuramoto_dynamics(phases + 0.5 * dt * k1, *args)
        k3 = self._kuramoto_dynamics(phases + 0.5 * dt * k2, *args)
        k4 = self._kuramoto_dynamics(phases + dt * k3, *args)
        return phases + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), k1

    def _compute_order_parameter(self, phases: np.ndarray) -> float:
        """Kuramoto order parameter ``r`` in ``[0, 1]``."""
        return float(order_parameter(phases))

    def _compute_phase_coherence(self, phases: np.ndarray) -> float:
        """``exp(-Var(theta))`` with phases wrapped to ``[-pi, pi)``."""
        return float(phase_coherence(phases))

    # ------------------------------------------------------------------ gym API
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Sample a new initial state.

        Phases are uniform in ``[-pi, pi)``, natural frequencies uniform in
        ``natural_freq_range`` and (dynamic mode) coupling strengths uniform in
        ``coupling_range``; control inputs start at zero.

        Parameters
        ----------
        seed:
            Seed for ``self.np_random``.
        options:
            Optional overrides of the sampled state: ``"phases"``,
            ``"natural_frequencies"`` (shape ``(N,)``) and, in dynamic mode,
            ``"coupling_strengths"`` (shape ``(M,)``).

        Returns
        -------
        (numpy.ndarray, dict)
            The observation and the info dictionary.
        """
        super().reset(seed=seed)
        overrides = self._parse_reset_options(options)
        n, rng = self.n_oscillators, self.np_random
        freq_low, freq_high = self.natural_freq_range
        phases = (rng.random(n) * 2 - 1) * np.pi
        natural_frequencies = rng.random(n) * (freq_high - freq_low) + freq_low
        coupling_strengths = None
        if self.coupling_mode == "dynamic":
            low, high = self.coupling_range
            coupling_strengths = rng.random(self.n_couplings) * (high - low) + low

        for key, value in overrides.items():
            if value.ndim != 1:
                raise ValueError(f"reset option {key!r} must be one-dimensional for this backend")
        if "phases" in overrides:
            phases = wrap_phases(overrides["phases"].copy())
        if "natural_frequencies" in overrides:
            natural_frequencies = overrides["natural_frequencies"].copy()
        if "coupling_strengths" in overrides:
            coupling_strengths = overrides["coupling_strengths"].copy()

        self.phases = phases
        self.natural_frequencies = natural_frequencies
        self.coupling_strengths = coupling_strengths
        self.control_inputs = np.zeros(n)
        self.step_count = 0
        self._last_reward = None
        if coupling_strengths is not None:
            self._current_coupling = self._coupling_matrix_from_actions(coupling_strengths)
        else:
            self._current_coupling = self.coupling_matrix
        self.phase_history = [phases.copy()]
        self._reset_renderer()

        info = {
            "order_parameter": self._compute_order_parameter(phases),
            "phase_coherence": self._compute_phase_coherence(phases),
            "natural_frequencies": natural_frequencies.copy(),
            "phases": phases.copy(),
            "coupling_matrix": self._current_coupling.copy(),
            "step_count": 0,
            "device": self.device,
            "n_agents": self.n_agents,
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the system by one time step ``dt``.

        Returns
        -------
        observation, reward, terminated, truncated, info
            ``terminated`` is ``True`` when the order parameter exceeds
            ``sync_threshold``; ``truncated`` is ``True`` once ``max_steps`` steps
            have been taken. ``info`` contains ``order_parameter``,
            ``phase_coherence``, ``step_count``, ``natural_frequencies``, ``phases``,
            ``coupling_matrix``, ``dphases_dt``, ``device``, ``n_agents`` and
            ``agent_rewards`` (shape ``(n_agents,)``).
        """
        if self.phases is None:
            raise ResetNeededError("Call reset() before step().")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"action must have shape {self.action_space.shape}, got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("action contains non-finite values")
        action = np.clip(action, self._action_low, self._action_high)
        n = self.n_oscillators
        self.control_inputs = action[:n]
        if self.coupling_mode == "dynamic":
            self.coupling_strengths = action[n:]
            coupling_matrix = self._coupling_matrix_from_actions(self.coupling_strengths)
        else:
            coupling_matrix = self.coupling_matrix

        if self.integration_method == "euler":
            dphases_dt = self._kuramoto_dynamics(
                self.phases, self.natural_frequencies, coupling_matrix, self.control_inputs
            )
            phases = self.phases + dphases_dt * self.dt
        else:
            phases, dphases_dt = self._integrate_rk4(
                self.phases, self.natural_frequencies, coupling_matrix, self.control_inputs
            )
        if self.noise_std > 0:
            phases = phases + self.np_random.normal(0.0, self.noise_std, n)
        self.phases = wrap_phases(phases)
        self._current_coupling = coupling_matrix
        self._append_history(self.phases.copy())
        self.step_count += 1

        r = self._compute_order_parameter(self.phases)
        coherence = self._compute_phase_coherence(self.phases)
        frequency_error = (
            float(np.mean(np.abs(dphases_dt - self.target_frequency)))
            if self.reward_type == "frequency_synchronization"
            else 0.0
        )
        reward = float(combine_reward(self.reward_type, r, coherence, frequency_error))
        terminated = bool(r > self.sync_threshold)
        if terminated and self.reward_type != "frequency_synchronization":
            reward += self.sync_bonus
        truncated = bool(self.step_count >= self.max_steps)
        self._last_reward = reward

        info = {
            "order_parameter": r,
            "phase_coherence": coherence,
            "step_count": self.step_count,
            "natural_frequencies": self.natural_frequencies.copy(),
            "phases": self.phases.copy(),
            "coupling_matrix": coupling_matrix.copy(),
            "device": self.device,
            "n_agents": self.n_agents,
            "dphases_dt": dphases_dt,
            "agent_rewards": np.full(self.n_agents, reward, dtype=np.float64),
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ internals
    def _get_obs(self) -> np.ndarray:
        parts = [self.phases, self.natural_frequencies]
        if self.coupling_mode == "dynamic":
            parts.append(self.coupling_strengths)
        parts.append(self.control_inputs)
        return np.concatenate(parts).astype(np.float32)

    def _frame(self):
        from env_lib.kos_env.rendering import KuramotoFrame

        return KuramotoFrame(
            phases=self.phases,
            natural_frequencies=self.natural_frequencies,
            coupling_matrix=self._current_coupling,
            history=self.phase_history,
            step=self.step_count,
            max_steps=self.max_steps,
            dt=self.dt,
            reward=self._last_reward,
        )
