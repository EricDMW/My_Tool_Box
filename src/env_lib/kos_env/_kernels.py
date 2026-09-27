"""Batch-first NumPy kernels of the Kuramoto model (private).

All functions of :class:`KuramotoKernel` work on arrays with any leading
batch shape ``(...)``: phases, natural frequencies and control inputs
``(..., N)``, coupling strengths ``(..., M)`` and coupling matrices
``(..., N, N)`` (or one shared ``(N, N)`` matrix).
:class:`~env_lib.kos_env.KuramotoOscillatorEnv` calls them with unbatched
arrays (no leading dimension, as cheap as dedicated scalar code) and
:class:`~env_lib.kos_env.vector.KuramotoOscillatorVectorEnv` with a leading
``num_envs`` dimension. Every system is computed with the same operations
whatever the batch shape is (one matrix-vector product per system), so a copy
of the vector environment reproduces the single environment bit for bit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from env_lib.kos_env._common import combine_reward, order_parameter, phase_coherence, wrap_phases

if TYPE_CHECKING:  # pragma: no cover
    from env_lib.kos_env._common import KuramotoEnvBase

__all__ = ["KuramotoKernel"]


class KuramotoKernel:
    """Dynamics, reward and observation kernels for a validated configuration.

    Parameters
    ----------
    env:
        A configured Kuramoto environment (its ``_init_common`` has run); the
        kernel copies the validated parameters it needs.
    """

    def __init__(self, env: KuramotoEnvBase) -> None:
        self.n = env.n_oscillators
        self.n_couplings = env.n_couplings
        self.dynamic = env.coupling_mode == "dynamic"
        self.rk4 = env.integration_method == "rk4"
        self.dt = env.dt
        self.normalize = env.normalize_coupling
        self.coupling_factor = env._coupling_factor
        self.constant = env._constant_np
        self.reward_type = env.reward_type
        self.frequency_reward = env.reward_type == "frequency_synchronization"
        self.target_frequency = env.target_frequency
        self.sync_threshold = env.sync_threshold
        self.sync_bonus = env.sync_bonus
        self.noise_std = env.noise_std
        self.natural_freq_range = env.natural_freq_range
        self.coupling_range = env.coupling_range
        self.action_low = env._action_low
        self.action_high = env._action_high
        self._edge_rows = env._edge_rows
        self._edge_cols = env._edge_cols

    # ------------------------------------------------------------------ state
    def sample_state(
        self, rng: np.random.Generator, count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Random initial ``(phases, natural_frequencies, coupling_strengths)``.

        Phases are uniform in ``[-pi, pi)``, natural frequencies uniform in
        ``natural_freq_range`` and (dynamic mode) coupling strengths uniform in
        ``coupling_range``. For ``count = 1`` the random stream is consumed as
        by earlier releases.
        """
        n = self.n
        freq_low, freq_high = self.natural_freq_range
        phases = (rng.random((count, n)) * 2 - 1) * np.pi
        natural_frequencies = rng.random((count, n)) * (freq_high - freq_low) + freq_low
        strengths = None
        if self.dynamic:
            low, high = self.coupling_range
            strengths = rng.random((count, self.n_couplings)) * (high - low) + low
        return phases, natural_frequencies, strengths

    def coupling_from_strengths(self, strengths: np.ndarray) -> np.ndarray:
        """Scatter ``(..., M)`` edge strengths into symmetric ``(..., N, N)`` matrices."""
        matrix = np.zeros(strengths.shape[:-1] + (self.n, self.n))
        matrix[..., self._edge_rows, self._edge_cols] = strengths
        matrix[..., self._edge_cols, self._edge_rows] = strengths
        return matrix

    def clip_actions(self, actions: np.ndarray) -> np.ndarray:
        """Clip float64 actions ``(..., action_dim)`` to the action bounds."""
        return np.clip(actions, self.action_low, self.action_high)

    # ------------------------------------------------------------------ dynamics
    def dynamics(
        self,
        phases: np.ndarray,
        natural_frequencies: np.ndarray,
        coupling: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        """Phase velocities ``omega_i + a_i + c * sum_j K_ij sin(theta_j - theta_i)``, ``(..., N)``.

        Uses ``sin(theta_j - theta_i) = sin(theta_j) cos(theta_i) - cos(theta_j) sin(theta_i)``,
        i.e. two matrix-vector products per copy instead of ``N^2`` sine evaluations.
        """
        sin, cos = np.sin(phases), np.cos(phases)
        if sin.ndim == 1:  # one system: plain matrix-vector products
            k_sin, k_cos = coupling @ sin, coupling @ cos
        else:  # the same matrix-vector product for every system
            k_sin = np.matmul(coupling, sin[..., None])[..., 0]
            k_cos = np.matmul(coupling, cos[..., None])[..., 0]
        interaction = cos * k_sin - sin * k_cos
        if self.normalize:
            interaction *= self.coupling_factor
        return natural_frequencies + control + interaction

    def integrate(
        self,
        phases: np.ndarray,
        natural_frequencies: np.ndarray,
        coupling: np.ndarray,
        control: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One Euler or RK4 step (without noise and wrapping).

        Returns
        -------
        tuple
            ``(new_phases, dphases_dt)`` where ``dphases_dt`` is the phase
            velocity at the start of the step.
        """
        dt = self.dt
        args = (natural_frequencies, coupling, control)
        k1 = self.dynamics(phases, *args)
        if not self.rk4:
            return phases + k1 * dt, k1
        k2 = self.dynamics(phases + 0.5 * dt * k1, *args)
        k3 = self.dynamics(phases + 0.5 * dt * k2, *args)
        k4 = self.dynamics(phases + dt * k3, *args)
        return phases + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), k1

    def add_noise(self, phases: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Add Gaussian phase noise (if ``noise_std > 0``) and wrap to ``[-pi, pi)``."""
        if self.noise_std > 0:
            phases = phases + rng.normal(0.0, self.noise_std, phases.shape)
        return wrap_phases(phases)

    # ------------------------------------------------------------------ reward
    def rewards(
        self, phases: np.ndarray, dphases_dt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Rewards and synchronisation measures of the new state.

        Returns
        -------
        tuple
            ``(rewards, order_parameter, coherence, terminated)``, each of the
            leading shape;
            rewards include ``sync_bonus`` where the copy synchronised (not for
            ``"frequency_synchronization"``).
        """
        r = order_parameter(phases)
        coherence = phase_coherence(phases)
        if self.frequency_reward:
            frequency_error = np.mean(np.abs(dphases_dt - self.target_frequency), axis=-1)
        else:
            frequency_error = 0.0
        rewards = combine_reward(self.reward_type, r, coherence, frequency_error)
        terminated = r > self.sync_threshold
        if not self.frequency_reward:
            if np.ndim(rewards) == 0:  # one system
                rewards = rewards + self.sync_bonus if terminated else rewards
            else:
                rewards = np.where(terminated, rewards + self.sync_bonus, rewards)
        return rewards, r, coherence, terminated

    # ------------------------------------------------------------------ observation
    def observe(
        self,
        phases: np.ndarray,
        natural_frequencies: np.ndarray,
        strengths: np.ndarray | None,
        control: np.ndarray,
    ) -> np.ndarray:
        """Observations ``(..., obs_dim)`` float32: phases, frequencies, strengths, controls."""
        parts = [phases, natural_frequencies]
        if strengths is not None:
            parts.append(strengths)
        parts.append(control)
        return np.concatenate(parts, axis=-1).astype(np.float32)
