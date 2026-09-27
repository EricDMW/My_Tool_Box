"""Natively batched Kuramoto oscillator environment (NumPy).

:class:`KuramotoOscillatorVectorEnv` simulates ``num_envs`` copies of
:class:`~env_lib.kos_env.KuramotoOscillatorEnv` as one batch with the shared
array kernels of :mod:`env_lib.kos_env._kernels`, so copy ``b`` reproduces a
single environment started from the same state exactly (bit for bit, without
phase noise). It is the ``vector_entry_point`` of the NumPy
``KuramotoOscillator-*`` ids::

    import env_lib

    envs = env_lib.make_vec("KuramotoOscillator-v0", num_envs=256)
    obs, infos = envs.reset(seed=0)            # obs: (256, 75) float32

For GPU simulation see
:class:`~env_lib.kos_env.kuramoto_env_torch.KuramotoOscillatorEnvTorch`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from env_lib.errors import ResetNeededError
from env_lib.kos_env._common import order_parameter, phase_coherence, wrap_phases
from env_lib.kos_env._kernels import KuramotoKernel
from env_lib.kos_env.kuramoto_env import KuramotoOscillatorEnv
from env_lib.utils.vector import BatchedVectorEnv

__all__ = ["KuramotoOscillatorVectorEnv"]

_PUBLIC_ATTRIBUTES = (
    "n_oscillators",
    "n_agents",
    "dt",
    "max_steps",
    "coupling_range",
    "control_input_range",
    "natural_freq_range",
    "device",
    "integration_method",
    "reward_type",
    "noise_std",
    "topology",
    "adj_matrix",
    "target_frequency",
    "coupling_mode",
    "constant_coupling_matrix",
    "coupling_strength",
    "normalize_coupling",
    "topology_seed",
    "sync_threshold",
    "sync_bonus",
    "reward_mode",
    "terminate_on_sync",
    "control_cost",
    "n_couplings",
    "coupling_indices",
    "topology_matrix",
    "coupling_matrix",
)


class KuramotoOscillatorVectorEnv(BatchedVectorEnv):
    """``num_envs`` copies of :class:`~env_lib.kos_env.KuramotoOscillatorEnv` as one NumPy batch.

    Every copy follows exactly the model, observation and action layouts,
    rewards, termination and info definitions of
    :class:`~env_lib.kos_env.KuramotoOscillatorEnv`. Batched shapes
    (``B = num_envs``):

    * observations ``(B, obs_dim)`` float32, actions ``(B, action_dim)``
      (converted to float32 by the base class, then clipped to the bounds);
    * rewards ``(B,)`` (following ``reward_mode``, including the
      synchronisation bonus of each copy);
    * infos (each with the Gymnasium ``"_key"`` mask): ``order_parameter``,
      ``phase_coherence`` ``(B,)``; ``synchronized`` ``(B,)`` bool (step
      infos); ``phases``, ``natural_frequencies``,
      ``dphases_dt`` ``(B, N)``; ``coupling_matrix`` ``(B, N, N)`` (a
      read-only broadcast view in constant mode); ``agent_rewards``
      ``(B, n_agents)``; ``step_count``, ``n_agents`` and ``device`` ``(B,)``.

    Initial states are drawn from the shared generator ``self.np_random``.
    ``reset(options=...)`` accepts the overrides of the single environment,
    ``"phases"``, ``"natural_frequencies"`` and (dynamic mode)
    ``"coupling_strengths"``, as arrays of shape ``(size,)`` (the same for
    every reset copy) or ``(num_envs, size)`` (row ``b`` for copy ``b``),
    together with the optional ``"reset_mask"``.

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
        All other keyword arguments of
        :class:`~env_lib.kos_env.KuramotoOscillatorEnv`, with the same
        defaults and validation.

    Examples
    --------
    >>> envs = KuramotoOscillatorVectorEnv(num_envs=64, n_oscillators=6, coupling_mode="constant")
    >>> obs, infos = envs.reset(seed=0)
    >>> obs.shape
    (64, 18)
    >>> obs, rewards, terminated, truncated, infos = envs.step(envs.action_space.sample())
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        num_envs: int = 1,
        *,
        autoreset_mode: str = "next_step",
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
        reward_mode: str = "dense",
        terminate_on_sync: bool = True,
        control_cost: float = 0.0,
    ):
        # A single environment validates the arguments, builds the topology
        # and the spaces, and draws copy 0 in render().
        self._single = KuramotoOscillatorEnv(
            n_oscillators=n_oscillators,
            n_agents=n_agents,
            dt=dt,
            max_steps=max_steps,
            coupling_range=coupling_range,
            control_input_range=control_input_range,
            natural_freq_range=natural_freq_range,
            device=device,
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
            reward_mode=reward_mode,
            terminate_on_sync=terminate_on_sync,
            control_cost=control_cost,
        )
        single = self._single
        for name in _PUBLIC_ATTRIBUTES:
            setattr(self, name, getattr(single, name))
        super().__init__(
            num_envs,
            single.observation_space,
            single.action_space,
            autoreset_mode=autoreset_mode,
            render_mode=render_mode,
        )
        self._kernel = KuramotoKernel(single)
        batch, n = self.num_envs, self.n_oscillators
        self._dynamic = self.coupling_mode == "dynamic"
        self._phases = np.zeros((batch, n))
        self._freqs = np.zeros((batch, n))
        self._control = np.zeros((batch, n))
        self._strengths = np.zeros((batch, self.n_couplings)) if self._dynamic else None
        self._matrix_shape = (batch, n, n)
        self._step = np.zeros(batch, dtype=np.int64)
        self._last_reward = np.full(batch, np.nan)
        self._previous_signal = np.zeros(batch)  # reward_mode="progress"
        self._device_info = np.full(batch, self.device, dtype=object)
        self._n_agents_info = np.full(batch, self.n_agents, dtype=np.int64)

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------
    @property
    def phases(self) -> np.ndarray:
        """Current phases in ``[-pi, pi)``, shape ``(num_envs, N)`` (a copy)."""
        self._require_reset()
        return self._phases.copy()

    @property
    def natural_frequencies(self) -> np.ndarray:
        """Natural frequencies, shape ``(num_envs, N)`` (a copy)."""
        self._require_reset()
        return self._freqs.copy()

    @property
    def coupling_strengths(self) -> np.ndarray | None:
        """Edge coupling strengths ``(num_envs, M)`` (dynamic mode), else ``None``."""
        self._require_reset()
        return None if self._strengths is None else self._strengths.copy()

    @property
    def control_inputs(self) -> np.ndarray:
        """Control inputs of the last step, shape ``(num_envs, N)`` (a copy)."""
        self._require_reset()
        return np.array(self._control, copy=True)

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
            Optional ``"reset_mask"`` and state overrides ``"phases"``,
            ``"natural_frequencies"``, ``"coupling_strengths"``. Unknown keys
            are ignored with a ``UserWarning``.

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
        """Advance every copy by one time step ``dt`` (see :class:`BatchedVectorEnv`)."""
        result = super().step(actions)
        if self.render_mode == "human":
            self.render()
        return result

    def render(self) -> np.ndarray | None:
        """Render copy 0 (``(560, 1000, 3)`` uint8 frame in ``"rgb_array"`` mode)."""
        if self.render_mode is None:
            return None
        self._require_reset()
        single = self._single
        single.phases = self._phases[0]
        single.natural_frequencies = self._freqs[0]
        single.control_inputs = np.asarray(self._control[0])
        single.coupling_strengths = None if self._strengths is None else self._strengths[0]
        single._current_coupling = self._coupling_matrices(slice(0, 1))[0]
        single.step_count = int(self._step[0])
        last = float(self._last_reward[0])
        single._last_reward = None if np.isnan(last) else last
        return single.render()

    def close_extras(self, **kwargs: Any) -> None:
        """Release the rendering resources."""
        single = getattr(self, "_single", None)  # absent if __init__ failed early
        if single is not None:
            single.close()

    # ------------------------------------------------------------------
    # BatchedVectorEnv hooks
    # ------------------------------------------------------------------
    def _reset_envs(self, mask: np.ndarray, options: dict[str, Any] | None) -> None:
        index = np.flatnonzero(mask)
        count = index.size
        if count == 0:
            return
        # Called by reset() -> BatchedVectorEnv.reset() -> here: warn at the caller.
        overrides = self._single._parse_reset_options(options, stacklevel=5)
        kernel = self._kernel
        phases, freqs, strengths = kernel.sample_state(self.np_random, count)
        for key, value in overrides.items():
            if value.ndim == 2:
                if value.shape[0] != self.num_envs:
                    raise ValueError(
                        f"reset option {key!r} has {value.shape[0]} rows but num_envs="
                        f"{self.num_envs}"
                    )
                value = value[index]
            value = np.broadcast_to(value, (count, value.shape[-1])).copy()
            if key == "phases":
                phases = wrap_phases(value)
            elif key == "natural_frequencies":
                freqs = value
            else:
                strengths = value

        # In-place row updates are safe: infos hold copies of these arrays. The
        # coupling matrices of dynamic mode are rebuilt from the strengths when
        # needed (reset infos, rendering), so arrays handed out earlier are
        # never modified.
        self._phases[index] = phases
        self._freqs[index] = freqs
        self._control[index] = 0.0
        if self._dynamic:
            self._strengths[index] = strengths
        self._step[index] = 0
        self._last_reward[index] = np.nan
        if self.reward_mode == "progress":
            # Signal of the initial state, row by row as in the single environment.
            coupling = (
                kernel.coupling_from_strengths(strengths) if self._dynamic else kernel.constant
            )
            dphases_dt = kernel.dynamics(phases, freqs, coupling, np.zeros_like(phases))
            self._previous_signal[index] = kernel.signal(phases, dphases_dt)[0]
        if mask[0] and self.render_mode is not None:
            self._single.phase_history = [self._phases[0].copy()]
            self._single._reset_renderer()

    def _step_envs(
        self, actions: np.ndarray, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        kernel = self._kernel
        n = self.n_oscillators
        action = kernel.clip_actions(np.asarray(actions, dtype=np.float64))
        control = action[:, :n]
        if self._dynamic:
            self._strengths = action[:, n:]
            coupling = kernel.coupling_from_strengths(self._strengths)
        else:
            coupling = kernel.constant
        phases, dphases_dt = kernel.integrate(self._phases, self._freqs, coupling, control)
        self._phases = kernel.add_noise(phases, self.np_random)
        self._control = control
        self._step += 1

        truncated = self._step >= self.max_steps
        rewards, signal, r, coherence, synchronized, terminated = kernel.rewards(
            self._phases,
            dphases_dt,
            control=control,
            previous=self._previous_signal,
            truncated=truncated,
        )
        if self.reward_mode == "progress":
            self._previous_signal = signal
        self._last_reward = rewards.copy()
        if self.render_mode is not None:
            self._single._append_history(self._phases[0].copy())
        if coupling.ndim == 2:  # constant mode: read-only view, no copy
            coupling = np.broadcast_to(coupling, self._matrix_shape)
        infos = self._infos(r, coherence, coupling)
        infos["synchronized"] = synchronized
        infos["dphases_dt"] = dphases_dt
        infos["agent_rewards"] = np.repeat(rewards[:, None], self.n_agents, axis=1)
        return rewards, terminated, truncated, infos

    def _observe(self) -> np.ndarray:
        return self._kernel.observe(self._phases, self._freqs, self._strengths, self._control)

    def _reset_infos(self, mask: np.ndarray) -> dict[str, Any]:
        # Only the rows of reset copies are reported (masks); compute just those.
        index = np.flatnonzero(mask)
        r = np.zeros(self.num_envs)
        coherence = np.zeros(self.num_envs)
        r[index] = order_parameter(self._phases[index])
        coherence[index] = phase_coherence(self._phases[index])
        if self._dynamic:
            coupling = np.zeros(self._matrix_shape)
            coupling[index] = self._kernel.coupling_from_strengths(self._strengths[index])
        else:
            coupling = self._coupling_matrices(slice(None))
        infos = self._infos(r, coherence, coupling)
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

    def _coupling_matrices(self, rows: slice) -> np.ndarray:
        """Current coupling matrices of the selected copies (read-only view in constant mode)."""
        if self._dynamic:
            return self._kernel.coupling_from_strengths(self._strengths[rows])
        count = len(range(self.num_envs)[rows])
        return np.broadcast_to(self._kernel.constant, (count, *self._matrix_shape[1:]))

    def _infos(self, r: np.ndarray, coherence: np.ndarray, coupling: np.ndarray) -> dict[str, Any]:
        return {
            "order_parameter": r,
            "phase_coherence": coherence,
            "step_count": self._step.copy(),
            "natural_frequencies": self._freqs.copy(),
            "phases": self._phases.copy(),
            "coupling_matrix": coupling,
            "device": self._device_info.copy(),
            "n_agents": self._n_agents_info.copy(),
        }
