"""Batched Kuramoto oscillator environment (PyTorch backend).

Simulates ``n_agents`` independent copies ("systems") of the Kuramoto network
of :class:`env_lib.kos_env.kuramoto_env.KuramotoOscillatorEnv` on a CPU or GPU
with PyTorch tensors. With identical initial states and actions both backends
produce the same trajectories (up to float32 rounding).

The Gymnasium interface (``reset``/``step``) exposes system 0: the observation,
the scalar reward and ``terminated`` refer to it, while ``info`` carries the
per-system arrays. Batched training code can use :meth:`get_batch_observations`,
:meth:`get_batch_rewards` and pass actions of shape ``(n_agents, action_dim)``.
"""

from __future__ import annotations

import math
from typing import Any, Union

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installation
    raise ImportError(
        "KuramotoOscillatorEnvTorch requires PyTorch: "
        'pip install "my-tool-box[torch]" (or use the NumPy KuramotoOscillator ids)'
    ) from exc

from env_lib.errors import ResetNeededError
from env_lib.kos_env._common import KuramotoEnvBase, check_int, combine_reward, wrap_phases

__all__ = ["KuramotoOscillatorEnvTorch"]

ArrayLike = Union[np.ndarray, torch.Tensor]


def _resolve_device(device: str | torch.device) -> torch.device:
    """Turn ``"auto"``/``"cpu"``/``"cuda[:k]"``/``torch.device`` into a usable device."""
    if isinstance(device, torch.device):
        resolved = device
    elif isinstance(device, str):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            resolved = torch.device(device)
        except RuntimeError:
            raise ValueError(
                f"device must be 'auto', 'cpu', 'cuda' or 'cuda:<index>', got {device!r}"
            ) from None
    else:
        raise TypeError(f"device must be a str or torch.device, got {type(device).__name__}")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"device={str(resolved)!r} was requested but CUDA is not available; "
            "use device='auto' or device='cpu'."
        )
    return resolved


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Copy a tensor to a NumPy array that does not share memory with the environment."""
    tensor = tensor.detach()
    if tensor.device.type == "cpu":
        return tensor.numpy().copy()  # cheaper than .to("cpu", copy=True) on the host
    return tensor.to("cpu").numpy()  # the device-to-host transfer already copies


class KuramotoOscillatorEnvTorch(KuramotoEnvBase):
    """Batched Kuramoto oscillator environment on PyTorch tensors.

    The model, observation layout, action layout and rewards are those of
    :class:`env_lib.kos_env.kuramoto_env.KuramotoOscillatorEnv`; all state
    tensors have a leading batch dimension of size ``n_agents``.

    * ``step(action)`` accepts an action of shape ``(action_dim,)`` (applied to
      every system) or ``(n_agents, action_dim)`` (one row per system), as a NumPy
      array or a tensor.
    * The returned observation, scalar reward and ``terminated`` flag refer to
      system 0; ``truncated`` is set when ``max_steps`` is reached.
    * ``info["agent_rewards"]`` holds the per-system rewards (shape ``(n_agents,)``,
      including the synchronisation bonus of each system); ``order_parameter``,
      ``phase_coherence``, ``phases``, ``natural_frequencies``, ``dphases_dt`` and
      ``coupling_matrix`` are per-system NumPy arrays.

    Parameters
    ----------
    **model arguments:
        ``n_oscillators``, ``dt``, ``max_steps``, ``coupling_range``,
        ``control_input_range``, ``natural_freq_range``, ``render_mode``,
        ``integration_method``, ``reward_type``, ``noise_std``, ``topology``,
        ``adj_matrix``, ``target_frequency``, ``coupling_mode``,
        ``constant_coupling_matrix``, ``coupling_strength``, ``normalize_coupling``,
        ``topology_seed``, ``sync_threshold`` and ``sync_bonus`` have the meaning and
        defaults documented in :class:`~env_lib.kos_env.kuramoto_env.KuramotoOscillatorEnv`.
        ``adj_matrix`` and ``constant_coupling_matrix`` may be NumPy arrays or tensors.
    n_agents:
        Number of independent systems simulated in parallel.
    device:
        ``"cpu"``, ``"cuda"``, ``"cuda:<k>"``, a :class:`torch.device`, or ``"auto"``
        (CUDA when available, else CPU).
    render_agent:
        Index of the system shown by :meth:`render`.

    Raises
    ------
    ValueError, TypeError
        If an argument is invalid (including a CUDA device without CUDA support).

    Examples
    --------
    >>> env = KuramotoOscillatorEnvTorch(n_oscillators=8, n_agents=4)
    >>> obs, info = env.reset(seed=0)
    >>> env.get_batch_observations().shape
    torch.Size([4, 52])
    """

    _backend_name = "torch"

    def __init__(
        self,
        n_oscillators: int = 10,
        n_agents: int = 1,
        dt: float = 0.01,
        max_steps: int = 1000,
        coupling_range: tuple[float, float] = (0.0, 5.0),
        control_input_range: tuple[float, float] = (-1.0, 1.0),
        natural_freq_range: tuple[float, float] = (0.5, 2.0),
        device: str | torch.device = "cpu",
        render_mode: str | None = None,
        integration_method: str = "euler",
        reward_type: str = "order_parameter",
        noise_std: float = 0.0,
        topology: str = "fully_connected",
        adj_matrix: ArrayLike | None = None,
        target_frequency: float = 1.0,
        coupling_mode: str = "dynamic",
        constant_coupling_matrix: ArrayLike | None = None,
        coupling_strength: float = 1.0,
        normalize_coupling: bool = False,
        topology_seed: int = 42,
        sync_threshold: float = 0.99,
        sync_bonus: float = 10.0,
        render_agent: int = 0,
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
        self.device = _resolve_device(device)
        self.render_agent = self._check_render_agent(render_agent)
        self._dtype = torch.float32
        tensor_kwargs = {"dtype": self._dtype, "device": self.device}

        self.topology_matrix = torch.as_tensor(self._topology_np, **tensor_kwargs)
        n = self.n_oscillators
        # Flat (row-major) positions of K_ij and K_ji for every edge, used by scatter_.
        edge_flat = np.concatenate(
            (self._edge_rows * n + self._edge_cols, self._edge_cols * n + self._edge_rows)
        )
        self._edge_flat_t = torch.as_tensor(edge_flat, dtype=torch.long, device=self.device)
        # scatter_ is markedly faster with a materialised (contiguous) index than
        # with a stride-0 expanded one; keep it when it is small.
        self._edge_index_batch: torch.Tensor | None = None
        if self.n_agents * edge_flat.size <= 1 << 20:
            self._edge_index_batch = self._edge_flat_t.expand(self.n_agents, -1).contiguous()
        self._action_low_t = torch.as_tensor(self._action_low, **tensor_kwargs)
        self._action_high_t = torch.as_tensor(self._action_high, **tensor_kwargs)
        if self._constant_np is not None:
            self._constant_t: torch.Tensor | None = torch.as_tensor(
                self._constant_np, **tensor_kwargs
            )
            # Batched view kept for backward compatibility (no copy).
            self.coupling_matrix = self._constant_t.unsqueeze(0).expand(self.n_agents, -1, -1)
        else:
            self._constant_t = None
            self.coupling_matrix = None

        self._rng = torch.Generator(device=self.device)
        self._coupling_buf: torch.Tensor | None = None
        self.phases: torch.Tensor | None = None
        self.natural_frequencies: torch.Tensor | None = None
        self.coupling_strengths: torch.Tensor | None = None
        self.control_inputs: torch.Tensor | None = None
        self._current_coupling: torch.Tensor | None = None
        self._last_dphases_dt: torch.Tensor | None = None
        self._last_agent_rewards: np.ndarray | None = None

    def _coupling_buffer(self) -> torch.Tensor:
        """Persistent ``(n_agents, N * N)`` buffer for the dynamic coupling matrices.

        Only the edge entries change from step to step, so the buffer is zeroed
        once and reused (``self._current_coupling`` is its only holder; info
        arrays are copies).
        """
        if self._coupling_buf is None:
            n = self.n_oscillators
            self._coupling_buf = torch.zeros(
                self.n_agents, n * n, dtype=self._dtype, device=self.device
            )
        return self._coupling_buf

    def _check_render_agent(self, render_agent: int) -> int:
        return check_int("render_agent", render_agent, minimum=0, maximum=self.n_agents - 1)

    def _render_subtitle(self) -> str:
        return (
            f"torch backend ({self.device.type}) | system {self.render_agent} of "
            f"{self.n_agents} | {self.coupling_mode} coupling | {self.topology} topology | "
            f"{self.integration_method} dt={self.dt:g}"
        )

    # ------------------------------------------------------------------ model
    def _coupling_matrix_from_actions(
        self, actions: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Scatter ``(B, M)`` edge strengths into symmetric ``(B, N, N)`` matrices.

        ``out`` is an optional ``(B, N * N)`` buffer whose non-edge entries are
        zero; every edge entry is overwritten.
        """
        n, batch = self.n_oscillators, actions.shape[0]
        if out is None:
            out = torch.zeros(batch, n * n, dtype=actions.dtype, device=actions.device)
        matrix = out
        index = self._edge_index_batch
        if index is None or index.shape[0] != batch:
            index = self._edge_flat_t.expand(batch, -1)
        matrix.scatter_(1, index, torch.cat((actions, actions), dim=1))
        return matrix.view(batch, n, n)

    def _kuramoto_dynamics(
        self,
        phases: torch.Tensor,
        natural_frequencies: torch.Tensor,
        coupling_matrix: torch.Tensor,
        control_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Phase velocities ``omega_i + a_i + c * sum_j K_ij sin(theta_j - theta_i)``.

        ``phases`` has shape ``(B, N)``; ``coupling_matrix`` is ``(B, N, N)`` or a
        shared ``(N, N)`` matrix.
        """
        sin, cos = torch.sin(phases), torch.cos(phases)
        projected = torch.matmul(coupling_matrix, torch.stack((sin, cos), dim=-1))
        coupling = cos * projected[..., 0] - sin * projected[..., 1]
        if self.normalize_coupling:
            coupling = coupling * self._coupling_factor
        return natural_frequencies + control_inputs + coupling

    def _integrate_rk4(
        self,
        phases: torch.Tensor,
        natural_frequencies: torch.Tensor,
        coupling_matrix: torch.Tensor,
        control_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Classical fourth-order Runge-Kutta step. Returns ``(new_phases, k1)``."""
        dt = self.dt
        args = (natural_frequencies, coupling_matrix, control_inputs)
        k1 = self._kuramoto_dynamics(phases, *args)
        k2 = self._kuramoto_dynamics(phases + 0.5 * dt * k1, *args)
        k3 = self._kuramoto_dynamics(phases + 0.5 * dt * k2, *args)
        k4 = self._kuramoto_dynamics(phases + dt * k3, *args)
        return phases + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), k1

    def _compute_order_parameter(self, phases: torch.Tensor) -> torch.Tensor:
        """Per-system order parameter, shape ``(B,)``."""
        return torch.hypot(torch.cos(phases).mean(dim=-1), torch.sin(phases).mean(dim=-1))

    def _compute_phase_coherence(self, phases: torch.Tensor) -> torch.Tensor:
        """Per-system ``exp(-Var(theta))`` (phases wrapped to ``[-pi, pi)``, ``ddof=0``)."""
        return torch.exp(-torch.var(wrap_phases(phases), dim=-1, correction=0))

    def _base_rewards(self, phases: torch.Tensor, dphases_dt: torch.Tensor | None):
        """Per-system rewards without bonus, plus ``(r, coherence)``."""
        r = self._compute_order_parameter(phases)
        coherence = self._compute_phase_coherence(phases)
        frequency_error = None
        if self.reward_type == "frequency_synchronization":
            if dphases_dt is None:
                dphases_dt = self._current_dphases_dt()
            frequency_error = torch.mean(torch.abs(dphases_dt - self.target_frequency), dim=-1)
        return combine_reward(self.reward_type, r, coherence, frequency_error), r, coherence

    def _current_dphases_dt(self) -> torch.Tensor:
        if self._last_dphases_dt is not None:
            return self._last_dphases_dt
        return self._kuramoto_dynamics(
            self.phases, self.natural_frequencies, self._current_coupling, self.control_inputs
        )

    def _apply_bonus(self, rewards: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.reward_type == "frequency_synchronization":
            return rewards
        return torch.where(r > self.sync_threshold, rewards + self.sync_bonus, rewards)

    # ------------------------------------------------------------------ gym API
    @torch.no_grad()
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Sample new initial states for all systems.

        Sampling uses a :class:`torch.Generator` owned by the environment and
        seeded from ``self.np_random``, so ``reset(seed=s)`` is reproducible and
        the global torch RNG is never touched.

        Parameters
        ----------
        seed:
            Seed for ``self.np_random``.
        options:
            Optional overrides ``"phases"``, ``"natural_frequencies"`` and (dynamic
            mode) ``"coupling_strengths"``, each of shape ``(size,)`` (broadcast to
            every system) or ``(n_agents, size)``. Unknown keys are ignored with
            a ``UserWarning``.

        Returns
        -------
        (numpy.ndarray, dict)
            Observation of system 0 and the info dictionary.
        """
        super().reset(seed=seed)
        overrides = self._parse_reset_options(options)
        self._rng.manual_seed(int(self.np_random.integers(0, 2**63 - 1)))
        batch, n = self.n_agents, self.n_oscillators
        kwargs = {"generator": self._rng, "dtype": self._dtype, "device": self.device}
        freq_low, freq_high = self.natural_freq_range
        phases = (torch.rand(batch, n, **kwargs) * 2 - 1) * math.pi
        natural_frequencies = torch.rand(batch, n, **kwargs) * (freq_high - freq_low) + freq_low
        coupling_strengths = None
        if self.coupling_mode == "dynamic":
            low, high = self.coupling_range
            coupling_strengths = torch.rand(batch, self.n_couplings, **kwargs) * (high - low) + low

        for key, value in overrides.items():
            if value.ndim == 2 and value.shape[0] != batch:
                raise ValueError(
                    f"reset option {key!r} has {value.shape[0]} rows but n_agents={batch}"
                )
            tensor = torch.as_tensor(value, dtype=self._dtype, device=self.device)
            tensor = tensor.expand(batch, -1).clone()
            if key == "phases":
                phases = wrap_phases(tensor)
            elif key == "natural_frequencies":
                natural_frequencies = tensor
            else:
                coupling_strengths = tensor

        self.phases = phases
        self.natural_frequencies = natural_frequencies
        self.coupling_strengths = coupling_strengths
        self.control_inputs = torch.zeros(batch, n, dtype=self._dtype, device=self.device)
        if coupling_strengths is not None:
            self._current_coupling = self._coupling_matrix_from_actions(
                coupling_strengths, self._coupling_buffer()
            )
        else:
            self._current_coupling = self._constant_t
        self.step_count = 0
        self._last_dphases_dt = None
        self._last_agent_rewards = None
        self.phase_history = [_to_numpy(phases)]
        self._reset_renderer()

        info = {
            "order_parameter": _to_numpy(self._compute_order_parameter(phases)),
            "phase_coherence": _to_numpy(self._compute_phase_coherence(phases)),
            "natural_frequencies": _to_numpy(natural_frequencies),
            "phases": _to_numpy(phases),
            "coupling_matrix": self._batched_coupling_numpy(),
            "step_count": 0,
            "device": str(self.device),
            "n_agents": self.n_agents,
        }
        if self.render_mode == "human":
            self.render()
        return self._create_observation(0), info

    @torch.no_grad()
    def step(self, action: ArrayLike) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance every system by one time step ``dt``.

        Parameters
        ----------
        action:
            Shape ``(action_dim,)`` (shared by all systems) or
            ``(n_agents, action_dim)``; NumPy array or tensor. Values are clipped to
            the action bounds.

        Returns
        -------
        observation, reward, terminated, truncated, info
            Observation, reward and ``terminated`` (order parameter above
            ``sync_threshold``) of system 0; ``truncated`` once ``max_steps`` steps
            have been taken; per-system arrays in ``info``.
        """
        if self.phases is None:
            raise ResetNeededError("Call reset() before step().")
        action_t = self._action_to_tensor(action)
        n = self.n_oscillators
        control_inputs = action_t[:, :n]
        if self.coupling_mode == "dynamic":
            self.coupling_strengths = action_t[:, n:]
            coupling_matrix = self._coupling_matrix_from_actions(
                self.coupling_strengths, self._coupling_buffer()
            )
        else:
            coupling_matrix = self._constant_t

        if self.integration_method == "euler":
            dphases_dt = self._kuramoto_dynamics(
                self.phases, self.natural_frequencies, coupling_matrix, control_inputs
            )
            phases = self.phases + dphases_dt * self.dt
        else:
            phases, dphases_dt = self._integrate_rk4(
                self.phases, self.natural_frequencies, coupling_matrix, control_inputs
            )
        if self.noise_std > 0:
            noise = torch.randn(
                phases.shape, generator=self._rng, dtype=self._dtype, device=self.device
            )
            phases = phases + noise * self.noise_std
        self.phases = wrap_phases(phases)
        self.control_inputs = control_inputs
        self._current_coupling = coupling_matrix
        self._last_dphases_dt = dphases_dt
        self.step_count += 1

        base, r, coherence = self._base_rewards(self.phases, dphases_dt)
        # Two device-to-host transfers for all per-system and per-oscillator info.
        r_np, coherence_np, rewards_np = _to_numpy(
            torch.stack((r, coherence, self._apply_bonus(base, r)))
        )
        phases_np, frequencies_np, dphases_np = _to_numpy(
            torch.stack((self.phases, self.natural_frequencies, dphases_dt))
        )
        agent_rewards = rewards_np.astype(np.float64)
        reward = float(agent_rewards[0])
        terminated = bool(r_np[0] > self.sync_threshold)
        truncated = bool(self.step_count >= self.max_steps)
        self._last_agent_rewards = agent_rewards

        self._append_history(phases_np.copy())
        info = {
            "order_parameter": r_np,
            "phase_coherence": coherence_np,
            "step_count": self.step_count,
            "natural_frequencies": frequencies_np,
            "phases": phases_np,
            "coupling_matrix": self._batched_coupling_numpy(),
            "device": str(self.device),
            "n_agents": self.n_agents,
            "dphases_dt": dphases_np,
            "agent_rewards": agent_rewards.copy(),
        }
        if self.render_mode == "human":
            self.render()
        # Observation of system 0 from the host copies (no extra device transfer
        # for phases and frequencies).
        n = self.n_oscillators
        action_0 = _to_numpy(action_t[0])
        parts = [phases_np[0], frequencies_np[0]]
        if self.coupling_mode == "dynamic":
            parts.append(action_0[n:])
        parts.append(action_0[:n])
        return np.concatenate(parts), reward, terminated, truncated, info

    # ------------------------------------------------------------------ batch API
    def get_batch_observations(self) -> torch.Tensor:
        """Observations of all systems, shape ``(n_agents, obs_dim)``, on ``self.device``."""
        if self.phases is None:
            raise ResetNeededError("Call reset() before get_batch_observations().")
        parts = [self.phases, self.natural_frequencies]
        if self.coupling_mode == "dynamic":
            parts.append(self.coupling_strengths)
        parts.append(self.control_inputs)
        return torch.cat(parts, dim=1)

    def get_batch_rewards(
        self,
        dphases_dt: torch.Tensor | None = None,
        include_bonus: bool = False,
    ) -> torch.Tensor:
        """Per-system rewards of the current state, shape ``(n_agents,)``.

        Parameters
        ----------
        dphases_dt:
            Phase velocities ``(n_agents, N)`` used by the
            ``"frequency_synchronization"`` reward. Defaults to the velocities of
            the last step (or, right after ``reset``, those of the current state
            with the current control inputs and coupling).
        include_bonus:
            Add ``sync_bonus`` for every synchronised system (as in
            ``info["agent_rewards"]``). The default ``False`` keeps the historical
            behaviour of this method.
        """
        if self.phases is None:
            raise ResetNeededError("Call reset() before get_batch_rewards().")
        base, r, _ = self._base_rewards(self.phases, dphases_dt)
        return self._apply_bonus(base, r) if include_bonus else base

    # ------------------------------------------------------------------ internals
    def _action_to_tensor(self, action: ArrayLike) -> torch.Tensor:
        action_dim = self.action_space.shape[0]
        if isinstance(action, torch.Tensor):
            action = action.detach()  # never carry a caller's autograd graph into the state
            host = None
        else:
            # Convert on the host first: checking finiteness with NumPy is much
            # cheaper than torch.isfinite on CPU and needs no device sync.
            host = np.asarray(action, dtype=np.float32)
            action = host
        action_t = torch.as_tensor(action, dtype=self._dtype, device=self.device)
        if action_t.ndim == 1:
            action_t = action_t.unsqueeze(0).expand(self.n_agents, -1)
        if action_t.shape != (self.n_agents, action_dim):
            raise ValueError(
                f"action must have shape ({action_dim},) or ({self.n_agents}, {action_dim}), "
                f"got {tuple(torch.as_tensor(action).shape)}"
            )
        finite = np.isfinite(host).all() if host is not None else torch.isfinite(action_t).all()
        if not bool(finite):
            raise ValueError("action contains non-finite values")
        return torch.clamp(action_t, self._action_low_t, self._action_high_t)

    def _batched_coupling_numpy(self) -> np.ndarray:
        coupling = self._current_coupling
        if coupling.ndim == 2:
            coupling = coupling.unsqueeze(0).expand(self.n_agents, -1, -1)
        return _to_numpy(coupling)

    def _create_observation(self, agent_idx: int = 0) -> np.ndarray:
        """Observation of one system as a ``float32`` NumPy array."""
        parts = [self.phases[agent_idx], self.natural_frequencies[agent_idx]]
        if self.coupling_mode == "dynamic":
            parts.append(self.coupling_strengths[agent_idx])
        parts.append(self.control_inputs[agent_idx])
        return torch.cat(parts).detach().cpu().numpy()

    def _frame(self):
        from env_lib.kos_env.rendering import KuramotoFrame

        k = self.render_agent
        coupling = self._current_coupling
        coupling = coupling if coupling.ndim == 2 else coupling[k]
        reward = None if self._last_agent_rewards is None else float(self._last_agent_rewards[k])
        return KuramotoFrame(
            phases=_to_numpy(self.phases[k]),
            natural_frequencies=_to_numpy(self.natural_frequencies[k]),
            coupling_matrix=_to_numpy(coupling),
            history=self.phase_history,
            history_index=k,
            step=self.step_count,
            max_steps=self.max_steps,
            dt=self.dt,
            reward=reward,
        )
