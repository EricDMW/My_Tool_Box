"""Private building blocks shared by the NumPy and PyTorch Kuramoto environments.

This module holds everything that does not depend on the array backend:
argument validation, network topologies, the edge list used by dynamic
coupling, NumPy synchronisation measures, the reward definition and the
:class:`KuramotoEnvBase` class that owns spaces, rendering and bookkeeping.
It is not part of the public API.
"""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env_lib.errors import ResetNeededError
from env_lib.utils.rendering import validate_render_mode

if TYPE_CHECKING:  # pragma: no cover
    from env_lib.kos_env.rendering import KuramotoFrame, KuramotoRenderer

__all__ = [
    "COUPLING_MODES",
    "INTEGRATION_METHODS",
    "REWARD_TYPES",
    "TOPOLOGIES",
    "KuramotoEnvBase",
    "build_topology",
    "combine_reward",
    "mean_field",
    "order_parameter",
    "phase_coherence",
    "topology_edges",
    "wrap_phases",
]

TOPOLOGIES: tuple[str, ...] = ("fully_connected", "ring", "star", "random", "custom")
INTEGRATION_METHODS: tuple[str, ...] = ("euler", "rk4")
REWARD_TYPES: tuple[str, ...] = (
    "order_parameter",
    "phase_coherence",
    "combined",
    "frequency_synchronization",
)
COUPLING_MODES: tuple[str, ...] = ("dynamic", "constant")
RESET_OPTION_KEYS: tuple[str, ...] = ("phases", "natural_frequencies", "coupling_strengths")

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
def check_choice(name: str, value: Any, choices: Sequence[str]) -> str:
    """Return ``value`` if it is one of ``choices``, otherwise raise ``ValueError``."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {tuple(choices)}, got {value!r}")
    return value


def check_int(name: str, value: Any, *, minimum: int, maximum: int | None = None) -> int:
    """Validate an integer argument (``bool`` is rejected)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        upper = "" if maximum is None else f" and <= {maximum}"
        raise ValueError(f"{name} must be >= {minimum}{upper}, got {value}")
    return value


def check_real(
    name: str,
    value: Any,
    *,
    low: float | None = None,
    strict: bool = False,
    allow_inf: bool = False,
) -> float:
    """Validate a real number, optionally bounded below by ``low``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if np.isnan(value) or (np.isinf(value) and not allow_inf):
        raise ValueError(f"{name} must be finite, got {value}")
    if low is not None and (value <= low if strict else value < low):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be {relation} {low}, got {value}")
    return value


def check_range(name: str, value: Any) -> tuple[float, float]:
    """Validate a ``(low, high)`` pair of finite numbers with ``low <= high``."""
    try:
        low, high = value
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a (low, high) pair, got {value!r}") from None
    low = check_real(f"{name}[0]", low)
    high = check_real(f"{name}[1]", high)
    if low > high:
        raise ValueError(f"{name} must satisfy low <= high, got {value!r}")
    return (low, high)


def as_square_matrix(name: str, matrix: Any, n: int) -> np.ndarray:
    """Return ``matrix`` as a finite float64 ``(n, n)`` array (always a copy).

    Accepts anything convertible with :func:`numpy.array`, including PyTorch
    tensors on any device.
    """
    if hasattr(matrix, "detach") and hasattr(matrix, "cpu"):  # torch.Tensor, duck-typed
        matrix = matrix.detach().cpu().numpy()
    try:
        array = np.array(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a numeric (n, n) array") from None
    if array.shape != (n, n):
        raise ValueError(
            f"{name} must have shape ({n}, {n}) to match n_oscillators, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


# ---------------------------------------------------------------------------
# Topologies
# ---------------------------------------------------------------------------
def build_topology(
    topology: str,
    n_oscillators: int,
    adj_matrix: Any = None,
    seed: int = 42,
) -> tuple[np.ndarray, str]:
    """Build the ``(n, n)`` float64 adjacency matrix of a network topology.

    Parameters
    ----------
    topology:
        One of :data:`TOPOLOGIES`. Ignored (reported as ``"custom"``) when
        ``adj_matrix`` is given.
    n_oscillators:
        Number of nodes.
    adj_matrix:
        Optional custom adjacency matrix; non-zero entries are edges.
    seed:
        Seed of the local ``numpy.random.RandomState`` used by the ``"random"``
        topology. The draw ``random((n, n)) > 0.5`` with zero diagonal and an OR
        symmetrisation reproduces the matrices of earlier releases.

    Returns
    -------
    (numpy.ndarray, str)
        The adjacency matrix and the effective topology name.
    """
    check_choice("topology", topology, TOPOLOGIES)
    n = n_oscillators
    if adj_matrix is not None:
        return as_square_matrix("adj_matrix", adj_matrix, n), "custom"
    if topology == "custom":
        raise ValueError("topology='custom' requires an adj_matrix")
    if topology == "fully_connected":
        matrix = np.ones((n, n))
        np.fill_diagonal(matrix, 0.0)
    elif topology == "ring":
        matrix = np.zeros((n, n))
        index = np.arange(n)
        matrix[index, (index - 1) % n] = 1.0
        matrix[index, (index + 1) % n] = 1.0
    elif topology == "star":
        matrix = np.zeros((n, n))
        matrix[0, 1:] = 1.0
        matrix[1:, 0] = 1.0
    else:  # random
        rng = np.random.RandomState(seed)
        matrix = (rng.random((n, n)) > 0.5).astype(np.float64)
        np.fill_diagonal(matrix, 0.0)
        matrix = ((matrix + matrix.T) > 0).astype(np.float64)
    return matrix, topology


def topology_edges(topology_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the undirected edges ``(rows, cols)`` of a topology, ``rows < cols``.

    A pair is an edge when either direction is non-zero. The edges are listed
    in row-major order, which fixes the layout of the coupling part of the
    action and observation vectors.
    """
    mask = (topology_matrix != 0) | (topology_matrix.T != 0)
    rows, cols = np.nonzero(np.triu(mask, 1))
    return rows.astype(np.int64), cols.astype(np.int64)


# ---------------------------------------------------------------------------
# Synchronisation measures (NumPy; the torch backend mirrors them)
# ---------------------------------------------------------------------------
def wrap_phases(phases: Any) -> Any:
    """Wrap phases to ``[-pi, pi)``. Works for NumPy arrays and torch tensors."""
    return (phases + np.pi) % TWO_PI - np.pi


def order_parameter(phases: np.ndarray) -> np.ndarray:
    """Kuramoto order parameter ``r = |mean_j exp(i theta_j)|`` along the last axis."""
    return np.hypot(np.cos(phases).mean(axis=-1), np.sin(phases).mean(axis=-1))


def mean_field(phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(r, psi)`` with ``r exp(i psi) = mean_j exp(i theta_j)``."""
    x = np.cos(phases).mean(axis=-1)
    y = np.sin(phases).mean(axis=-1)
    return np.hypot(x, y), np.arctan2(y, x)


def phase_coherence(phases: np.ndarray) -> np.ndarray:
    """Phase coherence ``exp(-Var(theta))`` with phases wrapped to ``[-pi, pi)``.

    The variance is the population variance (``ddof=0``) over the last axis.
    """
    return np.exp(-np.var(wrap_phases(phases), axis=-1))


def combine_reward(
    reward_type: str,
    order_param: Any,
    coherence: Any,
    frequency_error: Any,
) -> Any:
    """Base reward (without synchronisation bonus); backend agnostic.

    ``frequency_error`` is ``mean_i |dtheta_i/dt - target_frequency|`` and is
    only used by ``"frequency_synchronization"``.
    """
    if reward_type == "order_parameter":
        return order_param
    if reward_type == "phase_coherence":
        return coherence
    if reward_type == "combined":
        return order_param + coherence
    return -frequency_error


# ---------------------------------------------------------------------------
# Shared environment logic
# ---------------------------------------------------------------------------
class KuramotoEnvBase(gym.Env):
    """Backend-independent part of the Kuramoto environments.

    Subclasses call :meth:`_init_common` from their constructor and implement
    ``reset``, ``step`` and :meth:`_frame`.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    _backend_name = "numpy"

    def _init_common(
        self,
        *,
        n_oscillators: int,
        n_agents: int,
        dt: float,
        max_steps: int,
        coupling_range: tuple[float, float],
        control_input_range: tuple[float, float],
        natural_freq_range: tuple[float, float],
        render_mode: str | None,
        integration_method: str,
        reward_type: str,
        noise_std: float,
        topology: str,
        adj_matrix: Any,
        target_frequency: float,
        coupling_mode: str,
        constant_coupling_matrix: Any,
        coupling_strength: float,
        normalize_coupling: bool,
        topology_seed: int,
        sync_threshold: float,
        sync_bonus: float,
    ) -> None:
        validate_render_mode(render_mode, self.metadata["render_modes"])
        self.render_mode = render_mode
        self.n_oscillators = check_int("n_oscillators", n_oscillators, minimum=2)
        self.n_agents = check_int("n_agents", n_agents, minimum=1)
        self.dt = check_real("dt", dt, low=0.0, strict=True)
        self.max_steps = check_int("max_steps", max_steps, minimum=1)
        self.coupling_range = check_range("coupling_range", coupling_range)
        self.control_input_range = check_range("control_input_range", control_input_range)
        self.natural_freq_range = check_range("natural_freq_range", natural_freq_range)
        self.integration_method = check_choice(
            "integration_method", integration_method, INTEGRATION_METHODS
        )
        self.reward_type = check_choice("reward_type", reward_type, REWARD_TYPES)
        self.noise_std = check_real("noise_std", noise_std, low=0.0)
        self.target_frequency = check_real("target_frequency", target_frequency)
        self.coupling_mode = check_choice("coupling_mode", coupling_mode, COUPLING_MODES)
        self.coupling_strength = check_real("coupling_strength", coupling_strength)
        if not isinstance(normalize_coupling, (bool, np.bool_)):
            raise TypeError(
                f"normalize_coupling must be a bool, got {type(normalize_coupling).__name__}"
            )
        self.normalize_coupling = bool(normalize_coupling)
        self.topology_seed = check_int("topology_seed", topology_seed, minimum=0, maximum=2**32 - 1)
        self.sync_threshold = check_real(
            "sync_threshold", sync_threshold, low=0.0, strict=True, allow_inf=True
        )
        self.sync_bonus = check_real("sync_bonus", sync_bonus)
        # Kept verbatim for backward compatibility (the validated copies live below).
        self.adj_matrix = adj_matrix
        self.constant_coupling_matrix = constant_coupling_matrix

        n = self.n_oscillators
        self._topology_np, self.topology = build_topology(
            topology, n, adj_matrix, self.topology_seed
        )
        self._edge_rows, self._edge_cols = topology_edges(self._topology_np)
        self.n_couplings = int(self._edge_rows.size)
        self.coupling_indices = list(zip(self._edge_rows.tolist(), self._edge_cols.tolist()))
        if (
            self.coupling_mode == "dynamic"
            and adj_matrix is not None
            and not np.array_equal(self._topology_np != 0, self._topology_np.T != 0)
        ):
            warnings.warn(
                "adj_matrix is not symmetric; in dynamic coupling mode every connected pair "
                "is one undirected edge with a single, symmetric coupling strength.",
                UserWarning,
                stacklevel=3,
            )

        if self.coupling_mode == "constant":
            if constant_coupling_matrix is None:
                constant = self.coupling_strength * self._topology_np
            else:
                constant = as_square_matrix("constant_coupling_matrix", constant_coupling_matrix, n)
            np.fill_diagonal(constant, 0.0)  # self-coupling has no effect on the dynamics
            self._constant_np: np.ndarray | None = constant
        else:
            if constant_coupling_matrix is not None:
                warnings.warn(
                    "constant_coupling_matrix is ignored when coupling_mode='dynamic'; "
                    "pass coupling_mode='constant' to use it.",
                    UserWarning,
                    stacklevel=3,
                )
            self._constant_np = None

        # Spaces. Bounds are kept in float64 for clipping and exposed as float32.
        control_low, control_high = self.control_input_range
        coupling_low, coupling_high = self.coupling_range
        m = self.n_couplings if self.coupling_mode == "dynamic" else 0
        self._action_low = np.concatenate([np.full(n, control_low), np.full(m, coupling_low)])
        self._action_high = np.concatenate([np.full(n, control_high), np.full(m, coupling_high)])
        self.action_space = spaces.Box(
            low=self._action_low.astype(np.float32),
            high=self._action_high.astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 * n + m,), dtype=np.float32
        )
        self._coupling_factor = 1.0 / n if self.normalize_coupling else 1.0

        self.step_count = 0
        self.phase_history: list = []
        self._renderer: KuramotoRenderer | None = None
        self._warned_render_none = False

    # ------------------------------------------------------------------ helpers
    def _parse_reset_options(
        self, options: dict[str, Any] | None, stacklevel: int = 3
    ) -> dict[str, np.ndarray]:
        """Validate ``reset(options=...)`` overrides of the sampled initial state.

        Unknown keys are ignored with a ``UserWarning`` (generic tooling such as
        PettingZoo's API test passes arbitrary options); the values of known
        keys are validated strictly. ``stacklevel`` locates the user's
        ``reset`` call for the warning.
        """
        if not options:
            return {}
        unknown = sorted(set(options) - set(RESET_OPTION_KEYS))
        if unknown:
            warnings.warn(
                f"{type(self).__name__}.reset(): ignoring unknown reset option(s) {unknown}; "
                f"supported: {list(RESET_OPTION_KEYS)}",
                UserWarning,
                stacklevel=stacklevel,
            )
            options = {key: value for key, value in options.items() if key not in unknown}
        if "coupling_strengths" in options and self.coupling_mode != "dynamic":
            raise ValueError("reset option 'coupling_strengths' requires coupling_mode='dynamic'")
        parsed = {}
        for key, value in options.items():
            if value is None:
                continue
            if hasattr(value, "detach") and hasattr(value, "cpu"):
                value = value.detach().cpu().numpy()
            array = np.asarray(value, dtype=np.float64)
            size = self.n_couplings if key == "coupling_strengths" else self.n_oscillators
            if array.shape[-1:] != (size,) or array.ndim > 2:
                raise ValueError(
                    f"reset option {key!r} must have shape ({size},) or (n_agents, {size}), "
                    f"got {array.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"reset option {key!r} must be finite")
            parsed[key] = array
        return parsed

    def _append_history(self, phases: np.ndarray) -> None:
        """Record phases for rendering; bounded to one episode (``max_steps + 1``)."""
        self.phase_history.append(phases)
        if len(self.phase_history) > self.max_steps + 1:
            del self.phase_history[0]

    def _coupling_display_scale(self) -> float:
        if self._constant_np is not None:
            scale = float(np.max(np.abs(self._constant_np)))
        else:
            scale = float(max(abs(self.coupling_range[0]), abs(self.coupling_range[1])))
        return scale if scale > 0 else 1.0

    def _render_subtitle(self) -> str:
        return (
            f"{self._backend_name} backend | {self.coupling_mode} coupling | "
            f"{self.topology} topology | {self.integration_method} dt={self.dt:g}"
        )

    # ------------------------------------------------------------------ render
    def _frame(self) -> KuramotoFrame:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def render(self) -> np.ndarray | None:
        """Render the current state.

        Returns
        -------
        numpy.ndarray or None
            A ``(560, 1000, 3)`` ``uint8`` frame for ``render_mode="rgb_array"``;
            ``None`` for ``"human"`` (a window is updated) and when no render mode
            was set.
        """
        if self.render_mode is None:
            if not self._warned_render_none:
                warnings.warn(
                    "render() was called but render_mode is None; create the environment "
                    "with render_mode='rgb_array' or 'human'.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_render_none = True
            return None
        if getattr(self, "phases", None) is None:
            raise ResetNeededError("Call reset() before render().")
        if self._renderer is None:
            from env_lib.kos_env.rendering import KuramotoRenderer

            self._renderer = KuramotoRenderer(
                self.render_mode,
                n_oscillators=self.n_oscillators,
                natural_freq_range=self.natural_freq_range,
                coupling_scale=self._coupling_display_scale(),
                sync_threshold=self.sync_threshold,
                subtitle=self._render_subtitle(),
                fps=self.metadata["render_fps"],
            )
        return self._renderer.render(self._frame())

    def _reset_renderer(self) -> None:
        if self._renderer is not None:
            self._renderer.reset()

    def close(self) -> None:
        """Close the render window / figure. Safe to call several times."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    @property
    def fig(self):
        """The renderer's matplotlib figure, or ``None`` before the first render."""
        return None if self._renderer is None else self._renderer.fig

    @property
    def ax(self):
        """The phase-portrait axes of the renderer, or ``None`` before the first render."""
        return None if self._renderer is None else self._renderer.ax_phase
