"""State-estimation primitives for the AJLATT environment.

* SE(2) unicycle motion (:func:`se2_step`) for robots and targets.
* :class:`AgentState` (ground truth) and :class:`AgentEstimate` (mean and
  covariance propagated with an extended-Kalman-filter prediction step).
* The range-bearing :func:`measurement_model` and its Jacobians.
* Covariance intersection (:func:`covariance_intersection`), the information
  fusion rule used for cooperative localisation and target tracking.

Covariance intersection chooses convex weights ``c`` for information matrices
``S_i`` so that ``trace(inv(sum_i c_i S_i))`` is minimal. The default solver is
a projected Newton method on the probability simplex with analytic gradient
``g_i = -trace(P S_i P)`` and Hessian ``H_ij = 2 trace(P S_i P S_j P)``
(``P = inv(sum_i c_i S_i)``); it converges in a handful of iterations. The
``"slsqp"`` solver reproduces the original implementation (SciPy SLSQP with
finite-difference gradients) for exact reproducibility of earlier results.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

try:  # gufuncs behind numpy.linalg.inv / solve (called without their Python wrappers)
    from numpy.linalg import _umath_linalg
except ImportError:  # pragma: no cover
    _umath_linalg = None

__all__ = [
    "Agent",
    "AgentEstimate",
    "AgentState",
    "Agent_est",
    "SE2Dynamics",
    "cartesian2polar",
    "ci_weights",
    "covariance_intersection",
    "measurement_model",
    "pi_to_pi",
    "psd_inverse",
    "rotation_matrix",
    "se2_step",
]

_J = np.array([[0.0, -1.0], [1.0, 0.0]])
CI_SOLVERS = ("newton", "slsqp")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def pi_to_pi(angle):
    """Wrap an angle (or array of angles) to ``[-pi, pi)``."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def cartesian2polar(x, y) -> tuple[float, float]:
    """Return ``(range, bearing)`` of the vector ``(x, y)``."""
    return float(np.sqrt(np.sum(x**2 + y**2))), float(np.arctan2(y, x))


def rotation_matrix(theta) -> np.ndarray:
    """2-D rotation matrix for angle ``theta``."""
    theta = float(np.asarray(theta, dtype=np.float64).squeeze())
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def se2_step(x, dt: float, u) -> np.ndarray:
    """Propagate a unicycle pose ``x = (px, py, theta)`` with ``u = (v, omega)`` for ``dt``."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (3,):
        raise ValueError(f"expected a pose of shape (3,), got {x.shape}")
    new = x + np.array([u[0] * dt * np.cos(x[2]), u[0] * dt * np.sin(x[2]), dt * u[1]])
    new[2] = pi_to_pi(new[2])
    return new


SE2Dynamics = se2_step


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
class AgentState:
    """Ground-truth pose of a robot or target."""

    def __init__(self, dim: int = 3, sampling_period: float = 0.5):
        self.dim = dim
        self.sampling_period = sampling_period
        self.state = np.zeros(dim)

    def reset(self, init_state) -> None:
        self.state = np.array(init_state, dtype=np.float64)

    def propagate(self, control_input) -> None:
        """Advance the pose with the control ``(v, omega)``."""
        self.state = se2_step(self.state, self.sampling_period, control_input)


class AgentEstimate(AgentState):
    """Gaussian belief (mean and covariance) over a pose or a 2-D position."""

    def __init__(self, dim: int = 3, sampling_period: float = 0.5):
        super().__init__(dim, sampling_period)
        self.cov = np.eye(dim)

    def reset(self, init_state, init_cov, rng: np.random.Generator | None = None) -> None:
        """Sample the initial mean around ``init_state`` with covariance ``init_cov``."""
        init_cov = np.array(init_cov, dtype=np.float64)
        rng = np.random.default_rng() if rng is None else rng
        self.state = rng.normal(
            np.asarray(init_state, dtype=np.float64), np.sqrt(init_cov.diagonal())
        )
        self.cov = init_cov

    def propagate(self, control_input, sigma_v: float, sigma_w: float) -> None:  # type: ignore[override]
        """EKF prediction for the unicycle model with velocity noise ``(sigma_v, sigma_w)``."""
        dt = self.sampling_period
        new_state = se2_step(self.state, dt, control_input)
        phi = np.eye(3)
        phi[:2, 2] = _J @ (new_state[:2] - self.state[:2])
        g = np.array(
            [[dt * np.cos(self.state[2]), 0.0], [dt * np.sin(self.state[2]), 0.0], [0.0, dt]]
        )
        q = np.diag([sigma_v**2, sigma_w**2])
        cov = phi @ self.cov @ phi.T + g @ q @ g.T
        self.state = new_state
        self.cov = 0.5 * (cov + cov.T)


# Backwards-compatible names.
Agent = AgentState
Agent_est = AgentEstimate


# ---------------------------------------------------------------------------
# Measurement model
# ---------------------------------------------------------------------------
def measurement_model(xe_i, xe_j) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Range-bearing measurement of ``xe_j`` taken by the robot at pose ``xe_i``.

    Returns
    -------
    zhat:
        Predicted ``(range, bearing)``.
    Hi:
        Jacobian with respect to the observer pose, shape ``(2, 3)``.
    Hj:
        Jacobian with respect to the observed pose, shape ``(2, 3)`` (the last
        column is zero; slice ``[:, :2]`` for position-only states).
    """
    xe_i = np.asarray(xe_i, dtype=np.float64)
    xe_j = np.asarray(xe_j, dtype=np.float64)
    c = rotation_matrix(xe_i[2])
    delta = xe_j[:2] - xe_i[:2]
    p_ij = c.T @ delta
    # Guard against coincident poses (zero range), which would make the
    # Jacobians undefined.
    rho = max(float(np.linalg.norm(p_ij)), 1e-9)
    zhat = np.array([rho, np.arctan2(p_ij[1], p_ij[0])])

    h_l = np.empty((2, 2))
    h_l[0] = p_ij / rho
    h_l[1] = (p_ij @ _J.T) / rho**2
    hi_prior = np.hstack([np.eye(2), (_J @ delta).reshape(2, 1)])
    hi = -h_l @ c.T @ hi_prior
    hj = np.hstack([h_l @ c.T, np.zeros((2, 1))])
    return zhat, hi, hj


# ---------------------------------------------------------------------------
# Covariance intersection
# ---------------------------------------------------------------------------
def psd_inverse(matrix: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric positive semi-definite matrix.

    Uses a direct inverse when the matrix is non-singular and the
    Moore-Penrose pseudo-inverse otherwise (e.g. information matrices with an
    unobservable component).
    """
    if np.all(np.diag(matrix) > 0):
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            pass
    return np.linalg.pinv(matrix)


class _DegenerateCI(RuntimeError):
    pass


#: Floating-point errors of LAPACK on singular matrices are reported through
#: NaN results inside the Newton solver (as numpy.linalg does internally).
_QUIET_FP = {"invalid": "ignore", "divide": "ignore", "over": "ignore", "under": "ignore"}


def _raw_inv(matrix: np.ndarray) -> np.ndarray:
    """``numpy.linalg.inv`` of a float64 matrix without the Python wrapper.

    Calls the same LAPACK kernel (the gufunc behind ``numpy.linalg.inv``), so
    results are bitwise identical; a singular matrix gives a NaN-filled result
    instead of raising. Call inside ``np.errstate(**_QUIET_FP)``.
    """
    if _umath_linalg is not None:
        return _umath_linalg.inv(matrix, signature="d->d")
    try:  # pragma: no cover - NumPy without the private gufunc module
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.full(matrix.shape, np.nan)


def _raw_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``numpy.linalg.solve`` for a 1-D right-hand side, see :func:`_raw_inv`."""
    if _umath_linalg is not None:
        return _umath_linalg.solve1(matrix, rhs, signature="dd->d")
    try:  # pragma: no cover
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.full(rhs.shape, np.nan)


def _combine(S: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``sum_i c_i S_i`` for ``S`` of shape ``(n, d, d)``."""
    n, d, _ = S.shape
    return (c @ S.reshape(n, d * d)).reshape(d, d)


def _objective(S: np.ndarray, c: np.ndarray):
    n, d, _ = S.shape
    with np.errstate(**_QUIET_FP):
        return _flat_objective(S.reshape(n, d * d), c, d)


def _flat_objective(flat: np.ndarray, c: np.ndarray, d: int):
    """``(trace(P), P)`` with ``P = inv(sum_i c_i S_i)``, or ``(inf, None)``.

    ``flat`` is ``S.reshape(n, d * d)``. A singular combination gives a NaN
    trace and is rejected, like the ``LinAlgError`` of ``numpy.linalg.inv``.
    """
    inverse = _raw_inv((c @ flat).reshape(d, d))
    value = inverse.trace()
    if not 0.0 < value < math.inf:
        return math.inf, None
    return value, inverse


def _weights_newton(S: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Active-set projected Newton method on the probability simplex.

    The implementation avoids per-call overhead (direct LAPACK gufunc calls,
    one floating-point error context per solve) but performs exactly the
    same floating-point operations as the straightforward NumPy version, so
    its iterates are bitwise identical to it. This matters: nearly identical
    sources (common when neighbouring robots share estimates) make the KKT
    system ill-conditioned, and any rounding change would alter the weights.
    """
    n, d, _ = S.shape
    with np.errstate(**_QUIET_FP):
        return _newton_loop(S, S.reshape(n, d * d), n, d, max_iter)


def _newton_loop(S: np.ndarray, flat: np.ndarray, n: int, d: int, max_iter: int) -> np.ndarray:
    inv = _raw_inv
    diagonal = list(range(0, d * d, d + 1))
    c = np.full(n, 1.0 / n)
    f, p = _flat_objective(flat, c, d)
    if p is None:
        raise _DegenerateCI
    f = float(f)
    free = np.ones(n, dtype=bool)
    all_free = True
    # KKT matrix and right-hand side of the all-free case, reused across iterations.
    kkt_full = np.ones((n + 1, n + 1))
    kkt_full[n, n] = 0.0
    rhs_full = np.zeros(n + 1)

    for _ in range(max_iter):
        b = S @ p  # S_i P, shape (n, d, d)
        a = p @ b  # P S_i P
        trace_a = np.einsum("nii->n", a)
        grad = -trace_a
        hess = 2.0 * a.reshape(n, -1) @ b.transpose(0, 2, 1).reshape(n, -1).T

        if all_free:
            idx, m = None, n
            kkt, rhs = kkt_full, rhs_full
            kkt[:n, :n] = hess
            rhs[:n] = trace_a  # -grad
        else:
            idx = np.flatnonzero(free)
            m = idx.size
            kkt = np.empty((m + 1, m + 1))
            kkt[:m, :m] = hess[idx][:, idx]
            kkt[:m, m] = kkt[m, :m] = 1.0
            kkt[m, m] = 0.0
            rhs = np.empty(m + 1)
            rhs[:m] = -grad[idx]
            rhs[m] = 0.0
        solution = _raw_solve(kkt, rhs)
        if not np.isfinite(solution).all():  # singular or ill-conditioned KKT system
            solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        if idx is None:
            step = solution[:m].copy()
        else:
            step = np.zeros(n)
            step[idx] = solution[:m]
        nu = solution[m]
        decrement = -float(grad @ step)  # Newton decrement (>= 0 for convex f)

        accepted = False
        if decrement > 1e-14 * f and np.abs(step).max() > 1e-13:
            decreasing = step < 0
            alpha_max = 1.0
            if decreasing.any():
                alpha_max = min(1.0, float((c[decreasing] / -step[decreasing]).min()))
            alpha = alpha_max
            while alpha > 1e-12:
                trial = np.maximum(c + alpha * step, 0.0)
                trial /= trial.sum()
                # Objective trace(inv(sum_i trial_i S_i)), see _flat_objective.
                p_trial = inv((trial @ flat).reshape(d, d))
                f_trial = 0.0
                for k in diagonal:  # same summation order as ndarray.trace
                    f_trial += p_trial.item(k)
                if 0.0 < f_trial < math.inf and f_trial <= f - 1e-4 * alpha * decrement:
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                if alpha == alpha_max and alpha_max < 1.0:
                    blocked = decreasing & (c + alpha * step <= 1e-14)
                    free &= ~blocked
                    trial[blocked] = 0.0
                    trial /= trial.sum()
                    f_blocked, p_blocked = _flat_objective(flat, trial, d)
                    if p_blocked is not None:
                        f_trial, p_trial = float(f_blocked), p_blocked
                    else:
                        free |= blocked
                    all_free = bool(free.all())
                c, f, p = trial, f_trial, p_trial
                continue

        # No further progress on the current free set: check the KKT
        # multipliers of the variables fixed at zero and release the most
        # violated one.
        if not all_free:
            multipliers = grad + nu
            fixed = np.flatnonzero(~free)
            tolerance = 1e-9 * max(1.0, float(np.abs(grad).max()))
            if multipliers[fixed].min() < -tolerance:
                free[fixed[np.argmin(multipliers[fixed])]] = True
                all_free = bool(free.all())
                continue
        break
    return c


def _weights_slsqp(S: np.ndarray) -> np.ndarray:
    """Original solver: SLSQP, finite differences, pseudo-inverse objective."""
    n = S.shape[0]

    def objective(c):
        return np.trace(np.linalg.pinv(_combine(S, c)))

    solution = minimize(
        objective,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0 - 1e-10)] * n,
        constraints=[{"type": "eq", "fun": lambda c: np.sum(c) - 1.0}],
        options={"ftol": 1e-10},
    )
    return solution.x


def ci_weights(S: np.ndarray, solver: str = "newton") -> np.ndarray:
    """Optimal covariance-intersection weights for information matrices ``S``.

    Parameters
    ----------
    S:
        Array of shape ``(n_sources, d, d)`` of symmetric positive
        semi-definite information matrices.
    solver:
        ``"newton"`` (default) or ``"slsqp"`` (original implementation).

    Returns
    -------
    numpy.ndarray
        Weights of shape ``(n_sources,)`` on the probability simplex.
    """
    if solver not in CI_SOLVERS:
        raise ValueError(f"solver must be one of {CI_SOLVERS}, got {solver!r}")
    S = np.asarray(S, dtype=np.float64)
    n = S.shape[0]
    if n == 0:
        raise ValueError("covariance intersection needs at least one source")
    if n == 1:
        return np.ones(1)
    if solver == "slsqp":
        return _weights_slsqp(S)
    # Dimensions that carry no information in any source (e.g. the heading of a
    # target observed only through range and bearing) are dropped; the
    # pseudo-inverse objective ignores them as well.
    informative = (np.diagonal(S, axis1=1, axis2=2) != 0).any(axis=0)
    if informative.all():
        reduced = np.ascontiguousarray(S)
    else:
        support = np.flatnonzero(informative)
        reduced = S[:, support][:, :, support]
    try:
        return _weights_newton(reduced)
    except _DegenerateCI:
        return _weights_slsqp(S)


def covariance_intersection(
    S: np.ndarray,
    y: np.ndarray,
    information_form: bool = False,
    solver: str = "newton",
):
    """Fuse information pairs ``(S_i, y_i)`` by covariance intersection.

    Parameters
    ----------
    S:
        Information matrices, shape ``(n_sources, d, d)``.
    y:
        Information vectors, shape ``(d, n_sources)``.
    information_form:
        Return the fused information pair instead of mean and covariance.
    solver:
        Weight solver, see :func:`ci_weights`.

    Returns
    -------
    tuple
        ``(covariance, mean)`` or, with ``information_form=True``,
        ``(information_matrix, information_vector)``.
    """
    S = np.asarray(S, dtype=np.float64)
    c = ci_weights(S, solver=solver)
    fused = _combine(S, c)
    information = y @ c
    if information_form:
        return fused, information
    covariance = psd_inverse(fused)
    return covariance, covariance @ information
