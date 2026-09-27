"""Tabular reinforcement-learning tools.

Q-tables, value tables and stochastic policy tables for finite state and action
spaces, together with the classic tabular updates (Q-learning, SARSA, expected
SARSA) and exploration rules (epsilon-greedy, softmax/Boltzmann, UCB, Thompson
sampling).

All randomness comes from a :class:`numpy.random.Generator` owned by each table
(argument ``rng``: a generator, an integer seed, or ``None`` for fresh entropy);
the global NumPy and Python random states are never used.

Examples
--------
>>> from toolkit.neural_toolkit import DiscreteTools
>>> q = DiscreteTools.create_q_table(5, 2, rng=0)
>>> DiscreteTools.q_learning_update(q, state=0, action=1, reward=1.0, next_state=1, alpha=0.5)
>>> q.get_value(0, 1)
0.5
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np

__all__ = [
    "BaseDiscreteTable",
    "DiscreteEnvironment",
    "DiscreteTools",
    "PolicyTable",
    "QTable",
    "RNGLike",
    "ValueTable",
    "softmax",
]

RNGLike = Optional[Union[int, np.random.Generator]]
"""A ``numpy.random.Generator``, an integer seed, or ``None`` (fresh OS entropy)."""


def _check_size(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    try:
        size = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be a positive integer, got {value!r}") from None
    if size <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return size


def _check_index(name: str, value: Any, size: int) -> int:
    """Validate a state/action index (negative indices are rejected, not wrapped)."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer index, got {value!r}")
    try:
        index = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer index, got {value!r}") from None
    if not 0 <= index < size:
        raise IndexError(f"{name} {index} is out of range [0, {size})")
    return index


def _check_probability(name: str, value: float) -> float:
    p = float(value)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return p


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax of ``values / temperature`` (float64).

    Parameters
    ----------
    values : array_like
        Preferences, for example the Q-values of one state.
    temperature : float, default 1.0
        Positive temperature; small values approach the greedy distribution.

    Returns
    -------
    numpy.ndarray
        Probabilities summing to one.

    Raises
    ------
    ValueError
        If ``temperature`` is not positive.
    """
    if not temperature > 0:
        raise ValueError(f"temperature must be positive, got {temperature!r}")
    z = np.asarray(values, dtype=np.float64) / float(temperature)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


class BaseDiscreteTable(ABC):
    """Base class for tabular value containers.

    Parameters
    ----------
    state_space_size : int
        Number of states.
    action_space_size : int, optional
        Number of actions (required by action-indexed tables).
    initial_value : float, default 0.0
        Initial entry value.
    dtype : numpy dtype, default ``np.float32``
        Storage dtype of :attr:`table`.
    rng : Generator, int or None, optional
        Random generator used for sampling (see :data:`RNGLike`).

    Attributes
    ----------
    table : numpy.ndarray
        The underlying array.
    rng : numpy.random.Generator
        The generator used for all sampling.
    """

    table: np.ndarray

    def __init__(
        self,
        state_space_size: int,
        action_space_size: int | None = None,
        initial_value: float = 0.0,
        dtype: Any = np.float32,
        rng: RNGLike = None,
    ) -> None:
        self.state_space_size = _check_size("state_space_size", state_space_size)
        self.action_space_size = (
            None
            if action_space_size is None
            else _check_size("action_space_size", action_space_size)
        )
        self.initial_value = float(initial_value)
        self.dtype = dtype
        self.rng = np.random.default_rng(rng)
        self._initialize_table()

    def _state(self, state: Any) -> int:
        return _check_index("state", state, self.state_space_size)

    def _action(self, action: Any) -> int:
        if self.action_space_size is None:
            raise ValueError(f"{type(self).__name__} has no action dimension")
        return _check_index("action", action, self.action_space_size)

    def reset(self) -> None:
        """Re-initialise the table to its initial values."""
        self._initialize_table()

    @abstractmethod
    def _initialize_table(self) -> None:
        """Allocate :attr:`table`."""

    @abstractmethod
    def get_value(self, state: int, action: int | None = None) -> float:
        """Return the entry for ``state`` (and ``action``)."""

    @abstractmethod
    def set_value(self, state: int, value: float, action: int | None = None) -> None:
        """Overwrite the entry for ``state`` (and ``action``)."""

    @abstractmethod
    def update_value(
        self, state: int, value: float, action: int | None = None, learning_rate: float = 0.1
    ) -> None:
        """Move the entry towards ``value``: ``x <- x + learning_rate * (value - x)``."""


class QTable(BaseDiscreteTable):
    """Action-value table ``Q[s, a]`` of shape ``(state_space_size, action_space_size)``."""

    def _initialize_table(self) -> None:
        if self.action_space_size is None:
            raise ValueError("action_space_size must be specified for QTable")
        self.table = np.full(
            (self.state_space_size, self.action_space_size), self.initial_value, dtype=self.dtype
        )

    def get_value(self, state: int, action: int) -> float:  # type: ignore[override]
        """Return ``Q(state, action)``."""
        return float(self.table[self._state(state), self._action(action)])

    def set_value(self, state: int, value: float, action: int) -> None:  # type: ignore[override]
        """Set ``Q(state, action) = value``."""
        self.table[self._state(state), self._action(action)] = value

    def update_value(  # type: ignore[override]
        self, state: int, value: float, action: int, learning_rate: float = 0.1
    ) -> None:
        """Move ``Q(state, action)`` towards the target ``value`` with step ``learning_rate``."""
        s, a = self._state(state), self._action(action)
        current = float(self.table[s, a])
        self.table[s, a] = current + learning_rate * (value - current)

    def get_max_q_value(self, state: int) -> float:
        """Return ``max_a Q(state, a)``."""
        return float(np.max(self.table[self._state(state)]))

    def get_max_action(self, state: int) -> int:
        """Return ``argmax_a Q(state, a)`` (the lowest index among ties)."""
        return int(np.argmax(self.table[self._state(state)]))

    def get_q_values(self, state: int) -> np.ndarray:
        """Return a copy of the Q-values of ``state``."""
        return self.table[self._state(state)].copy()

    def get_policy(self, state: int, epsilon: float = 0.0) -> int:
        """Sample an epsilon-greedy action: uniform with probability ``epsilon``, else greedy."""
        epsilon = _check_probability("epsilon", epsilon)
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_space_size))
        return self.get_max_action(state)

    def get_epsilon_greedy_probs(self, state: int, epsilon: float = 0.0) -> np.ndarray:
        """Action probabilities of the epsilon-greedy policy (ties share the greedy mass)."""
        epsilon = _check_probability("epsilon", epsilon)
        q = self.table[self._state(state)]
        greedy = q == q.max()
        n = self.action_space_size
        return epsilon / n + (1.0 - epsilon) * greedy / greedy.sum()

    def get_softmax_probs(self, state: int, temperature: float = 1.0) -> np.ndarray:
        """Boltzmann action probabilities ``softmax(Q(state, .) / temperature)``."""
        return softmax(self.table[self._state(state)], temperature)

    def get_softmax_policy(self, state: int, temperature: float = 1.0) -> int:
        """Sample an action from the Boltzmann (softmax) policy."""
        probs = self.get_softmax_probs(state, temperature)
        return int(self.rng.choice(self.action_space_size, p=probs))


class ValueTable(BaseDiscreteTable):
    """State-value table ``V[s]`` of shape ``(state_space_size,)``."""

    def _initialize_table(self) -> None:
        self.table = np.full(self.state_space_size, self.initial_value, dtype=self.dtype)

    def get_value(self, state: int, action: int | None = None) -> float:
        """Return ``V(state)`` (``action`` is ignored)."""
        return float(self.table[self._state(state)])

    def set_value(self, state: int, value: float, action: int | None = None) -> None:
        """Set ``V(state) = value`` (``action`` is ignored)."""
        self.table[self._state(state)] = value

    def update_value(
        self, state: int, value: float, action: int | None = None, learning_rate: float = 0.1
    ) -> None:
        """Move ``V(state)`` towards ``value`` with step ``learning_rate``."""
        s = self._state(state)
        current = float(self.table[s])
        self.table[s] = current + learning_rate * (value - current)

    def get_values(self) -> np.ndarray:
        """Return a copy of all state values."""
        return self.table.copy()


class PolicyTable(BaseDiscreteTable):
    """Stochastic policy table ``pi[s, a]``; every row is a probability distribution.

    The table starts uniform. :meth:`set_value` and :meth:`update_value` change one
    probability and rescale the other actions of that state proportionally so that
    the row still sums to one (the value that was set is preserved exactly).
    """

    def _initialize_table(self) -> None:
        if self.action_space_size is None:
            raise ValueError("action_space_size must be specified for PolicyTable")
        self.table = np.full(
            (self.state_space_size, self.action_space_size),
            1.0 / self.action_space_size,
            dtype=self.dtype,
        )

    def get_value(self, state: int, action: int) -> float:  # type: ignore[override]
        """Return ``pi(action | state)``."""
        return float(self.table[self._state(state), self._action(action)])

    def set_value(self, state: int, value: float, action: int) -> None:  # type: ignore[override]
        """Set ``pi(action | state) = value`` and rescale the other actions to sum to ``1 - value``.

        Raises
        ------
        ValueError
            If ``value`` is outside ``[0, 1]``.
        """
        s, a = self._state(state), self._action(action)
        p = _check_probability("value", value)
        row = self.table[s].astype(np.float64)
        others = np.ones(self.action_space_size, dtype=bool)
        others[a] = False
        rest = row[others].sum()
        if rest > 0.0:
            row[others] *= (1.0 - p) / rest
        else:
            row[others] = (1.0 - p) / max(self.action_space_size - 1, 1)
        row[a] = p
        if self.action_space_size == 1:
            row[a] = 1.0
        self.table[s] = row

    def update_value(  # type: ignore[override]
        self, state: int, value: float, action: int, learning_rate: float = 0.1
    ) -> None:
        """Move ``pi(action | state)`` towards ``value`` and renormalise as in :meth:`set_value`."""
        current = self.get_value(state, action)
        target = _check_probability("value", value)
        self.set_value(state, current + learning_rate * (target - current), action)

    def _normalize_state(self, state: int) -> None:
        """Rescale the row of ``state`` to sum to one (uniform if it sums to zero)."""
        s = self._state(state)
        row = np.clip(self.table[s].astype(np.float64), 0.0, None)
        total = row.sum()
        self.table[s] = row / total if total > 0 else 1.0 / self.action_space_size

    def set_policy_probs(self, state: int, probs: np.ndarray) -> None:
        """Replace the distribution of ``state`` (non-negative weights, renormalised)."""
        s = self._state(state)
        p = np.asarray(probs, dtype=np.float64)
        if p.shape != (self.action_space_size,) or np.any(p < 0) or not np.isfinite(p).all():
            raise ValueError(
                f"probs must be {self.action_space_size} finite non-negative weights, got {probs!r}"
            )
        self.table[s] = p
        self._normalize_state(s)

    def get_policy(self, state: int) -> int:
        """Sample an action from ``pi(. | state)``."""
        probs = self.table[self._state(state)].astype(np.float64)
        return int(self.rng.choice(self.action_space_size, p=probs / probs.sum()))

    def get_policy_probs(self, state: int) -> np.ndarray:
        """Return a copy of ``pi(. | state)``."""
        return self.table[self._state(state)].copy()

    def set_deterministic_policy(self, state: int, action: int) -> None:
        """Make ``state`` choose ``action`` with probability one."""
        s, a = self._state(state), self._action(action)
        self.table[s] = 0.0
        self.table[s, a] = 1.0


class DiscreteTools:
    """Static helpers for tabular reinforcement learning."""

    @staticmethod
    def create_q_table(
        state_space_size: int,
        action_space_size: int,
        initial_value: float = 0.0,
        rng: RNGLike = None,
    ) -> QTable:
        """Create a :class:`QTable`."""
        return QTable(state_space_size, action_space_size, initial_value, rng=rng)

    @staticmethod
    def create_value_table(
        state_space_size: int, initial_value: float = 0.0, rng: RNGLike = None
    ) -> ValueTable:
        """Create a :class:`ValueTable`."""
        return ValueTable(state_space_size, initial_value=initial_value, rng=rng)

    @staticmethod
    def create_policy_table(
        state_space_size: int, action_space_size: int, rng: RNGLike = None
    ) -> PolicyTable:
        """Create a uniform :class:`PolicyTable`."""
        return PolicyTable(state_space_size, action_space_size, rng=rng)

    @staticmethod
    def q_learning_update(
        q_table: QTable,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        gamma: float = 0.99,
        alpha: float = 0.1,
        done: bool = False,
    ) -> None:
        """Q-learning: ``Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))``.

        When ``done`` is True (terminal transition) the bootstrap term is dropped.
        """
        bootstrap = 0.0 if done else q_table.get_max_q_value(next_state)
        q_table.update_value(state, reward + gamma * bootstrap, action, alpha)

    @staticmethod
    def sarsa_update(
        q_table: QTable,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        gamma: float = 0.99,
        alpha: float = 0.1,
        done: bool = False,
    ) -> None:
        """SARSA: ``Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))`` (no bootstrap if ``done``)."""
        bootstrap = 0.0 if done else q_table.get_value(next_state, next_action)
        q_table.update_value(state, reward + gamma * bootstrap, action, alpha)

    @staticmethod
    def expected_sarsa_update(
        q_table: QTable,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        policy_table: PolicyTable | None = None,
        gamma: float = 0.99,
        alpha: float = 0.1,
        epsilon: float = 0.0,
        done: bool = False,
    ) -> None:
        """Expected SARSA: target ``r + gamma * sum_a' pi(a'|s') Q(s',a')``.

        Parameters
        ----------
        policy_table : PolicyTable, optional
            Target policy ``pi``. If omitted, the epsilon-greedy policy of ``q_table``
            with the given ``epsilon`` is used (ties share the greedy mass).
        epsilon : float, default 0.0
            Exploration rate of the implicit epsilon-greedy target policy.
        done : bool, default False
            Terminal transition (no bootstrap).
        """
        if done:
            expected = 0.0
        else:
            q_next = q_table.get_q_values(next_state).astype(np.float64)
            if policy_table is None:
                probs = q_table.get_epsilon_greedy_probs(next_state, epsilon)
            else:
                if policy_table.action_space_size != q_table.action_space_size:
                    raise ValueError(
                        "policy_table and q_table must have the same number of actions, got "
                        f"{policy_table.action_space_size} and {q_table.action_space_size}"
                    )
                probs = policy_table.get_policy_probs(next_state).astype(np.float64)
                probs = probs / probs.sum()
            expected = float(np.dot(probs, q_next))
        q_table.update_value(state, reward + gamma * expected, action, alpha)

    @staticmethod
    def value_iteration_update(
        value_table: ValueTable, state: int, q_table: QTable, gamma: float = 0.99
    ) -> None:
        """Greedy backup ``V(s) = max_a Q(s, a)``.

        ``q_table`` must already contain one-step lookahead values
        ``r + gamma * V(s')``; ``gamma`` is accepted for backward compatibility
        and not used.
        """
        value_table.set_value(state, q_table.get_max_q_value(state))

    @staticmethod
    def policy_iteration_update(policy_table: PolicyTable, state: int, q_table: QTable) -> None:
        """Greedy policy improvement: ``pi(. | s)`` becomes deterministic on ``argmax_a Q(s, a)``."""
        policy_table.set_deterministic_policy(state, q_table.get_max_action(state))

    @staticmethod
    def epsilon_greedy_policy(q_table: QTable, state: int, epsilon: float) -> int:
        """Sample an epsilon-greedy action (uses ``q_table.rng``)."""
        return q_table.get_policy(state, epsilon)

    @staticmethod
    def softmax_policy(q_table: QTable, state: int, temperature: float) -> int:
        """Sample from ``softmax(Q(s, .) / temperature)`` (uses ``q_table.rng``)."""
        return q_table.get_softmax_policy(state, temperature)

    @staticmethod
    def boltzmann_policy(q_table: QTable, state: int, temperature: float) -> int:
        """Alias of :meth:`softmax_policy`."""
        return q_table.get_softmax_policy(state, temperature)

    @staticmethod
    def ucb_policy(
        q_table: QTable, state: int, visit_counts: np.ndarray, exploration_constant: float = 1.0
    ) -> int:
        """UCB1 action selection.

        Picks ``argmax_a Q(s,a) + c * sqrt(ln N(s) / N(s,a))`` with
        ``N(s) = sum_a N(s,a)``. Untried actions (``N(s,a) = 0``) are chosen first,
        uniformly at random among themselves (using ``q_table.rng``).

        Parameters
        ----------
        q_table : QTable
            Current value estimates.
        state : int
            Current state.
        visit_counts : numpy.ndarray
            Visit counts ``N(s, a)`` of shape ``(state_space_size, action_space_size)``.
        exploration_constant : float, default 1.0
            Exploration weight ``c``.
        """
        s = q_table._state(state)
        counts = np.asarray(visit_counts, dtype=np.float64)
        expected_shape = (q_table.state_space_size, q_table.action_space_size)
        if counts.shape != expected_shape:
            raise ValueError(f"visit_counts must have shape {expected_shape}, got {counts.shape}")
        n_sa = counts[s]
        untried = np.flatnonzero(n_sa <= 0)
        if untried.size:
            return int(q_table.rng.choice(untried))
        bonus = exploration_constant * np.sqrt(np.log(n_sa.sum()) / n_sa)
        return int(np.argmax(q_table.table[s].astype(np.float64) + bonus))

    @staticmethod
    def thompson_sampling_policy(
        q_table: QTable,
        state: int,
        visit_counts: np.ndarray,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> int:
        """Heuristic Beta-Bernoulli Thompson sampling.

        Each action's Q-value is squashed to a success probability
        ``p = sigmoid(Q(s,a))``; with ``n = N(s,a)`` visits the posterior is
        ``Beta(prior_alpha + n p, prior_beta + n (1 - p))``. One sample is drawn per
        action (using ``q_table.rng``) and the argmax is returned. This is exact
        Thompson sampling only for Bernoulli rewards with ``Q`` in logit space.
        """
        s = q_table._state(state)
        counts = np.asarray(visit_counts, dtype=np.float64)
        expected_shape = (q_table.state_space_size, q_table.action_space_size)
        if counts.shape != expected_shape:
            raise ValueError(f"visit_counts must have shape {expected_shape}, got {counts.shape}")
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be positive")
        q = q_table.table[s].astype(np.float64)
        p = 0.5 * (1.0 + np.tanh(0.5 * q))  # numerically stable sigmoid
        n = np.clip(counts[s], 0.0, None)
        samples = q_table.rng.beta(prior_alpha + n * p, prior_beta + n * (1.0 - p))
        return int(np.argmax(samples))


class DiscreteEnvironment(ABC):
    """Minimal base class for finite-state environments.

    Subclasses implement :meth:`reset` and :meth:`step` and may use
    :attr:`rng` for randomness.

    Parameters
    ----------
    state_space_size, action_space_size : int
        Sizes of the state and action spaces.
    rng : Generator, int or None, optional
        Random generator (see :data:`RNGLike`).
    """

    def __init__(self, state_space_size: int, action_space_size: int, rng: RNGLike = None) -> None:
        self.state_space_size = _check_size("state_space_size", state_space_size)
        self.action_space_size = _check_size("action_space_size", action_space_size)
        self.current_state = 0
        self.rng = np.random.default_rng(rng)

    @abstractmethod
    def reset(self) -> int:
        """Reset the environment and return the initial state."""

    @abstractmethod
    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        """Apply ``action`` and return ``(next_state, reward, done, info)``."""

    def get_state(self) -> int:
        """Return the current state."""
        return self.current_state
