"""Exceptions shared by the environments."""

from __future__ import annotations

from gymnasium.error import ResetNeeded

__all__ = ["ResetNeededError"]


class ResetNeededError(ResetNeeded, RuntimeError):
    """Raised when an environment is used before :meth:`reset` was called.

    It derives from both :class:`gymnasium.error.ResetNeeded` (raised by
    Gymnasium's order-enforcing wrapper) and :class:`RuntimeError`, so either
    base class can be caught.
    """
