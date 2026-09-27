"""Pistonball: cooperative multi-agent physics environment (requires ``pymunk``).

The environment is registered as ``"Pistonball-v0"`` by
:mod:`env_lib.registration`; ``pygame`` is only needed for rendering and for
:class:`ManualPolicy` (keyboard control).
"""

from __future__ import annotations

from env_lib.pistonball_env.manual_policy import ManualPolicy
from env_lib.pistonball_env.pistonball_env import PistonballEnv
from env_lib.pistonball_env.rendering import PistonballLayout, PistonballRenderer

__all__ = ["ManualPolicy", "PistonballEnv", "PistonballLayout", "PistonballRenderer"]
