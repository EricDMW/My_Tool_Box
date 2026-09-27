"""Wireless multiple-access environment (Gymnasium ids ``WirelessComm-v0`` and ``-v1``).

Registration with Gymnasium is handled centrally by :mod:`env_lib.registration`.
"""

from __future__ import annotations

import warnings

from env_lib.wireless_comm_env.wireless_comm_env import WirelessCommEnv

__all__ = ["WirelessCommEnv", "register"]


def register() -> None:
    """Register the ``env_lib`` environments with Gymnasium.

    .. deprecated::
        Registration happens automatically on ``import env_lib``; use
        :func:`env_lib.registration.register_envs` if you need to call it
        explicitly.
    """
    warnings.warn(
        "env_lib.wireless_comm_env.register() is deprecated; environments are registered on "
        "import. Use env_lib.registration.register_envs() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from env_lib.registration import register_envs

    register_envs()
