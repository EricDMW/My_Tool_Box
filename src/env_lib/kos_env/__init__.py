"""Kuramoto oscillator network environments.

Two interchangeable backends of the same controlled Kuramoto model:

* :class:`KuramotoOscillatorEnv` -- NumPy, one system per environment.
* :class:`KuramotoOscillatorEnvTorch` -- PyTorch, ``n_agents`` systems simulated
  in parallel on CPU or GPU. Requires the optional ``torch`` dependency
  (``pip install "my-tool-box[torch]"``) and is imported lazily.

:class:`KuramotoOscillatorVectorEnv` is the natively batched Gymnasium vector
environment of the NumPy backend (used by ``env_lib.make_vec``).

The Gymnasium ids (``KuramotoOscillator-v0``, ``KuramotoOscillatorTorch-v0``,
...) are registered centrally by :mod:`env_lib.registration` when
:mod:`env_lib` is imported.

Examples
--------
>>> import env_lib
>>> env = env_lib.make("KuramotoOscillator-v0", render_mode="rgb_array")
>>> obs, info = env.reset(seed=0)
"""

from __future__ import annotations

import warnings
from typing import Any

from env_lib.kos_env.kuramoto_env import KuramotoOscillatorEnv
from env_lib.kos_env.vector import KuramotoOscillatorVectorEnv

__all__ = [
    "KuramotoOscillatorEnv",
    "KuramotoOscillatorEnvTorch",
    "KuramotoOscillatorVectorEnv",
    "register",
]


def register() -> None:
    """Register the Kuramoto environments with Gymnasium (deprecated).

    Registration now happens automatically when :mod:`env_lib` is imported.
    This function only calls :func:`env_lib.registration.register_envs`.
    """
    warnings.warn(
        "env_lib.kos_env.register() is deprecated: all env_lib environments are registered "
        "when env_lib is imported. Use env_lib.registration.register_envs() if you need to "
        "register them explicitly.",
        DeprecationWarning,
        stacklevel=2,
    )
    from env_lib.registration import register_envs

    register_envs()


def __getattr__(name: str) -> Any:
    if name == "KuramotoOscillatorEnvTorch":
        from env_lib.kos_env.kuramoto_env_torch import KuramotoOscillatorEnvTorch

        return KuramotoOscillatorEnvTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
