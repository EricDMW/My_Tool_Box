"""env_lib: multi-agent reinforcement learning environments.

Every environment follows the Gymnasium API and is registered on import, so it
can be created either directly or through :func:`env_lib.make`::

    import env_lib

    env = env_lib.make("KuramotoOscillator-v0", render_mode="rgb_array")
    env = env_lib.KuramotoOscillatorEnv(n_oscillators=16)

Environment classes are imported lazily: optional dependencies such as
``pygame``/``pymunk`` (Pistonball) or ``torch`` (PyTorch Kuramoto backend) are
only required when the corresponding environment is used.
"""

from __future__ import annotations

import importlib
from typing import Any

from env_lib._version import __version__
from env_lib.errors import ResetNeededError
from env_lib.registration import get_spec, list_envs, make, make_vec, register_envs

register_envs()

# Public attribute -> (module, attribute) for lazy imports.
_LAZY_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "KuramotoOscillatorEnv": ("env_lib.kos_env.kuramoto_env", "KuramotoOscillatorEnv"),
    "KuramotoOscillatorEnvTorch": (
        "env_lib.kos_env.kuramoto_env_torch",
        "KuramotoOscillatorEnvTorch",
    ),
    "LineMsgEnv": ("env_lib.linemsg_env.linemsg_env", "LineMsgEnv"),
    "WirelessCommEnv": ("env_lib.wireless_comm_env.wireless_comm_env", "WirelessCommEnv"),
    "PistonballEnv": ("env_lib.pistonball_env.pistonball_env", "PistonballEnv"),
    "ConsensusEnv": ("env_lib.consensus_env.consensus_env", "ConsensusEnv"),
    "AJLATTEnv": ("env_lib.ajlatt_env.env", "AJLATTEnv"),
    "AJLATTConfig": ("env_lib.ajlatt_env.config", "AJLATTConfig"),
    "PowerGridEnv": ("env_lib.power_grid_env.power_grid_env", "PowerGridEnv"),
    "PowerGridVectorEnv": ("env_lib.power_grid_env.power_grid_env", "PowerGridVectorEnv"),
    "PlatoonEnv": ("env_lib.platoon_env.platoon_env", "PlatoonEnv"),
    "PlatoonVectorEnv": ("env_lib.platoon_env.platoon_env", "PlatoonVectorEnv"),
    "ConsensusVectorEnv": ("env_lib.consensus_env.vector", "ConsensusVectorEnv"),
    "KuramotoOscillatorVectorEnv": ("env_lib.kos_env.vector", "KuramotoOscillatorVectorEnv"),
    # Convenience API.
    "EnvInfo": ("env_lib.catalog", "EnvInfo"),
    "describe": ("env_lib.catalog", "describe"),
    "baseline_policy": ("env_lib.baselines", "baseline_policy"),
    "evaluate": ("env_lib.utils.evaluation", "evaluate"),
    "rollout": ("env_lib.utils.evaluation", "rollout"),
}

_SUBPACKAGES = (
    "ajlatt_env",
    "baselines",
    "catalog",
    "consensus_env",
    "kos_env",
    "linemsg_env",
    "pistonball_env",
    "platoon_env",
    "power_grid_env",
    "utils",
    "wireless_comm_env",
    "wrappers",
)

# Names that need an optional extra (torch, or pygame/pymunk). They stay
# reachable as attributes but are left out of ``__all__``, so that
# ``from env_lib import *`` works on a base installation.
_NEEDS_EXTRA = frozenset({"KuramotoOscillatorEnvTorch", "PistonballEnv", "pistonball_env"})

__all__: list[str] = [
    "ResetNeededError",
    "__version__",
    "get_spec",
    "list_envs",
    "make",
    "make_vec",
    "register_envs",
    *(name for name in sorted(_LAZY_ATTRIBUTES) if name not in _NEEDS_EXTRA),
    *(name for name in _SUBPACKAGES if name not in _NEEDS_EXTRA),
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRIBUTES:
        module_name, attribute = _LAZY_ATTRIBUTES[name]
        value = getattr(importlib.import_module(module_name), attribute)
        globals()[name] = value
        return value
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _NEEDS_EXTRA)
