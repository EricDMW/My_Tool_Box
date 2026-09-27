"""AJLATT: active joint localisation and target tracking.

Main entry points
-----------------
:class:`AJLATTEnv`
    The Gymnasium environment (joint multi-agent spaces).
:class:`AJLATTConfig`
    Dataclass holding every parameter (legacy names accepted by
    :meth:`AJLATTConfig.from_kwargs`).
:func:`make`
    Backwards-compatible factory accepting the original keyword names.
:class:`TeamRewardWrapper`
    Scalar team reward / termination view for single-agent algorithms.

For backwards compatibility the module itself is callable:
``env_lib.ajlatt_env(map_name="obstacles04", num_Robot=4)`` is equivalent to
:func:`make`.
"""

from __future__ import annotations

import sys
import types
import warnings
from typing import Any

import gymnasium as gym
import numpy as np

from env_lib.ajlatt_env.config import AJLATTConfig
from env_lib.ajlatt_env.env import AJLATTEnv
from env_lib.ajlatt_env.maps import available_maps, load_grid_map, load_map

__all__ = [
    "AJLATTConfig",
    "AJLATTEnv",
    "TeamRewardWrapper",
    "ajlatt_env",
    "available_maps",
    "load_grid_map",
    "load_map",
    "make",
]


def make(figID: int = 0, *args: Any, render_mode: str | None = None, **kwargs: Any) -> AJLATTEnv:
    """Create an :class:`AJLATTEnv` from (legacy) keyword arguments.

    Unknown keyword arguments are ignored with a warning, as the original
    factory silently ignored them. The legacy ``render=True`` flag maps to
    ``render_mode="human"``.

    Parameters
    ----------
    figID:
        Ignored (kept for signature compatibility).
    render_mode:
        ``None``, ``"human"`` or ``"rgb_array"``.
    **kwargs:
        :class:`AJLATTConfig` fields or their legacy names.
    """
    return _make(args, render_mode, kwargs, stacklevel=3)


def _make(
    args: tuple[Any, ...], render_mode: str | None, kwargs: dict[str, Any], *, stacklevel: int
) -> AJLATTEnv:
    # ``stacklevel`` is the warning stack level, seen from this function, of
    # the user code that called the public factory.
    if args:
        raise TypeError("make() accepts keyword arguments only")
    legacy_render = kwargs.pop("render", None)
    if legacy_render is not None:
        warnings.warn(
            "The 'render' flag is deprecated; pass render_mode='human' or 'rgb_array'",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        if render_mode is None and legacy_render:
            render_mode = "human"
    config = AJLATTConfig.from_kwargs(strict=False, _stacklevel=stacklevel + 1, **kwargs)
    return AJLATTEnv(config, render_mode=render_mode)


class TeamRewardWrapper(gym.Wrapper):
    """Expose the team reward (sum over robots) and a scalar ``terminated`` flag.

    The per-robot values stay available as ``info["agent_rewards"]`` and
    ``info["agent_terminated"]``.
    """

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["agent_terminated"] = np.asarray(terminated, dtype=bool)
        return observation, float(np.sum(reward)), bool(np.any(terminated)), truncated, info


class _CallableModule(types.ModuleType):
    """Allow ``env_lib.ajlatt_env(...)`` as an alias of :func:`make`."""

    def __call__(
        self, figID: int = 0, *args: Any, render_mode: str | None = None, **kwargs: Any
    ) -> AJLATTEnv:
        return _make(args, render_mode, kwargs, stacklevel=3)


#: Deprecated alias kept for ``from env_lib.ajlatt_env import ajlatt_env``.
ajlatt_env = make

sys.modules[__name__].__class__ = _CallableModule
