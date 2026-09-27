"""Utilities shared by the ``env_lib`` environments.

Rendering themes and figures, episode recording, graph helpers
(:mod:`env_lib.utils.graphs`), the batched vector-environment base class
(:class:`BatchedVectorEnv`) and policy evaluation (:func:`evaluate`,
:func:`rollout`).
"""

from __future__ import annotations

import importlib
from typing import Any

# Attributes are imported on first use, so that environments importing
# ``env_lib.utils.graphs`` do not pull in matplotlib (``rendering``).
_LAZY: dict[str, str] = {
    "DARK": "env_lib.utils.rendering",
    "LIGHT": "env_lib.utils.rendering",
    "FigureWindow": "env_lib.utils.rendering",
    "MatplotlibRenderer": "env_lib.utils.rendering",
    "Theme": "env_lib.utils.rendering",
    "figure_to_rgb": "env_lib.utils.rendering",
    "get_theme": "env_lib.utils.rendering",
    "register_theme": "env_lib.utils.rendering",
    "set_theme": "env_lib.utils.rendering",
    "style_axes": "env_lib.utils.rendering",
    "with_alpha": "env_lib.utils.rendering",
    "EvaluationResult": "env_lib.utils.evaluation",
    "Trajectory": "env_lib.utils.evaluation",
    "evaluate": "env_lib.utils.evaluation",
    "rollout": "env_lib.utils.evaluation",
    "record_episode": "env_lib.utils.recording",
    "save_animation": "env_lib.utils.recording",
    "BatchedVectorEnv": "env_lib.utils.vector",
}

__all__ = [
    "DARK",
    "LIGHT",
    "BatchedVectorEnv",
    "EvaluationResult",
    "FigureWindow",
    "MatplotlibRenderer",
    "Theme",
    "Trajectory",
    "evaluate",
    "figure_to_rgb",
    "get_theme",
    "graphs",
    "record_episode",
    "register_theme",
    "rollout",
    "save_animation",
    "set_theme",
    "style_axes",
    "with_alpha",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    if name == "graphs":
        return importlib.import_module(f"{__name__}.graphs")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
