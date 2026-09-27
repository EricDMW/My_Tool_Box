"""Utilities shared by the ``env_lib`` environments.

Rendering themes and figures, episode recording, graph helpers
(:mod:`env_lib.utils.graphs`), the batched vector-environment base class
(:class:`BatchedVectorEnv`) and policy evaluation (:func:`evaluate`,
:func:`rollout`).
"""

from env_lib.utils import graphs
from env_lib.utils.evaluation import EvaluationResult, Trajectory, evaluate, rollout
from env_lib.utils.recording import record_episode, save_animation
from env_lib.utils.rendering import (
    DARK,
    LIGHT,
    FigureWindow,
    MatplotlibRenderer,
    Theme,
    figure_to_rgb,
    get_theme,
    register_theme,
    set_theme,
    style_axes,
    with_alpha,
)
from env_lib.utils.vector import BatchedVectorEnv

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
