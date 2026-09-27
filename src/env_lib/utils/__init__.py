"""Utilities shared by the ``env_lib`` environments (rendering and recording)."""

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

__all__ = [
    "DARK",
    "LIGHT",
    "FigureWindow",
    "MatplotlibRenderer",
    "Theme",
    "figure_to_rgb",
    "get_theme",
    "record_episode",
    "register_theme",
    "save_animation",
    "set_theme",
    "style_axes",
    "with_alpha",
]
