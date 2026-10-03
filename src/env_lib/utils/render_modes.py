"""Render-mode helpers the environments need without importing matplotlib.

:mod:`env_lib.utils.rendering` re-exports everything defined here; the
environment modules import from this module so that creating an environment
without a ``render_mode`` does not import matplotlib.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["RENDER_MODES", "state_without_renderer", "validate_render_mode"]

RENDER_MODES: tuple[str, ...] = ("human", "rgb_array")


def validate_render_mode(render_mode: str | None, allowed: Sequence[str] = RENDER_MODES) -> None:
    """Raise ``ValueError`` if ``render_mode`` is not ``None`` or one of ``allowed``."""
    if render_mode is not None and render_mode not in allowed:
        raise ValueError(
            f"render_mode must be one of {tuple(allowed)} or None, got {render_mode!r}"
        )


def state_without_renderer(env: Any) -> dict[str, Any]:
    """``env.__dict__`` without its renderer, for ``pickle`` and ``copy.deepcopy``.

    Renderers hold matplotlib figures or pygame surfaces, which cannot be
    pickled; environments rebuild them on the next ``render()``, so a pickled
    or copied environment keeps its full simulation state and renders as
    before.
    """
    state = env.__dict__.copy()
    if state.get("_renderer") is not None:
        state["_renderer"] = None
    return state
