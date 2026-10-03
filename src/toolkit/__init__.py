"""toolkit: research utilities for reinforcement learning experiments.

Subpackages
-----------
plotkit
    Publication-quality plotting (learning curves with confidence bands,
    heatmaps, grouped bars, ...). Depends on matplotlib only.
neural_toolkit
    Configurable PyTorch building blocks for RL (policy, value and Q networks,
    encoders, decoders, tabular tools). Requires ``torch``.
parakit
    Parameter management for ``argparse`` parsers, with an optional Tk GUI.

Subpackages are imported lazily, so ``import toolkit`` does not require
``torch`` or ``tkinter``.
"""

from __future__ import annotations

import importlib
from typing import Any

from toolkit._version import __version__

__author__ = "Dongming Wang"

_SUBPACKAGES = ("neural_toolkit", "parakit", "plotkit")

# neural_toolkit needs torch; it is left out of ``__all__`` so that
# ``from toolkit import *`` works on a base installation.
__all__: list[str] = ["__version__", "parakit", "plotkit"]


def __getattr__(name: str) -> Any:
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_SUBPACKAGES))
