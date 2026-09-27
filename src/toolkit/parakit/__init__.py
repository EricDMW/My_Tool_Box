"""parakit: parameter management for ``argparse`` parsers.

Headless API (no GUI dependency):

* :func:`get_parameters` - current defaults of a parser as a dict.
* :func:`parse_value` - convert entry text or JSON values using the parser's
  declarations (types, booleans, ``nargs``, ``None``, ``choices``).
* :func:`save_parameters` / :func:`load_parameters` - JSON files, timestamped
  names when saving into a directory.
* :func:`apply_parameters` - validate values and install them as parser defaults.

GUI:

* :class:`ParameterTuner` (alias :data:`ParameterAdjuster`) - a Tk window for
  editing the defaults with auto-save. Tk is imported only when the window opens.

Examples
--------
>>> import argparse
>>> from toolkit.parakit import apply_parameters, get_parameters
>>> parser = argparse.ArgumentParser()
>>> _ = parser.add_argument("--batch_size", type=int, default=32)
>>> get_parameters(apply_parameters(parser, {"batch_size": "64"}))
{'batch_size': 64}
"""

from __future__ import annotations

from toolkit._version import __version__

from .params import (
    apply_parameters,
    describe_type,
    format_value,
    get_parameters,
    load_parameters,
    parse_bool,
    parse_value,
    save_parameters,
    tunable_actions,
)
from .tune_para import TK_MISSING_MESSAGE, ParameterAdjuster, ParameterTuner

__author__ = "Dongming Wang"
__email__ = "dongming.wang@email.ucr.edu"

__all__ = [
    "__version__",
    "ParameterTuner",
    "ParameterAdjuster",
    "TK_MISSING_MESSAGE",
    "apply_parameters",
    "describe_type",
    "format_value",
    "get_parameters",
    "load_parameters",
    "parse_bool",
    "parse_value",
    "save_parameters",
    "tunable_actions",
]
