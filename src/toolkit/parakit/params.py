"""Headless parameter management for ``argparse`` parsers.

These functions read, convert, validate, save, load and apply parameter values
for an :class:`argparse.ArgumentParser` without any GUI dependency. The Tk tuner
(:class:`toolkit.parakit.ParameterTuner`) is built on top of them.

Value conversion (:func:`parse_value`) follows the parser's own declarations:

* ``type=`` callables are applied to text; without a ``type`` the type is inferred
  from the default (``bool``, ``int``, ``float`` or ``str``).
* Booleans (``store_true``/``store_false``, ``BooleanOptionalAction``, ``type=bool``
  or a boolean default) accept ``true/false``, ``yes/no``, ``on/off`` and ``1/0``
  (case-insensitive), so the string ``"False"`` becomes ``False``.
* ``nargs='+'``, ``'*'``, ``N`` and ``append``/``extend`` actions take a JSON list
  (``[1.0, 2.0]``) or comma- or whitespace-separated tokens (``1.0, 2.0``); every
  element is converted and the ``nargs`` count is checked (``append``/``extend``
  accumulate any number of values). ``append`` with ``nargs`` (``'+'``, ``'*'`` or
  ``N``) holds a list of lists such as ``[[1, 2], [3, 4]]``; for ``nargs=N`` a flat
  list is split into groups of ``N``.
* Non-text values (e.g. from a JSON file) are passed to a custom ``type=`` callable
  as they are and, if it fails, as text, so converters written for command-line
  strings (``str2bool``, ``lambda s: int(s, 0)``) survive a save/load round trip.
* ``None``, ``null`` or an empty string give ``None`` when the default is
  ``None`` (or ``nargs='?'``).
* ``choices`` are enforced on the converted value(s).

Examples
--------
>>> import argparse
>>> parser = argparse.ArgumentParser()
>>> _ = parser.add_argument("--lr", type=float, default=0.01)
>>> _ = parser.add_argument("--layers", type=int, nargs="+", default=[64, 64])
>>> _ = parser.add_argument("--verbose", action="store_true")
>>> apply_parameters(parser, {"lr": "0.1", "layers": "32 32 16", "verbose": "yes"}).parse_args([])
Namespace(lr=0.1, layers=[32, 32, 16], verbose=True)
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import secrets
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Union

__all__ = [
    "PARAMETER_FILE_PREFIX",
    "ValidationCallback",
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

logger = logging.getLogger(__name__)

PathLike = Union[str, "os.PathLike[str]"]

ValidationCallback = Callable[[Any], Any]
"""A validator receives the converted value and returns a bool, ``(is_valid, message)``,
``(is_valid, converted_value, message)`` or ``None`` (valid); raising ``ValueError`` or
``TypeError`` marks the value invalid with the exception text as message."""

PARAMETER_FILE_PREFIX = "parameters"
"""File-name prefix of timestamped parameter files (``parameters_YYYYmmdd_HHMMSS.json``)."""

_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})
_NONE_STRINGS = frozenset({"", "none", "null"})
_SKIPPED_ACTIONS = (argparse._HelpAction, argparse._VersionAction, argparse._SubParsersAction)
_BOOL_ACTIONS: tuple[type, ...] = (argparse._StoreTrueAction, argparse._StoreFalseAction)
if hasattr(argparse, "BooleanOptionalAction"):
    _BOOL_ACTIONS += (argparse.BooleanOptionalAction,)
_EXTEND_ACTIONS: tuple[type, ...] = (
    (argparse._ExtendAction,) if hasattr(argparse, "_ExtendAction") else ()
)
_LIST_ACTIONS: tuple[type, ...] = (argparse._AppendAction, *_EXTEND_ACTIONS)
_TIMESTAMPED_NAME = re.compile(
    rf"{re.escape(PARAMETER_FILE_PREFIX)}_(\d{{8}}_\d{{6}})(?:_(\d+))?\.json"
)


# ---------------------------------------------------------------------------
# Action introspection
# ---------------------------------------------------------------------------


def tunable_actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """Map each tunable destination to its argparse action.

    Help, version and sub-parser actions and suppressed destinations are skipped.
    When several options share a destination (``--flag``/``--no-flag``) the first
    one is used.
    """
    if not isinstance(parser, argparse.ArgumentParser):
        raise TypeError(f"expected an argparse.ArgumentParser, got {type(parser).__name__}")
    actions: dict[str, argparse.Action] = {}
    for action in parser._actions:
        if isinstance(action, _SKIPPED_ACTIONS) or action.dest == argparse.SUPPRESS:
            continue
        if action.default is argparse.SUPPRESS:
            continue
        actions.setdefault(action.dest, action)
    return actions


def _is_bool_action(action: argparse.Action) -> bool:
    if isinstance(action, _BOOL_ACTIONS) or action.type is bool:
        return True
    return (
        action.type is None
        and action.nargs is None
        and isinstance(action, argparse._StoreAction)
        and isinstance(action.default, bool)
    )


def _is_list_action(action: argparse.Action) -> bool:
    return (
        isinstance(action, _LIST_ACTIONS)
        or action.nargs in ("*", "+", argparse.REMAINDER)
        or (
            isinstance(action.nargs, int)
            and not isinstance(action.nargs, bool)
            and action.nargs > 0
        )
    )


def _is_nested_action(action: argparse.Action) -> bool:
    """``append`` with ``nargs``: each occurrence adds a list, so the value is a list of lists."""
    return (
        isinstance(action, argparse._AppendAction)
        and not isinstance(action, _EXTEND_ACTIONS)
        and action.nargs not in (None, argparse.OPTIONAL)
    )


def _none_allowed(action: argparse.Action) -> bool:
    return action.default is None or action.nargs == "?"


def _infer_type(sample: Any) -> type | None:
    for candidate in (bool, int, float, str):
        if type(sample) is candidate:
            return candidate
    return None


def _element_type(action: argparse.Action) -> Callable[[Any], Any]:
    """Converter for one scalar element of ``action``."""
    if _is_bool_action(action):
        return bool
    if action.type is not None:
        return action.type
    if isinstance(action, argparse._CountAction):
        return int
    if action.choices:
        types = {_infer_type(c) for c in action.choices}
        if len(types) == 1 and None not in types:
            return types.pop()
    default = action.default
    if isinstance(default, (list, tuple)):
        default = next((d for d in default if d is not None), None)
    if default is None and isinstance(action, argparse._StoreConstAction):
        default = action.const
    return _infer_type(default) or str


def describe_type(action: argparse.Action) -> str:
    """Short human-readable type of an action, e.g. ``'float'`` or ``'list[int]'``."""
    converter = _element_type(action)
    name = getattr(converter, "__name__", type(converter).__name__)
    if name == "<lambda>":
        name = "custom"
    if _is_list_action(action):
        name = f"list[list[{name}]]" if _is_nested_action(action) else f"list[{name}]"
    if _none_allowed(action):
        name = f"{name} or None"
    return name


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def parse_bool(value: Any) -> bool:
    """Convert ``value`` to ``bool``.

    Accepts booleans, the integers 0 and 1, and the strings ``true/false``,
    ``yes/no``, ``on/off``, ``1/0`` (case-insensitive, surrounding blanks ignored).

    Raises
    ------
    ValueError
        For any other input.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _TRUE_STRINGS:
            return True
        if key in _FALSE_STRINGS:
            return False
    raise ValueError(f"invalid boolean value {value!r}; use true/false, yes/no, on/off or 1/0")


def _convert_text(text: str, converter: Callable[[Any], Any], value: Any) -> Any:
    """Apply ``converter`` to ``text``; any conversion failure becomes ``ValueError``."""
    try:
        return converter(text)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(str(exc)) from None
    except (TypeError, ValueError, AttributeError) as exc:
        name = getattr(converter, "__name__", "<lambda>")
        kind = "value" if name == "<lambda>" else f"{name} value"
        raise ValueError(f"invalid {kind} {value!r}: {exc}") from None


def _convert_scalar(value: Any, converter: Callable[[Any], Any]) -> Any:
    if converter is bool:
        return parse_bool(value)
    if isinstance(value, str):
        text = value.strip() if converter in (int, float) else value
        return _convert_text(text, converter, value)
    # Already-typed input, e.g. from a JSON file.
    if converter is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid int value {value!r}")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"invalid int value {value!r}: not an integer")
        return int(value)
    if converter is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid float value {value!r}")
        return float(value)
    if converter is str:
        if isinstance(value, (list, tuple, dict)):
            raise ValueError(f"invalid str value {value!r}")
        return str(value)
    # Custom ``type=`` callables are written for command-line text. Try the value
    # itself, then its text form (booleans as 'true'/'false', the canonical text).
    try:
        return converter(value)
    except (argparse.ArgumentTypeError, TypeError, ValueError, AttributeError):
        pass
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    return _convert_text(text, converter, value)


def _split_list(text: str) -> list[Any]:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] in "[(":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"cannot parse list {text!r}: {exc}") from None
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"expected a list, got {text!r}")
        return list(parsed)
    separator = r"\s*,\s*" if "," in stripped else r"\s+"
    return [token for token in re.split(separator, stripped) if token != ""]


def _check_count(action: argparse.Action, items: list[Any]) -> None:
    nargs = action.nargs
    if nargs == "+" and not items:
        raise ValueError("expected at least one value")
    if isinstance(nargs, int) and not isinstance(nargs, bool) and len(items) != nargs:
        raise ValueError(f"expected exactly {nargs} values, got {len(items)}")


def _group_items(action: argparse.Action, items: list[Any]) -> list[list[Any]]:
    """Split the items of an ``append`` + ``nargs`` action into one list per occurrence."""
    if all(isinstance(item, (list, tuple)) for item in items):
        return [list(item) for item in items]
    nargs = action.nargs
    flat = not any(isinstance(item, (list, tuple)) for item in items)
    if flat and isinstance(nargs, int) and not isinstance(nargs, bool) and nargs > 0:
        if len(items) % nargs == 0:
            return [items[i : i + nargs] for i in range(0, len(items), nargs)]
        raise ValueError(f"expected groups of {nargs} values, got {len(items)} values")
    raise ValueError(f"expected a list of value lists such as [[1, 2], [3, 4]], got {items!r}")


def _check_choices(action: argparse.Action, items: list[Any]) -> None:
    if not action.choices:
        return
    for item in items:
        if item not in action.choices:
            options = ", ".join(repr(c) for c in action.choices)
            raise ValueError(f"{item!r} is not a valid choice; value must be one of: {options}")


def _run_validator(validator: ValidationCallback, value: Any) -> None:
    try:
        result = validator(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc) or "validation failed") from None
    if result is None:
        return
    if isinstance(result, tuple):
        ok = bool(result[0]) if result else False
        message = str(result[-1]) if len(result) > 1 else ""
    else:
        ok, message = bool(result), ""
    if not ok:
        raise ValueError(message or f"validation failed for {value!r}")


def parse_value(
    action: argparse.Action, text: Any, validator: ValidationCallback | None = None
) -> Any:
    """Convert entry text (or an already-typed value) for ``action``.

    Parameters
    ----------
    action : argparse.Action
        The action the value belongs to (see :func:`tunable_actions`).
    text : str or any
        Text as typed by a user, or a value loaded from JSON (numbers, lists,
        booleans and ``None`` are accepted as they are when compatible).
    validator : callable, optional
        Custom check applied to the converted value (see :data:`ValidationCallback`).
        It is not called when the result is ``None``. A converted value returned by
        the validator is ignored: the result always follows the parser declaration.

    Returns
    -------
    Any
        The converted value (``list`` for multi-value actions).

    Raises
    ------
    ValueError
        If the input cannot be converted, violates ``nargs`` or ``choices``, or
        the validator rejects it.
    """
    is_list = _is_list_action(action)
    if text is None or (isinstance(text, str) and text.strip().lower() in _NONE_STRINGS):
        if _none_allowed(action):
            return None
        if text is None or (
            not is_list
            and text.strip().lower() in ("none", "null")
            and _element_type(action) is not str
        ):
            raise ValueError("None is not allowed for this parameter (its default is not None)")

    converter = _element_type(action)
    if is_list:
        if isinstance(text, str):
            items = _split_list(text)
        elif isinstance(text, (list, tuple)):
            items = list(text)
        else:
            raise ValueError(f"expected a list of values, got {text!r}")
        if _is_nested_action(action):
            value: Any = []
            for group in _group_items(action, items):
                _check_count(action, group)
                value.append([_convert_scalar(item, converter) for item in group])
            flat = [item for group in value for item in group]
        else:
            if not isinstance(action, _LIST_ACTIONS):  # append/extend accumulate values
                _check_count(action, items)
            value = flat = [_convert_scalar(item, converter) for item in items]
    else:
        if isinstance(text, (list, tuple)):
            raise ValueError(f"expected a single value, got {text!r}")
        value = _convert_scalar(text, converter)
        flat = [value]
    _check_choices(action, flat)
    if validator is not None:
        _run_validator(validator, value)
    return value


def format_value(action: argparse.Action, value: Any) -> str:
    """Text representation of ``value`` that :func:`parse_value` converts back.

    Lists (including lists of lists) are written as JSON, with elements that are
    not JSON types (e.g. :class:`pathlib.Path`) written as strings; ``None`` is
    written as ``'None'`` and booleans as ``'True'``/``'False'``.
    """
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        try:
            return json.dumps(list(value), default=str)
        except (TypeError, ValueError):  # e.g. circular references
            return json.dumps([str(v) for v in value])
    return str(value)


# ---------------------------------------------------------------------------
# Reading, writing and applying parameter sets
# ---------------------------------------------------------------------------


def get_parameters(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Current default values of all tunable parameters, keyed by destination.

    Lists are copied, so mutating the result does not change the parser.
    """
    return {
        dest: list(action.default) if isinstance(action.default, list) else action.default
        for dest, action in tunable_actions(parser).items()
    }


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):  # numpy arrays and scalars
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    return str(obj)


def _timestamped_path(directory: Path, prefix: str = PARAMETER_FILE_PREFIX) -> Path:
    stem = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    candidate = directory / f"{stem}.json"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{counter}.json"
        counter += 1
    return candidate


def _create_temp_file(target: Path) -> tuple[int, str]:
    """Create a unique temporary file next to ``target`` and return ``(fd, name)``.

    Unlike ``tempfile.mkstemp`` (mode 0600), the file is created with mode 0666
    filtered by the process umask, so the renamed file has the permissions of a file
    written with :func:`open`. The umask itself is not read or changed.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        name = target.parent / f".{target.name}.{secrets.token_hex(6)}.tmp"
        try:
            return os.open(name, flags, 0o666), str(name)
        except FileExistsError:
            continue
    raise FileExistsError(f"could not create a temporary file next to {target}")


def save_parameters(values: Mapping[str, Any], path_or_dir: PathLike) -> Path:
    """Write ``values`` as JSON and return the file path.

    Parameters
    ----------
    values : mapping
        Parameter values (JSON-serialisable; other objects are stored via ``str``).
    path_or_dir : str or path-like
        A file path (any path with a suffix, e.g. ``run.json``) or a directory
        (an existing directory or a path without suffix). For a directory a new
        file ``parameters_YYYYmmdd_HHMMSS.json`` is created (``_1``, ``_2``, ...
        is appended if that name exists). Missing directories are created.

    Returns
    -------
    pathlib.Path
        The written file. The write is atomic (temporary file + rename); the file
        gets the default permissions of new files (``0o666`` minus the umask).
    """
    target = Path(path_or_dir).expanduser()
    if target.is_dir() or not target.suffix:
        target.mkdir(parents=True, exist_ok=True)
        target = _timestamped_path(target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(values), indent=4, ensure_ascii=False, default=_json_default)
    fd, tmp_name = _create_temp_file(target)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    logger.debug("Saved %d parameters to %s", len(values), target)
    return target


def _file_rank(path: Path) -> tuple[bool, str, int, str]:
    """Sort key of parameter files: (timestamped, timestamp, counter, name)."""
    match = _TIMESTAMPED_NAME.fullmatch(path.name)
    if match is None:
        return (False, "", 0, path.name)
    return (True, match.group(1), int(match.group(2) or 0), path.name)


def load_parameters(path: PathLike) -> dict[str, Any]:
    """Read a JSON parameter file written by :func:`save_parameters`.

    Parameters
    ----------
    path : str or path-like
        A JSON file, or a directory, in which case the newest
        ``parameters_*.json`` file is read: files are ranked by the timestamp and
        counter in their name (``parameters_YYYYmmdd_HHMMSS_N.json``, so ``_10``
        is newer than ``_9``); other ``parameters_*.json`` names rank below them.

    Raises
    ------
    FileNotFoundError
        If the file, or any parameter file in the directory, does not exist.
    ValueError
        If the file does not contain a JSON object.
    """
    source = Path(path).expanduser()
    if source.is_dir():
        candidates = sorted(source.glob(f"{PARAMETER_FILE_PREFIX}_*.json"), key=_file_rank)
        if not candidates:
            raise FileNotFoundError(f"no {PARAMETER_FILE_PREFIX}_*.json files in {source}")
        source = candidates[-1]
    with open(source, encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{source} must contain a JSON object, got {type(data).__name__}")
    return data


def apply_parameters(
    parser: argparse.ArgumentParser,
    values: Mapping[str, Any],
    strict: bool = True,
    validation_callbacks: Mapping[str, ValidationCallback] | None = None,
) -> argparse.ArgumentParser:
    """Validate ``values`` and install them as the parser's defaults.

    Every value goes through :func:`parse_value` (type conversion, ``nargs``,
    ``choices`` and the optional validation callback) before
    ``parser.set_defaults(**converted)`` is called, so later ``parse_args`` calls
    see the new defaults while command-line flags still override them.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to update (modified in place and returned).
    values : mapping
        Parameter values keyed by destination (``dest``) name.
    strict : bool, default True
        Raise on keys that are not parser destinations; otherwise ignore them.
    validation_callbacks : mapping, optional
        Per-destination validators (see :data:`ValidationCallback`).

    Returns
    -------
    argparse.ArgumentParser
        ``parser`` itself.

    Raises
    ------
    ValueError
        Listing every unknown key (strict mode) and every invalid value. The
        parser is left unchanged in that case.
    """
    actions = tunable_actions(parser)
    callbacks = dict(validation_callbacks or {})
    unknown = [key for key in values if key not in actions]
    if unknown and strict:
        raise ValueError(
            f"Unknown parameter(s): {', '.join(sorted(map(str, unknown)))}; "
            f"valid parameters are: {', '.join(actions)}"
        )
    if unknown:
        logger.debug("Ignoring unknown parameters: %s", ", ".join(map(str, unknown)))

    converted: dict[str, Any] = {}
    errors: list[str] = []
    for key, raw in values.items():
        if key not in actions:
            continue
        try:
            converted[key] = parse_value(actions[key], raw, callbacks.get(key))
        except ValueError as exc:
            errors.append(f"{key}: {exc}")
    if errors:
        raise ValueError("Invalid parameter values:\n" + "\n".join(errors))
    parser.set_defaults(**converted)
    return parser
