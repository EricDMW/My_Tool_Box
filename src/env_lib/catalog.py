"""Discover the environments of ``env_lib``.

:func:`catalog` lists every registered environment with its family, space
types and shapes, number of agents, optional dependency and native vector
support; :func:`describe` prints everything about one environment::

    import env_lib

    print(env_lib.catalog(action_type="continuous"))
    print(env_lib.describe("Formation-v0"))

The same information is available on the command line (``env-lib list``,
``env-lib describe ID``).

The module itself is callable: ``env_lib.catalog`` may resolve to this module
(once it has been imported) or to the :func:`catalog` function, and
``env_lib.catalog(...)`` works in both cases.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect as _inspect
import logging
import math
import sys
import textwrap
import types
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from gymnasium import spaces

from env_lib.registration import ENV_SPECS, EnvSpec, get_spec

__all__ = ["Catalog", "EnvInfo", "catalog", "clear_cache", "describe"]

logger = logging.getLogger(__name__)

#: Import names needed by every optional-dependency extra.
_EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    "torch": ("torch",),
    "pistonball": ("pygame", "pymunk"),
    "video": ("imageio",),
}

_TYPES = ("continuous", "discrete")


@dataclass(frozen=True)
class EnvInfo:
    """Catalog entry of one registered environment.

    Attributes
    ----------
    id:
        Registered id (use with :func:`env_lib.make`).
    family:
        Environment family (the ``env_lib`` subpackage without ``_env``).
    description:
        One-line summary.
    observation_type, action_type:
        ``"continuous"``, ``"discrete"`` or ``"continuous|discrete"``.
    requires:
        Optional-dependency extra (``pip install "my-tool-box[<extra>]"``) or ``None``.
    native_vector:
        Whether :func:`env_lib.make_vec` uses a native batched implementation.
    available:
        Whether the optional dependencies and the environment module can be
        imported (``False`` for missing extras or environments still in
        development).
    n_agents:
        Number of agents (leading axis of the joint observation; 1 for a flat
        observation). ``None`` when not inspected.
    observation_shape, action_shape:
        Shapes of the joint spaces (``()`` for a ``Discrete`` action), or
        ``None`` when not inspected.
    kwargs:
        Constructor arguments of this registered configuration (read-only).
    entry_point, vector_entry_point:
        ``"module:Class"`` of the single and native vector implementation.
    observation_space, action_space:
        ``repr`` of the spaces, or ``None`` when not inspected.
    note:
        Why the environment is unavailable or could not be inspected.
    """

    id: str
    family: str
    description: str
    observation_type: str
    action_type: str
    requires: str | None
    native_vector: bool
    available: bool
    n_agents: int | None = None
    observation_shape: tuple[int, ...] | None = None
    action_shape: tuple[int, ...] | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict, hash=False, compare=False)
    entry_point: str = ""
    vector_entry_point: str | None = None
    observation_space: str | None = None
    action_space: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", MappingProxyType(dict(self.kwargs)))

    def matches(self, kind: str, *, action: bool = True) -> bool:
        """Whether the action (or observation) type includes ``kind``."""
        declared = self.action_type if action else self.observation_type
        return kind in declared.split("|")


class Catalog(list):
    """List of :class:`EnvInfo` with a readable text table.

    ``str(catalog)`` gives an aligned plain-text table, :meth:`to_markdown` a
    Markdown table and :meth:`ids` the list of ids.
    """

    _COLUMNS = ("id", "family", "agents", "obs shape", "action", "vector", "requires", "summary")

    def ids(self) -> list[str]:
        """The environment ids, in catalog order."""
        return [info.id for info in self]

    def get(self, env_id: str) -> EnvInfo:
        """Entry with the given id.

        Raises
        ------
        KeyError
            If the id is not in the catalog.
        """
        for info in self:
            if info.id == env_id:
                return info
        raise KeyError(f"{env_id!r} is not in this catalog")

    def _table(self) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
        """Header and rows; the shape columns are left out when nothing was inspected."""
        inspected = any(info.n_agents is not None for info in self)
        columns = [c for c in self._COLUMNS if inspected or c not in ("agents", "obs shape")]
        rows = []
        for info in self:
            action = info.action_type.replace("continuous", "cont").replace("discrete", "disc")
            if info.action_space is not None and info.action_space.startswith("Discrete("):
                action = f"{action} {info.action_space.split(':')[0]}"
            elif info.action_shape is not None:
                action = f"{action} {_shape(info.action_shape)}"
            requires = info.requires or "-"
            if info.requires and not info.available:
                requires += " (missing)"
            elif not info.available:
                requires = "unavailable"
            cells = {
                "id": info.id,
                "family": info.family,
                "agents": "-" if info.n_agents is None else str(info.n_agents),
                "obs shape": "-"
                if info.observation_shape is None
                else _shape(info.observation_shape),
                "action": action,
                "vector": "native" if info.native_vector else "sync",
                "requires": requires,
                "summary": info.description,
            }
            rows.append(tuple(cells[c] for c in columns))
        return tuple(columns), rows

    def __str__(self) -> str:
        if not self:
            return "(no matching environments)"
        columns, body = self._table()
        rows = [tuple(c.upper() for c in columns), *body]
        widths = [max(len(row[i]) for row in rows) for i in range(len(columns) - 1)]
        lines = []
        for row in rows:
            cells = [cell.ljust(width) for cell, width in zip(row[:-1], widths)]
            lines.append("  ".join([*cells, row[-1]]).rstrip())
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """The catalog as a GitHub-flavoured Markdown table."""
        columns, body = self._table()
        header = "| " + " | ".join(c.capitalize() for c in columns) + " |"
        rule = "|" + "|".join("---" for _ in columns) + "|"
        lines = []
        for row in body:
            cells = [f"`{row[0]}`", *row[1:]]
            lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
        return "\n".join([header, rule, *lines])

    def __repr__(self) -> str:
        return f"Catalog({self.ids()!r})"


def _shape(shape: tuple[int, ...]) -> str:
    return str(tuple(int(s) for s in shape))


# ---------------------------------------------------------------------------
# Availability and inspection (cached)
# ---------------------------------------------------------------------------
_AVAILABILITY: dict[str, tuple[bool, str | None]] = {}
_INSPECTION: dict[str, dict[str, Any]] = {}


def clear_cache() -> None:
    """Forget cached availability checks and inspection results."""
    _AVAILABILITY.clear()
    _INSPECTION.clear()


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _availability(spec: EnvSpec) -> tuple[bool, str | None]:
    if spec.id in _AVAILABILITY:
        return _AVAILABILITY[spec.id]
    result: tuple[bool, str | None] = (True, None)
    if spec.requires:
        missing = [
            m for m in _EXTRA_MODULES.get(spec.requires, (spec.requires,)) if not _module_exists(m)
        ]
        if missing:
            result = (
                False,
                f"missing optional dependency {', '.join(missing)}: "
                f'pip install "my-tool-box[{spec.requires}]"',
            )
    if result[0]:
        module = spec.entry_point.split(":")[0]
        if not _module_exists(module):
            result = (False, f"module {module} is not available")
    _AVAILABILITY[spec.id] = result
    return result


def _load_class(entry_point: str) -> type:
    module, _, name = entry_point.partition(":")
    return getattr(importlib.import_module(module), name)


def _agents_of(observation_space: spaces.Space) -> int:
    shape = observation_space.shape
    if isinstance(observation_space, spaces.Box) and shape is not None and len(shape) >= 2:
        return int(shape[0])
    return 1


def _inspect_env(spec: EnvSpec) -> dict[str, Any]:
    """Instantiate the environment once and record its spaces (cached)."""
    if spec.id in _INSPECTION:
        return _INSPECTION[spec.id]
    result: dict[str, Any] = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = _load_class(spec.entry_point)(**spec.kwargs)
        try:
            obs_space, act_space = env.observation_space, env.action_space
            result = {
                "n_agents": _agents_of(obs_space),
                "observation_shape": tuple(obs_space.shape or ()),
                "action_shape": tuple(act_space.shape or ()),
                "observation_space": _space_summary(obs_space),
                "action_space": _space_summary(act_space),
            }
        finally:
            env.close()
    except Exception as exc:  # never let one broken environment break the catalog
        logger.debug("could not inspect %s", spec.id, exc_info=True)
        result = {"note": f"could not be created: {type(exc).__name__}: {exc}"}
    _INSPECTION[spec.id] = result
    return result


def _info(spec: EnvSpec, inspect: bool) -> EnvInfo:
    available, note = _availability(spec)
    details: dict[str, Any] = {}
    if inspect and available:
        details = dict(_inspect_env(spec))
        if "note" in details:
            available = False
    return EnvInfo(
        id=spec.id,
        family=spec.family,
        description=spec.description,
        observation_type=spec.observation_type,
        action_type=spec.action_type,
        requires=spec.requires,
        native_vector=spec.vector_entry_point is not None,
        available=available,
        kwargs=dict(spec.kwargs),
        entry_point=spec.entry_point,
        vector_entry_point=spec.vector_entry_point,
        note=details.pop("note", note),
        **details,
    )


def _check_type(name: str, value: str | None) -> None:
    if value is not None and value not in _TYPES:
        raise ValueError(f"{name} must be one of {_TYPES} or None, got {value!r}")


def catalog(
    *,
    family: str | Iterable[str] | None = None,
    action_type: str | None = None,
    observation_type: str | None = None,
    native_vector: bool | None = None,
    available_only: bool = False,
    inspect: bool = True,
) -> Catalog:
    """List the registered environments, optionally filtered.

    Parameters
    ----------
    family:
        Keep only these families (a name or an iterable of names), e.g.
        ``"consensus"`` or ``("kuramoto", "power_grid")``.
    action_type, observation_type:
        ``"continuous"`` or ``"discrete"``. Environments that support both
        (``"continuous|discrete"``, e.g. Pistonball) match either.
    native_vector:
        Keep only environments with (``True``) or without (``False``) a native
        batched implementation.
    available_only:
        Drop environments whose optional dependencies or module are missing.
    inspect:
        Instantiate every environment once (results are cached for the
        session) to report the number of agents and the space shapes.
        Environments that cannot be created are marked unavailable with a
        ``note`` instead of raising.

    Returns
    -------
    Catalog
        A list of :class:`EnvInfo` whose ``str()`` is a text table.

    Raises
    ------
    ValueError
        For an unknown ``action_type`` or ``observation_type``.

    Examples
    --------
    >>> import env_lib
    >>> continuous = env_lib.catalog(action_type="continuous", inspect=False)
    >>> "Pistonball-v0" in continuous.ids()
    True
    >>> print(env_lib.catalog(family="consensus"))            # doctest: +SKIP
    """
    _check_type("action_type", action_type)
    _check_type("observation_type", observation_type)
    families = None
    if family is not None:
        families = {family} if isinstance(family, str) else set(family)
    entries = Catalog()
    for spec in ENV_SPECS:
        if families is not None and spec.family not in families:
            continue
        if action_type is not None and action_type not in spec.action_type.split("|"):
            continue
        if observation_type is not None and observation_type not in spec.observation_type.split(
            "|"
        ):
            continue
        if native_vector is not None and (spec.vector_entry_point is not None) != native_vector:
            continue
        info = _info(spec, inspect)
        if available_only and not info.available:
            continue
        entries.append(info)
    return entries


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------
def _format_values(values: np.ndarray) -> str:
    """Compact text for an array of bounds (run-length encoded when possible)."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return "[]"
    runs: list[tuple[float, int]] = []
    for value in flat:
        if runs and (runs[-1][0] == value or (math.isnan(value) and math.isnan(runs[-1][0]))):
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((float(value), 1))
    if len(runs) == 1:
        return _number(runs[0][0])
    if len(runs) <= 4:
        return (
            "[" + ", ".join(_number(v) if c == 1 else f"{_number(v)} x{c}" for v, c in runs) + "]"
        )
    return f"{_number(float(flat.min()))} .. {_number(float(flat.max()))} (varies)"


def _number(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6g}"


def _box_bounds(values: np.ndarray) -> str:
    array = np.asarray(values)
    if array.ndim >= 2 and np.all(array == array[:1]):
        row = _format_values(array[0])
        return row if not row.startswith("[") or array.shape[-1] == 1 else f"{row} per agent"
    return _format_values(array)


def _space_summary(space: spaces.Space) -> str:
    """One-line description of a space with its bounds."""
    if isinstance(space, spaces.Box):
        low, high = _box_bounds(space.low), _box_bounds(space.high)
        return f"Box{_shape(space.shape)} {np.dtype(space.dtype).name}, low {low}, high {high}"
    if isinstance(space, spaces.Discrete):
        start = int(space.start)
        return f"Discrete({int(space.n)}): integers {start} .. {start + int(space.n) - 1}"
    if isinstance(space, spaces.MultiDiscrete):
        return f"MultiDiscrete{_shape(space.shape)}: values per entry {_format_values(space.nvec)}"
    if isinstance(space, spaces.MultiBinary):
        return f"MultiBinary{_shape(space.shape)}"
    return repr(space)


def _summary_line(cls: type) -> str:
    doc = _inspect.getdoc(cls) or ""
    paragraph = doc.strip().split("\n\n")[0].replace("``", "")
    return " ".join(paragraph.split())


def _parameters(cls: type, registered: Mapping[str, Any]) -> list[str]:
    """``name=value`` for every constructor parameter (registered values marked with ``*``)."""
    try:
        signature = _inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return [f"{k}={v!r}*" for k, v in registered.items()]
    items = []
    seen = set()
    for name, parameter in signature.parameters.items():
        if name in ("self", "render_mode") or parameter.kind in (
            parameter.VAR_POSITIONAL,
            parameter.VAR_KEYWORD,
        ):
            continue
        seen.add(name)
        if name in registered:
            items.append(f"{name}={registered[name]!r}*")
        elif parameter.default is not parameter.empty:
            items.append(f"{name}={parameter.default!r}")
        else:
            items.append(name)
    items.extend(f"{k}={v!r}*" for k, v in registered.items() if k not in seen)
    return items


def _config_fields(env: Any) -> list[str]:
    """Fields of a dataclass ``env.config`` (AJLATT), as ``name=value``."""
    config = getattr(env, "config", None)
    fields = getattr(config, "__dataclass_fields__", None)
    if not fields:
        return []
    out = []
    for name in fields:
        value = getattr(config, name)
        text = repr(value)
        if len(text) > 40:
            text = f"<{type(value).__name__}>"
        out.append(f"{name}={text}")
    return out


def _layout_lines(env: Any) -> list[str]:
    layout = getattr(env, "observation_layout", None)
    if layout is None:
        return []
    try:
        layout = layout() if callable(layout) else layout
    except Exception:  # layout may need a reset; it is optional information
        return []
    lines = []
    width = max(len(name) for name in layout) if layout else 0
    for name, cols in layout.items():
        if isinstance(cols, slice):
            span = f"{cols.start}:{cols.stop}"
        else:
            span = str(cols)
        lines.append(f"{name.ljust(width)}  columns {span}")
    return lines


def _wrap(label: str, text: str, width: int = 88) -> list[str]:
    indent = " " * 14
    wrapped = textwrap.wrap(
        text, width=width, initial_indent=f"{label:<14}", subsequent_indent=indent
    )
    return wrapped or [label]


def describe(env_id: str) -> str:
    """Readable multi-line description of a registered environment.

    Includes the summary from the class docstring, the spaces with their
    bounds, the observation layout (when the environment defines
    ``observation_layout``), the constructor parameters of this configuration
    (values set by the registration are marked with ``*``), native vector
    support, the baseline controller and the optional dependency.

    Parameters
    ----------
    env_id:
        A registered id (see :func:`catalog`).

    Returns
    -------
    str

    Raises
    ------
    KeyError
        For an unknown id.

    Examples
    --------
    >>> import env_lib
    >>> print(env_lib.describe("Consensus-v0"))              # doctest: +SKIP
    """
    spec = get_spec(env_id)
    info = _info(spec, inspect=True)
    from env_lib.baselines import list_baselines

    lines = [f"{spec.id} -- {spec.description}", ""]
    lines += _wrap("family", spec.family)
    lines += _wrap("class", spec.entry_point)
    cls = None
    env = None
    if info.available:
        try:
            cls = _load_class(spec.entry_point)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                env = cls(**spec.kwargs)
        except Exception as exc:  # reported below as unavailable
            logger.debug("could not create %s", env_id, exc_info=True)
            info = EnvInfo(**{**info.__dict__, "available": False, "note": str(exc)})
    if cls is not None:
        lines += _wrap("summary", _summary_line(cls))
    lines += _wrap("agents", "-" if info.n_agents is None else str(info.n_agents))
    lines += _wrap("observation", f"{info.observation_type}; {info.observation_space or 'n/a'}")
    lines += _wrap("action", f"{info.action_type}; {info.action_space or 'n/a'}")
    if env is not None:
        layout = _layout_lines(env)
        if layout:
            lines.append("obs layout    " + layout[0])
            lines += [" " * 14 + line for line in layout[1:]]
    if cls is not None:
        params = _parameters(cls, spec.kwargs)
        config = _config_fields(env) if env is not None else []
        if config:  # configuration dataclass (AJLATT): list its fields instead of `config`
            params = [p for p in params if not p.startswith("config=")] + config
        lines += _wrap("parameters", ", ".join(params))
        if spec.kwargs:
            lines.append(" " * 14 + "(* = set by this id)")
    if spec.vector_entry_point:
        vector = f"native batched implementation {spec.vector_entry_point} (env_lib.make_vec)"
        if not _module_exists(spec.vector_entry_point.split(":")[0]):
            vector += "; module not available in this installation"
    else:
        vector = "no native implementation; env_lib.make_vec uses gymnasium SyncVectorEnv"
    lines += _wrap("vector", vector)
    baseline = list_baselines().get(spec.family)
    lines += _wrap("baseline", f"{baseline} (env_lib.baseline_policy)" if baseline else "none")
    if spec.requires:
        modules = _EXTRA_MODULES.get(spec.requires, (spec.requires,))
        status = "installed" if all(_module_exists(m) for m in modules) else "not installed"
        requires = f'extra "{spec.requires}" (pip install "my-tool-box[{spec.requires}]"), {status}'
    else:
        requires = "no optional dependencies"
    lines += _wrap("requires", requires)
    if not info.available and info.note:
        lines += _wrap("status", f"unavailable: {info.note}")
    if env is not None:
        env.close()
    return "\n".join(lines)


class _CallableModule(types.ModuleType):
    """Module type whose instances forward calls to :func:`catalog`."""

    def __call__(self, *args: Any, **kwargs: Any) -> Catalog:
        return catalog(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
