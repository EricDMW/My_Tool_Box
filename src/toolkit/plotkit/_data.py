"""Input normalisation for plotkit.

Every plot function accepts NumPy arrays, Python lists and tuples, pandas objects,
PyTorch tensors and TensorFlow / JAX arrays. The helpers in this module turn those
inputs into plain :class:`numpy.ndarray` objects and into lists of series following
one set of rules:

* a 1-D array-like of numbers (including a flat list of scalars) is **one** series;
* a list or tuple of array-likes is **several** series (one per element);
* a 2-D array is interpreted by the caller: either several series stacked as rows
  (``two_d="series"``) or one series whose rows are samples (``two_d="samples"``);
* a pandas ``DataFrame`` contributes one series per column, except with
  ``two_d="samples"``, where it is one 2-D sample matrix like a 2-D array;
* dates, times and durations (``datetime.date``, ``datetime.datetime``,
  ``datetime.time``, ``datetime.timedelta``, their pandas subclasses and NumPy
  ``datetime64`` / ``timedelta64`` values) are scalars, so a list of them is one series.
  As x values, dates are drawn on a matplotlib date axis and durations as seconds
  (float), since matplotlib has no duration axis.
"""

from __future__ import annotations

import datetime
import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "as_float_array",
    "as_series_list",
    "broadcast_x",
    "describe_shape",
    "is_scalar_like",
    "to_numpy",
]


def to_numpy(value: Any) -> np.ndarray:
    """Convert an array-like object to a :class:`numpy.ndarray`.

    Parameters
    ----------
    value : array-like
        NumPy array, list, tuple, scalar, pandas ``Series``/``DataFrame``/``Index``,
        PyTorch tensor (any device, with or without ``requires_grad``), TensorFlow
        eager tensor or JAX array.

    Returns
    -------
    numpy.ndarray
        The converted array. NumPy arrays are returned without copying.

    Examples
    --------
    >>> to_numpy([1, 2, 3]).shape
    (3,)
    """
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):  # PyTorch tensor
        tensor = detach()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        try:
            return tensor.numpy()
        except TypeError:  # dtypes without a NumPy equivalent, e.g. bfloat16
            return tensor.float().numpy()
    to_np = getattr(value, "to_numpy", None)
    if callable(to_np):  # pandas / xarray / polars
        return np.asarray(to_np())
    as_np = getattr(value, "numpy", None)
    if callable(as_np):  # TensorFlow eager tensors
        return np.asarray(as_np())
    return np.asarray(value)  # lists, scalars, JAX arrays (via __array__)


def as_float_array(value: Any, *, name: str = "data") -> np.ndarray:
    """Convert ``value`` to a float64 array; masked entries become NaN.

    Raises
    ------
    TypeError
        If the values cannot be interpreted as numbers.
    """
    if isinstance(value, np.ma.MaskedArray):
        return value.astype(float).filled(np.nan)
    arr = to_numpy(value)
    try:
        return np.asarray(arr, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numbers; got dtype {arr.dtype}") from exc


_SCALAR_TYPES = (
    str,
    bytes,
    numbers.Number,
    np.generic,  # includes numpy.datetime64 and numpy.timedelta64
    datetime.date,  # includes datetime.datetime and pandas.Timestamp
    datetime.time,
    datetime.timedelta,  # includes pandas.Timedelta
)


def is_scalar_like(value: Any) -> bool:
    """Return True for numbers, strings, dates, times, durations and 0-d arrays or tensors."""
    if isinstance(value, _SCALAR_TYPES):
        return True
    ndim = getattr(value, "ndim", None)
    if ndim is not None:
        try:
            return int(ndim) == 0
        except (TypeError, ValueError):
            return False
    return False


def describe_shape(value: Any) -> str:
    """Short description of an input's shape for error messages."""
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} of {len(value)} elements"
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"shape {tuple(shape)}"
    return type(value).__name__


def _is_dataframe(value: Any) -> bool:
    return hasattr(value, "columns") and hasattr(value, "iloc") and getattr(value, "ndim", 0) == 2


def _scalar_item(value: Any) -> Any:
    """Unwrap a scalar-like value (e.g. a 0-d array or tensor) to a scalar.

    Python scalars, dates and durations are returned unchanged (time zones are kept).
    """
    if isinstance(value, _SCALAR_TYPES):
        return value
    arr = to_numpy(value)
    return arr[()] if arr.ndim == 0 else arr


def _durations_to_seconds(arr: np.ndarray) -> np.ndarray:
    """Convert ``timedelta64`` or ``datetime.timedelta`` values to float seconds (NaT -> NaN).

    matplotlib has no converter for durations: ``timedelta`` objects raise and
    ``timedelta64`` values are drawn as raw integers in the array's unit.
    """
    if (
        arr.dtype == object
        and arr.size
        and all(isinstance(v, datetime.timedelta) for v in arr.flat)
    ):
        arr = arr.astype("timedelta64[us]")
    if arr.dtype.kind == "m":
        return arr / np.timedelta64(1, "s")
    return arr


def _convert(value: Any, numeric: bool, name: str) -> np.ndarray:
    return as_float_array(value, name=name) if numeric else to_numpy(value)


def as_series_list(
    value: Any,
    *,
    name: str = "y",
    two_d: str = "series",
    numeric: bool = True,
    max_element_ndim: int = 1,
) -> list[np.ndarray]:
    """Normalise ``value`` into a list of arrays, one per series.

    Parameters
    ----------
    value : array-like or sequence of array-likes
        Input following the rules in the module docstring.
    name : str, default "y"
        Argument name used in error messages.
    two_d : {"series", "samples"}, default "series"
        How a single 2-D array is read. ``"series"`` splits it into rows (one series
        per row); ``"samples"`` keeps it as one series whose samples are stacked.
        With ``"series"`` a ``DataFrame`` gives one series per column; with
        ``"samples"`` it is converted to one 2-D array (rows and columns kept).
    numeric : bool, default True
        Convert to float64 (masked entries become NaN). When False the dtype is kept,
        which allows dates or category labels.
    max_element_ndim : int, default 1
        Maximum dimensionality of each element of a list/tuple input.

    Returns
    -------
    list of numpy.ndarray
        At least one array. A 0-d input becomes a length-1 series.

    Raises
    ------
    ValueError
        If ``value`` is None or empty, mixes scalars and array-likes, or has too many
        dimensions.
    """
    if value is None:
        raise ValueError(f"{name} must not be None")
    if two_d not in ("series", "samples"):
        raise ValueError(f"two_d must be 'series' or 'samples', got {two_d!r}")

    if _is_dataframe(value):
        if two_d == "samples":
            return [_convert(value, numeric, name)]
        return [_convert(value.iloc[:, j], numeric, name) for j in range(value.shape[1])]

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(f"{name} is empty")
        if all(isinstance(v, (numbers.Number, np.generic)) for v in value):
            return [_convert(np.asarray(value), numeric, name)]
        scalar_flags = [is_scalar_like(v) for v in value]
        if all(scalar_flags):
            return [_convert(np.asarray([_scalar_item(v) for v in value]), numeric, name)]
        if any(scalar_flags):
            raise ValueError(
                f"{name} mixes scalars and array-likes; pass a flat sequence of numbers for "
                "one series or a sequence of array-likes for several series"
            )
        series = []
        for i, element in enumerate(value):
            if element is None:
                raise ValueError(f"{name}[{i}] is None")
            arr = _convert(element, numeric, name)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if arr.ndim > max_element_ndim:
                raise ValueError(
                    f"{name}[{i}] has shape {arr.shape}; expected at most "
                    f"{max_element_ndim} dimension(s) per element"
                )
            series.append(arr)
        return series

    arr = _convert(value, numeric, name)
    if arr.ndim == 0:
        return [arr.reshape(1)]
    if arr.ndim == 1:
        return [arr]
    if arr.ndim == 2:
        return [arr] if two_d == "samples" else list(arr)
    raise ValueError(
        f"{name} must be 1-D or 2-D, got shape {arr.shape}; pass a list of arrays "
        "for several series"
    )


def broadcast_x(
    x: Any,
    ys: Sequence[np.ndarray],
    *,
    lengths: Sequence[int] | None = None,
    func: str = "plot",
    y_input: Any = None,
    hint: str | None = None,
) -> list[np.ndarray]:
    """Return one x array per series, broadcasting a single x to all series.

    Parameters
    ----------
    x : array-like, sequence of array-likes or None
        None gives ``arange(n)`` per series. A single 1-D array is shared by all
        series; a list (or 2-D array) provides one x per series. Durations
        (``timedelta``) are converted to float seconds.
    ys : sequence of numpy.ndarray
        The y series (only their number and lengths are used).
    lengths : sequence of int, optional
        Number of points per series; defaults to ``len(y)`` for each series.
    func : str
        Name of the calling function, used in error messages.
    y_input : object, optional
        The original y argument, used to describe its shape in error messages.
    hint : str, optional
        Appended to a length-mismatch message when ``y_input`` is 2-D and the x length
        matches its other dimension (a likely transposition).

    Raises
    ------
    ValueError
        If the number of x arrays does not match the number of series, or an x array
        has a different length than its series.
    """
    n_series = len(ys)
    if lengths is None:
        lengths = [len(y) for y in ys]
    if x is None:
        return [np.arange(n) for n in lengths]
    xs = [_durations_to_seconds(xi) for xi in as_series_list(x, name="x", numeric=False)]
    if len(xs) == 1 and n_series > 1:
        xs = xs * n_series
    elif len(xs) != n_series:
        raise ValueError(
            f"{func}: got {len(xs)} x arrays for {n_series} series; pass a single x shared "
            "by all series or one x per series"
        )
    for i, (xi, n) in enumerate(zip(xs, lengths)):
        if len(xi) != n:
            msg = (
                f"{func}: series {i} has {n} points but its x has {len(xi)} (x {describe_shape(x)}"
            )
            if y_input is not None:
                msg += f", y {describe_shape(y_input)}"
            msg += ")"
            shape = getattr(y_input, "shape", None)
            if hint and shape is not None and len(shape) == 2 and len(xi) in tuple(shape):
                msg += f"; {hint}"
            raise ValueError(msg)
    return xs
