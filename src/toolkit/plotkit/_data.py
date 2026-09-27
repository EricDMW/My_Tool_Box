"""Input normalisation for plotkit.

Every plot function accepts NumPy arrays, Python lists and tuples, pandas objects,
PyTorch tensors and TensorFlow / JAX arrays. The helpers in this module turn those
inputs into plain :class:`numpy.ndarray` objects and into lists of series following
one set of rules:

* a 1-D array-like of numbers (including a flat list of scalars) is **one** series;
* a list or tuple of array-likes is **several** series (one per element);
* a 2-D array is interpreted by the caller: either several series stacked as rows
  (``two_d="series"``) or one series whose rows are samples (``two_d="samples"``);
* a pandas ``DataFrame`` contributes one series per column.
"""

from __future__ import annotations

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


def is_scalar_like(value: Any) -> bool:
    """Return True for numbers, strings and 0-d arrays or tensors."""
    if isinstance(value, (str, bytes, numbers.Number, np.generic)):
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
        return [_convert(value.iloc[:, j], numeric, name) for j in range(value.shape[1])]

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(f"{name} is empty")
        if all(isinstance(v, (numbers.Number, np.generic)) for v in value):
            return [_convert(np.asarray(value), numeric, name)]
        scalar_flags = [is_scalar_like(v) for v in value]
        if all(scalar_flags):
            return [_convert(np.asarray([to_numpy(v) for v in value]), numeric, name)]
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
        series; a list (or 2-D array) provides one x per series.
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
    xs = as_series_list(x, name="x", numeric=False)
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
