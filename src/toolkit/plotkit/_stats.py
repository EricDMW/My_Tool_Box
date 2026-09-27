"""NaN-robust summary statistics and smoothing used by the curve plots."""

from __future__ import annotations

import numbers
import warnings
from typing import Union

import numpy as np

__all__ = ["BANDS", "check_band", "check_smoothing", "ema", "moving_average", "smooth", "summarize"]

#: Band types understood by :func:`summarize`.
BANDS = ("std", "sem", "ci95", "minmax")

#: Two-sided 97.5 % quantile of the standard normal distribution.
_Z95 = 1.959963984540054

Smoothing = Union[None, int, float]


def check_band(band: str | None) -> str | None:
    """Validate a band name and return it in canonical (lower-case) form.

    Raises
    ------
    ValueError
        If ``band`` is not None or one of :data:`BANDS`.
    """
    if band is None:
        return None
    if isinstance(band, str):
        key = band.lower()
        if key == "none":
            return None
        if key in BANDS:
            return key
    raise ValueError(f"band must be None or one of {BANDS}, got {band!r}")


def check_smoothing(smoothing: Smoothing) -> Smoothing:
    """Validate a smoothing specification.

    Returns
    -------
    None, int or float
        None (no smoothing), an int moving-average window >= 1 or a float EMA weight
        in (0, 1).

    Raises
    ------
    TypeError
        If ``smoothing`` has an unsupported type.
    ValueError
        If the window is smaller than 1 or the weight is outside (0, 1).
    """
    if smoothing is None:
        return None
    if isinstance(smoothing, (bool, np.bool_)):
        raise TypeError("smoothing must be None, an int window or a float in (0, 1), not a bool")
    if isinstance(smoothing, numbers.Integral):
        if smoothing < 1:
            raise ValueError(f"smoothing window must be >= 1, got {smoothing}")
        return int(smoothing)
    if isinstance(smoothing, numbers.Real):
        if not 0.0 < float(smoothing) < 1.0:
            raise ValueError(
                f"float smoothing is an EMA weight and must lie in (0, 1), got {smoothing}; "
                "use an int for a moving-average window"
            )
        return float(smoothing)
    raise TypeError(
        f"smoothing must be None, an int window or a float in (0, 1), got {type(smoothing).__name__}"
    )


def summarize(
    samples: np.ndarray, axis: int = 0, band: str | None = "std"
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Mean curve and band edges of a 2-D sample matrix, ignoring NaN.

    Parameters
    ----------
    samples : numpy.ndarray
        2-D array; ``axis`` indexes the samples (e.g. seeds), the other axis the steps.
    axis : int, default 0
        Sample axis.
    band : {"std", "sem", "ci95", "minmax"} or None
        ``"std"``: mean +/- population standard deviation (``ddof=0``);
        ``"sem"``: mean +/- standard error, ``std(ddof=1) / sqrt(n)``;
        ``"ci95"``: mean +/- 1.96 standard errors (normal approximation);
        ``"minmax"``: the per-step minimum and maximum; None: no band.
        ``n`` is the number of non-NaN samples at each step.

    Returns
    -------
    mean, lower, upper : numpy.ndarray
        ``lower`` and ``upper`` are None when ``band`` is None. Steps without any valid
        sample are NaN; SEM-based bands are NaN where fewer than two samples exist.
    """
    band = check_band(band)
    a = np.asarray(samples, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN steps, n < 2 for ddof=1
        mean = np.nanmean(a, axis=axis)
        if band is None:
            return mean, None, None
        if band == "minmax":
            return mean, np.nanmin(a, axis=axis), np.nanmax(a, axis=axis)
        if band == "std":
            half = np.nanstd(a, axis=axis)
        else:
            count = np.sum(~np.isnan(a), axis=axis)
            half = np.nanstd(a, axis=axis, ddof=1) / np.sqrt(count)
            if band == "ci95":
                half = _Z95 * half
    return mean, mean - half, mean + half


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average along the last axis with the same output length.

    Each output ``out[t]`` is the mean of the finite values among
    ``values[max(0, t - window + 1) : t + 1]`` (the window shrinks at the start), i.e.
    ``pandas.Series(values).rolling(window, min_periods=1).mean()``. Positions whose
    window holds no finite value are NaN.
    """
    v = np.asarray(values, dtype=float)
    if window <= 1:
        return v.copy()
    valid = np.isfinite(v)
    csum = np.cumsum(np.where(valid, v, 0.0), axis=-1)
    ccount = np.cumsum(valid, axis=-1)
    csum[..., window:] = csum[..., window:] - csum[..., :-window]
    ccount[..., window:] = ccount[..., window:] - ccount[..., :-window]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = csum / ccount
    out[ccount == 0] = np.nan
    return out


def _ema_filter_loop(x: np.ndarray, weight: float) -> np.ndarray:
    """Reference implementation of ``y[t] = weight * y[t-1] + (1 - weight) * x[t]``."""
    out = np.empty_like(x)
    acc = np.zeros(x.shape[:-1])
    for t in range(x.shape[-1]):
        acc = weight * acc + (1.0 - weight) * x[..., t]
        out[..., t] = acc
    return out


def _ema_filter(x: np.ndarray, weight: float) -> np.ndarray:
    try:
        from scipy.signal import lfilter
    except ImportError:  # pragma: no cover - scipy is a core dependency of my-tool-box
        return _ema_filter_loop(x, weight)
    return lfilter([1.0 - weight], [1.0, -weight], x, axis=-1)


def ema(values: np.ndarray, weight: float) -> np.ndarray:
    """Debiased exponential moving average along the last axis (TensorBoard style).

    ``weight`` is the smoothing weight: 0 means no smoothing and values close to 1 are
    very smooth. With ``alpha = 1 - weight`` the result equals
    ``pandas.Series(values).ewm(alpha=alpha).mean()``: a weighted mean of all past
    finite values with weights ``weight ** age``. Non-finite values are skipped (the
    previous estimate is carried forward); leading positions without any finite value
    are NaN.
    """
    v = np.asarray(values, dtype=float)
    valid = np.isfinite(v)
    num = _ema_filter(np.where(valid, v, 0.0), weight)
    den = _ema_filter(valid.astype(float), weight)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den <= 0] = np.nan
    return out


def smooth(values: np.ndarray, smoothing: Smoothing) -> np.ndarray:
    """Apply ``smoothing`` (see :func:`check_smoothing`) along the last axis."""
    smoothing = check_smoothing(smoothing)
    if smoothing is None:
        return np.asarray(values, dtype=float)
    if isinstance(smoothing, int):
        return moving_average(values, smoothing)
    return ema(values, smoothing)
