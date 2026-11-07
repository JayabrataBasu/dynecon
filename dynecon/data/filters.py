"""Macro detrending filters: Hodrick-Prescott, Baxter-King and Hamilton.

Economics note:
Business-cycle analysis separates a slow-moving trend from cyclical fluctuations. The
filters differ in what they call "cycle": HP is a smoothing spline, Baxter-King a band-pass
for 6-32 quarter fluctuations, and Hamilton the unpredictable part of a regression forecast.
Filters distort the ends of the sample (HP) or discard observations (BK, Hamilton).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from dynecon.core.errors import ValidationError
from dynecon.utils.validation import check_finite, check_int

__all__ = ["baxter_king", "hamilton_filter", "hp_filter"]

Array = npt.NDArray[np.float64]


def _as_array(y: Any) -> tuple[Array, Any]:
    index = getattr(y, "index", None)
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise ValidationError("filters need a 1-D series")
    if not np.all(np.isfinite(arr)):
        raise ValidationError("series contains non-finite values")
    return arr, index


def _wrap(values: Array, index: Any) -> Any:
    if index is None:
        return values
    import pandas as pd

    return pd.Series(values, index=index)


def hp_filter(y: Any, lamb: float = 1600.0) -> tuple[Any, Any]:
    """Hodrick-Prescott filter: ``(trend, cycle)`` minimising squared cycle plus ``lamb`` times
    squared second differences of the trend.

    Conventional ``lamb`` is 1600 (quarterly), 100 (annual), 14400 (monthly); 1600 is the
    default. The trend at the ends of the sample is less well determined.
    """
    arr, index = _as_array(y)
    n = arr.shape[0]
    lamb = check_finite(lamb, "lamb", positive=True)
    if n < 4:
        raise ValidationError("need at least 4 observations")
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    K = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    A = sparse.identity(n, format="csc") + lamb * (K.T @ K)
    trend = np.asarray(spsolve(A.tocsc(), arr), dtype=np.float64)
    if not np.all(np.isfinite(trend)):
        raise ValidationError("lamb or the scale of y is too large for a stable solve")
    return _wrap(trend, index), _wrap(arr - trend, index)


def baxter_king(y: Any, low: int = 6, high: int = 32, K: int = 12) -> Any:
    """Baxter-King band-pass filter keeping cycles between ``low`` and ``high`` periods.

    Weights are the ideal band-pass weights truncated at lag ``K`` and shifted to sum to zero
    (removing constants and linear trends). The first and last ``K`` observations are lost.
    """
    arr, index = _as_array(y)
    n = arr.shape[0]
    low = check_int(low, "low", 2)
    high = check_int(high, "high", low + 1)
    K = check_int(K, "K", 1)
    if n <= 2 * K:
        raise ValidationError("need len(y) > 2K")
    w_lo, w_hi = 2 * np.pi / high, 2 * np.pi / low
    j = np.arange(1, K + 1)
    b = np.concatenate(
        [[(w_hi - w_lo) / np.pi], (np.sin(j * w_hi) - np.sin(j * w_lo)) / (np.pi * j)]
    )
    theta = (b[0] + 2 * b[1:].sum()) / (2 * K + 1)
    weights = np.concatenate([b[:0:-1], b]) - theta
    out = np.convolve(arr, weights, mode="valid")
    return _wrap(out, None if index is None else index[K:-K])


def hamilton_filter(y: Any, h: int = 8, p: int = 4) -> tuple[Any, Any]:
    """Hamilton (2018) regression filter: ``(trend, cycle)`` for ``t = p - 1 + h, ..., T - 1``.

    Regresses ``y_{t+h}`` on a constant and ``y_t, ..., y_{t-p+1}``; the fitted value is the
    trend and the residual the cycle. The first ``h + p - 1`` observations are lost.
    """
    arr, index = _as_array(y)
    n = arr.shape[0]
    h, p = check_int(h, "h", 1), check_int(p, "p", 1)
    if n <= h + p + 1:
        raise ValidationError("not enough observations for this h and p")
    rows = n - h - p + 1
    X = np.column_stack([np.ones(rows)] + [arr[p - 1 - k : p - 1 - k + rows] for k in range(p)])
    target = arr[p - 1 + h :]
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    fitted = X @ beta
    if not (np.all(np.isfinite(fitted)) and np.all(np.isfinite(target - fitted))):
        raise ValidationError("the regression overflowed: rescale the series")
    idx = None if index is None else index[p - 1 + h :]
    return _wrap(fitted, idx), _wrap(target - fitted, idx)
