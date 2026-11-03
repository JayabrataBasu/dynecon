"""Sample moments and standard-error helpers for simulated series and panels.

Economics note:
Simulated moments (means, volatilities, persistence) are what structural estimation
matches to data; their sampling error determines how informative a moment is.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from econkit.core.errors import ValidationError
from econkit.utils.rng import as_generator

__all__ = [
    "autocorrelation",
    "block_bootstrap_se",
    "correlation",
    "newey_west_variance",
    "percentile",
    "regression_slope",
    "sample_mean",
    "sample_std",
]

Array = npt.NDArray[np.float64]


def _arr(x: npt.ArrayLike) -> Array:
    return np.asarray(x, dtype=np.float64)


def sample_mean(x: npt.ArrayLike) -> float:
    """Mean pooled over all periods and agents."""
    return float(np.mean(_arr(x)))


def sample_std(x: npt.ArrayLike) -> float:
    """Standard deviation (population, ``ddof=0``) pooled over periods and agents."""
    return float(np.std(_arr(x)))


def percentile(x: npt.ArrayLike, q: float) -> float:
    """Pooled percentile ``q`` in [0, 100]."""
    return float(np.percentile(_arr(x), q))


def autocorrelation(x: npt.ArrayLike, lag: int = 1) -> float:
    """Lag-``lag`` autocorrelation; for a ``(T, N)`` panel pairs are formed within each agent."""
    a = _arr(x)
    if lag < 1:
        raise ValidationError("lag must be >= 1")
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] <= lag:
        raise ValidationError("series too short for the requested lag")
    lead, lagged = a[lag:].ravel(), a[:-lag].ravel()
    if np.std(lead) == 0.0 or np.std(lagged) == 0.0:
        return 0.0
    return float(np.corrcoef(lead, lagged)[0, 1])


def correlation(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Pooled correlation between two series of identical shape."""
    a, b = _arr(x).ravel(), _arr(y).ravel()
    if a.shape != b.shape:
        raise ValidationError("x and y must have the same shape")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def regression_slope(y: npt.ArrayLike, x: npt.ArrayLike) -> float:
    """OLS slope of ``y`` on ``x`` (with intercept), pooled."""
    a, b = _arr(x).ravel(), _arr(y).ravel()
    if a.shape != b.shape:
        raise ValidationError("x and y must have the same shape")
    vx = np.var(a)
    if vx == 0.0:
        raise ValidationError("x has zero variance")
    return float(np.cov(a, b, bias=True)[0, 1] / vx)


def block_bootstrap_se(
    series: npt.ArrayLike,
    statistic_fn: object,
    block_len: int,
    n_boot: int = 200,
    rng: np.random.Generator | int | None = None,
) -> float:
    """Moving-block bootstrap standard error of ``statistic_fn(series)`` for a 1-D series."""
    x = _arr(series)
    if x.ndim != 1:
        raise ValidationError("series must be 1-D")
    n = x.shape[0]
    if not 1 <= block_len <= n:
        raise ValidationError("block_len must lie in [1, len(series)]")
    if not callable(statistic_fn):
        raise ValidationError("statistic_fn must be callable")
    gen = as_generator(rng)
    n_blocks = -(-n // block_len)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        starts = gen.integers(0, n - block_len + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel()[:n]
        stats[b] = statistic_fn(x[idx])
    return float(np.std(stats, ddof=1))


def newey_west_variance(x: npt.ArrayLike, lags: int | None = None) -> float:
    """Newey-West long-run variance of the sample mean of ``x`` (Bartlett kernel)."""
    a = _arr(x).ravel()
    n = a.shape[0]
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    d = a - a.mean()
    var = float(d @ d) / n
    for ell in range(1, lags + 1):
        var += 2.0 * (1.0 - ell / (lags + 1.0)) * float(d[ell:] @ d[:-ell]) / n
    return var / n
