"""Grid constructors (cold path, NumPy only).

Economics note:
Grid curvature concentrates points near the borrowing constraint, where policy
functions have their kink and interpolation error is largest.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dynecon.core.errors import ValidationError

__all__ = ["cartesian", "exp_grid", "make_grid", "uniform_grid"]

Array = npt.NDArray[np.float64]


def _check(lo: float, hi: float, n: int) -> None:
    if not (np.isfinite(lo) and np.isfinite(hi)) or not hi > lo:
        raise ValidationError(f"grid bounds must be finite with hi > lo, got ({lo}, {hi})")
    if int(n) != n or n < 2:
        raise ValidationError(f"grid size n must be an integer >= 2, got {n}")


def uniform_grid(lo: float, hi: float, n: int) -> Array:
    """``n`` equally spaced points on ``[lo, hi]`` with exact endpoints."""
    _check(lo, hi, n)
    g = np.linspace(lo, hi, int(n), dtype=np.float64)
    g[0], g[-1] = lo, hi
    return g


def exp_grid(lo: float, hi: float, n: int, curvature: float = 3.0) -> Array:
    """Exponentially curved grid, dense near ``lo``.

    ``x = lo + (hi - lo) * (exp(k u) - 1) / (exp(k) - 1)`` with ``u`` uniform on [0, 1].
    Curvature ``k -> 0`` recovers the uniform grid; negative ``k`` makes it dense near ``hi``.
    """
    _check(lo, hi, n)
    if not np.isfinite(curvature):
        raise ValidationError(f"curvature must be finite, got {curvature}")
    u = np.linspace(0.0, 1.0, int(n), dtype=np.float64)
    if abs(curvature) < 1e-12:
        g = lo + (hi - lo) * u
    else:
        g = lo + (hi - lo) * np.expm1(curvature * u) / np.expm1(curvature)
    g[0], g[-1] = lo, hi
    return g


def cartesian(*grids: npt.ArrayLike) -> Array:
    """Cartesian product of 1-D grids: rows ``(prod n_i, d)``, last grid varying fastest."""
    if not grids:
        raise ValidationError("need at least one grid")
    arrs = [np.asarray(g, dtype=np.float64).ravel() for g in grids]
    if any(a.size == 0 for a in arrs):
        raise ValidationError("grids must be non-empty")
    mesh = np.meshgrid(*arrs, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def make_grid(kind: str, lo: float, hi: float, n: int, **kw: float) -> Array:
    """Build a grid by name: ``"uniform"`` or ``"exp"`` (``curvature`` keyword)."""
    if kind == "uniform":
        return uniform_grid(lo, hi, n)
    if kind == "exp":
        return exp_grid(lo, hi, n, **kw)
    raise ValidationError(f"unknown grid kind {kind!r}; use 'uniform' or 'exp'")
