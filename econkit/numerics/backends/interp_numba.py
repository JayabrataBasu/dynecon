"""Numba kernels for 1-D linear interpolation.

``interp_linear_scalar`` is the single bracketing/weighting implementation, inlined by
the array operation here and by the compiled Bellman, simulation and histogram kernels.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from econkit.utils.jit import jit, prange

Array = npt.NDArray[np.float64]


@jit
def interp_linear_scalar(x: float, xp: Array, fp: Array, mode: int) -> float:
    """Interpolate at one point. ``mode`` 0 clamps outside the grid, 1 extrapolates."""
    idx = np.searchsorted(xp, x, side="right") - 1
    idx = max(0, min(idx, xp.shape[0] - 2))
    x0 = xp[idx]
    t = (x - x0) / (xp[idx + 1] - x0)
    if mode == 0:
        t = min(1.0, max(0.0, t))
    f0 = fp[idx]
    return float(f0 + t * (fp[idx + 1] - f0))


@jit
def interp_linear_numba(x: Array, xp: Array, fp: Array, mode: int) -> Array:
    out = np.empty(x.shape[0], dtype=np.float64)
    for i in range(x.shape[0]):
        out[i] = interp_linear_scalar(x[i], xp, fp, mode)
    return out


@jit(parallel=True)
def interp_columns_numba(xq: Array, xk: Array, fk: Array, mode: int) -> Array:
    out = np.empty_like(xq)
    for c in prange(xq.shape[0]):
        for i in range(xq.shape[1]):
            out[c, i] = interp_linear_scalar(xq[c, i], xk[c], fk[c], mode)
    return out
