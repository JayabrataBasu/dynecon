"""NumPy reference kernel for 1-D linear interpolation (permanent correctness baseline)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def interp_linear_numpy(x: Array, xp: Array, fp: Array, mode: int) -> Array:
    """Piecewise-linear interpolation. ``mode`` 0 clamps outside the grid, 1 extrapolates."""
    idx = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, xp.shape[0] - 2)
    x0 = xp[idx]
    t = (x - x0) / (xp[idx + 1] - x0)
    if mode == 0:
        t = np.clip(t, 0.0, 1.0)
    f0 = fp[idx]
    return np.asarray(f0 + t * (fp[idx + 1] - f0), dtype=np.float64)


def interp_columns_numpy(xq: Array, xk: Array, fk: Array, mode: int) -> Array:
    """Row-wise interpolation: row ``c`` of ``xq`` is evaluated on the knots ``(xk[c], fk[c])``."""
    out = np.empty_like(xq)
    for c in range(xq.shape[0]):
        out[c] = interp_linear_numpy(xq[c], xk[c], fk[c], mode)
    return out
