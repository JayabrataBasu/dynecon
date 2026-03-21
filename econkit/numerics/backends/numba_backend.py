"""Numba kernels.

Numba support is optional at runtime.
"""

import numpy as np
import numpy.typing as npt

try:
    import numba
except Exception:  # pragma: no cover - optional dependency
    numba = None


if numba is not None:

    @numba.njit(cache=True)
    def _linear_interp_scalar(
        x: float,
        xp: npt.NDArray[np.float64],
        fp: npt.NDArray[np.float64],
    ) -> float:
        idx = np.searchsorted(xp, x) - 1
        idx = max(0, min(idx, xp.shape[0] - 2))
        x0 = xp[idx]
        x1 = xp[idx + 1]
        y0 = fp[idx]
        y1 = fp[idx + 1]
        if x1 == x0:
            return float(y0)
        t = (x - x0) / (x1 - x0)
        return float(y0 * (1.0 - t) + y1 * t)

    @numba.njit(cache=True)
    def linear_interp_1d_numba(
        x: npt.NDArray[np.float64],
        xp: npt.NDArray[np.float64],
        fp: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = _linear_interp_scalar(x[i], xp, fp)
        return out

else:

    def linear_interp_1d_numba(
        x: npt.NDArray[np.float64],
        xp: npt.NDArray[np.float64],
        fp: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:  # pragma: no cover
        raise RuntimeError("numba backend requested but numba is not installed")
