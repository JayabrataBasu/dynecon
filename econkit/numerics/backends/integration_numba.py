"""Numba integration/expectation kernels.

Numba support is optional at runtime.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from econkit.numerics.backends.integration_numpy import (
    gauss_hermite_normal_numpy,
    gauss_laguerre_exp_numpy,
)

try:
    import numba
except Exception:  # pragma: no cover - optional dependency
    numba = None

Array = npt.NDArray[np.float64]

if numba is not None:

    @numba.njit(cache=True)
    def expectation_discrete_numba(values: Array, probs: Array) -> float:
        if values.shape != probs.shape:
            raise ValueError("values and probs must have identical shapes")

        s = 0.0
        for i in range(probs.shape[0]):
            s += probs[i]
        if abs(s - 1.0) > 1e-10:
            raise ValueError("probabilities must sum to one")

        out = 0.0
        for i in range(values.shape[0]):
            out += values[i] * probs[i]
        return out

    def gauss_hermite_normal_numba(order: int) -> tuple[Array, Array]:
        # Node/weight generation is delegated to NumPy polynomial routines.
        return gauss_hermite_normal_numpy(order)

    def gauss_laguerre_exp_numba(order: int) -> tuple[Array, Array]:
        # Node/weight generation is delegated to NumPy polynomial routines.
        return gauss_laguerre_exp_numpy(order)

else:

    def expectation_discrete_numba(values: Array, probs: Array) -> float:  # pragma: no cover
        raise RuntimeError("numba backend requested but numba is not installed")

    def gauss_hermite_normal_numba(order: int) -> tuple[Array, Array]:  # pragma: no cover
        raise RuntimeError("numba backend requested but numba is not installed")

    def gauss_laguerre_exp_numba(order: int) -> tuple[Array, Array]:  # pragma: no cover
        raise RuntimeError("numba backend requested but numba is not installed")
