"""Numba discretization kernels for AR(1) processes.

Numba support is optional at runtime.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

try:
    import numba
except Exception:  # pragma: no cover - optional dependency
    numba = None

Array = npt.NDArray[np.float64]

if numba is not None:

    @numba.njit(cache=True)
    def _std_norm_cdf_numba(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @numba.njit(cache=True)
    def tauchen_numba(
        rho: float,
        sigma_e: float,
        n_states: int,
        m: float = 3.0,
    ) -> tuple[Array, Array]:
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        if not (-1.0 < rho < 1.0):
            raise ValueError("rho must lie in (-1, 1)")
        if sigma_e <= 0.0:
            raise ValueError("sigma_e must be positive")

        sigma_z = sigma_e / math.sqrt(1.0 - rho * rho)
        z_max = m * sigma_z
        z_min = -z_max
        grid = np.linspace(z_min, z_max, n_states)
        step = grid[1] - grid[0]

        transition = np.zeros((n_states, n_states))
        for i in range(n_states):
            for j in range(n_states):
                if j == 0:
                    upper = (grid[0] - rho * grid[i] + 0.5 * step) / sigma_e
                    transition[i, j] = _std_norm_cdf_numba(upper)
                elif j == n_states - 1:
                    lower = (grid[n_states - 1] - rho * grid[i] - 0.5 * step) / sigma_e
                    transition[i, j] = 1.0 - _std_norm_cdf_numba(lower)
                else:
                    upper = (grid[j] - rho * grid[i] + 0.5 * step) / sigma_e
                    lower = (grid[j] - rho * grid[i] - 0.5 * step) / sigma_e
                    transition[i, j] = _std_norm_cdf_numba(upper) - _std_norm_cdf_numba(lower)

        row_sums = transition.sum(axis=1)
        for i in range(n_states):
            transition[i, :] = transition[i, :] / row_sums[i]

        return grid, transition

else:

    def tauchen_numba(
        rho: float,
        sigma_e: float,
        n_states: int,
        m: float = 3.0,
    ) -> tuple[Array, Array]:  # pragma: no cover
        raise RuntimeError("numba backend requested but numba is not installed")
