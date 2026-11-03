"""Simulation and histogram kernels: NumPy reference and Numba implementations.

Both backends receive only float64/int64 arrays. Random numbers are drawn in NumPy
beforehand, so results never depend on thread count.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dynecon.numerics.backends.interp_numba import interp_linear_scalar
from dynecon.numerics.backends.interp_numpy import interp_linear_numpy
from dynecon.utils.jit import jit, prange

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def simulate_saving_numpy(
    grid: Array, pol_a: Array, pol_c: Array, shock_idx: IntArray, a0: Array
) -> tuple[Array, Array]:
    """Assets (start of period) and consumption paths, each ``(T, N)``.

    ``pol_a[:, j]`` / ``pol_c[:, j]`` are next-period assets and consumption on ``grid`` when
    the shock state is ``j``; they are linearly interpolated (clamped) at the agents' assets.
    """
    n_t, n_agents = shock_idx.shape
    n_z = pol_a.shape[1]
    assets = np.empty((n_t, n_agents), dtype=np.float64)
    cons = np.empty((n_t, n_agents), dtype=np.float64)
    a = np.array(a0, dtype=np.float64)
    for t in range(n_t):
        assets[t] = a
        nxt = np.empty(n_agents, dtype=np.float64)
        for j in range(n_z):
            mask = shock_idx[t] == j
            if mask.any():
                nxt[mask] = interp_linear_numpy(a[mask], grid, pol_a[:, j], 0)
                cons[t, mask] = interp_linear_numpy(a[mask], grid, pol_c[:, j], 0)
        a = nxt
    return assets, cons


@jit(parallel=True)
def simulate_saving_numba(
    grid: Array, pol_a: Array, pol_c: Array, shock_idx: IntArray, a0: Array
) -> tuple[Array, Array]:
    n_t, n_agents = shock_idx.shape
    assets = np.empty((n_t, n_agents), dtype=np.float64)
    cons = np.empty((n_t, n_agents), dtype=np.float64)
    for n in prange(n_agents):
        a = a0[n]
        for t in range(n_t):
            j = shock_idx[t, n]
            assets[t, n] = a
            cons[t, n] = interp_linear_scalar(a, grid, pol_c[:, j], 0)
            a = interp_linear_scalar(a, grid, pol_a[:, j], 0)
    return assets, cons


def distribution_lottery_numpy(mu: Array, k: IntArray, w: Array) -> Array:
    """Move mass ``mu[i, j]`` to asset nodes ``k[i, j]`` (weight ``w``) and ``k + 1`` (1 - w)."""
    n_a, n_z = mu.shape
    cols = np.broadcast_to(np.arange(n_z), mu.shape)
    lo = (k * n_z + cols).ravel()
    hi = ((k + 1) * n_z + cols).ravel()
    size = n_a * n_z
    out = np.bincount(lo, weights=(mu * w).ravel(), minlength=size)
    out += np.bincount(hi, weights=(mu * (1.0 - w)).ravel(), minlength=size)
    return out.reshape(n_a, n_z)


@jit
def distribution_lottery_numba(mu: Array, k: IntArray, w: Array) -> Array:
    n_a, n_z = mu.shape
    out = np.zeros((n_a, n_z), dtype=np.float64)
    for i in range(n_a):
        for j in range(n_z):
            m = mu[i, j]
            out[k[i, j], j] += m * w[i, j]
            out[k[i, j] + 1, j] += m * (1.0 - w[i, j])
    return out
