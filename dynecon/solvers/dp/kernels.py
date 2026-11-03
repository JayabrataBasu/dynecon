"""Bellman maximisation kernels for ``saving`` problems: NumPy reference and Numba.

Layout: ``U[j, i, k]`` is the payoff at shock ``j``, assets ``i``, next assets ``k``
(``-inf`` if infeasible); ``EVT[j, k] = beta * E[V(a'_k, z') | z_j]``. Kernels see only
float64/int64 arrays, never user callables.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dynecon.utils.jit import jit, prange

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def bellman_saving_numpy(U: Array, EVT: Array, monotone: bool) -> tuple[Array, IntArray]:
    """Exhaustive maximisation over ``k`` (``monotone`` is accepted but not needed here)."""
    del monotone
    total = U + EVT[:, None, :]
    idx = total.argmax(axis=2)
    v = np.take_along_axis(total, idx[:, :, None], axis=2)[:, :, 0]
    return np.ascontiguousarray(v.T), np.ascontiguousarray(idx.T.astype(np.int64))


@jit(parallel=True)
def bellman_saving_numba(U: Array, EVT: Array, monotone: bool) -> tuple[Array, IntArray]:
    n_z, n_a, n_k = U.shape
    v_new = np.empty((n_a, n_z), dtype=np.float64)
    pol = np.empty((n_a, n_z), dtype=np.int64)
    for j in prange(n_z):
        k_lo = 0
        for i in range(n_a):
            best = -np.inf
            arg = k_lo
            for k in range(k_lo if monotone else 0, n_k):
                val = U[j, i, k] + EVT[j, k]
                if val > best:
                    best = val
                    arg = k
            v_new[i, j] = best
            pol[i, j] = arg
            k_lo = arg
    return v_new, pol
