"""Markov-chain path kernels: NumPy reference and Numba implementation.

Both take pre-drawn uniforms so results never depend on thread count or backend.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from econkit.utils.jit import jit, prange

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def markov_paths_numpy(cdf: Array, u: Array, init: IntArray) -> IntArray:
    """Paths ``s[t+1] = #{k : cdf[s[t], k] <= u[t]}``; ``u`` is (T-1, N), result (T, N)."""
    n_steps, n_agents = u.shape
    out = np.empty((n_steps + 1, n_agents), dtype=np.int64)
    out[0] = init
    last = cdf.shape[1] - 1
    for t in range(n_steps):
        rows = cdf[out[t]]
        out[t + 1] = np.minimum((rows <= u[t][:, None]).sum(axis=1), last)
    return out


@jit(parallel=True)
def markov_paths_numba(cdf: Array, u: Array, init: IntArray) -> IntArray:
    n_steps, n_agents = u.shape
    n_states = cdf.shape[1]
    out = np.empty((n_steps + 1, n_agents), dtype=np.int64)
    for a in prange(n_agents):
        s = init[a]
        out[0, a] = s
        for t in range(n_steps):
            k = 0
            while k < n_states - 1 and cdf[s, k] <= u[t, a]:
                k += 1
            s = k
            out[t + 1, a] = s
    return out
