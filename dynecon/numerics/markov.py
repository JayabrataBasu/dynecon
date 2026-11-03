"""Markov-chain utilities: exact stationary distribution and chain simulation.

Economics note:
The stationary distribution of the shock chain weights the cross-section of
households; simulated chains drive panel simulation and SMM moments.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dynecon.core.errors import ValidationError
from dynecon.numerics._dispatch import resolve_backend
from dynecon.numerics._registry import register
from dynecon.numerics.backends.markov_kernels import markov_paths_numba, markov_paths_numpy
from dynecon.utils.jit import NUMBA_AVAILABLE
from dynecon.utils.rng import as_generator
from dynecon.utils.validation import check_stochastic_matrix

__all__ = ["markov_paths", "simulate_markov", "stationary_distribution"]

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

register("markov_paths", "numpy", markov_paths_numpy)
if NUMBA_AVAILABLE:
    register("markov_paths", "numba", markov_paths_numba)


def stationary_distribution(P: Array) -> Array:
    """Stationary distribution by the Grassmann-Taksar-Heyman algorithm (no eigensolver).

    GTH eliminates states using only additions of non-negative numbers, so it is
    accurate to machine precision even for nearly-decomposable chains.
    """
    A = check_stochastic_matrix(P, "P").copy()
    n = A.shape[0]
    for k in range(n - 1, 0, -1):
        s = A[k, :k].sum()
        if s <= 0.0:
            raise ValidationError("P has a reducible structure; stationary distribution not unique")
        A[:k, k] /= s
        A[:k, :k] += np.outer(A[:k, k], A[k, :k])
    pi = np.zeros(n, dtype=np.float64)
    pi[0] = 1.0
    for j in range(1, n):
        pi[j] = pi[:j] @ A[:j, j]
    return np.asarray(pi / pi.sum(), dtype=np.float64)


def markov_paths(
    P: Array,
    n_periods: int,
    init: IntArray,
    u: Array,
    backend: str | None = None,
) -> IntArray:
    """Simulate ``len(init)`` chains for ``n_periods`` using pre-drawn uniforms ``u`` (T-1, N)."""
    Pm = check_stochastic_matrix(P, "P")
    init = np.asarray(init, dtype=np.int64)
    u = np.ascontiguousarray(u, dtype=np.float64)
    if u.shape != (n_periods - 1, init.shape[0]):
        raise ValidationError(f"u must have shape {(n_periods - 1, init.shape[0])}, got {u.shape}")
    if np.any(init < 0) or np.any(init >= Pm.shape[0]):
        raise ValidationError("init states out of range")
    cdf = np.cumsum(Pm, axis=1)
    cdf[:, -1] = 1.0
    _, impl = resolve_backend("markov_paths", backend=backend)
    return np.asarray(impl(cdf, u, init), dtype=np.int64)


def simulate_markov(
    P: Array,
    n: int,
    rng: np.random.Generator | int | None = None,
    init: int | None = None,
    backend: str | None = None,
) -> IntArray:
    """Simulate one chain of length ``n``; the initial state is drawn from the stationary law."""
    if n < 1:
        raise ValidationError("n must be >= 1")
    gen = as_generator(rng)
    Pm = check_stochastic_matrix(P, "P")
    s0 = (
        int(init)
        if init is not None
        else int(gen.choice(Pm.shape[0], p=stationary_distribution(Pm)))
    )
    u = gen.random((n - 1, 1))
    return markov_paths(Pm, n, np.array([s0], dtype=np.int64), u, backend=backend)[:, 0]
