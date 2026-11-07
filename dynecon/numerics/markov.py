"""Markov-chain utilities: exact stationary distribution and chain simulation.

Economics note:
The stationary distribution of the shock chain weights the cross-section of
households; simulated chains drive panel simulation and SMM moments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from dynecon.core.errors import ValidationError
from dynecon.numerics._dispatch import resolve_backend
from dynecon.numerics._registry import register
from dynecon.numerics.backends.markov_kernels import markov_paths_numba, markov_paths_numpy
from dynecon.utils.jit import NUMBA_AVAILABLE
from dynecon.utils.rng import as_generator
from dynecon.utils.validation import check_int, check_stochastic_matrix

__all__ = ["MarkovChain", "markov_paths", "simulate_markov", "stationary_distribution"]

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
    n = check_int(n, "n", 1, 10**8)
    gen = as_generator(rng)
    Pm = check_stochastic_matrix(P, "P")
    s0 = (
        check_int(init, "init", 0, Pm.shape[0] - 1)
        if init is not None
        else int(gen.choice(Pm.shape[0], p=stationary_distribution(Pm)))
    )
    u = gen.random((n - 1, 1))
    return markov_paths(Pm, n, np.array([s0], dtype=np.int64), u, backend=backend)[:, 0]


def _classes(P: Array) -> tuple[list[IntArray], list[bool]]:
    """Communication classes (strongly connected components) and whether each is closed."""
    _, label = connected_components(csr_matrix(P > 0), directed=True, connection="strong")
    classes = [np.flatnonzero(label == c).astype(np.int64) for c in range(label.max() + 1)]
    closed = [
        bool(np.all(P[np.ix_(c, np.flatnonzero(~np.isin(np.arange(len(P)), c)))] == 0))
        for c in classes
    ]
    return classes, closed


def _period(P: Array, cls: IntArray) -> int:
    """Period of one communication class: gcd of ``d[u] + 1 - d[v]`` over its edges (BFS levels)."""
    sub = P[np.ix_(cls, cls)] > 0
    dist = np.full(len(cls), -1)
    dist[0] = 0
    frontier = [0]
    while frontier:
        nxt: list[int] = []
        for u in frontier:
            for v in np.flatnonzero(sub[u] & (dist < 0)):
                dist[v] = dist[u] + 1
                nxt.append(int(v))
        frontier = nxt
    g = 0
    for u, v in zip(*np.nonzero(sub), strict=True):
        g = math.gcd(g, int(dist[u] + 1 - dist[v]))
    return g


@dataclass(frozen=True)
class MarkovChain:
    """Finite Markov chain: structure (classes, period) and stationary distributions.

    ``stationary_distributions`` returns one distribution per recurrent class (a unique one when
    the chain has a single recurrent class), which :func:`stationary_distribution` cannot do for
    reducible chains. ``state_values`` labels the states (defaults to ``0, ..., n-1``).
    """

    P: Array
    state_values: Array = field(default_factory=lambda: np.empty(0))

    def __post_init__(self) -> None:
        P = check_stochastic_matrix(self.P, "P")
        object.__setattr__(self, "P", P)
        given = np.asarray(self.state_values)
        sv = np.arange(len(P), dtype=np.float64) if given.size == 0 else given
        if np.shape(sv)[0] != len(P):
            raise ValidationError("state_values must have one entry per state")
        object.__setattr__(self, "state_values", np.asarray(sv))

    @property
    def communication_classes(self) -> list[IntArray]:
        """State indices of each communication class."""
        return _classes(self.P)[0]

    @property
    def recurrent_classes(self) -> list[IntArray]:
        """Closed communication classes (the chain is eventually absorbed in one of them)."""
        classes, closed = _classes(self.P)
        return [c for c, is_closed in zip(classes, closed, strict=True) if is_closed]

    @property
    def is_irreducible(self) -> bool:
        """Whether every state can reach every other state."""
        return len(self.communication_classes) == 1

    @property
    def period(self) -> int:
        """Period of an irreducible chain (1 means aperiodic)."""
        if not self.is_irreducible:
            raise ValidationError("the period is defined per recurrent class for reducible chains")
        return _period(self.P, np.arange(len(self.P)))

    @property
    def is_aperiodic(self) -> bool:
        """Whether every recurrent class has period one."""
        return all(_period(self.P, c) == 1 for c in self.recurrent_classes)

    @property
    def stationary_distributions(self) -> Array:
        """Rows: the stationary distribution supported on each recurrent class."""
        out = np.zeros((len(self.recurrent_classes), len(self.P)))
        for k, c in enumerate(self.recurrent_classes):
            sub = self.P[np.ix_(c, c)]
            out[k, c] = stationary_distribution(sub) if len(c) > 1 else 1.0
        return out

    def simulate(
        self, n: int, rng: np.random.Generator | int | None = None, init: int | None = None
    ) -> Array:
        """Simulate ``n`` periods; returns the visited ``state_values``."""
        if init is None and not self.is_irreducible:
            raise ValidationError("pass init for a reducible chain (no unique stationary law)")
        return np.asarray(self.state_values[simulate_markov(self.P, n, rng=rng, init=init)])
