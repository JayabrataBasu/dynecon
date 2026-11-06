"""Young (2010) histogram distributions, aggregates and inequality measures.

Economics note:
Heterogeneous-agent equilibria need the exact stationary cross-section of wealth. The
histogram method splits each household's mass between the two grid nodes bracketing its
policy, preserving the policy mean and avoiding simulation noise.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from dynecon.core.errors import ConvergenceWarning, GridWarning, ValidationError
from dynecon.core.solution import Solution
from dynecon.model.problems import SavingProblem
from dynecon.numerics._dispatch import resolve_backend
from dynecon.numerics._registry import register
from dynecon.numerics.fixedpoint import fixed_point
from dynecon.numerics.markov import stationary_distribution
from dynecon.simulation.kernels import distribution_lottery_numba, distribution_lottery_numpy
from dynecon.utils.jit import NUMBA_AVAILABLE

TOP_MASS_WARN = 0.01  # mass on the last asset node above which the grid is called too small

__all__ = [
    "aggregate",
    "build_transition_matrix",
    "cdf",
    "gini",
    "iterate_distribution",
    "lorenz_curve",
    "marginal_distribution",
    "quantile",
    "stationary_distribution_young",
]

if TYPE_CHECKING:
    from scipy import sparse

Array = npt.NDArray[np.float64]

register("distribution_lottery", "numpy", distribution_lottery_numpy)
if NUMBA_AVAILABLE:
    register("distribution_lottery", "numba", distribution_lottery_numba)


def _parts(solution: Solution) -> tuple[Array, Array, Array]:
    model = solution.model
    prob = None if model is None else model.problem
    if model is None or not isinstance(prob, SavingProblem):
        raise ValidationError("distributions need a Solution of a 'saving' problem")
    shock = model.shock_process(prob.shock)
    return solution.grids[prob.asset], solution.policy[f"{prob.asset}_next"], shock.P


def lottery_weights(solution: Solution) -> tuple[npt.NDArray[np.int64], Array, int]:
    """Bracket index ``k`` and weight ``w`` on node ``k`` (``1 - w`` on ``k + 1``) per state.

    Policy values outside the grid are clamped to its ends; the number clamped is returned
    (zero for well-chosen grids).
    """
    a, a_next, _ = _parts(solution)
    clamped = int(np.sum((a_next < a[0] - 1e-9) | (a_next > a[-1] + 1e-9)))
    a_c = np.clip(a_next, a[0], a[-1])
    k = np.clip(np.searchsorted(a, a_c, side="right") - 1, 0, a.shape[0] - 2).astype(np.int64)
    w = (a[k + 1] - a_c) / (a[k + 1] - a[k])
    return k, np.clip(w, 0.0, 1.0), clamped


def build_transition_matrix(solution: Solution) -> sparse.csr_matrix:
    """Row-stochastic sparse transition matrix over flattened ``(asset, shock)`` states.

    This is the NumPy/SciPy reference for the matrix-free histogram step.
    """
    from scipy import sparse

    a, _, P = _parts(solution)
    n_a, n_z = solution.policy[f"{_asset_name(solution)}_next"].shape
    k, w, _ = lottery_weights(solution)
    i_idx, j_idx, jp_idx = np.meshgrid(
        np.arange(n_a), np.arange(n_z), np.arange(n_z), indexing="ij"
    )
    rows = (i_idx * n_z + j_idx).ravel()
    pj = P[j_idx, jp_idx]
    lo = (k[i_idx, j_idx] * n_z + jp_idx).ravel()
    hi = ((k[i_idx, j_idx] + 1) * n_z + jp_idx).ravel()
    wl = w[i_idx, j_idx]
    data_lo = (wl * pj).ravel()
    data_hi = ((1.0 - wl) * pj).ravel()
    size = n_a * n_z
    mat = sparse.coo_matrix(
        (np.concatenate([data_lo, data_hi]), (np.tile(rows, 2), np.concatenate([lo, hi]))),
        shape=(size, size),
    ).tocsr()
    del a
    return mat


def _asset_name(solution: Solution) -> str:
    model = solution.model
    assert model is not None and isinstance(model.problem, SavingProblem)
    return model.problem.asset


def iterate_distribution(
    mu: Array, solution: Solution, n: int = 1, backend: str | None = None
) -> Array:
    """Apply the policy-induced transition ``n`` times to the mass ``mu`` (``(n_a, n_z)``)."""
    _, _, P = _parts(solution)
    k, w, _ = lottery_weights(solution)
    _, lottery = resolve_backend("distribution_lottery", backend=backend)
    out = np.ascontiguousarray(mu, dtype=np.float64)
    if out.shape != k.shape:
        raise ValidationError(f"mu must have shape {k.shape}, got {out.shape}")
    for _ in range(n):
        out = np.asarray(lottery(out, k, w), dtype=np.float64) @ P
    return out


def stationary_distribution_young(
    solution: Solution,
    tol: float = 1e-12,
    max_iter: int = 2_000,
    backend: str | None = None,
) -> Array:
    """Stationary mass ``mu[i, j]`` on the (asset, shock) grid.

    Iterates the histogram step on :func:`dynecon.numerics.fixed_point`; when the iteration has
    not reached ``tol`` within ``max_iter`` steps (slowly mixing wealth distributions near
    ``beta (1 + r) = 1``) it falls back to a sparse direct solve of ``mu = T' mu``.
    """
    _, a_next, P = _parts(solution)
    k, w, _ = lottery_weights(solution)
    _, lottery = resolve_backend("distribution_lottery", backend=backend)
    pi = stationary_distribution(P) if P.shape[0] > 1 else np.ones(1)
    mu0 = np.outer(np.full(a_next.shape[0], 1.0 / a_next.shape[0]), pi)

    def step(mu: Array) -> Array:
        return np.asarray(lottery(mu, k, w), dtype=np.float64) @ P

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = fixed_point(step, mu0, tol=tol, max_iter=max_iter, norm="l1", name="young")
    mu = res.x
    if not res.converged:
        mu = _direct_stationary(build_transition_matrix(solution)).reshape(mu0.shape)
    mu = np.asarray(mu / np.sum(mu), dtype=np.float64)
    if (top := float(mu[-1].sum())) > TOP_MASS_WARN:
        warnings.warn(
            f"{top:.1%} of the stationary mass sits on the top asset-grid node: the grid ceiling "
            "binds and aggregates are biased. Raise a_max (or the grid size).",
            GridWarning,
            stacklevel=2,
        )
    return mu


def _direct_stationary(T: sparse.csr_matrix) -> Array:
    """Solve ``mu = T' mu`` with ``sum(mu) = 1`` by a sparse LU factorisation."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    n = T.shape[0]
    lhs = (sparse.identity(n, format="csr") - T.T).tolil()
    lhs[n - 1, :] = np.ones(n)
    rhs = np.zeros(n)
    rhs[n - 1] = 1.0
    mu = np.asarray(spsolve(lhs.tocsc(), rhs), dtype=np.float64)
    mu = np.clip(mu, 0.0, None)
    return np.asarray(mu / mu.sum(), dtype=np.float64)


def aggregate(mu: Array, values: npt.ArrayLike) -> float:
    """Expectation ``sum(mu * values)``; ``values`` broadcasts against ``mu``."""
    return float(np.sum(mu * np.asarray(values, dtype=np.float64)))


def marginal_distribution(mu: Array, axis: int) -> Array:
    """Marginal mass: ``axis=0`` over assets (sums shocks out), ``axis=1`` over shocks."""
    if axis not in (0, 1):
        raise ValidationError("axis must be 0 (assets) or 1 (shocks)")
    return np.asarray(mu.sum(axis=1 - axis), dtype=np.float64)


def cdf(mass: Array) -> Array:
    """Cumulative distribution of a 1-D mass vector."""
    return np.asarray(np.cumsum(mass), dtype=np.float64)


def quantile(mass: Array, grid: Array, q: float) -> float:
    """Generalised inverse CDF: the smallest grid value with cumulative mass >= ``q``."""
    if not 0.0 <= q <= 1.0:
        raise ValidationError("q must lie in [0, 1]")
    if mass.size == 0 or mass.shape != grid.shape:
        raise ValidationError("mass and grid must be non-empty arrays of the same shape")
    c = cdf(mass) / float(np.sum(mass))
    return float(grid[min(int(np.searchsorted(c, q - 1e-14, side="left")), grid.shape[0] - 1)])


def lorenz_curve(
    values: npt.ArrayLike, weights: npt.ArrayLike | None = None
) -> tuple[Array, Array]:
    """Lorenz curve ``(population share, value share)``, both from 0 to 1."""
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if v.shape != w.shape:
        raise ValidationError("values and weights must have the same size")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(w))) or np.any(w < 0):
        raise ValidationError("values must be finite and weights finite and non-negative")
    if np.any(v < 0):
        raise ValidationError("Lorenz curve needs non-negative values (shift wealth by the limit)")
    keep = w > 0
    v, w = v[keep].ravel(), w[keep].ravel()
    order = np.argsort(v, kind="stable")
    v, w = v[order].ravel(), w[order].ravel()
    total = float(np.sum(w * v))
    if total <= 0.0:
        raise ValidationError("values must have a positive weighted sum")
    pop = np.concatenate([[0.0], np.cumsum(w) / np.sum(w)])
    share = np.concatenate([[0.0], np.cumsum(w * v) / total])
    return pop, share


def gini(values: npt.ArrayLike, weights: npt.ArrayLike | None = None) -> float:
    """Gini coefficient ``1 - 2 * area under the Lorenz curve`` (weighted, exact)."""
    pop, share = lorenz_curve(values, weights)
    return float(1.0 - np.sum(np.diff(pop) * (share[1:] + share[:-1])))
