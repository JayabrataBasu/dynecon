"""Solver diagnostics.

Economics note:
Bellman residuals, policy monotonicity and the share of households at the borrowing
constraint tell a researcher whether a computed policy is trustworthy economically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from econkit.core.errors import ValidationError
from econkit.model.problems import SavingProblem
from econkit.numerics.interpolation import linear_interp_1d

if TYPE_CHECKING:
    from econkit.core.solution import Solution

__all__ = ["bellman_residual", "euler_error_grid", "euler_errors", "policy_monotone_in_assets"]


def bellman_residual(
    value_new: npt.NDArray[np.float64], value_old: npt.NDArray[np.float64]
) -> float:
    """Sup-norm distance between successive value iterates (the VFI stopping diagnostic)."""
    new = np.asarray(value_new, dtype=np.float64)
    old = np.asarray(value_old, dtype=np.float64)
    if new.shape != old.shape:
        raise ValueError("value_new and value_old must have the same shape")
    return float(np.max(np.abs(new - old)))


def policy_monotone_in_assets(policy: npt.NDArray[np.float64], tol: float = 1e-12) -> bool:
    """Whether a policy ``(n_assets, n_shocks)`` is non-decreasing along the asset axis."""
    return bool(np.all(np.diff(policy, axis=0) >= -tol))


def euler_errors(solution: Solution) -> npt.NDArray[np.float64]:
    """Interior Euler errors as a flat vector (see :func:`euler_error_grid`)."""
    grid = euler_error_grid(solution)
    return np.asarray(grid[np.isfinite(grid)], dtype=np.float64)


def euler_error_grid(solution: Solution) -> npt.NDArray[np.float64]:
    """Unit-free Euler-equation errors ``log10 |1 - c_implied / c|`` over interior states.

    ``c_implied = (u')^{-1}(beta E[R(a', z') u'(c(a', z'))])``. Constrained states (next
    assets at the lower bound) and states whose next assets leave the grid are excluded.
    Valid for solutions from any grid method.

    Economics note:
    The Euler error is the standard accuracy test for consumption-savings solutions: it
    measures the consumption-equivalent mistake a household makes by following the policy.
    """
    model = solution.model
    prob = None if model is None else model.problem
    if model is None or not isinstance(prob, SavingProblem) or not prob.supports_egm:
        raise ValidationError("euler_errors needs a solution of a 'saving' problem with marginals")
    assert prob.marginal_utility and prob.inverse_marginal_utility and prob.marginal_return
    p = model.p
    a = model.grids[prob.asset]
    shock = model.shock_process(prob.shock)
    cons = solution.policy[prob.control]
    a_next = solution.policy[f"{prob.asset}_next"]
    beta = model.parameters.values[prob.discount]
    interior = (a_next > a[0] + 1e-9 * (1.0 + abs(a[0]))) & (a_next < a[-1]) & (cons > 0.0)
    rhs = np.zeros_like(cons)
    for jp in range(shock.values.shape[0]):
        c_next = linear_interp_1d(a_next, a, cons[:, jp])
        gross = np.broadcast_to(
            np.asarray(prob.marginal_return(a_next, shock.values[jp], p)), cons.shape
        )
        rhs += shock.P[:, jp][None, :] * gross * np.asarray(prob.marginal_utility(c_next, p))
    c_imp = np.asarray(prob.inverse_marginal_utility(beta * rhs, p), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.log10(np.maximum(np.abs(1.0 - c_imp / cons), 1e-16))
    return np.where(interior & np.isfinite(err), err, np.nan)
