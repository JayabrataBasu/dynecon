"""Endogenous grid method (Carroll 2006) for ``saving`` problems.

Economics note:
EGM inverts the Euler equation on a grid of end-of-period assets, so the optimal
consumption for each level of cash-on-hand is recovered in closed form with no root
finding. A constrained segment (consume everything above the borrowing limit) handles
the borrowing constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt

from econkit.core.errors import ModelSpecificationError, ValidationError
from econkit.core.solution import Solution
from econkit.model.compiled import CompiledModel
from econkit.model.problems import SavingProblem
from econkit.numerics._dispatch import resolve_backend
from econkit.numerics.fixedpoint import fixed_point
from econkit.numerics.interpolation import linear_interp_1d
from econkit.solvers.diagnostics import euler_errors, policy_monotone_in_assets
from econkit.solvers.registry import SOLVERS, resolve_config
from econkit.utils.provenance import collect_provenance

__all__ = ["EGMConfig", "solve_egm"]

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class EGMConfig:
    """Options of :func:`solve_egm`; the single source of EGM option names and defaults."""

    tol: float = 1e-10
    max_iter: int = 10000
    damping: float = 1.0

    def __post_init__(self) -> None:
        if not self.tol > 0.0:
            raise ValidationError("tol must be strictly positive")
        if self.max_iter < 1:
            raise ValidationError("max_iter must be >= 1")
        if not 0.0 < self.damping <= 1.0:
            raise ValidationError("damping must lie in (0, 1]")


def _require_egm(model: CompiledModel) -> SavingProblem:
    prob = model.problem
    if not isinstance(prob, SavingProblem):
        raise ValidationError("EGM needs a 'saving' problem")
    if not prob.supports_egm:
        raise ValidationError(
            "EGM needs marginal_utility, inverse_marginal_utility and marginal_return; "
            "use method='vfi'"
        )
    return prob


def solve_egm(model: CompiledModel, *, backend: str | None = None, **options: Any) -> Solution:
    """Solve a ``saving`` problem by EGM (see :class:`EGMConfig` for options)."""
    cfg = resolve_config(EGMConfig, options)
    prob = _require_egm(model)
    mu_fn, inv_mu_fn = prob.marginal_utility, prob.inverse_marginal_utility
    assert mu_fn is not None and inv_mu_fn is not None and prob.marginal_return is not None
    t0 = perf_counter()
    p = model.p
    a = model.grids[prob.asset]
    shock = model.shock_process(prob.shock)
    z, P = shock.values, shock.P
    n_a, n_z = a.shape[0], z.shape[0]
    beta = model.parameters.values[prob.discount]
    a_min = a[0]
    backend_used, _ = resolve_backend("linear_interp_1d", backend=backend)

    m_grid = np.broadcast_to(
        np.asarray(prob.resources(a[:, None], z[None, :], p)), (n_a, n_z)
    ).copy()
    if np.any(np.diff(m_grid, axis=0) <= 0.0):
        raise ModelSpecificationError(
            "resources must be strictly increasing in assets for every shock state"
        )
    gross = np.broadcast_to(np.asarray(prob.marginal_return(a[:, None], z[None, :], p)), (n_a, n_z))

    def operator(C: Array) -> Array:
        mu_next = np.asarray(mu_fn(C, p))  # u'(c(a'_k, z'_j')) on the grid
        rhs = beta * ((gross * mu_next) @ P.T)  # [k, j]
        c_end = np.asarray(inv_mu_fn(rhs, p), dtype=np.float64)
        m_end = c_end + a[:, None]
        if np.any(np.diff(m_end, axis=0) <= 0.0):
            raise ModelSpecificationError(
                "endogenous grid is not monotone; the problem is non-concave, use method='vfi'"
            )
        out = np.empty_like(C)
        for j in range(n_z):
            xk, fk = m_end[:, j], c_end[:, j]
            if m_grid[0, j] < xk[0]:  # constrained segment c = m - a_min
                xk = np.concatenate(([m_grid[0, j]], xk))
                fk = np.concatenate(([m_grid[0, j] - a_min], fk))
            out[:, j] = linear_interp_1d(
                m_grid[:, j], xk, fk, extrapolate="linear", backend=backend
            )
        return out

    C0 = np.maximum(m_grid - a_min, 1e-8)
    res = fixed_point(
        operator, C0, tol=cfg.tol, max_iter=cfg.max_iter, damping=cfg.damping, name="egm"
    )
    cons = res.x
    a_next = m_grid - cons
    sol = Solution(
        method="egm",
        converged=res.converged,
        iterations=res.iterations,
        residual=res.residual,
        elapsed=perf_counter() - t0,
        backend=backend_used,
        model_name=model.name,
        params=model.parameters.values,
        grids={prob.asset: a, **({prob.shock: z} if prob.shock else {})},
        policy={prob.control: cons, f"{prob.asset}_next": a_next, "cash_on_hand": m_grid},
        value=None,
        diagnostics={
            "monotone_policy": policy_monotone_in_assets(a_next),
            "max_feasibility_violation": float(max(0.0, -np.min(cons))),
            "n_binding": float(np.mean(a_next <= a_min + 1e-10)),
            "residual_history": list(res.history),
        },
        metadata={
            "fingerprint": model.fingerprint(),
            "provenance": collect_provenance(),
            "config": cfg,
        },
        model=model,
    )
    err = euler_errors(sol)
    if err.size:
        sol.diagnostics["euler_error_max"] = float(np.max(err))
        sol.diagnostics["euler_error_mean"] = float(np.mean(err))
    return sol


SOLVERS["egm"] = solve_egm
