"""Steady state and linearisation of a ``dsge`` problem.

Economics note:
Perturbation approximates the model around its deterministic steady state, the point
where all variables stay constant in the absence of shocks.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dynecon.core.errors import ConvergenceError
from dynecon.model.problems import DSGEProblem, Vars
from dynecon.model.specs import ParamView
from dynecon.numerics.jacobian import numerical_jacobian

__all__ = ["linearize", "steady_state"]

Array = npt.NDArray[np.float64]


def _levels(v: Array, log_mask: npt.NDArray[np.bool_]) -> Array:
    return np.where(log_mask, np.exp(v), v)


def steady_state(prob: DSGEProblem, p: ParamView, tol: float = 1e-12) -> Array:
    """Steady state in levels: analytic ``steady_state(p)`` or numerically from ``guess(p)``."""
    from scipy.optimize import root

    names = prob.variables
    if prob.steady_state is not None:
        y = np.asarray(prob.steady_state(p), dtype=np.float64)
    else:
        assert prob.guess is not None
        log_mask = np.array([v in prob.log_variables for v in names])
        start = np.asarray(prob.guess(p), dtype=np.float64)

        def resid(v: Array) -> Array:
            y_ = _levels(v, log_mask)
            return np.asarray(prob.equations(Vars(names, y_), Vars(names, y_), p), dtype=np.float64)

        x0 = np.where(log_mask, np.log(np.where(log_mask, start, 1.0)), start)
        sol = root(resid, x0, method="hybr", tol=tol)
        y = _levels(sol.x, log_mask)
    res = np.asarray(prob.equations(Vars(names, y), Vars(names, y), p), dtype=np.float64)
    if not np.all(np.isfinite(res)) or np.max(np.abs(res)) > max(tol * 1e3, 1e-9):
        raise ConvergenceError(f"steady state not found; final residual vector: {res}")
    return y


def linearize(
    prob: DSGEProblem, p: ParamView, y_ss: Array, fd_eps: float = 1e-6
) -> tuple[Array, Array, Array]:
    """First-order expansion ``A E[yhat'] = B yhat`` and the shock loadings.

    Derivatives are central differences with respect to ``log y`` for ``log_variables`` and
    ``y`` otherwise, so reported deviations are percent deviations for logged variables.
    """
    names = prob.variables
    log_mask = np.array([v in prob.log_variables for v in names])
    y0 = np.where(log_mask, np.log(np.where(log_mask, y_ss, 1.0)), y_ss)

    def f_next(v: Array) -> Array:
        return np.asarray(
            prob.equations(Vars(names, _levels(v, log_mask)), Vars(names, y_ss), p),
            dtype=np.float64,
        )

    def f_now(v: Array) -> Array:
        return np.asarray(
            prob.equations(Vars(names, y_ss), Vars(names, _levels(v, log_mask)), p),
            dtype=np.float64,
        )

    A = numerical_jacobian(f_next, y0, eps=fd_eps)
    B = -numerical_jacobian(f_now, y0, eps=fd_eps)
    sigma = np.asarray(prob.shock_loadings(p), dtype=np.float64)
    return A, B, sigma
