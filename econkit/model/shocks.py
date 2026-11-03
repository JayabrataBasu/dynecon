"""Discretisation of declared shocks into ``(values, P)`` pairs.

Economics note:
Every exogenous risk (productivity, income, preference) ends up as a finite Markov
chain, the object dynamic-programming solvers can take expectations over.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from econkit.core.errors import ValidationError
from econkit.model.specs import AR1Shock, IIDShock, MarkovShock, ShockProcess
from econkit.numerics.discretization import rouwenhorst, tauchen
from econkit.numerics.integration import gauss_hermite_normal, gauss_legendre
from econkit.utils.validation import check_array, check_stochastic_matrix

__all__ = ["discretize_shock"]


def _resolve(x: float | str, params: Mapping[str, float], what: str) -> float:
    if isinstance(x, str):
        if x not in params:
            raise ValidationError(f"{what} refers to unknown parameter {x!r}")
        return float(params[x])
    return float(x)


def discretize_shock(
    shock: AR1Shock | MarkovShock | IIDShock, params: Mapping[str, float]
) -> ShockProcess:
    """Discretise a shock descriptor using the current parameter values."""
    if isinstance(shock, AR1Shock):
        rho = _resolve(shock.rho, params, "AR1Shock.rho")
        sig = _resolve(shock.sigma_e, params, "AR1Shock.sigma_e")
        if shock.method == "rouwenhorst":
            grid, P = rouwenhorst(rho, sig, shock.n_states)
        elif shock.method == "tauchen":
            grid, P = tauchen(rho, sig, shock.n_states)
        else:
            raise ValidationError(f"unknown AR1 method {shock.method!r}")
        values = grid + shock.mean
        return ShockProcess(np.exp(values) if shock.log else values, P)
    if isinstance(shock, MarkovShock):
        values = check_array(shock.values, "MarkovShock.values", ndim=1)
        P = check_stochastic_matrix(shock.P, "MarkovShock.P")
        if P.shape[0] != values.shape[0]:
            raise ValidationError("MarkovShock.values and P must have matching sizes")
        return ShockProcess(values, P)
    if isinstance(shock, IIDShock):
        pr = shock.params
        if shock.kind in ("normal", "lognormal"):
            mu, sd = float(pr.get("mu", 0.0)), float(pr.get("sigma", 1.0))
            if not sd > 0:
                raise ValidationError("IIDShock sigma must be positive")
            x, w = gauss_hermite_normal(shock.n_nodes)
            values = mu + sd * x
            if shock.kind == "lognormal":
                values = np.exp(values)
        elif shock.kind == "uniform":
            lo, hi = float(pr.get("low", 0.0)), float(pr.get("high", 1.0))
            values, w = gauss_legendre(shock.n_nodes, lo, hi)
        else:
            raise ValidationError(f"unknown IIDShock kind {shock.kind!r}")
        n = values.shape[0]
        if not math.isclose(float(w.sum()), 1.0, rel_tol=1e-12):
            raise ValidationError("quadrature weights do not sum to one")
        return ShockProcess(values, np.tile(w, (n, 1)))
    raise ValidationError(f"unsupported shock type {type(shock).__name__}")
