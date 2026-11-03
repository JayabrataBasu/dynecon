"""Two-period overlapping-generations model (Diamond): log utility, Cobb-Douglas, full depreciation.

Economics note:
Young agents save a constant share of their wage; the capital stock per young worker
follows an exact one-dimensional law of motion with a unique, globally stable steady state.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from econkit.model import Model, Param
from econkit.models.registry import register, split_overrides

__all__ = [
    "ANALYTIC",
    "CALIBRATION_SOURCE",
    "PARAMS",
    "build",
    "euler_steady_state",
    "iterate_capital",
    "steady_state_capital",
]

CALIBRATION_SOURCE = (
    "Diamond (1965); one period = about 30 years (beta = 0.96^30, n = 1.01^30 - 1)."
)

PARAMS: dict[str, Param] = {
    "alpha": Param(0.3, bounds=(0.05, 0.95), description="Capital share"),
    "beta": Param(0.96**30, bounds=(0.01, 1.0), description="Per-generation discount factor"),
    "n": Param(1.01**30 - 1.0, bounds=(-0.5, 5.0), description="Population growth per period"),
}


def steady_state_capital(alpha: float, beta: float, n: float) -> float:
    """``k* = [beta (1 - alpha) / ((1 + beta)(1 + n))]^(1/(1 - alpha))``."""
    return float((beta * (1.0 - alpha) / ((1.0 + beta) * (1.0 + n))) ** (1.0 / (1.0 - alpha)))


def _next_capital(k: Any, alpha: float, beta: float, n: float) -> Any:
    """Law of motion ``k' = beta/(1+beta) (1 - alpha) k^alpha / (1 + n)``."""
    return beta / (1.0 + beta) * (1.0 - alpha) * k**alpha / (1.0 + n)


def iterate_capital(k0: float, periods: int, alpha: float, beta: float, n: float) -> np.ndarray:
    """Exact capital path ``k_0, ..., k_periods`` from the law of motion."""
    path = np.empty(periods + 1)
    path[0] = k0
    for t in range(periods):
        path[t + 1] = _next_capital(path[t], alpha, beta, n)
    return path


def euler_steady_state(alpha: float, beta: float, n: float) -> float:
    """Steady state from the household Euler equation by root finding (independent check)."""

    from scipy.optimize import brentq

    def euler(k: float) -> float:
        wage = (1.0 - alpha) * k**alpha
        saving = (1.0 + n) * k
        return float(1.0 / (wage - saving) - beta / saving)

    upper = ((1.0 - alpha) / (1.0 + n)) ** (1.0 / (1.0 - alpha)) * (1.0 - 1e-9)
    return float(brentq(euler, 1e-12, upper, xtol=1e-14, rtol=1e-14))


ANALYTIC = {
    "steady_state_capital": steady_state_capital,
    "euler_steady_state": euler_steady_state,
}


def build(**overrides: Any) -> Model:
    """OLG model as a deterministic ``dsge`` problem in capital per young worker."""
    opts = split_overrides(overrides, {k: v.value for k, v in PARAMS.items()}, "model")
    m = Model("olg2")
    m.params(**{k: Param(opts[k], p.bounds, p.free, p.description) for k, p in PARAMS.items()})
    m.set_problem(
        "dsge",
        variables=["k"],
        states=["k"],
        shocks=[],
        log_variables=["k"],
        equations=lambda yp, y, p: [
            (1.0 + p.n) * yp.k - p.beta / (1.0 + p.beta) * (1.0 - p.alpha) * y.k**p.alpha
        ],
        shock_loadings=lambda p: np.zeros((1, 0)),
        steady_state=lambda p: np.array([steady_state_capital(p.alpha, p.beta, p.n)]),
    )
    return m


register("olg2", build)
