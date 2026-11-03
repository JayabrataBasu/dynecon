"""Brock-Mirman stochastic growth: log utility, Cobb-Douglas, full depreciation.

Economics note:
With log utility and full depreciation the planner's problem has a closed form: a constant
fraction ``alpha * beta`` of output is invested, ``k' = alpha beta z k^alpha``. It is the
canonical test case for every global and local solution method.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dynecon.core.errors import ValidationError
from dynecon.model import AR1Shock, Control, Model, Param, State
from dynecon.models.registry import register, split_overrides
from dynecon.models.utility import log

__all__ = ["ANALYTIC", "CALIBRATION_SOURCE", "DEFAULTS", "PARAMS", "build", "steady_state_capital"]

CALIBRATION_SOURCE = "Brock and Mirman (1972); textbook calibration."

PARAMS: dict[str, Param] = {
    "alpha": Param(0.36, bounds=(0.05, 0.95), description="Capital share"),
    "beta": Param(0.96, bounds=(0.5, 0.9999), description="Discount factor"),
    "rho": Param(0.9, bounds=(0.0, 0.99), description="TFP persistence (log AR(1))"),
    "sigma_e": Param(0.02, bounds=(0.001, 0.5), description="TFP innovation std"),
}
DEFAULTS: dict[str, Any] = {"problem": "saving", "n_k": 200, "n_z": 7}


def steady_state_capital(alpha: float, beta: float) -> float:
    """Deterministic steady-state capital ``(alpha beta)^(1/(1-alpha))``."""
    return float((alpha * beta) ** (1.0 / (1.0 - alpha)))


def _policy_capital(k: Any, z: Any, alpha: float, beta: float) -> Any:
    return alpha * beta * z * k**alpha


def _policy_consumption(k: Any, z: Any, alpha: float, beta: float) -> Any:
    return (1.0 - alpha * beta) * z * k**alpha


ANALYTIC = {"capital_policy": _policy_capital, "consumption_policy": _policy_consumption}


def build(**overrides: Any) -> Model:
    """Brock-Mirman model as a ``saving`` problem (default) or a ``dsge`` problem.

    ``problem="dsge"`` declares the log-linearisable system for perturbation; override
    parameters or ``n_k``, ``n_z`` (grid sizes of the ``saving`` form).
    """
    opts = split_overrides(
        overrides, {**{k: v.value for k, v in PARAMS.items()}, **DEFAULTS}, "model"
    )
    form = opts["problem"]
    m = Model("brock_mirman")
    m.params(**{k: Param(opts[k], p.bounds, p.free, p.description) for k, p in PARAMS.items()})
    if form == "saving":
        kstar = steady_state_capital(opts["alpha"], opts["beta"])
        m.add_state(State("k", bounds=(0.05 * kstar, 2.0 * kstar), n_points=opts["n_k"]))
        m.add_state(State("z", shock=AR1Shock("rho", "sigma_e", opts["n_z"], log=True)))
        m.add_control(Control("c"))
        m.set_problem(
            "saving",
            asset="k",
            shock="z",
            control="c",
            discount="beta",
            utility=lambda c, p: np.log(c),
            marginal_utility=lambda c, p: 1.0 / c,
            inverse_marginal_utility=lambda x, p: 1.0 / x,
            resources=lambda k, z, p: z * k**p.alpha,
            marginal_return=lambda k, z, p: p.alpha * z * k ** (p.alpha - 1.0),
        )
    elif form == "dsge":
        m.set_problem(
            "dsge",
            variables=["k", "z", "c"],
            states=["k", "z"],
            shocks=["eps_z"],
            log_variables=["k", "z", "c"],
            equations=lambda yp, y, p: [
                yp.k - y.z * y.k**p.alpha + y.c,
                log(yp.z) - p.rho * log(y.z),
                1.0 / y.c - p.beta * p.alpha * yp.z * yp.k ** (p.alpha - 1.0) / yp.c,
            ],
            shock_loadings=lambda p: np.array([[0.0], [p.sigma_e]]),
            steady_state=lambda p: _steady_state(p.alpha, p.beta),
        )
    else:
        raise ValidationError("problem must be 'saving' or 'dsge'")
    return m


def _steady_state(alpha: float, beta: float) -> np.ndarray:
    k = steady_state_capital(alpha, beta)
    return np.array([k, 1.0, k**alpha - k])


register("brock_mirman", build)
