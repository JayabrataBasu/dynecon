"""Real business cycle model with inelastic labour.

Economics note:
A representative household consumes and accumulates capital; productivity follows a log
AR(1). The model has a known steady state (``k/y = alpha / (1/beta - 1 + delta)``) and is
the workhorse of quantitative macroeconomics, solved here by perturbation or EGM.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from econkit.core.errors import ValidationError
from econkit.model import AR1Shock, Control, Model, Param, State
from econkit.models.registry import register, split_overrides
from econkit.models.utility import crra, crra_marginal, crra_marginal_inverse, log

__all__ = ["ANALYTIC", "CALIBRATION_SOURCE", "DEFAULTS", "PARAMS", "build", "steady_state"]

CALIBRATION_SOURCE = "Textbook quarterly calibration (e.g. Cooley and Prescott 1995 style)."

PARAMS: dict[str, Param] = {
    "alpha": Param(0.33, bounds=(0.05, 0.95), description="Capital share"),
    "beta": Param(0.99, bounds=(0.5, 0.9999), description="Discount factor"),
    "delta": Param(0.025, bounds=(0.001, 1.0), description="Depreciation rate"),
    "sigma": Param(1.0, bounds=(0.2, 10.0), description="CRRA coefficient (1 = log)"),
    "rho": Param(0.95, bounds=(0.0, 0.999), description="TFP persistence"),
    "sigma_e": Param(0.007, bounds=(0.0001, 0.5), description="TFP innovation std"),
}
DEFAULTS: dict[str, Any] = {"problem": "dsge", "n_k": 200, "n_z": 7}


def steady_state(alpha: float, beta: float, delta: float) -> dict[str, float]:
    """Deterministic steady state: capital, output, investment, consumption."""
    k = (alpha * beta / (1.0 - beta * (1.0 - delta))) ** (1.0 / (1.0 - alpha))
    y = k**alpha
    return {"k": k, "y": y, "i": delta * k, "c": y - delta * k, "k_over_y": k / y}


ANALYTIC = {"steady_state": steady_state}


def _ss_vector(p: Any) -> np.ndarray:
    s = steady_state(p.alpha, p.beta, p.delta)
    return np.array([s["k"], 1.0, s["c"], s["y"], s["i"]])


def build(**overrides: Any) -> Model:
    """RBC model as a ``dsge`` problem (default) or a ``saving`` problem (solved by EGM)."""
    opts = split_overrides(
        overrides, {**{k: v.value for k, v in PARAMS.items()}, **DEFAULTS}, "model"
    )
    m = Model("rbc")
    m.params(**{k: Param(opts[k], p.bounds, p.free, p.description) for k, p in PARAMS.items()})
    form = opts["problem"]
    if form == "dsge":
        m.set_problem(
            "dsge",
            variables=["k", "z", "c", "y", "i"],
            states=["k", "z"],
            shocks=["eps_z"],
            log_variables=["k", "z", "c", "y", "i"],
            equations=lambda yp, y, p: [
                yp.k - (1.0 - p.delta) * y.k - y.i,
                log(yp.z) - p.rho * log(y.z),
                y.c ** (-p.sigma)
                - p.beta
                * yp.c ** (-p.sigma)
                * (p.alpha * yp.z * yp.k ** (p.alpha - 1.0) + 1.0 - p.delta),
                y.y - y.z * y.k**p.alpha,
                y.i - (y.y - y.c),
            ],
            shock_loadings=lambda p: np.array([[0.0], [p.sigma_e]]),
            steady_state=_ss_vector,
        )
    elif form == "saving":
        s = steady_state(opts["alpha"], opts["beta"], opts["delta"])
        m.add_state(State("k", bounds=(0.05 * s["k"], 2.0 * s["k"]), n_points=opts["n_k"]))
        m.add_state(State("z", shock=AR1Shock("rho", "sigma_e", opts["n_z"], log=True)))
        m.add_control(Control("c"))
        m.set_problem(
            "saving",
            asset="k",
            shock="z",
            control="c",
            discount="beta",
            utility=lambda c, p: crra(c, p.sigma),
            marginal_utility=lambda c, p: crra_marginal(c, p.sigma),
            inverse_marginal_utility=lambda x, p: crra_marginal_inverse(x, p.sigma),
            resources=lambda k, z, p: z * k**p.alpha + (1.0 - p.delta) * k,
            marginal_return=lambda k, z, p: p.alpha * z * k ** (p.alpha - 1.0) + 1.0 - p.delta,
        )
    else:
        raise ValidationError("problem must be 'dsge' or 'saving'")
    return m


register("rbc", build)
