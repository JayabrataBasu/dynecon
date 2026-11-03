"""Three-equation New Keynesian model.

Economics note:
A Phillips curve (inflation), a dynamic IS curve (output gap) and an interest-rate rule.
The Taylor principle (``phi_pi > 1`` when ``phi_x = 0``) is necessary for a unique
equilibrium; otherwise sunspot fluctuations are possible.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from econkit.model import Model, Param
from econkit.models.registry import register, split_overrides

__all__ = ["ANALYTIC", "CALIBRATION_SOURCE", "PARAMS", "build", "closed_form_coefficients"]

CALIBRATION_SOURCE = "Textbook (Galí 2015 style) quarterly calibration."

PARAMS: dict[str, Param] = {
    "beta": Param(0.99, bounds=(0.5, 0.9999), description="Discount factor"),
    "kappa": Param(0.15, bounds=(0.001, 5.0), description="Phillips curve slope"),
    "sigma": Param(1.0, bounds=(0.1, 10.0), description="Inverse EIS"),
    "phi_pi": Param(1.5, bounds=(0.0, 10.0), description="Taylor rule on inflation"),
    "phi_x": Param(0.125, bounds=(0.0, 10.0), description="Taylor rule on output gap"),
    "rho_v": Param(0.5, bounds=(0.0, 0.999), description="Policy shock persistence"),
}


def closed_form_coefficients(
    beta: float, kappa: float, sigma: float, phi_pi: float, rho_v: float
) -> tuple[float, float]:
    """``(a, b)`` with ``pi = a v`` and ``x = b v`` when ``phi_x = 0``."""
    b = -1.0 / (sigma * (1.0 - rho_v) + kappa * (phi_pi - rho_v) / (1.0 - beta * rho_v))
    return kappa * b / (1.0 - beta * rho_v), b


ANALYTIC = {"closed_form_coefficients": closed_form_coefficients}


def build(**overrides: Any) -> Model:
    """Three-equation NK model as a ``dsge`` problem (variables v, pi, x, i)."""
    opts = split_overrides(overrides, {k: v.value for k, v in PARAMS.items()}, "model")
    m = Model("nk3")
    m.params(**{k: Param(opts[k], p.bounds, p.free, p.description) for k, p in PARAMS.items()})
    m.set_problem(
        "dsge",
        variables=["v", "pi", "x", "i"],
        states=["v"],
        shocks=["eps_v"],
        equations=lambda yp, y, p: [
            yp.v - p.rho_v * y.v,
            y.pi - p.beta * yp.pi - p.kappa * y.x,
            y.x - yp.x + (y.i - yp.pi) / p.sigma,
            y.i - p.phi_pi * y.pi - p.phi_x * y.x - y.v,
        ],
        shock_loadings=lambda p: np.array([[1.0]]),
        steady_state=lambda p: np.zeros(4),
    )
    return m


register("nk3", build)
