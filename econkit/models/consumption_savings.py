"""Income-fluctuation (consumption-savings) problem with a borrowing limit.

Economics note:
A household with CRRA preferences earns stochastic income, saves in one risk-free asset
at rate ``r`` and cannot borrow below ``a_min``. Precautionary saving and the
borrowing constraint make consumption concave in wealth.
"""

from __future__ import annotations

from typing import Any

from econkit.model import AR1Shock, Control, Model, Param, State
from econkit.models.registry import register, split_overrides
from econkit.models.utility import crra, crra_marginal, crra_marginal_inverse

__all__ = ["ANALYTIC", "CALIBRATION_SOURCE", "DEFAULTS", "PARAMS", "build", "mpc_perfect_foresight"]

CALIBRATION_SOURCE = "Textbook calibration (annual); not tied to a specific paper."

PARAMS: dict[str, Param] = {
    "beta": Param(0.96, bounds=(0.5, 0.9999), description="Discount factor"),
    "sigma": Param(2.0, bounds=(0.2, 10.0), description="CRRA coefficient"),
    "r": Param(0.03, bounds=(-0.1, 0.2), description="Net interest rate"),
    "rho": Param(0.9, bounds=(0.0, 0.99), description="Income persistence (log AR(1))"),
    "sigma_e": Param(0.1, bounds=(0.001, 1.0), description="Income innovation std"),
}
DEFAULTS: dict[str, Any] = {
    "a_min": 0.0,
    "a_max": 50.0,
    "n_a": 200,
    "n_z": 7,
    "curvature": 3.0,
}


def mpc_perfect_foresight(beta: float, sigma: float, r: float) -> float:
    """Marginal propensity to consume out of wealth, infinite horizon, no income risk."""
    return float(1.0 - beta ** (1.0 / sigma) * (1.0 + r) ** ((1.0 - sigma) / sigma))


ANALYTIC = {"mpc_perfect_foresight": mpc_perfect_foresight}


def build(**overrides: Any) -> Model:
    """Consumption-savings model; override parameters or ``a_min, a_max, n_a, n_z, curvature``."""
    opts = split_overrides(
        overrides, {**{k: v.value for k, v in PARAMS.items()}, **DEFAULTS}, "model"
    )
    m = Model("consumption_savings")
    m.params(**{k: Param(opts[k], p.bounds, p.free, p.description) for k, p in PARAMS.items()})
    m.add_state(
        State(
            "assets",
            bounds=(opts["a_min"], opts["a_max"]),
            grid_type="exp",
            n_points=opts["n_a"],
            grid_curvature=opts["curvature"],
        )
    )
    m.add_state(State("income", shock=AR1Shock("rho", "sigma_e", opts["n_z"], log=True)))
    m.add_control(Control("consumption"))
    m.set_problem(
        "saving",
        asset="assets",
        shock="income",
        control="consumption",
        discount="beta",
        utility=lambda c, p: crra(c, p.sigma),
        marginal_utility=lambda c, p: crra_marginal(c, p.sigma),
        inverse_marginal_utility=lambda x, p: crra_marginal_inverse(x, p.sigma),
        resources=lambda a, z, p: (1.0 + p.r) * a + z,
        marginal_return=lambda a, z, p: 1.0 + p.r,
    )
    return m


register("consumption_savings", build)
