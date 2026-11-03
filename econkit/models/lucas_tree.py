"""Lucas (1978) tree: asset pricing in an endowment economy.

Economics note:
A representative household consumes the dividend of a tree. With log utility the
price-dividend ratio is constant at ``beta / (1 - beta)`` whatever the dividend process:
income and substitution effects of dividend news exactly offset.
"""

from __future__ import annotations

from typing import Any

from econkit.model import AR1Shock, Model, Param, State
from econkit.models.registry import register, split_overrides
from econkit.models.utility import crra_marginal

__all__ = [
    "ANALYTIC",
    "CALIBRATION_SOURCE",
    "DEFAULTS",
    "PARAMS",
    "build",
    "log_utility_price_dividend",
]

CALIBRATION_SOURCE = "Lucas (1978); textbook log-AR(1) dividend calibration."

PARAMS: dict[str, Param] = {
    "beta": Param(0.96, bounds=(0.5, 0.9999), description="Discount factor"),
    "sigma": Param(1.0, bounds=(0.2, 10.0), description="CRRA coefficient (1 = log)"),
    "rho": Param(0.9, bounds=(0.0, 0.99), description="Dividend persistence (log AR(1))"),
    "sigma_e": Param(0.05, bounds=(0.001, 0.5), description="Dividend innovation std"),
}
DEFAULTS: dict[str, Any] = {"n_z": 9}


def log_utility_price_dividend(beta: float) -> float:
    """Price-dividend ratio under log utility: ``beta / (1 - beta)``."""
    return float(beta / (1.0 - beta))


ANALYTIC = {"log_utility_price_dividend": log_utility_price_dividend}


def build(**overrides: Any) -> Model:
    """Lucas tree as a ``pricing`` problem with a log-AR(1) dividend."""
    opts = split_overrides(
        overrides, {**{k: v.value for k, v in PARAMS.items()}, **DEFAULTS}, "model"
    )
    m = Model("lucas_tree")
    m.params(**{k: Param(opts[k], p.bounds, p.free, p.description) for k, p in PARAMS.items()})
    m.add_state(State("dividend", shock=AR1Shock("rho", "sigma_e", opts["n_z"], log=True)))
    m.set_problem(
        "pricing",
        shock="dividend",
        discount="beta",
        marginal_utility=lambda c, p: crra_marginal(c, p.sigma),
    )
    return m


register("lucas_tree", build)
