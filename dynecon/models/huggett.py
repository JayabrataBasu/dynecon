"""Huggett (1993) bond economy: incomplete markets with a zero-net-supply bond.

Economics note:
Households trade a risk-free bond that is in zero net supply; the equilibrium interest
rate makes aggregate bond holdings zero. Precautionary saving demands lie below the
complete-markets rate, and a tighter borrowing limit lowers the equilibrium rate further.
"""

from __future__ import annotations

from typing import Any

from dynecon.core.errors import ValidationError
from dynecon.model import Control, MarkovShock, Model, Param, State
from dynecon.models._equilibrium_helpers import asset_upper_bound, unit_mean_income_chain
from dynecon.models.registry import register, split_overrides
from dynecon.models.utility import crra, crra_marginal, crra_marginal_inverse
from dynecon.simulation.distribution import aggregate
from dynecon.solvers.equilibrium import EquilibriumSolution, solve_stationary_equilibrium

__all__ = ["ANALYTIC", "CALIBRATION_SOURCE", "DEFAULTS", "PARAMS", "build", "equilibrium"]

CALIBRATION_SOURCE = "Huggett (1993) style; textbook calibration (annual)."

PARAMS: dict[str, Param] = {
    "beta": Param(0.96, bounds=(0.5, 0.9999), description="Discount factor"),
    "sigma": Param(2.0, bounds=(0.2, 10.0), description="CRRA coefficient"),
}
DEFAULTS: dict[str, Any] = {
    "r": 0.02,
    "rho": 0.9,
    "sigma_e": 0.2,
    "phi": 2.0,
    "n_a": 200,
    "n_z": 7,
}

ANALYTIC = {"complete_markets_rate": lambda beta: 1.0 / beta - 1.0}


def build(**overrides: Any) -> Model:
    """Household problem at bond return ``r`` with borrowing limit ``-phi`` (mean income = 1)."""
    opts = split_overrides(
        overrides, {**{k: v.value for k, v in PARAMS.items()}, **DEFAULTS}, "model"
    )
    r = float(opts["r"])
    z, P = unit_mean_income_chain(opts["rho"], opts["sigma_e"], opts["n_z"])
    m = Model("huggett")
    m.params(**{k: Param(opts[k], p.bounds, p.free, p.description) for k, p in PARAMS.items()})
    m.params(r=Param(r, bounds=(-0.5, 1.0), description="Net bond return (price)"))
    m.add_state(
        State(
            "assets",
            bounds=(-opts["phi"], asset_upper_bound(opts["beta"], r, floor=20.0)),
            grid_type="exp",
            n_points=opts["n_a"],
            grid_curvature=4.0,
        )
    )
    m.add_state(State("income", shock=MarkovShock(z, P)))
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
        resources=lambda a, zz, p: (1.0 + p.r) * a + zz,
        marginal_return=lambda a, zz, p: 1.0 + p.r,
    )
    return m


def equilibrium(**overrides: Any) -> EquilibriumSolution:
    """Bond-market clearing return and the stationary wealth distribution."""
    opts = split_overrides(
        overrides, {**{k: v.value for k, v in PARAMS.items()}, **DEFAULTS}, "model"
    )
    if "r" in overrides:
        raise ValidationError("r is the equilibrium price; do not override it")
    kw = {k: v for k, v in opts.items() if k != "r"}

    def excess(r: float, sol: Any, mu: Any) -> tuple[float, dict[str, float]]:
        bonds = aggregate(mu, sol.grids["assets"][:, None])
        return bonds, {"bond_holdings": bonds, "mass_at_top": float(mu[-1].sum())}

    return solve_stationary_equilibrium(
        lambda r: build(r=r, **kw).compile(),
        "r",
        (-0.1, 1.0 / opts["beta"] - 1.0 - 1e-6),
        excess,
        tol=1e-10,
    )


register("huggett", build)
