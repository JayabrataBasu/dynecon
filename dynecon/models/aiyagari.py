"""Aiyagari (1994) economy: incomplete markets, idiosyncratic income risk, capital market.

Economics note:
Households self-insure against uninsurable income risk by holding capital. Precautionary
saving pushes the equilibrium interest rate below the complete-markets rate ``1/beta - 1``,
and more income risk lowers it further.
"""

from __future__ import annotations

from typing import Any

from dynecon.core.errors import ValidationError
from dynecon.model import AR1Shock, Control, Model, Param, State
from dynecon.models._equilibrium_helpers import asset_upper_bound
from dynecon.models.registry import register, split_overrides
from dynecon.models.utility import crra, crra_marginal, crra_marginal_inverse
from dynecon.simulation.distribution import aggregate
from dynecon.solvers.equilibrium import EquilibriumSolution, solve_stationary_equilibrium

__all__ = [
    "ANALYTIC",
    "CALIBRATION_SOURCE",
    "DEFAULTS",
    "PARAMS",
    "build",
    "capital_demand",
    "equilibrium",
    "excess_demand",
    "wage",
]

CALIBRATION_SOURCE = "Aiyagari (1994) Table II style (sigma=3, rho=0.6, sigma_e=0.2); textbook."

PARAMS: dict[str, Param] = {
    "beta": Param(0.96, bounds=(0.5, 0.9999), description="Discount factor"),
    "sigma": Param(3.0, bounds=(0.2, 10.0), description="CRRA coefficient"),
    "alpha": Param(0.36, bounds=(0.05, 0.95), description="Capital share"),
    "delta": Param(0.08, bounds=(0.0, 1.0), description="Depreciation"),
    "rho": Param(0.6, bounds=(0.0, 0.99), description="Income persistence"),
    "sigma_e": Param(0.2, bounds=(0.01, 1.0), description="Income innovation sd"),
}
# a_max=None: upper asset bound from beta and r (asset_upper_bound); a number fixes the grid.
DEFAULTS: dict[str, Any] = {"r": 0.03, "n_a": 200, "n_z": 7, "a_max": None}


def capital_demand(r: Any, alpha: Any, delta: Any, labour: float = 1.0) -> Any:
    """Firm demand ``K_d = L ((r + delta) / alpha)^(1/(alpha - 1))``."""
    return labour * ((r + delta) / alpha) ** (1.0 / (alpha - 1.0))


def wage(r: Any, alpha: Any, delta: Any) -> Any:
    """Wage ``(1 - alpha) (K/L)^alpha`` implied by the interest rate (``L = 1``)."""
    return (1.0 - alpha) * capital_demand(r, alpha, delta) ** alpha


ANALYTIC = {
    "capital_demand": capital_demand,
    "wage": wage,
    "complete_markets_rate": lambda beta: 1.0 / beta - 1.0,
}


def build(**overrides: Any) -> Model:
    """Household problem at interest rate ``r`` (wage from the firm's first-order conditions)."""
    opts = split_overrides(
        overrides, {**{k: v.value for k, v in PARAMS.items()}, **DEFAULTS}, "model"
    )
    r = float(opts["r"])
    m = Model("aiyagari")
    m.params(**{k: Param(opts[k], p.bounds, p.free, p.description) for k, p in PARAMS.items()})
    m.params(r=Param(r, bounds=(-opts["delta"], 1.0), description="Interest rate (price)"))
    m.add_state(
        State(
            "assets",
            bounds=(0.0, opts["a_max"] or asset_upper_bound(opts["beta"], r)),
            grid_type="exp",
            n_points=opts["n_a"],
            grid_curvature=4.0,
        )
    )
    m.add_state(
        State("income", shock=AR1Shock("rho", "sigma_e", opts["n_z"], log=True, normalize=True))
    )
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
        resources=lambda a, zz, p: (1.0 + p.r) * a + wage(p.r, p.alpha, p.delta) * zz,
        marginal_return=lambda a, zz, p: 1.0 + p.r,
    )
    return m


def excess_demand(mu: Any, _c: Any, a_next: Any, _a_grid: Any, _z: Any, p: Any) -> Any:
    """Capital excess demand ``E[a'] - K_d(r)``; JAX-traceable (see ``equilibrium_moments``)."""
    return (mu * a_next).sum() - capital_demand(p.r, p.alpha, p.delta)


def equilibrium(**overrides: Any) -> EquilibriumSolution:
    """Stationary equilibrium interest rate, aggregates and wealth distribution."""
    opts = split_overrides(
        overrides, {**{k: v.value for k, v in PARAMS.items()}, **DEFAULTS}, "model"
    )
    if "r" in overrides:
        raise ValidationError("r is the equilibrium price; do not override it")
    kw = {k: v for k, v in opts.items() if k != "r"}
    alpha, delta, beta = opts["alpha"], opts["delta"], opts["beta"]

    def excess(r: float, sol: Any, mu: Any) -> tuple[float, dict[str, float]]:
        supply = aggregate(mu, sol.grids["assets"][:, None])
        demand = capital_demand(r, alpha, delta)
        return supply - demand, {
            "capital_supply": supply,
            "capital_demand": demand,
            "output": demand**alpha,
            "wage": wage(r, alpha, delta),
            "mass_at_top": float(mu[-1].sum()),
        }

    return solve_stationary_equilibrium(
        lambda r: build(r=r, **kw).compile(),
        "r",
        (-delta + 1e-4, 1.0 / beta - 1.0 - 1e-6),
        excess,
        tol=1e-10,
    )


register("aiyagari", build)
