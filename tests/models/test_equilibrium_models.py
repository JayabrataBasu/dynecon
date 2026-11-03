"""Aiyagari and Huggett equilibria: structural properties from the plan (P8.4)."""

import pytest

from dynecon.models import aiyagari, huggett

COMPLETE = 1 / 0.96 - 1
KW = dict(sigma=2.0, rho=0.9, n_a=80, n_z=5)


@pytest.fixture(scope="module")
def aiy():
    return aiyagari.equilibrium(sigma_e=0.2, **KW)


def test_precautionary_saving_and_market_clearing(aiy) -> None:
    assert aiy.price < COMPLETE
    assert aiy.price > -0.08
    agg = aiy.aggregates
    assert abs(agg["capital_supply"] - agg["capital_demand"]) / agg["capital_demand"] < 1e-6
    capital_output = agg["capital_demand"] / agg["output"]
    assert abs(capital_output - 0.36 / (aiy.price + 0.08)) < 1e-8
    assert agg["mass_at_top"] < 1e-6  # grid upper bound is not binding


def test_risk_lowers_the_interest_rate() -> None:
    low = aiyagari.equilibrium(sigma_e=0.01, **KW).price
    mid = aiyagari.equilibrium(sigma_e=0.1, **KW).price
    high = aiyagari.equilibrium(sigma_e=0.2, **KW).price
    assert low > mid > high
    assert COMPLETE - low < 5e-4  # near the complete-markets rate for tiny risk
    assert low < COMPLETE


def test_warm_start_reduces_inner_iterations() -> None:
    kw = dict(sigma_e=0.1, **KW)
    warm = aiyagari.equilibrium(**kw)
    from dynecon.solvers.equilibrium import solve_stationary_equilibrium

    cold = solve_stationary_equilibrium(
        lambda r: aiyagari.build(r=r, **kw).compile(),
        "r",
        (-0.08 + 1e-4, COMPLETE - 1e-6),
        lambda r, sol, mu: (
            float((mu * sol.grids["assets"][:, None]).sum())
            - aiyagari.capital_demand(r, 0.36, 0.08),
            {},
        ),
        tol=1e-10,
        warm_start=False,
    )
    assert warm.history["inner_iterations_total"][0] < cold.history["inner_iterations_total"][0]
    assert abs(warm.price - cold.price) < 1e-8


def test_huggett_clears_bonds_and_borrowing_limit_matters() -> None:
    loose = huggett.equilibrium(n_a=80, n_z=5, phi=2.0)
    tight = huggett.equilibrium(n_a=80, n_z=5, phi=0.5)
    assert loose.price < COMPLETE
    assert abs(loose.aggregates["bond_holdings"]) < 1e-6
    assert tight.price < loose.price
    assert loose.aggregates["mass_at_top"] < 1e-6


def test_wealth_more_unequal_than_income() -> None:
    from dynecon.simulation import gini, marginal_distribution

    eq = huggett.equilibrium(n_a=80, n_z=5, phi=2.0)
    a = eq.grids["assets"]
    z = eq.model.shocks["income"].values
    wealth_gini = gini(a - a[0], marginal_distribution(eq.distribution, 0))
    income_gini = gini(z, marginal_distribution(eq.distribution, 1))
    assert wealth_gini > income_gini


def test_equilibrium_rejects_price_override() -> None:
    from dynecon.core.errors import ValidationError

    with pytest.raises(ValidationError):
        aiyagari.equilibrium(r=0.03)
    with pytest.raises(ValidationError):
        huggett.equilibrium(r=0.03)


@pytest.mark.slow
def test_full_grid_aiyagari_converges() -> None:
    eq = aiyagari.equilibrium(sigma=3.0, n_a=500, n_z=7)
    assert eq.converged and eq.price < COMPLETE
    assert eq.elapsed < 60.0
