"""Stationary-equilibrium engine: plumbing, errors and warm starts."""

import numpy as np
import pytest
from builders import consumption_savings

import econkit as ek
from econkit.core.errors import ConvergenceError, ValidationError
from econkit.simulation import aggregate, stationary_distribution_young
from econkit.solvers.equilibrium import EquilibriumSolution, solve_stationary_equilibrium


def factory(r: float):
    return consumption_savings(r=r, n_a=60, n_z=3, a_max=40.0).compile()


def assets_supply(sol, mu) -> float:
    return aggregate(mu, sol.grids["assets"][:, None])


def test_trivial_root_is_found() -> None:
    eq = solve_stationary_equilibrium(
        factory, "r", (0.0, 0.03), lambda p, sol, mu: (p - 0.0123, {"p": p}), tol=1e-10
    )
    assert isinstance(eq, EquilibriumSolution)
    assert abs(eq.price - 0.0123) < 1e-9 and eq.converged
    assert eq.household_solution is not None and eq.distribution is not None
    assert abs(eq.distribution.sum() - 1.0) < 1e-12
    assert eq.aggregates["p"] == pytest.approx(eq.price)
    assert eq.price_name == "r" and len(eq.history["price"]) >= 3
    assert eq.method.startswith("equilibrium")


def test_no_sign_change_raises_with_endpoints() -> None:
    with pytest.raises(ConvergenceError, match="no sign change") as err:
        solve_stationary_equilibrium(factory, "r", (0.0, 0.03), lambda p, sol, mu: (1.0 + p, {}))
    assert "lower end" in str(err.value)
    with pytest.raises(ValidationError):
        solve_stationary_equilibrium(factory, "r", (0.03, 0.0), lambda p, sol, mu: (p, {}))


def test_market_clearing_and_warm_start_saves_iterations() -> None:
    def supply(r: float) -> float:
        sol = ek.solve(factory(r))
        return assets_supply(sol, stationary_distribution_young(sol))

    target = 0.5 * (supply(0.0) + supply(0.03))

    def excess(p, sol, mu):
        a = assets_supply(sol, mu)
        return a - target, {"assets": a}

    warm = solve_stationary_equilibrium(factory, "r", (0.0, 0.03), excess, tol=1e-6)
    cold = solve_stationary_equilibrium(
        factory, "r", (0.0, 0.03), excess, tol=1e-6, warm_start=False
    )
    assert abs(warm.excess_demand) < 1e-4 * target
    assert abs(warm.price - cold.price) < 1e-6
    assert warm.history["inner_iterations_total"][0] < cold.history["inner_iterations_total"][0]
    np.testing.assert_allclose(warm.policy["consumption"], cold.policy["consumption"], atol=1e-6)


def test_warm_start_shape_mismatch_ignored() -> None:
    model = consumption_savings(n_a=30, n_z=3).compile()
    sol = ek.solve(model, method="egm", initial=np.ones((5, 5)))
    assert sol.converged
    sol2 = ek.solve(model, method="vfi", initial=np.zeros((30, 3)))
    assert sol2.converged
