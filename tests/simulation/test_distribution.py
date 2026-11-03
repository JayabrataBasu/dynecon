"""Young histogram distributions, aggregates and inequality."""

import math

import numpy as np
import pytest
from builders import consumption_savings, permanent_income
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm

import dynecon as ek
from dynecon.core.errors import ValidationError
from dynecon.numerics.markov import stationary_distribution
from dynecon.simulation import (
    aggregate,
    build_transition_matrix,
    cdf,
    gini,
    iterate_distribution,
    lorenz_curve,
    marginal_distribution,
    quantile,
    stationary_distribution_young,
)
from dynecon.simulation.distribution import lottery_weights
from dynecon.simulation.kernels import distribution_lottery_numpy


@pytest.fixture(scope="module")
def sol():
    return ek.solve(consumption_savings(n_a=100, n_z=5, a_max=60.0).compile())


def test_mass_conserved_and_lottery_preserves_policy_mean(sol) -> None:
    mu = np.random.default_rng(0).random(sol.policy["assets_next"].shape)
    mu /= mu.sum()
    for _ in range(3):
        mu = iterate_distribution(mu, sol, 1)
        assert abs(mu.sum() - 1.0) < 1e-14 and mu.min() >= 0
    a = sol.grids["assets"]
    k, w, clamped = lottery_weights(sol)
    assert clamped == 0
    mean_next = w * a[k] + (1 - w) * a[k + 1]
    np.testing.assert_allclose(mean_next, sol.policy["assets_next"], atol=1e-13)


def test_stationary_matches_sparse_matrix_and_shock_marginal(sol) -> None:
    mu = stationary_distribution_young(sol)
    T = build_transition_matrix(sol)
    np.testing.assert_allclose(np.asarray(T.sum(axis=1)).ravel(), 1.0, atol=1e-13)
    np.testing.assert_allclose((T.T @ mu.ravel()).reshape(mu.shape), mu, atol=1e-11)
    pi = stationary_distribution(sol.model.shocks["income"].P)
    np.testing.assert_allclose(marginal_distribution(mu, 1), pi, atol=1e-10)
    assert marginal_distribution(mu, 0).shape == (100,)
    with pytest.raises(ValidationError):
        marginal_distribution(mu, 2)


@pytest.mark.numba
def test_numba_lottery_parity(sol) -> None:
    pytest.importorskip("numba")
    mu = np.random.default_rng(1).random(sol.policy["assets_next"].shape)
    mu /= mu.sum()
    a = iterate_distribution(mu, sol, 5, backend="numpy")
    b = iterate_distribution(mu, sol, 5, backend="numba")
    np.testing.assert_allclose(a, b, atol=1e-14)
    T = build_transition_matrix(sol)
    ref = mu.ravel()
    for _ in range(5):
        ref = T.T @ ref
    np.testing.assert_allclose(a.ravel(), ref, atol=1e-13)


def test_permanent_income_distribution_invariant() -> None:
    sol = ek.solve(permanent_income(n_a=41, a_max=4.0).compile())
    mu = np.random.default_rng(2).random((41, 1))
    mu /= mu.sum()
    # a'(a) = a only up to the EGM solve tolerance (policy error ~1e-8), so mass moves ~1e-8
    np.testing.assert_allclose(iterate_distribution(mu, sol, 10), mu, atol=1e-6)


def test_agreement_with_monte_carlo(sol) -> None:
    a = sol.grids["assets"]
    mu = stationary_distribution_young(sol)
    exact = aggregate(mu, a[:, None])
    sim = ek.simulate(sol, n_periods=1500, n_agents=2000, burn_in=500, rng=0)
    per_agent = sim.paths["assets"].mean(axis=0)
    se = per_agent.std(ddof=1) / math.sqrt(per_agent.size)
    assert abs(per_agent.mean() - exact) < 4 * se


def test_iterate_validation(sol) -> None:
    with pytest.raises(ValidationError):
        iterate_distribution(np.ones((3, 3)), sol)


def test_quantile_and_cdf() -> None:
    grid = np.array([1.0, 2.0, 3.0, 4.0])
    mass = np.array([0.1, 0.4, 0.4, 0.1])
    assert cdf(mass)[-1] == pytest.approx(1.0)
    assert quantile(mass, grid, 0.05) == 1.0
    assert quantile(mass, grid, 0.5) == 2.0
    assert quantile(mass, grid, 0.6) == 3.0
    assert quantile(mass, grid, 1.0) == 4.0
    with pytest.raises(ValidationError):
        quantile(mass, grid, 1.5)


def test_gini_known_cases() -> None:
    n = 4000
    uniform = (np.arange(n) + 0.5) / n
    assert abs(gini(uniform) - 1 / 3) < 1e-4
    assert gini(np.ones(10)) == 0.0
    sigma = 0.8
    p = (np.arange(2000) + 0.5) / 2000
    ln = np.exp(sigma * norm.ppf(p))
    assert abs(gini(ln) - (2 * norm.cdf(sigma / math.sqrt(2)) - 1)) < 5e-3
    assert abs(gini([1.0, 3.0], weights=[1.0, 1.0]) - 0.25) < 1e-12


def test_lorenz_properties_and_validation() -> None:
    pop, share = lorenz_curve(np.random.default_rng(0).lognormal(size=500))
    assert pop[0] == 0 and share[0] == 0 and pop[-1] == 1 and abs(share[-1] - 1) < 1e-12
    assert np.all(np.diff(share, 2) >= -1e-12)  # convex
    assert np.all(share <= pop + 1e-12)
    with pytest.raises(ValidationError):
        lorenz_curve([-1.0, 1.0])
    with pytest.raises(ValidationError):
        lorenz_curve([1.0, 2.0], [1.0])
    with pytest.raises(ValidationError):
        lorenz_curve([0.0, 0.0])


@settings(deadline=None, max_examples=60, derandomize=True)
@given(n_a=st.integers(3, 30), n_z=st.integers(1, 4), seed=st.integers(0, 5000))
def test_lottery_conserves_mass_and_policy_mean(n_a: int, n_z: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    grid = np.sort(rng.uniform(0, 10, n_a)) + np.arange(n_a) * 0.01
    target = rng.uniform(grid[0], grid[-1], (n_a, n_z))
    k = np.clip(np.searchsorted(grid, target, side="right") - 1, 0, n_a - 2).astype(np.int64)
    w = (grid[k + 1] - target) / (grid[k + 1] - grid[k])
    mu = rng.random((n_a, n_z))
    mu /= mu.sum()
    out = distribution_lottery_numpy(mu, k, w)
    assert abs(out.sum() - 1.0) < 1e-12 and out.min() >= 0
    assert abs(np.sum(out * grid[:, None]) - np.sum(mu * target)) < 1e-10
