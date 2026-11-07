"""Degenerate inputs give a clear ValidationError, not a raw numpy/Python exception."""

import numpy as np
import pytest

from dynecon.core.errors import ConvergenceError, ValidationError
from dynecon.numerics import (
    exp_grid,
    fixed_point,
    gauss_chebyshev,
    gauss_hermite_normal,
    gauss_legendre,
    lognormal_nodes,
    mvnormal_nodes,
    rouwenhorst,
    stationary_distribution,
    tauchen,
    trapezoid_rule,
)
from dynecon.simulation import gini, quantile


def test_empty_inputs() -> None:
    with pytest.raises(ValidationError):
        stationary_distribution(np.zeros((0, 0)))
    with pytest.raises(ValidationError):
        fixed_point(lambda x: x, np.zeros(0), tol=1e-8, max_iter=10)
    with pytest.raises(ValidationError):
        quantile(np.array([]), np.array([]), 0.5)
    with pytest.raises(ValidationError):
        quantile(np.ones(3) / 3, np.ones(4), 0.5)


def test_bad_weights_and_non_finite_curvature() -> None:
    with pytest.raises(ValidationError):
        gini(np.array([1.0, 2.0]), np.array([-0.5, 1.5]))
    with pytest.raises(ValidationError):
        gini(np.array([1.0, np.nan]))
    for k in (np.nan, np.inf):
        with pytest.raises(ValidationError):
            exp_grid(0.0, 1.0, 5, curvature=k)
    assert np.allclose(exp_grid(0.0, 1.0, 5, curvature=0.0), np.linspace(0, 1, 5))


def test_klein_rejects_bad_matrices() -> None:
    from dynecon.solvers.perturbation.klein import klein_solve

    for A, B, n_x in (
        (np.eye(3), np.eye(2), 1),
        (np.eye(2), np.eye(2), 5),
        (np.full((2, 2), np.nan), np.eye(2), 1),
    ):
        with pytest.raises(ValidationError):
            klein_solve(A, B, n_x)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan, 0.0, -1.0, None, "0.1"])
def test_discretisation_rejects_bad_sigma(bad) -> None:
    with pytest.raises(ValidationError, match="sigma_e"):
        tauchen(0.9, bad, 5)
    with pytest.raises(ValidationError, match="sigma_e"):
        rouwenhorst(0.9, bad, 5)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1.5, 1, "5", None])
def test_discretisation_rejects_bad_state_count(bad) -> None:
    with pytest.raises(ValidationError, match="n_states"):
        tauchen(0.9, 0.1, bad)
    with pytest.raises(ValidationError, match="n_states"):
        rouwenhorst(0.9, 0.1, bad)


def test_discretisation_rejects_overflowing_grid_and_bad_width() -> None:
    with pytest.raises(ValidationError, match="overflows"):
        tauchen(0.99, 1e308, 5)
    with pytest.raises(ValidationError, match="overflows"):
        rouwenhorst(0.99, 1e308, 50)
    for m in (np.nan, np.inf, 0.0, -1.0):
        with pytest.raises(ValidationError, match="m"):
            tauchen(0.9, 0.1, 5, m)
    assert tauchen(0.9, 0.1, 5.0)[0].shape == (5,)  # a whole float count is accepted


@pytest.mark.parametrize("order", [100, 101, 300, 500, 1000])
def test_gauss_hermite_is_finite_and_exact_at_high_order(order) -> None:
    x, w = gauss_hermite_normal(order)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(w))
    assert abs(w.sum() - 1.0) < 1e-12 and abs(w @ x**2 - 1.0) < 1e-10 and abs(w @ x**4 - 3.0) < 1e-9


def test_quadrature_rejects_bad_orders_and_intervals() -> None:
    for bad in (0, -1, np.nan, np.inf, 2.5, "3"):
        with pytest.raises(ValidationError, match="order"):
            gauss_hermite_normal(bad)
    for rule in (gauss_legendre, gauss_chebyshev):
        for a, b in ((0.0, np.inf), (-np.inf, 1.0), (np.nan, 1.0), (1.0, 1.0), (2.0, 1.0)):
            with pytest.raises(ValidationError):
                rule(5, a, b)
    with pytest.raises(ValidationError):
        trapezoid_rule(5, 0.0, np.inf)
    with pytest.raises(ValidationError, match="order"):
        trapezoid_rule(1)
    with pytest.raises(ValidationError, match="mean"):
        mvnormal_nodes(3, [np.nan, 0.0], np.eye(2))
    with pytest.raises(ValidationError, match="order"):
        mvnormal_nodes([3, np.nan], [0.0, 0.0], np.eye(2))
    with pytest.raises(ValidationError, match="overflows"):
        lognormal_nodes(5, [1000.0], [[1.0]])


def test_fixed_point_stops_on_non_finite_values() -> None:
    with pytest.raises(ConvergenceError, match="non-finite"):
        fixed_point(lambda x: x * np.nan, np.ones(3), tol=1e-8, max_iter=10)
    with pytest.raises(ConvergenceError, match="non-finite"):
        fixed_point(lambda x: x * np.inf, np.ones(3), tol=1e-8, max_iter=10)
    with pytest.raises(ConvergenceError):
        fixed_point(lambda x: x * np.nan, np.ones(3), tol=1e-8, max_iter=10, anderson=2)
    with pytest.raises(ValidationError, match="x0"):
        fixed_point(lambda x: x, np.array([1.0, np.nan]), tol=1e-8, max_iter=10)


@pytest.mark.parametrize(
    "kw",
    [
        {"tol": np.inf},
        {"tol": np.nan},
        {"tol": 0.0},
        {"max_iter": 2.5},
        {"max_iter": np.nan},
        {"damping": np.nan},
        {"anderson": 1.5},
        {"lower": np.nan},
    ],
)
def test_fixed_point_rejects_bad_options(kw) -> None:
    opts = {"tol": 1e-8, "max_iter": 10, **kw}
    with pytest.raises(ValidationError):
        fixed_point(lambda x: 0.5 * x, np.ones(3), **opts)


def test_quantile_and_hamilton_refuse_bad_values() -> None:
    from dynecon.data import hamilton_filter

    for mass, grid, q in (
        ([0.0], [np.inf], 0.0),
        ([0.0], [1.0], 0.5),
        ([-1.0, 2.0], [1.0, 2.0], 0.5),
        ([0.5, 0.5], [1.0, np.nan], 0.5),
        ([0.5, 0.5], [1.0, 2.0], np.nan),
    ):
        with pytest.raises(ValidationError):
            quantile(np.array(mass), np.array(grid), q)
    assert quantile(np.array([0.5, 0.5]), np.array([1.0, 2.0]), 0.5) == 1.0
    big = 1.348e308
    overflowing = [0.0, 0.0, 1e300, -big, 0.0, -big, big, -big, -big, big, -big, -big, 1e300, 1.0]
    with pytest.raises(ValidationError, match="overflowed"):  # found by the fuzz contract
        hamilton_filter(np.array(overflowing), h=2, p=2)
