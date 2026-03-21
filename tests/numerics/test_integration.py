"""Numerics integration tests.

Economics rationale:
Expectations are central to dynamic programming in stochastic models. These
checks ensure quadrature and discrete expectation utilities preserve core
probability properties and produce economically meaningful moments.
"""

import numpy as np

from econkit.numerics.integration import (
    expectation_discrete,
    gauss_hermite_normal,
    gauss_laguerre_exp,
)


def test_gauss_hermite_weights_sum_to_one() -> None:
    _, w = gauss_hermite_normal(order=7, backend="numpy")
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)


def test_gauss_laguerre_exp_mean_near_one() -> None:
    x, w = gauss_laguerre_exp(order=20, backend="numpy")
    mean = float(np.dot(w, x))
    np.testing.assert_allclose(mean, 1.0, rtol=0.0, atol=1e-8)


def test_expectation_discrete_matches_manual_dot() -> None:
    values = np.array([0.0, 2.0, 5.0], dtype=np.float64)
    probs = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    got = expectation_discrete(values, probs, backend="numpy")
    want = float(np.dot(values, probs))
    np.testing.assert_allclose(got, want, atol=1e-12)
