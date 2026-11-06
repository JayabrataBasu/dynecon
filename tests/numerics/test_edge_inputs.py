"""Degenerate inputs give a clear ValidationError, not a raw numpy/Python exception."""

import numpy as np
import pytest

from dynecon.core.errors import ValidationError
from dynecon.numerics import exp_grid, fixed_point, stationary_distribution
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
