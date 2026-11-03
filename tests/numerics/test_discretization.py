"""AR(1) discretisation against the exact stationary moments."""

import numpy as np
import pytest

from econkit.core.errors import ValidationError
from econkit.numerics.discretization import rouwenhorst, tauchen
from econkit.numerics.markov import stationary_distribution


def _moments(grid: np.ndarray, P: np.ndarray) -> tuple[float, float, float]:
    pi = stationary_distribution(P)
    mean = float(pi @ grid)
    var = float(pi @ (grid - mean) ** 2)
    cov = float(sum(pi[i] * grid[i] * (P[i] @ grid) for i in range(len(grid))))
    return mean, var, cov / var


@pytest.mark.parametrize("n", [3, 7, 9])
def test_rouwenhorst_matches_ar1_moments(n: int) -> None:
    rho, s = 0.95, 0.05
    grid, P = rouwenhorst(rho, s, n)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-14)
    mean, var, ac = _moments(grid, P)
    assert abs(mean) < 1e-12
    assert abs(var / (s**2 / (1 - rho**2)) - 1) < 1e-10
    assert abs(ac - rho) < 1e-10


def test_tauchen_close_to_ar1_moments() -> None:
    rho, s = 0.9, 0.1
    grid, P = tauchen(rho, s, 21, m=3.0)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
    mean, var, _ = _moments(grid, P)
    assert abs(mean) < 1e-10
    assert abs(var / (s**2 / (1 - rho**2)) - 1) < 0.05


@pytest.mark.parametrize("args", [(1.0, 0.1, 5), (0.5, -1.0, 5), (0.5, 0.1, 1)])
def test_invalid_args(args: tuple[float, float, int]) -> None:
    with pytest.raises(ValidationError):
        tauchen(*args)
    with pytest.raises(ValidationError):
        rouwenhorst(*args)
