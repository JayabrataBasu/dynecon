import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from dynecon.core.errors import ValidationError
from dynecon.numerics.grids import exp_grid, make_grid, uniform_grid


def test_uniform_grid_basic() -> None:
    g = uniform_grid(0.0, 1.0, 5)
    np.testing.assert_allclose(g, [0, 0.25, 0.5, 0.75, 1.0])


def test_exp_grid_small_curvature_is_uniform() -> None:
    np.testing.assert_allclose(
        exp_grid(-1.0, 3.0, 50, 1e-9), uniform_grid(-1.0, 3.0, 50), atol=1e-8
    )


def test_exp_grid_dense_near_lower_bound() -> None:
    g = exp_grid(0.0, 10.0, 20, 4.0)
    assert np.diff(g)[0] < np.diff(g)[-1]


def test_make_grid_dispatch_and_errors() -> None:
    assert make_grid("uniform", 0, 1, 4).shape == (4,)
    assert make_grid("exp", 0, 1, 4, curvature=2.0).shape == (4,)
    with pytest.raises(ValidationError):
        make_grid("cheby", 0, 1, 4)
    with pytest.raises(ValidationError):
        uniform_grid(1.0, 0.0, 4)
    with pytest.raises(ValidationError):
        uniform_grid(0.0, 1.0, 1)


@given(
    lo=st.floats(-1e3, 1e3),
    width=st.floats(1e-3, 1e3),
    n=st.integers(2, 200),
    k=st.floats(-6.0, 6.0),
)
def test_grid_properties(lo: float, width: float, n: int, k: float) -> None:
    hi = lo + width
    for g in (uniform_grid(lo, hi, n), exp_grid(lo, hi, n, k)):
        assert g.shape == (n,)
        assert g[0] == lo and g[-1] == hi
        assert np.all(np.diff(g) > 0)


def test_exp_grid_reduces_interpolation_error_near_constraint() -> None:
    sigma = 2.0
    lo, hi, n = 0.05, 10.0, 200

    def u(c: np.ndarray) -> np.ndarray:
        return c ** (1 - sigma) / (1 - sigma)

    from dynecon.numerics.interpolation import linear_interp_1d

    fine = np.linspace(lo, lo + 0.1 * (hi - lo), 2000)
    errs = []
    for grid in (exp_grid(lo, hi, n, 4.0), uniform_grid(lo, hi, n)):
        errs.append(np.max(np.abs(linear_interp_1d(fine, grid, u(grid)) - u(fine))))
    assert errs[0] < errs[1]
