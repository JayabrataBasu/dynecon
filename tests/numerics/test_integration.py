import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from dynecon.core.errors import ValidationError
from dynecon.numerics.integration import gauss_hermite_normal, gauss_legendre


def test_gauss_hermite_moments() -> None:
    x, w = gauss_hermite_normal(7)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-13)
    np.testing.assert_allclose(w @ x**2, 1.0, atol=1e-12)
    mu, sd = 0.3, 0.4
    x10, w10 = gauss_hermite_normal(10)
    np.testing.assert_allclose(w10 @ np.exp(mu + sd * x10), math.exp(mu + sd**2 / 2), rtol=1e-10)


def test_gauss_legendre_polynomial_exactness() -> None:
    order = 5
    x, w = gauss_legendre(order, 1.0, 3.0)
    for deg in range(2 * order):
        exact = (3.0 ** (deg + 1) - 1.0) / ((deg + 1) * 2.0)
        np.testing.assert_allclose(w @ x**deg, exact, rtol=1e-12)


def test_invalid() -> None:
    with pytest.raises(ValidationError):
        gauss_hermite_normal(0)
    with pytest.raises(ValidationError):
        gauss_legendre(3, 1.0, 0.0)


@given(order=st.integers(1, 40))
def test_weights_positive_sum_one(order: int) -> None:
    for rule in (gauss_hermite_normal(order), gauss_legendre(order)):
        assert np.all(rule[1] > 0)
        assert abs(rule[1].sum() - 1.0) < 1e-12
