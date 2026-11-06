import itertools

import numpy as np
import pytest

from dynecon.core.errors import ValidationError
from dynecon.numerics import (
    cartesian,
    gauss_chebyshev,
    lognormal_nodes,
    mvnormal_nodes,
    tensor_rule,
    trapezoid_rule,
)

MEAN = np.array([0.5, -1.0])
COV = np.array([[1.0, 0.6], [0.6, 2.0]])


def test_mvnormal_integrates_polynomials_exactly() -> None:
    x, w = mvnormal_nodes(5, MEAN, COV)
    assert x.shape == (25, 2) and abs(w.sum() - 1) < 1e-14
    np.testing.assert_allclose(w @ x, MEAN, atol=1e-13)
    d = x - MEAN
    np.testing.assert_allclose(
        (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0), COV, atol=1e-12
    )
    # fourth moment E[(x1 - m1)^2 (x2 - m2)^2] = s11 s22 + 2 s12^2 for a Gaussian
    assert (
        abs(w @ (d[:, 0] ** 2 * d[:, 1] ** 2) - (COV[0, 0] * COV[1, 1] + 2 * COV[0, 1] ** 2))
        < 1e-11
    )


def test_lognormal_mean_and_unequal_orders() -> None:
    x, w = lognormal_nodes([9, 7], MEAN, COV)
    assert x.shape == (63, 2) and np.all(x > 0)
    np.testing.assert_allclose(w @ x, np.exp(MEAN + 0.5 * np.diag(COV)), rtol=1e-5)


def test_one_dimensional_rules() -> None:
    x, w = gauss_chebyshev(6, 2.0, 4.0)
    assert np.all(np.diff(x) > 0) and abs(w.sum() - 1) < 1e-14
    assert abs(w @ x - 3.0) < 1e-13  # mean of the arcsine law on [2, 4]
    assert abs(w @ (x - 3.0) ** 2 - 0.5) < 1e-13  # variance (b - a)^2 / 8
    t, v = trapezoid_rule(11, 0.0, 2.0)
    assert abs(v @ t - 1.0) < 1e-14 and abs(v @ t**2 - 4 / 3) < 0.02  # exact for linear


def test_tensor_rule_and_cartesian_follow_product_order() -> None:
    a, b = np.array([1.0, 2.0]), np.array([10.0, 20.0, 30.0])
    nodes, w = tensor_rule([a, b], [np.array([0.5, 0.5]), np.array([0.2, 0.3, 0.5])])
    assert [tuple(r) for r in nodes] == list(itertools.product(a, b))
    np.testing.assert_allclose(w, [0.1, 0.15, 0.25] * 2)
    np.testing.assert_array_equal(cartesian(a, b), nodes)
    assert cartesian([1, 2, 3]).shape == (3, 1)


def test_bad_inputs() -> None:
    for bad in (
        lambda: mvnormal_nodes(3, [0, 0], [[1, 2], [2, 1]]),  # not positive definite
        lambda: mvnormal_nodes(3, [0, 0], [[1, 0.1], [0.2, 1]]),  # not symmetric
        lambda: mvnormal_nodes([3], [0, 0], np.eye(2)),
        lambda: gauss_chebyshev(0),
        lambda: gauss_chebyshev(3, 1.0, 1.0),
        lambda: trapezoid_rule(1),
        lambda: tensor_rule([np.ones(2)], []),
        lambda: cartesian(),
        lambda: cartesian([1.0], []),
    ):
        with pytest.raises(ValidationError):
            bad()


def test_numpy_integer_orders_are_accepted() -> None:
    x, w = mvnormal_nodes(np.int64(3), MEAN, COV)
    assert x.shape == (9, 2) and abs(w.sum() - 1) < 1e-14
