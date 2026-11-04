import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import dynecon as ek
from dynecon.core.errors import ConvergenceError, ConvergenceWarning, ValidationError
from dynecon.numerics.fixedpoint import fixed_point
from dynecon.numerics.jacobian import numerical_jacobian


def test_cos_fixed_point() -> None:
    res = fixed_point(lambda x: np.cos(x), np.array([1.0]), tol=1e-13, max_iter=500)
    assert res.converged
    assert abs(res.x[0] - 0.7390851332151607) < 1e-10
    assert len(res.history) == res.iterations


def test_damping_still_converges_but_slower() -> None:
    fast = fixed_point(lambda x: 0.5 * x + 1, np.array([0.0]), tol=1e-10, max_iter=1000)
    slow = fixed_point(
        lambda x: 0.5 * x + 1, np.array([0.0]), tol=1e-10, max_iter=1000, damping=0.5
    )
    assert slow.converged and slow.iterations > fast.iterations
    assert abs(slow.x[0] - 2.0) < 1e-8


def test_callback_stops_without_warning() -> None:
    res = fixed_point(
        lambda x: x * 0.99,
        np.array([1.0]),
        tol=1e-12,
        max_iter=1000,
        callbacks=[lambda i, x, r: i >= 3],
    )
    assert res.iterations == 3 and not res.converged


def test_nonconvergence_warns_and_strict_raises() -> None:
    with pytest.warns(ConvergenceWarning):
        res = fixed_point(lambda x: x + 1, np.array([0.0]), tol=1e-8, max_iter=5)
    assert not res.converged
    ek.config.strict_convergence = True
    try:
        with pytest.raises(ConvergenceError):
            fixed_point(lambda x: x + 1, np.array([0.0]), tol=1e-8, max_iter=5)
    finally:
        ek.config.strict_convergence = False


def test_l1_norm_and_validation() -> None:
    res = fixed_point(lambda x: 0.5 * x, np.array([1.0, 1.0]), tol=1e-9, max_iter=200, norm="l1")
    assert res.converged
    for kw in ({"tol": 0.0}, {"max_iter": 0}, {"damping": 0.0}, {"norm": "l2"}):
        args = {"tol": 1e-8, "max_iter": 5} | kw
        with pytest.raises(ValidationError):
            fixed_point(lambda x: x, np.array([1.0]), **args)
    with pytest.raises(ValidationError):
        fixed_point(lambda x: np.zeros(3), np.array([1.0]), tol=1e-8, max_iter=5)


@settings(deadline=None, max_examples=50, derandomize=True)
@given(seed=st.integers(0, 10_000), n=st.integers(1, 6))
def test_contraction_residual_bound(seed: int, n: int) -> None:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    A *= 0.8 / max(np.linalg.norm(A, 2), 1e-9)
    b = rng.standard_normal(n)
    res = fixed_point(lambda x: A @ x + b, np.zeros(n), tol=1e-10, max_iter=2000)
    assert res.converged
    assert np.max(np.abs(A @ res.x + b - res.x)) <= 1e-8


def test_jacobian_polynomial_map() -> None:
    def F(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 * x[1], x[0] + 3 * x[1] ** 3])

    x = np.array([1.5, -0.7])
    J = numerical_jacobian(F, x)
    exact = np.array([[2 * x[0] * x[1], x[0] ** 2], [1.0, 9 * x[1] ** 2]])
    np.testing.assert_allclose(J, exact, atol=1e-8)
    np.testing.assert_allclose(numerical_jacobian(F, x, eps=1e-5, relative=False), J, atol=1e-6)


def test_anderson_same_fixed_point_in_fewer_steps() -> None:
    a = np.diag(np.linspace(0.5, 0.995, 20))  # slow contraction
    op = lambda x: a @ x + 1.0  # noqa: E731
    plain = fixed_point(op, np.zeros(20), tol=1e-10, max_iter=100_000)
    fast = fixed_point(op, np.zeros(20), tol=1e-10, max_iter=100_000, anderson=5)
    assert fast.converged and fast.iterations < plain.iterations / 3
    np.testing.assert_allclose(fast.x, plain.x, atol=1e-7)


def test_anderson_undoes_a_domain_violation() -> None:
    def op(x: np.ndarray) -> np.ndarray:
        if x.min() < 0.0:
            raise ValidationError("outside the domain")
        return 0.9 * x + 0.1

    res = fixed_point(op, np.full(3, 0.5), tol=1e-10, max_iter=2000, anderson=3, lower=0.0)
    assert res.converged and abs(res.x[0] - 1.0) < 1e-8


def test_anderson_rejects_damping() -> None:
    with pytest.raises(ValidationError):
        fixed_point(lambda x: x, np.zeros(1), tol=1e-8, max_iter=5, anderson=2, damping=0.5)
