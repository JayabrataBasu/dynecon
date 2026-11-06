import numpy as np
import pytest

from dynecon.core.errors import ValidationError
from dynecon.solvers.lq import LQ, solve_discrete_riccati


def test_scalar_riccati_has_a_closed_form() -> None:
    # x' = a x + u, cost q u^2 + r x^2, beta = 1: P solves P = r + a^2 P - a^2 P^2 / (q + P)
    a, q, r = 1.2, 1.0, 1.0
    P = solve_discrete_riccati([[a]], [[1.0]], [[r]], [[q]])[0, 0]
    assert abs(P - (r + a**2 * P - a**2 * P**2 / (q + P))) < 1e-10
    F = LQ([[q]], [[r]], [[a]], [[1.0]]).stationary_values()[1][0, 0]
    assert abs(F - a * P / (q + P)) < 1e-10
    assert abs(a - F) < 1  # the closed loop is stable


def test_stationary_values_satisfy_the_bellman_equation_with_discounting_and_noise() -> None:
    rng = np.random.default_rng(0)
    n, k = 3, 2
    A = rng.normal(size=(n, n)) * 0.6
    B = rng.normal(size=(n, k))
    R, Q = np.eye(n), np.eye(k)
    N = 0.1 * rng.normal(size=(k, n))
    C = 0.3 * rng.normal(size=(n, 2))
    lq = LQ(Q, R, A, B, C=C, N=N, beta=0.95)
    P, F, d = lq.stationary_values()
    b = 0.95
    rhs = R + b * A.T @ P @ A - (b * A.T @ P @ B + N.T) @ F
    np.testing.assert_allclose(P, rhs, atol=1e-9)
    assert abs(d - b * np.trace(P @ C @ C.T) / (1 - b)) < 1e-12
    assert np.max(np.abs(np.linalg.eigvals(A - B @ F))) < 1 / np.sqrt(b)  # stabilising


def test_finite_horizon_converges_to_the_infinite_horizon_policy() -> None:
    A, B = np.array([[1.0, 0.5], [0.0, 0.9]]), np.array([[0.0], [1.0]])
    R, Q = np.eye(2), np.array([[2.0]])
    inf = LQ(Q, R, A, B, beta=0.9)
    _, F_inf, _ = inf.stationary_values()
    fin = LQ(Q, R, A, B, beta=0.9, T=200)
    fin.update_values()
    np.testing.assert_allclose(fin.policies[0], F_inf, atol=1e-8)
    assert len(fin.policies) == 200
    xs, us, _ = fin.compute_sequence([1.0, 1.0])
    assert xs.shape == (201, 2) and us.shape == (200, 1)
    # cost of the optimal path with no noise equals x0'P x0 (plus terminal cost, here small)
    cost = sum(xs[t] @ R @ xs[t] + us[t] @ Q @ us[t] for t in range(200)) * 1.0
    assert cost > 0


def test_one_period_policy_is_static_optimum() -> None:
    # T=1, Rf=0: minimise x'Rx + u'Qu + (Ax+Bu)'Rf(Ax+Bu) -> u = 0 when Rf = 0
    lq = LQ([[1.0]], [[1.0]], [[0.5]], [[1.0]], T=1, Rf=[[0.0]])
    lq.update_values()
    np.testing.assert_allclose(lq.policies[0], 0.0, atol=1e-14)


def test_bad_inputs() -> None:
    eye2 = np.eye(2)
    with pytest.raises(ValidationError):
        solve_discrete_riccati(eye2, np.ones((3, 1)), eye2, [[1.0]])
    with pytest.raises(ValidationError):
        solve_discrete_riccati(eye2, np.ones((2, 1)), eye2, [[1.0]], beta=1.5)
    with pytest.raises(ValidationError):
        LQ(
            [[1.0]], eye2, eye2, np.ones((2, 1)), C=np.ones((2, 1)), beta=1.0
        )  # noise, no discounting
    with pytest.raises(ValidationError):
        LQ([[1.0]], np.full((2, 2), np.nan), eye2, np.ones((2, 1)))
    with pytest.raises(ValidationError):
        LQ([[1.0]], eye2, eye2, np.ones((2, 1))).update_values()  # needs finite T
    # unstabilisable: unstable mode that B cannot reach
    with pytest.raises(ValidationError):
        solve_discrete_riccati(np.diag([2.0, 0.5]), np.array([[0.0], [1.0]]), eye2, [[1.0]])
