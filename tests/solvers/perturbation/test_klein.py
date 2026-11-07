"""Klein solver: Blanchard-Kahn classification and exactness on constructed systems."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dynecon.core.errors import DeterminacyError
from dynecon.solvers.perturbation.klein import klein_solve


def test_scalar_classification_written_first() -> None:
    # x_{t+1} = 0.5 x_t is stable: one predetermined variable, one stable root
    F, P, eig = klein_solve(np.array([[1.0]]), np.array([[0.5]]), 1)
    assert P[0, 0] == pytest.approx(0.5) and F.shape == (0, 1) and eig[0] == pytest.approx(0.5)
    # x_{t+1} = 2 x_t is unstable: explosive when x is predetermined ...
    with pytest.raises(DeterminacyError, match="explosive"):
        klein_solve(np.array([[1.0]]), np.array([[2.0]]), 1)
    # ... but fine when x is a jump variable (forced to zero)
    F, P, _ = klein_solve(np.array([[1.0]]), np.array([[2.0]]), 0)
    assert F.shape == (1, 0)
    # stable root with a jump variable: indeterminate
    with pytest.raises(DeterminacyError, match="indeterminate"):
        klein_solve(np.array([[1.0]]), np.array([[0.5]]), 0)


def test_unit_root_counts_as_unstable() -> None:
    with pytest.raises(DeterminacyError):
        klein_solve(np.array([[1.0]]), np.array([[1.0]]), 1)


def test_bad_arguments() -> None:
    with pytest.raises(ValueError):
        klein_solve(np.eye(2), np.eye(3), 1)
    with pytest.raises(ValueError):
        klein_solve(np.eye(2), np.eye(2), 3)


@settings(deadline=None, max_examples=60, derandomize=True)
@given(n_x=st.integers(1, 3), n_u=st.integers(1, 3), seed=st.integers(0, 10_000))
def test_recovers_constructed_solution_under_equation_transformations(
    n_x: int, n_u: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    P0 = rng.standard_normal((n_x, n_x))
    P0 *= 0.8 / max(np.max(np.abs(np.linalg.eigvals(P0))), 1e-9)
    F0 = rng.standard_normal((n_u, n_x))
    n = n_x + n_u
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    A[:n_x, :n_x] = np.eye(n_x)
    B[:n_x, :n_x] = P0
    B[n_x:, :n_x] = F0
    B[n_x:, n_x:] = -np.eye(n_u)
    Q = rng.standard_normal((n, n)) + 3 * np.eye(n)
    F, P, _ = klein_solve(Q @ A, Q @ B, n_x)
    np.testing.assert_allclose(P, P0, atol=1e-8)
    np.testing.assert_allclose(F, F0, atol=1e-8)


def test_singular_pencil_and_ill_conditioning_raise_determinacy_error() -> None:
    zero = np.zeros((1, 1))
    with pytest.raises(DeterminacyError, match="singular"):
        klein_solve(zero, zero, 0)
    ones = np.ones((2, 2))
    with pytest.raises(DeterminacyError, match="singular"):
        klein_solve(ones, ones, 1)
    huge = np.full((2, 2), 1.34e154)
    huge[1, 1] = 1.0
    try:
        klein_solve(huge, np.eye(2), 1)
    except DeterminacyError:
        pass  # a clear refusal is acceptable; a raw scipy ValueError is not


def test_bad_matrix_types_and_counts_raise_validation_error() -> None:
    from dynecon.core.errors import ValidationError

    for A, B, n_x in (
        (None, np.eye(2), 1),
        (np.eye(2), np.eye(3), 1),
        (np.eye(2), np.eye(2), 1.5),
        (np.eye(2), np.eye(2), np.nan),
        (np.zeros((0, 0)), np.zeros((0, 0)), 0),
        (np.array([[np.nan]]), np.eye(1), 0),
    ):
        with pytest.raises(ValidationError):
            klein_solve(A, B, n_x)


def test_subnormal_pencil_is_refused_not_returned_as_nan() -> None:
    tiny = 2.22507386e-313  # found by the fuzz contract
    B = np.array([[tiny, tiny], [1.0, tiny]])
    try:
        F, P, _ = klein_solve(np.array([[0.0, 1.0], [1.0, 1.0]]), B, 1)
    except DeterminacyError:
        return
    assert np.isfinite(F).all() and np.isfinite(P).all()
