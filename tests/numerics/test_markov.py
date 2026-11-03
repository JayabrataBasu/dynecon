import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from econkit.core.errors import ValidationError
from econkit.numerics.markov import markov_paths, simulate_markov, stationary_distribution


def test_two_state_closed_form() -> None:
    p, q = 0.3, 0.1
    P = np.array([[1 - p, p], [q, 1 - q]])
    np.testing.assert_allclose(stationary_distribution(P), [q / (p + q), p / (p + q)], atol=1e-14)


def test_reducible_chain_rejected() -> None:
    with pytest.raises(ValidationError):
        stationary_distribution(np.eye(3))


@settings(deadline=None, max_examples=60, derandomize=True)
@given(n=st.integers(2, 8), seed=st.integers(0, 10_000))
def test_stationary_properties(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    P = rng.random((n, n)) + 0.05
    P /= P.sum(axis=1, keepdims=True)
    pi = stationary_distribution(P)
    assert np.all(pi >= 0) and abs(pi.sum() - 1) < 1e-12
    np.testing.assert_allclose(pi @ P, pi, atol=1e-10)
    path = simulate_markov(P, 50, rng=seed)
    assert path.min() >= 0 and path.max() < n


def test_long_run_frequencies_within_standard_errors() -> None:
    P = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]])
    pi = stationary_distribution(P)
    path = simulate_markov(P, 200_000, rng=5)
    for k in range(3):
        blocks = (path == k).astype(float).reshape(100, 2000).mean(axis=1)
        se = blocks.std(ddof=1) / np.sqrt(100)
        assert abs(blocks.mean() - pi[k]) < 4 * se + 1e-3


@pytest.mark.numba
def test_backend_parity_exact() -> None:
    pytest.importorskip("numba")
    rng = np.random.default_rng(1)
    P = rng.random((5, 5))
    P /= P.sum(axis=1, keepdims=True)
    u = rng.random((200, 64))
    init = rng.integers(0, 5, 64)
    a = markov_paths(P, 201, init, u, backend="numpy")
    b = markov_paths(P, 201, init, u, backend="numba")
    np.testing.assert_array_equal(a, b)


def test_input_validation() -> None:
    P = np.eye(2)
    with pytest.raises(ValidationError):
        markov_paths(P, 5, np.array([0]), np.zeros((3, 1)))
    with pytest.raises(ValidationError):
        markov_paths(P, 5, np.array([2]), np.zeros((4, 1)))
    with pytest.raises(ValidationError):
        simulate_markov(P, 0)
