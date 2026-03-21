"""Solver VFI scaffold tests.

Economics rationale:
VFI is the baseline dynamic programming workhorse. These tests validate fixed
point convergence behavior and residual diagnostics on controlled toy problems.
"""

import math

import numpy as np
import pytest

from econkit.solvers import VFIConfig, bellman_residual, value_function_iteration


def test_vfi_converges_on_linear_contraction() -> None:
    beta = 0.9
    utility = np.linspace(1.0, 2.0, 6, dtype=np.float64)
    expected = utility / (1.0 - beta)

    def bellman_operator(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return utility + beta * value, np.full_like(value, 0.5)

    solution = value_function_iteration(
        initial_value=np.zeros_like(utility),
        bellman_operator=bellman_operator,
        config=VFIConfig(tol=1e-10, max_iter=500),
        model_name="toy_contraction",
    )

    assert solution.converged
    assert solution.policy is not None
    np.testing.assert_allclose(solution.value_function, expected, atol=1e-6)
    assert solution.metadata["algorithm"] == "vfi"
    assert solution.metadata["model_name"] == "toy_contraction"


def test_vfi_respects_iteration_cap_when_not_yet_converged() -> None:
    beta = 0.99
    utility = np.ones(4, dtype=np.float64)

    def bellman_operator(value: np.ndarray) -> tuple[np.ndarray, None]:
        return utility + beta * value, None

    solution = value_function_iteration(
        initial_value=np.zeros_like(utility),
        bellman_operator=bellman_operator,
        config=VFIConfig(tol=1e-14, max_iter=3),
    )

    assert not solution.converged
    assert solution.iterations == 3
    assert solution.residual > solution.metadata["tolerance"]


def test_bellman_residual_norms() -> None:
    old = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    new = np.array([1.5, 1.0, 3.0], dtype=np.float64)

    sup = bellman_residual(new, old, norm="sup")
    l2 = bellman_residual(new, old, norm="l2")

    assert abs(sup - 1.0) < 1e-12
    expected_l2 = math.sqrt((0.5**2 + 1.0**2 + 0.0**2) / 3.0)
    assert abs(l2 - expected_l2) < 1e-12


def test_bellman_residual_requires_shape_match() -> None:
    with pytest.raises(ValueError):
        bellman_residual(np.array([1.0]), np.array([1.0, 2.0]))


def test_howard_policy_evaluation_steps_are_applied() -> None:
    utility = np.linspace(0.5, 1.0, 5, dtype=np.float64)
    expected = utility + 2.0

    def bellman_operator(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return utility + 0.5 * value, np.zeros_like(value)

    def policy_evaluation(value: np.ndarray, policy: np.ndarray) -> np.ndarray:
        del policy
        return value + 1.0

    solution = value_function_iteration(
        initial_value=np.zeros_like(utility),
        bellman_operator=bellman_operator,
        policy_evaluation_operator=policy_evaluation,
        config=VFIConfig(tol=10.0, max_iter=1, howard_policy_improvement_steps=2),
    )

    np.testing.assert_allclose(solution.value_function, expected, atol=1e-12)
    assert solution.metadata["howard_policy_improvement_steps"] == 2


def test_howard_policy_evaluation_shape_is_validated() -> None:
    def bellman_operator(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return value + 1.0, np.zeros_like(value)

    def bad_policy_evaluation(value: np.ndarray, policy: np.ndarray) -> np.ndarray:
        del value, policy
        return np.zeros((2,), dtype=np.float64)

    with pytest.raises(ValueError):
        value_function_iteration(
            initial_value=np.zeros((3,), dtype=np.float64),
            bellman_operator=bellman_operator,
            policy_evaluation_operator=bad_policy_evaluation,
            config=VFIConfig(max_iter=1, howard_policy_improvement_steps=1),
        )
