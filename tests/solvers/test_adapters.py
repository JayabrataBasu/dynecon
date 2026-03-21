"""Compiled-model solver adapter tests.

Economics rationale:
Adapters should let model authors move from equation declarations to solver
execution with minimal glue code while preserving transparent semantics.
"""

import numpy as np

from econkit.model import Control, Model, Param, State
from econkit.solvers import VFIConfig, solve_compiled_model_vfi


def test_solve_compiled_model_vfi_with_bellman_tuple_output() -> None:
    model = Model(name="adapter_tuple")
    model.params(beta=Param(value=0.9), u=Param(value=1.0))
    model.add_state(State("assets", bounds=(0.0, 1.0), n_points=4))
    model.add_control(Control("consumption", bounds=(0.0, 1.0)))

    @model.equation("bellman")
    def bellman(value: np.ndarray, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        return params["u"] + params["beta"] * value, np.full_like(value, 0.25)

    compiled = model.compile()
    initial = np.zeros(4, dtype=np.float64)
    expected = np.ones(4, dtype=np.float64) / (1.0 - 0.9)

    solution = solve_compiled_model_vfi(
        compiled,
        initial_value=initial,
        config=VFIConfig(tol=1e-10, max_iter=500),
    )

    assert solution.converged
    assert solution.policy is not None
    np.testing.assert_allclose(solution.value_function, expected, atol=1e-6)


def test_solve_compiled_model_vfi_with_value_only_output() -> None:
    model = Model(name="adapter_value_only")
    model.params(beta=Param(value=0.5), u=Param(value=2.0))
    model.add_state(State("assets", bounds=(0.0, 1.0), n_points=3))
    model.add_control(Control("consumption", bounds=(0.0, 1.0)))

    @model.equation("bellman")
    def bellman(value: np.ndarray, params: dict[str, float]) -> np.ndarray:
        return params["u"] + params["beta"] * value

    compiled = model.compile()
    solution = solve_compiled_model_vfi(
        compiled,
        initial_value=np.zeros(3, dtype=np.float64),
        config=VFIConfig(tol=1e-10, max_iter=300),
    )

    np.testing.assert_allclose(solution.value_function, np.full(3, 4.0), atol=1e-6)
    assert solution.policy is None
