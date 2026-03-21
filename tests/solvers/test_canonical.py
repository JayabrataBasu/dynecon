"""Canonical consumption-savings solver helper tests.

Economics rationale:
The canonical helper should provide a low-boilerplate baseline solve for
standard consumption-savings problems while preserving feasibility constraints.
"""

import numpy as np

from econkit.model import Control, Model, Param, State
from econkit.solvers import VFIConfig, solve_consumption_savings_vfi


def test_canonical_helper_solves_and_returns_policy_grid() -> None:
    model = Model(name="canonical_cs")
    model.params(
        beta=Param(value=0.95),
        sigma=Param(value=2.0),
        r=Param(value=0.02),
        y=Param(value=1.0),
    )
    model.add_state(State("assets", bounds=(0.0, 8.0), n_points=40))
    model.add_control(Control("consumption", bounds=(1e-8, None)))

    @model.equation("bellman")
    def bellman_placeholder(value: np.ndarray, params: dict[str, float]) -> np.ndarray:
        del value, params
        return np.zeros(40, dtype=np.float64)

    compiled = model.compile()
    solution = solve_consumption_savings_vfi(
        compiled,
        config=VFIConfig(tol=1e-6, max_iter=200),
    )

    assert solution.converged
    assert solution.policy is not None
    assert solution.policy.shape == solution.value_function.shape
    assert float(solution.policy.min()) >= 0.0
    assert float(solution.policy.max()) <= 8.0
    assert solution.metadata["canonical_model"] == "consumption_savings"


def test_canonical_helper_requires_assets_state() -> None:
    model = Model(name="missing_assets")
    model.params(beta=Param(value=0.95), sigma=Param(value=2.0), r=Param(value=0.02))
    model.add_state(State("income", bounds=(0.0, 1.0), n_points=5))
    model.add_control(Control("consumption", bounds=(0.0, 1.0)))

    @model.equation("bellman")
    def bellman_placeholder(value: np.ndarray, params: dict[str, float]) -> np.ndarray:
        del params
        return value

    compiled = model.compile()
    try:
        solve_consumption_savings_vfi(compiled)
    except ValueError as exc:
        assert "assets" in str(exc)
    else:
        raise AssertionError("expected canonical helper to reject missing assets state")
