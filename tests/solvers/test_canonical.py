"""Canonical consumption-savings solver helper tests.

Economics rationale:
The canonical helper should provide a low-boilerplate baseline solve for
standard consumption-savings problems while preserving feasibility constraints.
"""

import numpy as np

from econkit.model import AR1Shock, Control, Model, Param, State
from econkit.solvers import (
    VFIConfig,
    solve_consumption_savings_liquid_illiquid_vfi,
    solve_consumption_savings_stochastic_vfi,
    solve_consumption_savings_vfi,
)


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


def test_stochastic_canonical_helper_solves_and_returns_policy_grid() -> None:
    model = Model(name="canonical_cs_stochastic")
    model.params(
        beta=Param(value=0.94),
        sigma=Param(value=2.0),
        r=Param(value=0.02),
        y=Param(value=1.0),
    )
    model.add_state(State("assets", bounds=(0.0, 8.0), n_points=25))
    model.add_state(State("income", shock=AR1Shock(rho=0.9, sigma_e=0.1, n_states=5)))
    model.add_control(Control("consumption", bounds=(1e-8, None)))

    @model.equation("bellman")
    def bellman_placeholder(value: np.ndarray, params: dict[str, float]) -> np.ndarray:
        del value, params
        return np.zeros((25, 5), dtype=np.float64)

    compiled = model.compile()
    solution = solve_consumption_savings_stochastic_vfi(
        compiled,
        config=VFIConfig(tol=1e-6, max_iter=200),
        discretization_method="tauchen",
        shock_backend="numpy",
    )

    assert solution.converged
    assert solution.policy is not None
    assert solution.policy.shape == solution.value_function.shape
    assert solution.value_function.shape == (25, 5)
    assert solution.metadata["canonical_model"] == "consumption_savings_stochastic"
    assert solution.metadata["discretization_method"] == "tauchen"


def test_stochastic_helper_requires_income_shock() -> None:
    model = Model(name="missing_income_shock")
    model.params(beta=Param(value=0.95), sigma=Param(value=2.0), r=Param(value=0.02))
    model.add_state(State("assets", bounds=(0.0, 8.0), n_points=10))
    model.add_state(State("income", bounds=(0.0, 1.0), n_points=3))
    model.add_control(Control("consumption", bounds=(0.0, 1.0)))

    @model.equation("bellman")
    def bellman_placeholder(value: np.ndarray, params: dict[str, float]) -> np.ndarray:
        del params
        return value

    compiled = model.compile()
    try:
        solve_consumption_savings_stochastic_vfi(compiled)
    except ValueError as exc:
        assert "AR(1) shock" in str(exc)
    else:
        raise AssertionError("expected stochastic helper to reject missing income shock")


def test_canonical_helper_emits_monotonicity_metadata() -> None:
    model = Model(name="canonical_monotonic")
    model.params(beta=Param(value=0.95), sigma=Param(value=2.0), r=Param(value=0.01))
    model.add_state(State("assets", bounds=(0.0, 5.0), n_points=15))
    model.add_control(Control("consumption", bounds=(1e-8, None)))

    @model.equation("bellman")
    def bellman_placeholder(value: np.ndarray, params: dict[str, float]) -> np.ndarray:
        del value, params
        return np.zeros(15, dtype=np.float64)

    compiled = model.compile()
    solution = solve_consumption_savings_vfi(
        compiled,
        config=VFIConfig(tol=1e-6, max_iter=120),
        check_monotonic_policy=True,
    )

    assert "policy_monotonicity" in solution.metadata


def test_liquid_illiquid_helper_solves_prototype() -> None:
    model = Model(name="liquid_illiquid")
    model.params(
        beta=Param(value=0.93),
        sigma=Param(value=2.0),
        r_liquid=Param(value=0.01),
        r_illiquid=Param(value=0.03),
        y=Param(value=1.0),
        adjustment_cost=Param(value=0.01),
    )
    model.add_state(State("liquid_assets", bounds=(0.0, 3.0), n_points=10))
    model.add_state(State("illiquid_assets", bounds=(0.0, 3.0), n_points=8))
    model.add_control(Control("consumption", bounds=(1e-8, None)))

    @model.equation("bellman")
    def bellman_placeholder(value: np.ndarray, params: dict[str, float]) -> np.ndarray:
        del value, params
        return np.zeros((10, 8), dtype=np.float64)

    compiled = model.compile()
    solution = solve_consumption_savings_liquid_illiquid_vfi(
        compiled,
        config=VFIConfig(tol=1e-6, max_iter=150),
        check_monotonic_policy=True,
        policy_search_window=2,
    )

    assert solution.converged
    assert solution.policy is not None
    assert solution.value_function.shape == (10, 8)
    assert solution.policy.shape == (10, 8)
    assert solution.metadata["canonical_model"] == "consumption_savings_liquid_illiquid"
    assert "policy_illiquid" in solution.metadata
    assert solution.metadata["policy_search_window"] == 2
