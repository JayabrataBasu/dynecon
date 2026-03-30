"""End-to-end solve-simulate-moments pipeline helpers."""

from __future__ import annotations

import numpy as np

from econkit.estimation.moments import (
    MomentTarget,
    estimation_stub_report,
    extract_moments_from_simulation,
)
from econkit.model import Control, Model, Param, State
from econkit.simulation import simulate_markov_asset_policy
from econkit.solvers import VFIConfig, solve_consumption_savings_vfi


def run_solve_simulate_moments_pipeline(
    periods: int = 200,
    random_seed: int = 0,
) -> dict[str, float]:
    """Run a full solve -> simulate -> moments -> objective demo pipeline."""
    model = Model(name="session13_pipeline")
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
    solution = solve_consumption_savings_vfi(compiled, config=VFIConfig(tol=1e-6, max_iter=200))

    if solution.policy is None:
        raise RuntimeError("pipeline expected a valid policy output")

    assets_state = compiled.states["assets"]
    if assets_state.bounds is None or assets_state.n_points is None:
        raise RuntimeError("pipeline assets state missing bounds or n_points")

    asset_grid = np.linspace(
        assets_state.bounds[0],
        assets_state.bounds[1],
        assets_state.n_points,
        dtype=np.float64,
    )
    policy_assets = np.asarray(solution.policy, dtype=np.float64)[:, np.newaxis]
    transition = np.array([[1.0]], dtype=np.float64)

    simulation = simulate_markov_asset_policy(
        policy_assets=policy_assets,
        asset_grid=asset_grid,
        shock_transition=transition,
        periods=periods,
        random_seed=random_seed,
    )

    moments = extract_moments_from_simulation(simulation)
    targets = [
        MomentTarget(name="mean_assets", value=2.0, weight=1.0),
        MomentTarget(name="std_assets", value=1.0, weight=0.5),
    ]
    report = estimation_stub_report(moments, targets)
    report["converged"] = 1.0 if solution.converged else 0.0
    report["iterations"] = float(solution.iterations)
    return report
