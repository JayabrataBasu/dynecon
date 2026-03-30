"""Simulation vertical-slice tests.

Economics rationale:
A solved policy should generate reproducible state paths and summary moments,
which is the minimum viable simulation capability for model workflows.
"""

import numpy as np

from econkit.simulation import (
    simulation_diagnostics_report,
    simulate_markov_asset_policy,
    summarize_simulation,
)


def test_simulate_markov_asset_policy_shapes_and_bounds() -> None:
    asset_grid = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    policy_assets = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=np.float64,
    )
    transition = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)

    result = simulate_markov_asset_policy(
        policy_assets=policy_assets,
        asset_grid=asset_grid,
        shock_transition=transition,
        periods=50,
        asset_initial_index=1,
        shock_initial_index=0,
        random_seed=123,
    )

    assert result.assets.shape == (50,)
    assert result.asset_indices.shape == (50,)
    assert result.shock_indices.shape == (50,)
    assert float(result.assets.min()) >= float(asset_grid.min())
    assert float(result.assets.max()) <= float(asset_grid.max())


def test_simulation_summary_contains_moments() -> None:
    asset_grid = np.array([0.0, 1.0], dtype=np.float64)
    policy_assets = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    transition = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    result = simulate_markov_asset_policy(
        policy_assets=policy_assets,
        asset_grid=asset_grid,
        shock_transition=transition,
        periods=10,
        random_seed=42,
    )
    summary = summarize_simulation(result)

    assert set(summary.keys()) == {"mean_assets", "std_assets", "min_assets", "max_assets"}


def test_simulation_diagnostics_report_shapes_and_bounds() -> None:
    asset_grid = np.linspace(0.0, 2.0, 5)
    policy_assets = np.full((asset_grid.size, 2), 1.0)
    transition = np.array([[0.8, 0.2], [0.1, 0.9]])

    result = simulate_markov_asset_policy(
        policy_assets=policy_assets,
        asset_grid=asset_grid,
        shock_transition=transition,
        periods=200,
        random_seed=99,
    )
    diagnostics = simulation_diagnostics_report(result, n_shocks=2)

    assert diagnostics.shock_transition_frequencies.shape == (2, 2)
    assert -1.0 <= diagnostics.asset_lag1_autocorr <= 1.0
    assert diagnostics.moments["mean_assets"] >= 0.0
    row_sums = diagnostics.shock_transition_frequencies.sum(axis=1)
    assert np.all((row_sums <= 1.0 + 1e-12) & (row_sums >= 0.0))


def test_simulation_diagnostics_invalid_n_shocks_raises() -> None:
    asset_grid = np.linspace(0.0, 1.0, 4)
    policy_assets = np.full((asset_grid.size, 1), 0.5)
    transition = np.array([[1.0]])
    result = simulate_markov_asset_policy(
        policy_assets=policy_assets,
        asset_grid=asset_grid,
        shock_transition=transition,
        periods=5,
        random_seed=1,
    )

    try:
        simulation_diagnostics_report(result, n_shocks=0)
        assert False, "expected ValueError"
    except ValueError:
        assert True
