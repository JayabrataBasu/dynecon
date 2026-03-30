"""Simulation vertical-slice tests.

Economics rationale:
A solved policy should generate reproducible state paths and summary moments,
which is the minimum viable simulation capability for model workflows.
"""

import numpy as np

from econkit.simulation import simulate_markov_asset_policy, summarize_simulation


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
