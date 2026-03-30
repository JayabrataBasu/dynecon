"""Policy-based simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class SimulationResult:
    """Simulation artifact for a Markov-policy asset path."""

    asset_indices: IndexArray
    shock_indices: IndexArray
    assets: Array


def simulate_markov_asset_policy(
    policy_assets: Array,
    asset_grid: Array,
    shock_transition: Array,
    periods: int,
    asset_initial_index: int = 0,
    shock_initial_index: int = 0,
    random_seed: int | None = None,
) -> SimulationResult:
    """Simulate asset dynamics under a discrete Markov shock and policy map.

    Economics note:
    This is the first simulation vertical slice: given a solved policy function
    indexed by current assets and shock state, simulate household asset paths.
    """
    if periods < 1:
        raise ValueError("periods must be >= 1")
    if policy_assets.ndim != 2:
        raise ValueError("policy_assets must be 2D with shape (n_assets, n_shocks)")
    if asset_grid.ndim != 1:
        raise ValueError("asset_grid must be 1D")
    if shock_transition.ndim != 2:
        raise ValueError("shock_transition must be 2D")

    n_assets, n_shocks = policy_assets.shape
    if asset_grid.shape[0] != n_assets:
        raise ValueError("asset_grid length must match policy_assets first dimension")
    if shock_transition.shape != (n_shocks, n_shocks):
        raise ValueError("shock_transition shape must be (n_shocks, n_shocks)")

    row_sums = shock_transition.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-10):
        raise ValueError("each shock_transition row must sum to one")

    if not (0 <= asset_initial_index < n_assets):
        raise ValueError("asset_initial_index out of bounds")
    if not (0 <= shock_initial_index < n_shocks):
        raise ValueError("shock_initial_index out of bounds")

    rng = np.random.default_rng(seed=random_seed)

    asset_indices = np.empty(periods, dtype=np.int64)
    shock_indices = np.empty(periods, dtype=np.int64)
    assets = np.empty(periods, dtype=np.float64)

    asset_idx = int(asset_initial_index)
    shock_idx = int(shock_initial_index)

    for t in range(periods):
        asset_indices[t] = asset_idx
        shock_indices[t] = shock_idx
        assets[t] = asset_grid[asset_idx]

        target_asset = policy_assets[asset_idx, shock_idx]
        next_asset_idx = int(np.argmin(np.abs(asset_grid - target_asset)))

        probs = shock_transition[shock_idx]
        next_shock_idx = int(rng.choice(n_shocks, p=probs))

        asset_idx = next_asset_idx
        shock_idx = next_shock_idx

    return SimulationResult(
        asset_indices=asset_indices,
        shock_indices=shock_indices,
        assets=assets,
    )


def summarize_simulation(result: SimulationResult) -> dict[str, float]:
    """Return basic summary moments for a simulated asset path."""
    return {
        "mean_assets": float(np.mean(result.assets)),
        "std_assets": float(np.std(result.assets)),
        "min_assets": float(np.min(result.assets)),
        "max_assets": float(np.max(result.assets)),
    }
