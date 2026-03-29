"""Estimation stub tests.

Economics rationale:
Moment-based distance objectives are the minimal bridge from simulation output
to calibration/estimation workflows.
"""

import numpy as np
import pytest

from econkit.estimation import (
    MomentTarget,
    estimation_stub_report,
    extract_moments_from_simulation,
    quadratic_moment_loss,
)
from econkit.simulation import SimulationResult


def test_extract_moments_from_simulation() -> None:
    result = SimulationResult(
        asset_indices=np.array([0, 1, 1], dtype=np.int64),
        shock_indices=np.array([0, 0, 1], dtype=np.int64),
        assets=np.array([0.0, 1.0, 1.0], dtype=np.float64),
    )
    moments = extract_moments_from_simulation(result)

    assert moments["mean_assets"] > 0.0
    assert moments["max_assets"] == 1.0


def test_quadratic_moment_loss_and_report() -> None:
    simulated = {"mean_assets": 1.2, "std_assets": 0.4}
    targets = [
        MomentTarget(name="mean_assets", value=1.0, weight=2.0),
        MomentTarget(name="std_assets", value=0.5, weight=1.0),
    ]

    loss = quadratic_moment_loss(simulated, targets)
    report = estimation_stub_report(simulated, targets)

    assert loss > 0.0
    assert report["objective"] == loss
    assert report["n_moments"] == 2.0


def test_quadratic_moment_loss_rejects_missing_target() -> None:
    with pytest.raises(ValueError):
        quadratic_moment_loss({"mean_assets": 1.0}, [MomentTarget("std_assets", 0.2)])
