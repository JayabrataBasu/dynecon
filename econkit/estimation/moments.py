"""Estimation stubs for moment-based workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from econkit.simulation import SimulationResult, summarize_simulation


@dataclass(frozen=True)
class MomentTarget:
    """Named target moment used in simple estimation objectives."""

    name: str
    value: float
    weight: float = 1.0


def extract_moments_from_simulation(result: SimulationResult) -> dict[str, float]:
    """Extract baseline estimable moments from simulation output."""
    return summarize_simulation(result)


def quadratic_moment_loss(
    simulated: dict[str, float],
    targets: list[MomentTarget],
) -> float:
    """Compute weighted quadratic moment distance objective."""
    if not targets:
        raise ValueError("targets must contain at least one moment")

    loss = 0.0
    for target in targets:
        if target.name not in simulated:
            raise ValueError(f"simulated moments missing target: {target.name}")
        gap = simulated[target.name] - target.value
        loss += target.weight * gap * gap
    return float(loss)


def estimation_stub_report(
    simulated: dict[str, float],
    targets: list[MomentTarget],
) -> dict[str, float]:
    """Build a compact report for estimation-stub diagnostics."""
    objective = quadratic_moment_loss(simulated, targets)
    simulated_values = np.array([simulated[t.name] for t in targets], dtype=np.float64)
    target_values = np.array([t.value for t in targets], dtype=np.float64)
    rmse = float(np.sqrt(np.mean((simulated_values - target_values) ** 2)))
    return {
        "objective": objective,
        "rmse": rmse,
        "n_moments": float(len(targets)),
    }
