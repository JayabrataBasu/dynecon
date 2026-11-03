"""Monte Carlo replications and bootstrap confidence intervals.

Economics note:
Simulation studies and bootstrap intervals quantify sampling uncertainty of estimators and
of model-implied statistics without relying on asymptotic formulas.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from econkit.core.errors import ValidationError
from econkit.simulation.moments import bootstrap_statistics
from econkit.utils.rng import spawn

__all__ = ["bootstrap_ci", "monte_carlo"]


def monte_carlo(
    fn: Callable[[np.random.Generator], Any],
    n_reps: int,
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[Any]:
    """Run ``fn(generator)`` for ``n_reps`` independent, reproducible random streams."""
    if n_reps < 1:
        raise ValidationError("n_reps must be >= 1")
    return np.asarray([fn(g) for g in spawn(rng, n_reps)])


def bootstrap_ci(
    data: npt.ArrayLike,
    stat: Callable[[npt.NDArray[np.float64]], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    block_len: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``stat``; moving-block when ``block_len`` is given."""
    if not 0.0 < alpha < 1.0:
        raise ValidationError("alpha must lie in (0, 1)")
    stats = bootstrap_statistics(data, stat, block_len or 1, n_boot, rng)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
