"""Shared pieces of the Aiyagari and Huggett economies (income chain, grid bounds)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from econkit.numerics.discretization import rouwenhorst
from econkit.numerics.markov import stationary_distribution

__all__ = ["asset_upper_bound", "unit_mean_income_chain"]

Array = npt.NDArray[np.float64]


def unit_mean_income_chain(rho: float, sigma_e: float, n_z: int) -> tuple[Array, Array]:
    """Rouwenhorst chain for ``log z`` with ``z = exp(.)`` normalised to unit stationary mean."""
    grid, P = rouwenhorst(rho, sigma_e, n_z)
    z = np.exp(grid)
    return z / float(stationary_distribution(P) @ z), P


def asset_upper_bound(
    beta: float, r: float, scale: float = 1.0, floor: float = 30.0, cap: float = 400.0
) -> float:
    """Upper asset-grid bound growing as the interest rate approaches ``1/beta - 1``.

    Households' wealth explodes as ``beta (1 + r) -> 1``; a bound proportional to
    ``1 / (1/beta - 1 - r)`` keeps the stationary mass off the top node.
    """
    gap = max(1.0 / beta - 1.0 - r, 1e-4)
    return float(min(cap, floor + scale / gap))
