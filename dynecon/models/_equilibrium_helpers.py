"""Shared pieces of the Aiyagari and Huggett economies (grid bounds)."""

from __future__ import annotations

__all__ = ["asset_upper_bound"]


def asset_upper_bound(
    beta: float, r: float, scale: float = 1.0, floor: float = 30.0, cap: float = 400.0
) -> float:
    """Upper asset-grid bound growing as the interest rate approaches ``1/beta - 1``.

    Households' wealth explodes as ``beta (1 + r) -> 1``; a bound proportional to
    ``1 / (1/beta - 1 - r)`` keeps the stationary mass off the top node.
    """
    gap = max(1.0 / beta - 1.0 - r, 1e-4)
    return float(min(cap, floor + scale / gap))
