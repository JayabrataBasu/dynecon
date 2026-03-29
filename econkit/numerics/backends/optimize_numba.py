"""Numba optimization backend wrappers.

Numba support is optional at runtime.
"""

from __future__ import annotations

from collections.abc import Callable

from econkit.numerics.backends.optimize_numpy import (
    OptimizeResult,
    brent_max_numpy,
    golden_section_max_numpy,
)

try:
    import numba
except Exception:  # pragma: no cover - optional dependency
    numba = None


def golden_section_max_numba(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> OptimizeResult:
    """Numba backend entrypoint for bounded golden-section maximization.

    Economics note:
    This backend keeps API parity for strict backend selection while enabling
    future specialization for nopython-compatible objectives.
    """
    if numba is None:  # pragma: no cover
        raise RuntimeError("numba backend requested but numba is not installed")
    return golden_section_max_numpy(func, a, b, tol, max_iter)


def brent_max_numba(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> OptimizeResult:
    """Numba backend entrypoint for bounded Brent-style maximization."""
    if numba is None:  # pragma: no cover
        raise RuntimeError("numba backend requested but numba is not installed")
    return brent_max_numpy(func, a, b, tol, max_iter)
