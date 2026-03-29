"""Bounded optimization APIs routed through backend dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from econkit.numerics._dispatch import resolve_backend
from econkit.numerics._registry import register
from econkit.numerics.backends.optimize_numba import brent_max_numba, golden_section_max_numba
from econkit.numerics.backends.optimize_numpy import (
    OptimizeResult,
    brent_max_numpy,
    golden_section_max_numpy,
)

register("golden_section_max", "numpy", golden_section_max_numpy)
register("brent_max", "numpy", brent_max_numpy)
register("golden_section_max", "numba", golden_section_max_numba)
register("brent_max", "numba", brent_max_numba)


def golden_section_max(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 200,
    backend: str | None = None,
) -> OptimizeResult:
    _, impl = resolve_backend("golden_section_max", backend=backend)
    result = impl(func, a, b, tol, max_iter)
    return cast(OptimizeResult, result)


def brent_max(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 200,
    backend: str | None = None,
) -> OptimizeResult:
    _, impl = resolve_backend("brent_max", backend=backend)
    result = impl(func, a, b, tol, max_iter)
    return cast(OptimizeResult, result)
