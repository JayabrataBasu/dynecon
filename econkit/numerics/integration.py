"""Quadrature rules (cold path, NumPy).

Economics note:
Expectations over continuous shocks in Bellman and Euler operators are replaced by
weighted sums over quadrature nodes.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from econkit.core.errors import ValidationError

__all__ = ["gauss_hermite_normal", "gauss_legendre"]

Array = npt.NDArray[np.float64]


def _check_order(order: int) -> None:
    if int(order) != order or order < 1:
        raise ValidationError(f"order must be an integer >= 1, got {order}")


def gauss_hermite_normal(order: int) -> tuple[Array, Array]:
    """Nodes and weights for ``E[f(X)]``, ``X ~ N(0, 1)`` (weights sum to one)."""
    _check_order(order)
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return nodes * math.sqrt(2.0), weights / math.sqrt(math.pi)


def gauss_legendre(order: int, a: float = 0.0, b: float = 1.0) -> tuple[Array, Array]:
    """Nodes and weights for the *average* of ``f`` over ``U(a, b)`` (weights sum to one)."""
    _check_order(order)
    if not b > a:
        raise ValidationError(f"need b > a, got ({a}, {b})")
    x, w = np.polynomial.legendre.leggauss(order)
    return 0.5 * (b - a) * x + 0.5 * (a + b), w / 2.0
