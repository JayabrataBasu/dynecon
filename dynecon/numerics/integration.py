"""Quadrature rules (cold path, NumPy).

Economics note:
Expectations over continuous shocks in Bellman and Euler operators are replaced by
weighted sums over quadrature nodes.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from dynecon.core.errors import ValidationError

__all__ = [
    "gauss_chebyshev",
    "gauss_hermite_normal",
    "gauss_legendre",
    "lognormal_nodes",
    "mvnormal_nodes",
    "tensor_rule",
    "trapezoid_rule",
]

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


def gauss_chebyshev(order: int, a: float = 0.0, b: float = 1.0) -> tuple[Array, Array]:
    """Gauss-Chebyshev nodes and weights for ``E[f(X)]`` with ``X`` on ``[a, b]`` with the
    arcsine (Chebyshev) density ``1 / (pi sqrt((x - a)(b - x)))``; weights sum to one."""
    _check_order(order)
    if not b > a:
        raise ValidationError(f"need b > a, got ({a}, {b})")
    k = np.arange(order, 0, -1)
    x = np.cos(np.pi * (2 * k - 1) / (2 * order))  # ascending
    return 0.5 * (b - a) * x + 0.5 * (a + b), np.full(order, 1.0 / order)


def trapezoid_rule(order: int, a: float = 0.0, b: float = 1.0) -> tuple[Array, Array]:
    """Trapezoid nodes and weights for the *average* of ``f`` over ``U(a, b)`` (sum to one)."""
    if int(order) != order or order < 2:
        raise ValidationError(f"order must be an integer >= 2, got {order}")
    if not b > a:
        raise ValidationError(f"need b > a, got ({a}, {b})")
    w = np.full(order, 1.0 / (order - 1))
    w[[0, -1]] *= 0.5
    return np.linspace(a, b, order), w


def tensor_rule(nodes: list[Array], weights: list[Array]) -> tuple[Array, Array]:
    """Product rule of univariate rules: nodes ``(prod n_i, d)`` (last dimension varies fastest),
    weights multiplied. The rows follow ``itertools.product`` order."""
    if len(nodes) != len(weights) or not nodes:
        raise ValidationError("need one weight vector per node vector")
    grids = np.meshgrid(*nodes, indexing="ij")
    w = np.ones(())
    for i, wi in enumerate(weights):
        shape = [1] * len(nodes)
        shape[i] = len(wi)
        w = w * np.asarray(wi).reshape(shape)
    return np.stack([g.ravel() for g in grids], axis=1), np.asarray(w).ravel()


def mvnormal_nodes(
    orders: int | list[int], mean: npt.ArrayLike, cov: npt.ArrayLike
) -> tuple[Array, Array]:
    """Product Gauss-Hermite rule for ``X ~ N(mean, cov)``: nodes ``(prod orders, d)``, weights
    sum to one. A Cholesky factor of ``cov`` correlates the independent standard-normal nodes."""
    mu = np.atleast_1d(np.asarray(mean, dtype=np.float64))
    S = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    d = mu.size
    if S.shape != (d, d) or not np.all(np.isfinite(S)) or not np.allclose(S, S.T):
        raise ValidationError("cov must be a finite symmetric matrix matching mean")
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError as err:
        raise ValidationError("cov must be positive definite") from err
    ns = [int(orders)] * d if isinstance(orders, int) else [int(o) for o in orders]
    if len(ns) != d:
        raise ValidationError("orders must be an int or have one entry per dimension")
    rules = [gauss_hermite_normal(n) for n in ns]
    z, w = tensor_rule([r[0] for r in rules], [r[1] for r in rules])
    return mu + z @ L.T, w


def lognormal_nodes(
    orders: int | list[int], mean: npt.ArrayLike, cov: npt.ArrayLike
) -> tuple[Array, Array]:
    """Product rule for ``exp(X)`` with ``X ~ N(mean, cov)`` (``mean`` and ``cov`` are those of
    the underlying normal): nodes are ``exp`` of :func:`mvnormal_nodes`."""
    x, w = mvnormal_nodes(orders, mean, cov)
    return np.exp(x), w
