"""NumPy reference kernels for integration and expectation operators."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def gauss_hermite_normal_numpy(order: int) -> tuple[Array, Array]:
    """Nodes and weights for expectations under N(0, 1).

    Economics note:
    This is used for expectations over Gaussian shocks in Bellman and Euler
    operators, where E[f(epsilon)] is approximated by weighted sums.
    """
    if order < 1:
        raise ValueError("order must be >= 1")

    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes = nodes.astype(np.float64) * math.sqrt(2.0)
    weights = weights.astype(np.float64) / math.sqrt(math.pi)
    return nodes, weights


def gauss_laguerre_exp_numpy(order: int) -> tuple[Array, Array]:
    """Nodes and weights for expectations under Exp(1)."""
    if order < 1:
        raise ValueError("order must be >= 1")

    nodes, weights = np.polynomial.laguerre.laggauss(order)
    return nodes.astype(np.float64), weights.astype(np.float64)


def expectation_discrete_numpy(values: Array, probs: Array) -> float:
    """Discrete expectation helper: E[X] = sum_i p_i x_i."""
    if values.shape != probs.shape:
        raise ValueError("values and probs must have identical shapes")
    s = probs.sum()
    if not np.isclose(s, 1.0, atol=1e-10):
        raise ValueError("probabilities must sum to one")
    return float(np.dot(values, probs))
