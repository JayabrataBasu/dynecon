"""Integration and expectation APIs routed through the backend dispatcher."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from econkit.numerics._dispatch import resolve_backend
from econkit.numerics._registry import register
from econkit.numerics.backends.integration_numpy import (
    expectation_discrete_numpy,
    gauss_hermite_normal_numpy,
    gauss_laguerre_exp_numpy,
)

Array = npt.NDArray[np.float64]


## I will document the economics rationale for these kernels in a LaTeX which I will create. 
register("gauss_hermite_normal", "numpy", gauss_hermite_normal_numpy)
register("gauss_laguerre_exp", "numpy", gauss_laguerre_exp_numpy)
register("expectation_discrete", "numpy", expectation_discrete_numpy)


def gauss_hermite_normal(order: int, backend: str | None = None) -> tuple[Array, Array]:
    _, impl = resolve_backend("gauss_hermite_normal", backend=backend)
    nodes, weights = impl(order)
    return np.asarray(nodes, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def gauss_laguerre_exp(order: int, backend: str | None = None) -> tuple[Array, Array]:
    _, impl = resolve_backend("gauss_laguerre_exp", backend=backend)
    nodes, weights = impl(order)
    return np.asarray(nodes, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def expectation_discrete(
    values: Array,
    probs: Array,
    backend: str | None = None,
) -> float:
    _, impl = resolve_backend("expectation_discrete", backend=backend)
    return float(impl(values, probs))
