"""Discretization APIs routed through the backend dispatcher."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from econkit.numerics._dispatch import resolve_backend
from econkit.numerics._registry import register
from econkit.numerics.backends.discretization_jax import (
    JAX_AVAILABLE,
    rouwenhorst_jax,
    tauchen_jax,
)
from econkit.numerics.backends.discretization_numba import rouwenhorst_numba, tauchen_numba
from econkit.numerics.backends.discretization_numpy import rouwenhorst_numpy, tauchen_numpy

Array = npt.NDArray[np.float64]

register("tauchen", "numpy", tauchen_numpy)
register("rouwenhorst", "numpy", rouwenhorst_numpy)
register("tauchen", "numba", tauchen_numba)
register("rouwenhorst", "numba", rouwenhorst_numba)
if JAX_AVAILABLE:
    register("tauchen", "jax", tauchen_jax)
    register("rouwenhorst", "jax", rouwenhorst_jax)


def tauchen(
    rho: float,
    sigma_e: float,
    n_states: int,
    m: float = 3.0,
    backend: str | None = None,
) -> tuple[Array, Array]:
    _, impl = resolve_backend("tauchen", backend=backend)
    grid, transition = impl(rho, sigma_e, n_states, m)
    return np.asarray(grid, dtype=np.float64), np.asarray(transition, dtype=np.float64)


def rouwenhorst(
    rho: float,
    sigma_e: float,
    n_states: int,
    backend: str | None = None,
) -> tuple[Array, Array]:
    _, impl = resolve_backend("rouwenhorst", backend=backend)
    grid, transition = impl(rho, sigma_e, n_states)
    return np.asarray(grid, dtype=np.float64), np.asarray(transition, dtype=np.float64)
