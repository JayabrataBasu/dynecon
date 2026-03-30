"""JAX pilot entries for discretization APIs.

JAX support is optional at runtime.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from econkit.numerics.backends.discretization_numpy import rouwenhorst_numpy, tauchen_numpy

try:
    import jax.numpy as jnp  # type: ignore[import-not-found]

    JAX_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    jnp = None
    JAX_AVAILABLE = False

Array = npt.NDArray[np.float64]


if JAX_AVAILABLE:

    def tauchen_jax(
        rho: float,
        sigma_e: float,
        n_states: int,
        m: float = 3.0,
    ) -> tuple[Array, Array]:
        grid_np, trans_np = tauchen_numpy(rho, sigma_e, n_states, m)
        grid = np.asarray(jnp.asarray(grid_np), dtype=np.float64)
        transition = np.asarray(jnp.asarray(trans_np), dtype=np.float64)
        return grid, transition

    def rouwenhorst_jax(
        rho: float,
        sigma_e: float,
        n_states: int,
    ) -> tuple[Array, Array]:
        grid_np, trans_np = rouwenhorst_numpy(rho, sigma_e, n_states)
        grid = np.asarray(jnp.asarray(grid_np), dtype=np.float64)
        transition = np.asarray(jnp.asarray(trans_np), dtype=np.float64)
        return grid, transition

else:

    def tauchen_jax(
        rho: float,
        sigma_e: float,
        n_states: int,
        m: float = 3.0,
    ) -> tuple[Array, Array]:  # pragma: no cover
        del rho, sigma_e, n_states, m
        raise RuntimeError("jax backend requested but jax is not installed")

    def rouwenhorst_jax(
        rho: float,
        sigma_e: float,
        n_states: int,
    ) -> tuple[Array, Array]:  # pragma: no cover
        del rho, sigma_e, n_states
        raise RuntimeError("jax backend requested but jax is not installed")
