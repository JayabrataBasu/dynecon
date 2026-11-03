"""JAX implementations of the hot operations (registered lazily when JAX is requested).

JAX is not claimed to be faster than Numba on CPU. It is here for exact derivatives,
vectorisation across parameters (``vmap``) and GPU execution. Kernels are written
functionally (no mutation) and return NumPy arrays so callers are backend-agnostic.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from econkit import _jax_setup  # noqa: F401  (enables float64)
from econkit.numerics._registry import register

Array = npt.NDArray[np.float64]


def interp_linear(x: Any, xp: Any, fp: Any, mode: int) -> Any:
    """Traceable linear interpolation; ``mode`` 0 clamps, 1 extrapolates linearly."""
    idx = jnp.clip(jnp.searchsorted(xp, x, side="right") - 1, 0, xp.shape[0] - 2)
    x0 = xp[idx]
    t = (x - x0) / (xp[idx + 1] - x0)
    t = jnp.where(mode == 0, jnp.clip(t, 0.0, 1.0), t)
    return fp[idx] + t * (fp[idx + 1] - fp[idx])


@functools.partial(jax.jit, static_argnames=("mode",))
def _interp_jit(x: Any, xp: Any, fp: Any, mode: int) -> Any:
    return interp_linear(x, xp, fp, mode)


def linear_interp_1d_jax(x: Array, xp: Array, fp: Array, mode: int) -> Array:
    return np.asarray(_interp_jit(jnp.asarray(x), jnp.asarray(xp), jnp.asarray(fp), mode))


@jax.jit
def _bellman_jit(U: Any, EVT: Any) -> tuple[Any, Any]:
    total = U + EVT[:, None, :]
    idx = jnp.argmax(total, axis=2)
    val = jnp.take_along_axis(total, idx[:, :, None], axis=2)[:, :, 0]
    return val.T, idx.T


def bellman_saving_jax(U: Array, EVT: Array, monotone: bool) -> tuple[Array, Array]:
    del monotone  # exhaustive maximisation over the next-asset index
    v, i = _bellman_jit(jnp.asarray(U), jnp.asarray(EVT))
    return np.asarray(v), np.asarray(i, dtype=np.int64)


@jax.jit
def _lottery_jit(mu: Any, k: Any, w: Any) -> Any:
    n_a, n_z = mu.shape
    cols = jnp.broadcast_to(jnp.arange(n_z), mu.shape)
    out = jnp.zeros((n_a, n_z), dtype=mu.dtype)
    out = out.at[k, cols].add(mu * w)
    return out.at[k + 1, cols].add(mu * (1.0 - w))


def distribution_lottery_jax(mu: Array, k: Array, w: Array) -> Array:
    return np.asarray(_lottery_jit(jnp.asarray(mu), jnp.asarray(k), jnp.asarray(w)))


@jax.jit
def _markov_jit(cdf: Any, u: Any, init: Any) -> Any:
    last = cdf.shape[1] - 1

    def step(s: Any, u_t: Any) -> tuple[Any, Any]:
        nxt = jnp.minimum(jnp.sum(cdf[s] <= u_t[:, None], axis=1), last)
        return nxt, nxt

    _, rest = jax.lax.scan(step, init, u)
    return jnp.concatenate([init[None, :], rest], axis=0)


def markov_paths_jax(cdf: Array, u: Array, init: Array) -> Array:
    out = _markov_jit(jnp.asarray(cdf), jnp.asarray(u), jnp.asarray(init))
    return np.asarray(out, dtype=np.int64)


@functools.partial(jax.jit, static_argnames=("mode",))
def _interp_columns_jit(xq: Any, xk: Any, fk: Any, mode: int) -> Any:
    return jax.vmap(lambda a, b, c: interp_linear(a, b, c, mode))(xq, xk, fk)


def linear_interp_columns_jax(xq: Array, xk: Array, fk: Array, mode: int) -> Array:
    out = _interp_columns_jit(jnp.asarray(xq), jnp.asarray(xk), jnp.asarray(fk), mode)
    return np.asarray(out)


register("linear_interp_1d", "jax", linear_interp_1d_jax)
register("linear_interp_columns", "jax", linear_interp_columns_jax)
register("bellman_saving", "jax", bellman_saving_jax)
register("distribution_lottery", "jax", distribution_lottery_jax)
register("markov_paths", "jax", markov_paths_jax)
