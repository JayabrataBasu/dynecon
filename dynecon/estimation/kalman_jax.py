"""Differentiable Kalman-filter log-likelihood (JAX), for gradient-based likelihood methods.

Economics note:
With exact gradients of the likelihood, DSGE and state-space models can be estimated
by quasi-Newton methods and sampled by gradient-based MCMC.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dynecon import _jax_setup  # noqa: F401  (enables float64)

__all__ = ["kalman_loglik_jax"]


def kalman_loglik_jax(y: Any, Z: Any, H: Any, T: Any, Q: Any, a0: Any, P0: Any) -> Any:
    """Log-likelihood of fully observed ``y`` (no NaNs), traceable by ``jax.grad``."""
    y = jnp.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    n_y = y.shape[1]

    def step(carry: tuple[Any, Any], obs: Any) -> tuple[tuple[Any, Any], Any]:
        a, P = carry
        v = obs - Z @ a
        F = Z @ P @ Z.T + H
        _, logdet = jnp.linalg.slogdet(F)
        ll = -0.5 * (n_y * jnp.log(2 * jnp.pi) + logdet + v @ jnp.linalg.solve(F, v))
        K = P @ Z.T @ jnp.linalg.inv(F)
        a_f = a + K @ v
        P_f = P - K @ Z @ P
        P_next = T @ P_f @ T.T + Q
        return (T @ a_f, 0.5 * (P_next + P_next.T)), ll

    _, lls = jax.lax.scan(step, (jnp.asarray(a0), jnp.asarray(P0)), y)
    return jnp.sum(lls)
