"""Maximum likelihood with numerical-Hessian standard errors.

Economics note:
Likelihood methods use the full distribution implied by a model; the curvature of the
log-likelihood at the maximum measures how precisely the data pin down each parameter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from econkit.core.errors import ValidationError
from econkit.estimation.results import MLEResult

__all__ = ["mle", "numerical_hessian"]

Array = npt.NDArray[np.float64]


def numerical_hessian(f: Callable[[Array], float], x: Array, eps: float = 1e-4) -> Array:
    """Central-difference Hessian of a scalar function (step ``eps * max(1, |x_i|)``)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    h = eps * np.maximum(1.0, np.abs(x))
    H = np.empty((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        H[i, i] = (f(x + ei) - 2.0 * f0 + f(x - ei)) / h[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h[j]
            H[i, j] = H[j, i] = (
                f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
            ) / (4.0 * h[i] * h[j])
    return H


def mle(
    loglik: Callable[[Array], float],
    theta0: npt.ArrayLike,
    bounds: list[tuple[float | None, float | None]] | None = None,
    method: str = "L-BFGS-B",
    gradient: str | None = None,
) -> MLEResult:
    """Maximise ``loglik``; standard errors from the inverse of ``-H`` at the optimum.

    ``gradient="jax"`` uses exact JAX derivatives (gradient and Hessian); ``loglik`` must then
    be JAX-traceable. Otherwise derivatives are finite differences.
    """
    from scipy.optimize import minimize

    x0 = np.asarray(theta0, dtype=np.float64)
    if x0.ndim != 1:
        raise ValidationError("theta0 must be a vector")

    def neg(x: Array) -> float:
        v = float(loglik(x))
        return -v if np.isfinite(v) else 1e12

    jac = None
    hess_fn = None
    if gradient == "jax":
        import jax
        import jax.numpy as jnp

        from econkit import _jax_setup  # noqa: F401  (enables float64)

        def neg_j(x: Any) -> Any:
            return -loglik(x)

        grad_j, hess_j = jax.jit(jax.grad(neg_j)), jax.jit(jax.hessian(neg_j))

        def jac(x: Array) -> Array:
            return np.asarray(grad_j(jnp.asarray(x)), dtype=np.float64)

        def hess_fn(x: Array) -> Array:
            return np.asarray(hess_j(jnp.asarray(x)), dtype=np.float64)

    elif gradient is not None:
        raise ValidationError("gradient must be None or 'jax'")
    kw = {"bounds": bounds} if method in ("L-BFGS-B", "TNC", "SLSQP", "Powell") else {}
    opts = {"ftol": 1e-14, "gtol": 1e-10} if method == "L-BFGS-B" else {}
    res = minimize(neg, x0, method=method, options=opts, jac=jac, **kw)
    theta = np.asarray(res.x, dtype=np.float64)
    # Hessian of the log-likelihood
    H = -hess_fn(theta) if hess_fn is not None else -numerical_hessian(neg, theta)
    cov = np.linalg.pinv(-H)
    return MLEResult(
        estimates=theta,
        std_errors=np.sqrt(np.clip(np.diag(cov), 0.0, None)),
        loglik=-float(res.fun),
        hessian=H,
        covariance=cov,
        converged=bool(res.success),
        n_iter=int(getattr(res, "nit", 0)),
    )
