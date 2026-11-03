"""CRRA utility and its marginal/inverse-marginal, shared by the built-in models.

Economics note:
Constant relative risk aversion ``sigma`` pins down both the curvature of utility and the
elasticity of intertemporal substitution ``1 / sigma``; ``sigma = 1`` is log utility.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["crra", "crra_marginal", "crra_marginal_inverse", "exp", "log"]


def crra(c: Any, sigma: float) -> Any:
    """CRRA utility ``c^(1-sigma) / (1-sigma)`` (``log c`` when ``sigma == 1``)."""
    if sigma == 1.0:
        return np.log(c)
    return c ** (1.0 - sigma) / (1.0 - sigma)


def crra_marginal(c: Any, sigma: float) -> Any:
    """Marginal utility ``c^(-sigma)``."""
    return c ** (-sigma)


def crra_marginal_inverse(x: Any, sigma: float) -> Any:
    """Inverse marginal utility ``x^(-1/sigma)``."""
    return x ** (-1.0 / sigma)


def _is_jax(x: Any) -> bool:
    return type(x).__module__.split(".")[0] in ("jax", "jaxlib")


def log(x: Any) -> Any:
    """Natural log that works on NumPy values and on JAX tracers (for autodiff)."""
    if _is_jax(x):
        import jax.numpy as jnp

        return jnp.log(x)
    return np.log(x)


def exp(x: Any) -> Any:
    """Exponential that works on NumPy values and on JAX tracers (for autodiff)."""
    if _is_jax(x):
        import jax.numpy as jnp

        return jnp.exp(x)
    return np.exp(x)
