"""Exact (automatic-differentiation) linearisation of ``dsge`` problems with JAX.

Economics note:
Exact derivatives remove the truncation error of finite differences, so the linearised
model, and with it every impulse response and likelihood, is accurate to machine precision.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from dynecon import _jax_setup  # noqa: F401  (enables float64)
from dynecon.core.errors import ValidationError
from dynecon.model.problems import DSGEProblem, Vars
from dynecon.model.specs import ParamView

__all__ = ["linearize_jax"]

Array = npt.NDArray[np.float64]


def linearize_jax(prob: DSGEProblem, p: ParamView, y_ss: Array) -> tuple[Array, Array, Array]:
    """``(A, B, Sigma)`` as in :func:`linearize` but with ``jax.jacfwd`` derivatives.

    The equations must be JAX-traceable (use ``dynecon.models.utility.log``/``exp`` instead of
    ``numpy.log``/``numpy.exp`` inside them).
    """
    names = prob.variables
    log_mask = jnp.array([v in prob.log_variables for v in names])
    y_ss_j = jnp.asarray(y_ss)
    y0 = jnp.where(log_mask, jnp.log(jnp.where(log_mask, y_ss_j, 1.0)), y_ss_j)

    def levels(v: Any) -> Any:
        return jnp.where(log_mask, jnp.exp(v), v)

    def residuals(yp: Any, y: Any) -> Any:
        return jnp.stack(
            [jnp.asarray(r) for r in prob.equations(Vars(names, yp), Vars(names, y), p)]
        )

    try:
        A = jax.jacfwd(lambda v: residuals(levels(v), y_ss_j))(y0)
        B = -jax.jacfwd(lambda v: residuals(y_ss_j, levels(v)))(y0)
    except (TypeError, jax.errors.JAXTypeError) as exc:
        raise ValidationError(
            f"equations are not JAX-traceable ({type(exc).__name__}); use dynecon.models.utility "
            "log/exp rather than numpy functions inside equations"
        ) from exc
    sigma = np.asarray(prob.shock_loadings(p), dtype=np.float64)
    return np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64), sigma
