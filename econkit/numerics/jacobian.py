"""Numerical Jacobians by central differences.

Economics note:
Linearising a DSGE system around its steady state and computing moment
sensitivities in SMM both need derivatives of model-implied maps.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from econkit.utils.validation import check_array

__all__ = ["numerical_jacobian"]

Array = npt.NDArray[np.float64]


def numerical_jacobian(
    F: Callable[[Array], Array], x: Array, eps: float = 1e-6, relative: bool = True
) -> Array:
    """Central-difference Jacobian of ``F`` at ``x`` (shape ``(len(F(x)), len(x))``).

    The step for coordinate ``i`` is ``eps * max(1, |x_i|)`` when ``relative`` else ``eps``.
    """
    x0 = check_array(x, "x", ndim=1)
    f0 = np.atleast_1d(np.asarray(F(x0), dtype=np.float64))
    jac = np.empty((f0.shape[0], x0.shape[0]), dtype=np.float64)
    for i in range(x0.shape[0]):
        h = eps * max(1.0, abs(x0[i])) if relative else eps
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        jac[:, i] = (np.atleast_1d(F(xp)) - np.atleast_1d(F(xm))) / (2.0 * h)
    return jac
