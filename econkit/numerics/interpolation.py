"""Interpolation APIs routed through the backend dispatcher.

Economics note:
Interpolation is central to dynamic programming because continuation values
are evaluated off-grid during policy optimization and simulation.
"""

import numpy as np
import numpy.typing as npt

from econkit.numerics._dispatch import resolve_backend
from econkit.numerics._registry import register
from econkit.numerics.backends.jax_backend import JAX_AVAILABLE, linear_interp_1d_jax
from econkit.numerics.backends.numba_backend import linear_interp_1d_numba
from econkit.numerics.backends.numpy_backend import linear_interp_1d_numpy

register("linear_interp_1d", "numpy", linear_interp_1d_numpy)
register("linear_interp_1d", "numba", linear_interp_1d_numba)
if JAX_AVAILABLE:
    register("linear_interp_1d", "jax", linear_interp_1d_jax)


def linear_interp_1d(
    x: npt.NDArray[np.float64],
    xp: npt.NDArray[np.float64],
    fp: npt.NDArray[np.float64],
    backend: str | None = None,
) -> npt.NDArray[np.float64]:
    """Interpolate values on a 1D grid using the selected backend."""
    _, impl = resolve_backend("linear_interp_1d", backend=backend)
    out = impl(x, xp, fp)
    return np.asarray(out, dtype=np.float64)
