"""JAX backend pilot kernels.

JAX support is optional at runtime.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

try:
    import jax.numpy as jnp  # type: ignore[import-not-found]

    JAX_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    jnp = None
    JAX_AVAILABLE = False


if JAX_AVAILABLE:

    def linear_interp_1d_jax(
        x: npt.NDArray[np.float64],
        xp: npt.NDArray[np.float64],
        fp: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        x_j = jnp.asarray(x)
        xp_j = jnp.asarray(xp)
        fp_j = jnp.asarray(fp)
        out = jnp.interp(x_j, xp_j, fp_j)
        return np.asarray(out, dtype=np.float64)

else:

    def linear_interp_1d_jax(
        x: npt.NDArray[np.float64],
        xp: npt.NDArray[np.float64],
        fp: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:  # pragma: no cover
        del x, xp, fp
        raise RuntimeError("jax backend requested but jax is not installed")
