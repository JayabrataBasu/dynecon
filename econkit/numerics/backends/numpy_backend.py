"""NumPy reference kernels.

These implementations are the permanent correctness baseline.
"""

import numpy as np
import numpy.typing as npt


def linear_interp_1d_numpy(
    x: npt.NDArray[np.float64],
    xp: npt.NDArray[np.float64],
    fp: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    out = np.interp(x, xp, fp)
    return np.asarray(out, dtype=np.float64)
