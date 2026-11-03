"""1-D linear interpolation routed through the backend dispatcher.

Economics note:
Continuation values and policies are evaluated off-grid during policy iteration,
EGM and simulation; interpolation quality directly affects welfare comparisons.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from econkit.core.errors import ValidationError
from econkit.numerics._dispatch import resolve_backend
from econkit.numerics._registry import register
from econkit.numerics.backends.interp_numba import interp_linear_numba
from econkit.numerics.backends.interp_numpy import interp_linear_numpy
from econkit.utils.jit import NUMBA_AVAILABLE
from econkit.utils.validation import check_array, check_increasing

__all__ = ["linear_interp_1d"]

register("linear_interp_1d", "numpy", interp_linear_numpy)
if NUMBA_AVAILABLE:
    register("linear_interp_1d", "numba", interp_linear_numba)

_MODES = {"clamp": 0, "linear": 1}


def linear_interp_1d(
    x: Any,
    xp: Any,
    fp: Any,
    *,
    extrapolate: str = "clamp",
    backend: str | None = None,
) -> npt.NDArray[np.float64]:
    """Piecewise-linear interpolation of ``(xp, fp)`` at ``x`` (any shape).

    ``extrapolate="clamp"`` holds the end values outside the grid (used by simulation);
    ``"linear"`` continues the end segments (used by EGM).

    Economics note:
    Linear extrapolation keeps high-wealth consumption unbiased; clamping would
    flatten the policy above the grid.
    """
    if extrapolate not in _MODES:
        raise ValidationError(f"extrapolate must be one of {sorted(_MODES)}, got {extrapolate!r}")
    xp_a = check_array(xp, "xp", ndim=1)
    fp_a = check_array(fp, "fp", ndim=1)
    if xp_a.shape != fp_a.shape or xp_a.shape[0] < 2:
        raise ValidationError("xp and fp must be 1-D arrays of equal length >= 2")
    check_increasing(xp_a, "xp", strict=True)
    x_a = np.asarray(x, dtype=np.float64)
    _, impl = resolve_backend("linear_interp_1d", backend=backend)
    out = impl(np.ascontiguousarray(x_a.ravel()), xp_a, fp_a, _MODES[extrapolate])
    result: npt.NDArray[np.float64] = np.asarray(out, dtype=np.float64).reshape(x_a.shape)
    return result
