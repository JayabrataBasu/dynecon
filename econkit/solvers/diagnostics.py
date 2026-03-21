"""Solver diagnostics utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

NormName = Literal["sup", "l2"]


def bellman_residual(
    value_new: npt.NDArray[np.float64],
    value_old: npt.NDArray[np.float64],
    norm: NormName = "sup",
) -> float:
    """Compute Bellman residual between successive value iterates.

    Economics note:
    The Bellman residual measures policy/value fixed-point error and is the
    standard stopping diagnostic for value-function iteration.
    """
    new_arr = np.asarray(value_new, dtype=np.float64)
    old_arr = np.asarray(value_old, dtype=np.float64)

    if new_arr.shape != old_arr.shape:
        raise ValueError("value_new and value_old must have the same shape")

    delta = new_arr - old_arr
    if norm == "sup":
        return float(np.max(np.abs(delta)))
    if norm == "l2":
        return float(np.sqrt(np.mean(delta * delta)))
    raise ValueError(f"Unsupported residual norm: {norm}")
