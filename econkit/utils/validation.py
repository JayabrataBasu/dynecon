"""Input validation helpers raising ``ValidationError`` that name the argument.

Economics note:
Catching a malformed grid or transition matrix at the boundary prevents silently wrong
economic results.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from econkit.core.errors import ValidationError

__all__ = ["check_array", "check_increasing", "check_stochastic_matrix"]


def check_array(
    x: Any,
    name: str,
    ndim: int | None = None,
    shape: tuple[int | None, ...] | None = None,
    finite: bool = True,
) -> npt.NDArray[np.float64]:
    """Convert to a float64 array and validate dimension, shape and finiteness."""
    try:
        arr = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a numeric array: {exc}") from exc
    if ndim is not None and arr.ndim != ndim:
        raise ValidationError(f"{name} must have {ndim} dimension(s), got shape {arr.shape}")
    if shape is not None:
        ok = arr.ndim == len(shape) and all(
            s is None or s == d for s, d in zip(shape, arr.shape, strict=True)
        )
        if not ok:
            raise ValidationError(f"{name} must have shape {shape}, got {arr.shape}")
    if finite and not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} contains non-finite values")
    return arr


def check_increasing(x: npt.NDArray[np.float64], name: str, strict: bool = True) -> None:
    """Require a 1-D array to be (strictly) increasing."""
    d = np.diff(x)
    viol = d <= 0.0 if strict else d < 0.0
    if np.any(viol):
        kind = "strictly increasing" if strict else "non-decreasing"
        raise ValidationError(f"{name} must be {kind}; violated at index {int(np.argmax(viol))}")


def check_stochastic_matrix(P: Any, name: str, tol: float = 1e-10) -> npt.NDArray[np.float64]:
    """Require a square, non-negative matrix whose rows sum to one."""
    arr = check_array(P, name, ndim=2)
    if arr.shape[0] != arr.shape[1]:
        raise ValidationError(f"{name} must be square, got shape {arr.shape}")
    if np.any(arr < -tol):
        raise ValidationError(f"{name} has negative entries (min {np.min(arr):.3g})")
    err = float(np.max(np.abs(arr.sum(axis=1) - 1.0)))
    if err > tol:
        raise ValidationError(f"{name} rows must sum to one (max deviation {err:.3g})")
    return arr
