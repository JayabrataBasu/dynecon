"""Solver diagnostics utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

NormName = Literal["sup", "l2"]


@dataclass(frozen=True)
class ResidualHistorySummary:
    """Summary statistics for residual convergence history."""

    count: int
    first: float
    last: float
    min_value: float
    max_value: float
    monotone_nonincreasing: bool
    avg_log10_drop_per_step: float


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


def summarize_residual_history(history: list[float] | tuple[float, ...]) -> ResidualHistorySummary:
    """Return compact convergence diagnostics for a residual history."""
    if not history:
        raise ValueError("history must contain at least one residual value")

    arr = np.asarray(history, dtype=np.float64)
    if np.any(arr < 0.0):
        raise ValueError("residual history cannot contain negative values")

    count = int(arr.shape[0])
    first = float(arr[0])
    last = float(arr[-1])
    min_value = float(arr.min())
    max_value = float(arr.max())
    monotone_nonincreasing = bool(np.all(arr[1:] <= arr[:-1])) if count > 1 else True

    safe = np.maximum(arr, np.finfo(np.float64).tiny)
    if count > 1:
        avg_log10_drop_per_step = float((np.log10(safe[0]) - np.log10(safe[-1])) / (count - 1))
    else:
        avg_log10_drop_per_step = 0.0

    return ResidualHistorySummary(
        count=count,
        first=first,
        last=last,
        min_value=min_value,
        max_value=max_value,
        monotone_nonincreasing=monotone_nonincreasing,
        avg_log10_drop_per_step=avg_log10_drop_per_step,
    )


def format_convergence_report(history: list[float] | tuple[float, ...]) -> str:
    """Format a human-readable convergence report for logs and notebooks."""
    summary = summarize_residual_history(history)
    return (
        "Residual convergence report | "
        f"steps={summary.count}, "
        f"first={summary.first:.3e}, "
        f"last={summary.last:.3e}, "
        f"min={summary.min_value:.3e}, "
        f"max={summary.max_value:.3e}, "
        f"monotone_nonincreasing={summary.monotone_nonincreasing}, "
        f"avg_log10_drop_per_step={summary.avg_log10_drop_per_step:.3f}"
    )
