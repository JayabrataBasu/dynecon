"""Plots of solutions: policy, value, Euler errors and convergence."""

from __future__ import annotations

from typing import Any

import numpy as np

from econkit.core.errors import ValidationError
from econkit.core.solution import Solution
from econkit.model.problems import SavingProblem
from econkit.solvers.diagnostics import euler_error_grid
from econkit.viz.style import get_axes

__all__ = ["plot_convergence", "plot_euler_errors", "plot_policy", "plot_value"]


def _asset_grid(solution: Solution) -> tuple[Any, str]:
    model = solution.model
    prob = None if model is None else model.problem
    if not isinstance(prob, SavingProblem):
        raise ValidationError("this plot needs a Solution of a 'saving' problem")
    return solution.grids[prob.asset], prob.asset


def plot_policy(
    solution: Solution, var: str, state_slice: int | None = None, ax: Any | None = None
) -> Any:
    """Policy ``var`` against the asset grid, one line per shock state (or one: ``state_slice``).

    Economics note:
    The kink at the borrowing constraint and the concavity of consumption are the visible
    signatures of precautionary saving.
    """
    grid, asset = _asset_grid(solution)
    if var not in solution.policy:
        raise ValidationError(f"no policy {var!r}; available: {sorted(solution.policy)}")
    values = np.asarray(solution.policy[var])
    ax = get_axes(ax)
    cols = range(values.shape[1]) if state_slice is None else [state_slice]
    for j in cols:
        ax.plot(grid, values[:, j], label=f"state {j}")
    ax.set_xlabel(asset)
    ax.set_ylabel(var)
    ax.legend(loc="best", fontsize="small")
    return ax


def plot_value(solution: Solution, ax: Any | None = None) -> Any:
    """Value function against the asset grid (needs a method that computes it, e.g. VFI)."""
    grid, asset = _asset_grid(solution)
    if solution.value is None:
        raise ValidationError("this solution has no value function (EGM does not compute one)")
    ax = get_axes(ax)
    for j in range(solution.value.shape[1]):
        ax.plot(grid, solution.value[:, j], label=f"state {j}")
    ax.set_xlabel(asset)
    ax.set_ylabel("value")
    ax.legend(loc="best", fontsize="small")
    return ax


def plot_euler_errors(solution: Solution, ax: Any | None = None) -> Any:
    """Euler-equation errors (log10) against assets at interior states."""
    grid, asset = _asset_grid(solution)
    err = euler_error_grid(solution)
    ax = get_axes(ax)
    for j in range(err.shape[1]):
        ok = np.isfinite(err[:, j])
        ax.plot(grid[ok], err[ok, j], ".-", label=f"state {j}", markersize=3)
    ax.set_xlabel(asset)
    ax.set_ylabel("log10 Euler error")
    ax.legend(loc="best", fontsize="small")
    return ax


def plot_convergence(solution: Solution, ax: Any | None = None) -> Any:
    """Residual history on a log scale."""
    hist = solution.diagnostics.get("residual_history")
    if not hist:
        raise ValidationError("this solution has no residual history")
    ax = get_axes(ax)
    ax.plot(np.arange(1, len(hist) + 1), hist)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual")
    return ax
