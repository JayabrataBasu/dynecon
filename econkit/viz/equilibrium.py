"""Plots for equilibrium and parameter sweeps."""

from __future__ import annotations

from typing import Any

import numpy as np

from econkit.core.errors import ValidationError
from econkit.experiment.sweep import SweepResult
from econkit.solvers.equilibrium import EquilibriumSolution
from econkit.viz.style import get_axes

__all__ = ["plot_equilibrium", "plot_sweep"]


def plot_equilibrium(eq: EquilibriumSolution, ax: Any | None = None) -> Any:
    """Excess demand against the price over the root-finder's evaluations; root marked."""
    price = np.asarray(eq.history.get("price", []))
    excess = np.asarray(eq.history.get("excess_demand", []))
    if price.size == 0:
        raise ValidationError("equilibrium solution has no evaluation history")
    order = np.argsort(price)
    ax = get_axes(ax)
    ax.plot(price[order], excess[order], "o-", label="excess demand")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(eq.price, color="C1", ls="--", label=f"{eq.price_name}* = {eq.price:.4f}")
    ax.set_xlabel(eq.price_name)
    ax.set_ylabel("excess demand")
    ax.legend(loc="best")
    return ax


def plot_sweep(result: SweepResult, x: str, y: str, ax: Any | None = None) -> Any:
    """Metric ``y`` against parameter ``x`` over the successful points of a sweep."""
    xs, ys = [], []
    for pt, met in zip(result.points, result.metrics, strict=True):
        if met is not None:
            if x not in pt or y not in met:
                raise ValidationError(f"sweep has no parameter {x!r} or metric {y!r}")
            xs.append(pt[x])
            ys.append(met[y])
    if not xs:
        raise ValidationError("no successful sweep points to plot")
    order = np.argsort(xs)
    ax = get_axes(ax)
    ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], "o-")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax
