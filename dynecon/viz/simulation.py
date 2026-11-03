"""Plots of simulations and impulse responses.

Economics note:
Simulated paths and impulse responses show the dynamics a model generates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dynecon.core.errors import ValidationError
from dynecon.simulation.result import SimulationResult
from dynecon.viz.style import dynecon_style, get_axes, get_pyplot

__all__ = ["plot_irf", "plot_simulation"]


def plot_simulation(
    result: SimulationResult,
    vars: Sequence[str] | None = None,
    agent: int = 0,
    ax: Any | None = None,
) -> Any:
    """Simulated paths of one agent for the chosen variables."""
    names = [v for v in (vars or result.paths) if np.issubdtype(result.paths[v].dtype, np.floating)]
    if not names:
        raise ValidationError("no floating-point variables to plot")
    ax = get_axes(ax)
    for name in names:
        ax.plot(result.paths[name][:, agent], label=name)
    ax.set_xlabel("period")
    ax.set_ylabel("level")
    ax.legend(loc="best", fontsize="small")
    return ax


def plot_irf(irf: Mapping[str, Any], variables: Sequence[str] | None = None) -> Any:
    """Impulse responses in a grid of panels, one per variable. Returns the figure."""
    names = list(variables or irf)
    plt = get_pyplot()
    ncols = min(3, len(names))
    nrows = -(-len(names) // ncols)
    with dynecon_style():
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(4 * ncols, 2.8 * nrows))
    for ax, name in zip(axes.ravel(), names, strict=False):
        y = np.asarray(irf[name])
        ax.plot(np.arange(y.size), y)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_title(name)
        ax.set_xlabel("horizon")
    for ax in axes.ravel()[len(names) :]:
        ax.set_visible(False)
    return fig
