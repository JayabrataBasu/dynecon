"""Plots of distributions: marginal mass with CDF, and Lorenz curves.

Economics note:
The shape of the wealth distribution and its Lorenz curve summarise inequality.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from econkit.simulation.distribution import cdf, gini, lorenz_curve
from econkit.viz.style import econkit_style, get_axes, get_pyplot

__all__ = ["plot_distribution", "plot_lorenz"]


def plot_distribution(mass: npt.ArrayLike, grid: npt.ArrayLike) -> Any:
    """Marginal distribution (mass) and its CDF side by side. Returns the figure."""
    m, g = np.asarray(mass, dtype=np.float64), np.asarray(grid, dtype=np.float64)
    plt = get_pyplot()
    with econkit_style():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    a1.plot(g, m)
    a1.set_xlabel("value")
    a1.set_ylabel("mass")
    a2.plot(g, cdf(m))
    a2.set_xlabel("value")
    a2.set_ylabel("CDF")
    return fig


def plot_lorenz(
    values: npt.ArrayLike, weights: npt.ArrayLike | None = None, ax: Any | None = None
) -> Any:
    """Lorenz curve with the 45-degree line; the Gini coefficient is in the legend."""
    pop, share = lorenz_curve(values, weights)
    ax = get_axes(ax)
    ax.plot(pop, share, label=f"Lorenz (Gini = {gini(values, weights):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="equality")
    ax.set_xlabel("cumulative population share")
    ax.set_ylabel("cumulative value share")
    ax.legend(loc="best")
    return ax
