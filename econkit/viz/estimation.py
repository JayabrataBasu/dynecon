"""Plots for estimation: moment fit and MCMC diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from econkit.core.errors import ValidationError
from econkit.estimation.results import MCMCResult, SMMResult
from econkit.viz.style import get_axes

__all__ = ["plot_mcmc_trace", "plot_moment_fit", "plot_posterior"]


def plot_moment_fit(
    result: SMMResult, labels: list[str] | None = None, ax: Any | None = None
) -> Any:
    """Data moments (with standard-error bars) against model moments at the estimate."""
    q = result.data_moments.size
    se = np.sqrt(np.clip(np.diag(result.omega), 0.0, None) / result.n_obs)
    x = np.arange(q)
    ax = get_axes(ax)
    ax.errorbar(x - 0.08, result.data_moments, yerr=se, fmt="o", capsize=3, label="data")
    ax.plot(x + 0.08, result.model_moments, "s", label="model")
    ax.set_xticks(x, labels or [f"m{i}" for i in range(q)])
    ax.set_ylabel("moment")
    ax.legend(loc="best")
    return ax


def _param_index(result: MCMCResult, param: str | int) -> int:
    if isinstance(param, int):
        return param
    if param not in result.names:
        raise ValidationError(f"unknown parameter {param!r}; have {result.names}")
    return result.names.index(param)


def plot_mcmc_trace(result: MCMCResult, param: str | int = 0, ax: Any | None = None) -> Any:
    """Trace of one parameter, one line per chain."""
    i = _param_index(result, param)
    ax = get_axes(ax)
    for c in range(result.draws.shape[0]):
        ax.plot(result.draws[c, :, i], lw=0.6, label=f"chain {c}")
    ax.set_xlabel("draw")
    ax.set_ylabel(result.names[i])
    return ax


def plot_posterior(
    result: MCMCResult, param: str | int = 0, prior: Any | None = None, ax: Any | None = None
) -> Any:
    """Posterior histogram of one parameter, with the prior density overlaid if given."""
    i = _param_index(result, param)
    draws = result.pooled[:, i]
    ax = get_axes(ax)
    ax.hist(draws, bins=50, density=True, alpha=0.6, label="posterior")
    if prior is not None:
        grid = np.linspace(np.min(draws), np.max(draws), 300)
        ax.plot(grid, [np.exp(prior.logpdf(float(v))) for v in grid], label="prior")
    ax.set_xlabel(result.names[i])
    ax.set_ylabel("density")
    ax.legend(loc="best")
    return ax
