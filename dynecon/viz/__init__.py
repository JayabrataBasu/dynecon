"""Matplotlib plots (optional ``viz`` extra); importing this package imports matplotlib lazily.

Economics note:
Plots make the economic content of results (policy kinks, distributions, impulse
responses) visible and comparable.
"""

from dynecon.viz.distribution import plot_distribution, plot_lorenz
from dynecon.viz.equilibrium import plot_equilibrium, plot_sweep
from dynecon.viz.estimation import plot_mcmc_trace, plot_moment_fit, plot_posterior
from dynecon.viz.policy import plot_convergence, plot_euler_errors, plot_policy, plot_value
from dynecon.viz.simulation import plot_irf, plot_simulation
from dynecon.viz.style import dynecon_style

__all__ = [
    "dynecon_style",
    "plot_convergence",
    "plot_distribution",
    "plot_equilibrium",
    "plot_euler_errors",
    "plot_irf",
    "plot_lorenz",
    "plot_mcmc_trace",
    "plot_moment_fit",
    "plot_policy",
    "plot_posterior",
    "plot_simulation",
    "plot_sweep",
    "plot_value",
]
