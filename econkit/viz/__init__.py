"""Matplotlib plots (optional ``viz`` extra); importing this package imports matplotlib lazily.

Economics note:
Plots make the economic content of results (policy kinks, distributions, impulse
responses) visible and comparable.
"""

from econkit.viz.distribution import plot_distribution, plot_lorenz
from econkit.viz.equilibrium import plot_equilibrium, plot_sweep
from econkit.viz.estimation import plot_mcmc_trace, plot_moment_fit, plot_posterior
from econkit.viz.policy import plot_convergence, plot_euler_errors, plot_policy, plot_value
from econkit.viz.simulation import plot_irf, plot_simulation
from econkit.viz.style import econkit_style

__all__ = [
    "econkit_style",
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
