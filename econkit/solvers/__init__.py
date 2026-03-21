"""Solver APIs."""

from econkit.solvers.adapters import build_compiled_bellman_operator, solve_compiled_model_vfi
from econkit.solvers.canonical import solve_consumption_savings_vfi
from econkit.solvers.diagnostics import (
    ResidualHistorySummary,
    bellman_residual,
    format_convergence_report,
    summarize_residual_history,
)
from econkit.solvers.vfi import VFIConfig, VFISolution, VFISolver, value_function_iteration

__all__ = [
    "build_compiled_bellman_operator",
    "solve_compiled_model_vfi",
    "solve_consumption_savings_vfi",
    "ResidualHistorySummary",
    "bellman_residual",
    "summarize_residual_history",
    "format_convergence_report",
    "VFIConfig",
    "VFISolution",
    "VFISolver",
    "value_function_iteration",
]
