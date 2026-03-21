"""Solver APIs."""

from econkit.solvers.adapters import build_compiled_bellman_operator, solve_compiled_model_vfi
from econkit.solvers.canonical import (
    solve_consumption_savings_stochastic_vfi,
    solve_consumption_savings_vfi,
)
from econkit.solvers.diagnostics import (
    ResidualHistorySummary,
    bellman_residual,
    build_convergence_artifact,
    convergence_artifact_to_json,
    format_convergence_report,
    summarize_residual_history,
)
from econkit.solvers.vfi import VFIConfig, VFISolution, VFISolver, value_function_iteration

__all__ = [
    "build_compiled_bellman_operator",
    "solve_compiled_model_vfi",
    "solve_consumption_savings_vfi",
    "solve_consumption_savings_stochastic_vfi",
    "ResidualHistorySummary",
    "bellman_residual",
    "summarize_residual_history",
    "format_convergence_report",
    "build_convergence_artifact",
    "convergence_artifact_to_json",
    "VFIConfig",
    "VFISolution",
    "VFISolver",
    "value_function_iteration",
]
