"""EconKit public package entrypoint."""

from econkit.config import config
from econkit.model import AR1Shock, Control, Model, Param, State
from econkit.numerics.interpolation import linear_interp_1d
from econkit.solvers import (
    ResidualHistorySummary,
    VFIConfig,
    VFISolver,
    bellman_residual,
    build_convergence_artifact,
    build_compiled_bellman_operator,
    convergence_artifact_to_json,
    format_convergence_report,
    solve_consumption_savings_stochastic_vfi,
    solve_consumption_savings_vfi,
    solve_compiled_model_vfi,
    summarize_residual_history,
    value_function_iteration,
)

__all__ = [
    "config",
    "linear_interp_1d",
    "Model",
    "Param",
    "State",
    "Control",
    "AR1Shock",
    "VFIConfig",
    "VFISolver",
    "value_function_iteration",
    "bellman_residual",
    "build_compiled_bellman_operator",
    "solve_compiled_model_vfi",
    "solve_consumption_savings_vfi",
    "solve_consumption_savings_stochastic_vfi",
    "ResidualHistorySummary",
    "summarize_residual_history",
    "format_convergence_report",
    "build_convergence_artifact",
    "convergence_artifact_to_json",
]
__version__ = "0.1.0.dev0"
