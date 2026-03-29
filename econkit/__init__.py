"""EconKit public package entrypoint."""

from econkit.config import config
from econkit.estimation import (
    MomentTarget,
    estimation_stub_report,
    extract_moments_from_simulation,
    quadratic_moment_loss,
)
from econkit.model import AR1Shock, Control, Model, Param, State
from econkit.numerics.interpolation import linear_interp_1d
from econkit.simulation import SimulationResult, simulate_markov_asset_policy, summarize_simulation
from econkit.solvers import (
    PolicyMonotonicitySummary,
    ResidualHistorySummary,
    VFIConfig,
    VFISolver,
    bellman_residual,
    build_convergence_artifact,
    build_compiled_bellman_operator,
    convergence_artifact_to_json,
    format_convergence_report,
    solve_consumption_savings_liquid_illiquid_vfi,
    solve_consumption_savings_stochastic_vfi,
    solve_consumption_savings_vfi,
    solve_compiled_model_vfi,
    summarize_residual_history,
    summarize_policy_monotonicity,
    validate_convergence_artifact_schema,
    validate_convergence_schema_version,
    value_function_iteration,
    write_convergence_artifact,
)

__all__ = [
    "config",
    "linear_interp_1d",
    "SimulationResult",
    "simulate_markov_asset_policy",
    "summarize_simulation",
    "MomentTarget",
    "extract_moments_from_simulation",
    "quadratic_moment_loss",
    "estimation_stub_report",
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
    "solve_consumption_savings_liquid_illiquid_vfi",
    "ResidualHistorySummary",
    "PolicyMonotonicitySummary",
    "summarize_residual_history",
    "summarize_policy_monotonicity",
    "validate_convergence_artifact_schema",
    "validate_convergence_schema_version",
    "format_convergence_report",
    "build_convergence_artifact",
    "convergence_artifact_to_json",
    "write_convergence_artifact",
]
__version__ = "0.1.0.dev0"
