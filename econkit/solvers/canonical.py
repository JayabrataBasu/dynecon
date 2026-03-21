"""Canonical model helpers built on top of generic solver primitives."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from econkit.model.compiled import CompiledModel
from econkit.model.specs import State
from econkit.solvers.vfi import VFIConfig, VFISolution, value_function_iteration

Array = npt.NDArray[np.float64]


def solve_consumption_savings_vfi(
    compiled_model: CompiledModel,
    config: VFIConfig | None = None,
) -> VFISolution:
    """Solve a simple deterministic consumption-savings problem via VFI.

    Economics note:
    This helper provides a practical baseline for a one-asset consumption-
    savings model with CRRA utility and deterministic resource dynamics.
    """
    cfg = config or VFIConfig(tol=1e-8, max_iter=400)

    assets_state = _require_assets_state(compiled_model)
    beta = _require_param(compiled_model, "beta")
    sigma = _require_param(compiled_model, "sigma")
    interest = _require_param(compiled_model, "r")
    income = _optional_param(compiled_model, "y", default=1.0)
    borrowing_floor = _optional_param(compiled_model, "a_min", default=0.0)

    if assets_state.bounds is None:
        raise ValueError("assets state must define bounds for canonical helper")
    if assets_state.n_points is None or assets_state.n_points < 2:
        raise ValueError("assets state must define n_points >= 2 for canonical helper")

    a_lo = max(assets_state.bounds[0], borrowing_floor)
    a_hi = assets_state.bounds[1]
    if not (a_hi > a_lo):
        raise ValueError("assets bounds must satisfy upper > lower")

    grid = np.linspace(a_lo, a_hi, assets_state.n_points, dtype=np.float64)

    def bellman_operator(value: Array) -> tuple[Array, Array]:
        next_value = np.empty_like(value)
        policy_idx = np.empty_like(value)

        for i in range(grid.shape[0]):
            a_now = grid[i]
            resources = (1.0 + interest) * a_now + income

            best_val = -np.inf
            best_j = 0
            for j in range(grid.shape[0]):
                a_next = grid[j]
                c = resources - a_next
                if c <= 0.0:
                    continue
                util = _crra_utility(c, sigma)
                candidate = util + beta * value[j]
                if candidate > best_val:
                    best_val = candidate
                    best_j = j

            if not math.isfinite(best_val):
                raise ValueError("No feasible consumption choice found on grid")

            next_value[i] = best_val
            policy_idx[i] = float(best_j)

        return next_value, policy_idx

    initial = np.zeros_like(grid)
    solution = value_function_iteration(
        initial_value=initial,
        bellman_operator=bellman_operator,
        config=cfg,
        model_name=compiled_model.name,
    )

    metadata = dict(solution.metadata)
    metadata["canonical_model"] = "consumption_savings"
    metadata["grid_size"] = int(grid.shape[0])
    metadata["asset_grid_min"] = float(a_lo)
    metadata["asset_grid_max"] = float(a_hi)

    policy_grid = None
    if solution.policy is not None:
        idx = np.asarray(np.rint(solution.policy), dtype=np.int64)
        idx = np.clip(idx, 0, grid.shape[0] - 1)
        policy_grid = grid[idx]

    return VFISolution(
        value_function=solution.value_function,
        policy=policy_grid,
        converged=solution.converged,
        iterations=solution.iterations,
        residual=solution.residual,
        residual_history=solution.residual_history,
        metadata=metadata,
    )


def _crra_utility(consumption: float, sigma: float) -> float:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if abs(sigma - 1.0) < 1e-12:
        return math.log(consumption)
    out = (consumption ** (1.0 - sigma) - 1.0) / (1.0 - sigma)
    return float(out)


def _require_assets_state(compiled_model: CompiledModel) -> State:
    assets_state = compiled_model.states.get("assets")
    if assets_state is None:
        raise ValueError("canonical helper requires a state named 'assets'")
    return assets_state


def _require_param(compiled_model: CompiledModel, name: str) -> float:
    param = compiled_model.parameters.get(name)
    if param is None:
        raise ValueError(f"canonical helper requires parameter '{name}'")
    return float(param.value)


def _optional_param(compiled_model: CompiledModel, name: str, default: float) -> float:
    param = compiled_model.parameters.get(name)
    return default if param is None else float(param.value)
