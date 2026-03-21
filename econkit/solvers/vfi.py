"""Value function iteration scaffold for EconKit solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from econkit.solvers.diagnostics import bellman_residual

BellmanOperator = Callable[
    [npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None],
]
PolicyEvaluationOperator = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
]
IterationHook = Callable[
    [int, float, npt.NDArray[np.float64], npt.NDArray[np.float64] | None], None
]


@dataclass(frozen=True)
class VFIConfig:
    """Runtime options for value function iteration."""

    tol: float = 1e-8
    max_iter: int = 500
    relaxation: float = 1.0
    howard_policy_improvement_steps: int = 0
    store_history: bool = True

    def __post_init__(self) -> None:
        if self.tol <= 0.0:
            raise ValueError("tol must be strictly positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if not (0.0 < self.relaxation <= 1.0):
            raise ValueError("relaxation must be in (0, 1]")
        if self.howard_policy_improvement_steps < 0:
            raise ValueError("howard_policy_improvement_steps must be >= 0")


@dataclass(frozen=True)
class VFISolution:
    """Solution artifact produced by the VFI scaffold."""

    value_function: npt.NDArray[np.float64]
    policy: npt.NDArray[np.float64] | None
    converged: bool
    iterations: int
    residual: float
    residual_history: tuple[float, ...]
    metadata: dict[str, Any]


@dataclass
class VFISolver:
    """Thin solver object wrapping the VFI routine."""

    config: VFIConfig = field(default_factory=VFIConfig)

    def solve(
        self,
        initial_value: npt.NDArray[np.float64],
        bellman_operator: BellmanOperator,
        policy_evaluation_operator: PolicyEvaluationOperator | None = None,
        iteration_hook: IterationHook | None = None,
        model_name: str | None = None,
    ) -> VFISolution:
        return value_function_iteration(
            initial_value=initial_value,
            bellman_operator=bellman_operator,
            config=self.config,
            policy_evaluation_operator=policy_evaluation_operator,
            iteration_hook=iteration_hook,
            model_name=model_name,
        )


def value_function_iteration(
    initial_value: npt.NDArray[np.float64],
    bellman_operator: BellmanOperator,
    config: VFIConfig | None = None,
    policy_evaluation_operator: PolicyEvaluationOperator | None = None,
    iteration_hook: IterationHook | None = None,
    model_name: str | None = None,
) -> VFISolution:
    """Run value function iteration for a user-provided Bellman operator.

    Economics note:
    This scaffold implements the standard contraction-mapping loop used in many
    dynamic programming models. The Bellman operator is user supplied so model
    specifics stay outside the solver core.
    """
    cfg = config or VFIConfig()
    value = np.asarray(initial_value, dtype=np.float64).copy()

    residual_history: list[float] = []
    policy: npt.NDArray[np.float64] | None = None
    converged = False
    residual = float("inf")

    started_at = perf_counter()
    iterations = 0

    for iteration in range(1, cfg.max_iter + 1):
        previous_value = value.copy()
        candidate_value, candidate_policy = bellman_operator(value)
        next_value = np.asarray(candidate_value, dtype=np.float64)

        if next_value.shape != value.shape:
            raise ValueError("Bellman operator returned value with incompatible shape")

        relaxed_value = cfg.relaxation * next_value + (1.0 - cfg.relaxation) * value
        updated_value = relaxed_value
        iterations = iteration

        if candidate_policy is not None:
            policy_array = np.asarray(candidate_policy, dtype=np.float64)
            if policy_array.shape != updated_value.shape:
                raise ValueError("Bellman operator returned policy with incompatible shape")
            policy = policy_array

            if policy_evaluation_operator is not None and cfg.howard_policy_improvement_steps > 0:
                for _ in range(cfg.howard_policy_improvement_steps):
                    evaluated_value = policy_evaluation_operator(updated_value, policy)
                    evaluated_array = np.asarray(evaluated_value, dtype=np.float64)
                    if evaluated_array.shape != updated_value.shape:
                        raise ValueError(
                            "Policy evaluation operator returned value with incompatible shape"
                        )
                    updated_value = evaluated_array

        residual = bellman_residual(updated_value, previous_value, norm="sup")
        if cfg.store_history:
            residual_history.append(residual)

        value = updated_value

        if iteration_hook is not None:
            hook_policy = None if policy is None else policy.copy()
            iteration_hook(iteration, residual, value.copy(), hook_policy)

        if residual < cfg.tol:
            converged = True
            break

    elapsed = perf_counter() - started_at

    metadata: dict[str, Any] = {
        "algorithm": "vfi",
        "tolerance": cfg.tol,
        "max_iter": cfg.max_iter,
        "relaxation": cfg.relaxation,
        "howard_policy_improvement_steps": cfg.howard_policy_improvement_steps,
        "solve_time_seconds": elapsed,
    }
    if model_name is not None:
        metadata["model_name"] = model_name

    history_tuple = tuple(residual_history) if cfg.store_history else ()
    return VFISolution(
        value_function=value,
        policy=policy,
        converged=converged,
        iterations=iterations,
        residual=residual,
        residual_history=history_tuple,
        metadata=metadata,
    )
