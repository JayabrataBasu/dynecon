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


@dataclass(frozen=True)
class VFIConfig:
    """Runtime options for value function iteration."""

    tol: float = 1e-8
    max_iter: int = 500
    relaxation: float = 1.0
    store_history: bool = True

    def __post_init__(self) -> None:
        if self.tol <= 0.0:
            raise ValueError("tol must be strictly positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if not (0.0 < self.relaxation <= 1.0):
            raise ValueError("relaxation must be in (0, 1]")


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
        model_name: str | None = None,
    ) -> VFISolution:
        return value_function_iteration(
            initial_value=initial_value,
            bellman_operator=bellman_operator,
            config=self.config,
            model_name=model_name,
        )


def value_function_iteration(
    initial_value: npt.NDArray[np.float64],
    bellman_operator: BellmanOperator,
    config: VFIConfig | None = None,
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
        candidate_value, candidate_policy = bellman_operator(value)
        next_value = np.asarray(candidate_value, dtype=np.float64)

        if next_value.shape != value.shape:
            raise ValueError("Bellman operator returned value with incompatible shape")

        relaxed_value = cfg.relaxation * next_value + (1.0 - cfg.relaxation) * value
        residual = bellman_residual(relaxed_value, value, norm="sup")
        iterations = iteration

        if cfg.store_history:
            residual_history.append(residual)

        value = relaxed_value

        if candidate_policy is not None:
            policy_array = np.asarray(candidate_policy, dtype=np.float64)
            if policy_array.shape != value.shape:
                raise ValueError("Bellman operator returned policy with incompatible shape")
            policy = policy_array

        if residual < cfg.tol:
            converged = True
            break

    elapsed = perf_counter() - started_at

    metadata: dict[str, Any] = {
        "algorithm": "vfi",
        "tolerance": cfg.tol,
        "max_iter": cfg.max_iter,
        "relaxation": cfg.relaxation,
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
