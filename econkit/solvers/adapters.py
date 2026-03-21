"""Adapters bridging compiled models and solver routines."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from econkit.model.compiled import CompiledModel
from econkit.solvers.vfi import VFIConfig, VFISolution, value_function_iteration

BellmanOperator = Callable[
    [npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None],
]


def build_compiled_bellman_operator(
    compiled_model: CompiledModel,
    equation_name: str = "bellman",
) -> BellmanOperator:
    """Create a Bellman operator from a compiled model equation.

    Economics note:
    This adapter provides a narrow bridge from model-layer equations to solver
    iteration, reducing boilerplate in canonical VFI experiments.
    """

    def _operator(
        value: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
        parameter_values = {name: param.value for name, param in compiled_model.parameters.items()}

        result: Any
        try:
            result = compiled_model.call_equation(
                equation_name,
                np.asarray(value, dtype=np.float64),
                parameter_values,
            )
        except TypeError:
            result = compiled_model.call_equation(
                equation_name,
                np.asarray(value, dtype=np.float64),
            )

        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError("Bellman equation tuple output must be (value, policy)")
            new_value = np.asarray(result[0], dtype=np.float64)
            raw_policy = result[1]
            policy = None if raw_policy is None else np.asarray(raw_policy, dtype=np.float64)
            return new_value, policy

        return np.asarray(result, dtype=np.float64), None

    return _operator


def solve_compiled_model_vfi(
    compiled_model: CompiledModel,
    initial_value: npt.NDArray[np.float64],
    equation_name: str = "bellman",
    config: VFIConfig | None = None,
) -> VFISolution:
    """Solve a compiled model with the VFI scaffold via equation adapter."""
    operator = build_compiled_bellman_operator(compiled_model, equation_name=equation_name)
    return value_function_iteration(
        initial_value=np.asarray(initial_value, dtype=np.float64),
        bellman_operator=operator,
        config=config,
        model_name=compiled_model.name,
    )
