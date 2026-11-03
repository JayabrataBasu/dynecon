"""First-order perturbation (Klein 2000) for ``dsge`` problems.

Economics note:
First-order perturbation approximates a DSGE model around its steady state, the workhorse
of quantitative macroeconomics.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from dynecon.core.errors import ValidationError
from dynecon.model.compiled import CompiledModel
from dynecon.model.problems import DSGEProblem
from dynecon.solvers.perturbation.analysis import (
    PerturbationSolution,
    TheoreticalMoments,
    irf,
    simulate_linear,
    theoretical_moments,
)
from dynecon.solvers.perturbation.klein import klein_solve
from dynecon.solvers.perturbation.steady_state import linearize, steady_state
from dynecon.solvers.registry import SOLVERS, resolve_config
from dynecon.utils.provenance import collect_provenance

__all__ = [
    "PerturbationConfig",
    "PerturbationSolution",
    "TheoreticalMoments",
    "irf",
    "simulate_linear",
    "solve_perturbation",
    "theoretical_moments",
]


@dataclass(frozen=True)
class PerturbationConfig:
    """Options of :func:`solve_perturbation`."""

    ss_tol: float = 1e-12
    fd_eps: float = 1e-6

    def __post_init__(self) -> None:
        if not self.ss_tol > 0.0 or not self.fd_eps > 0.0:
            raise ValidationError("ss_tol and fd_eps must be strictly positive")


def solve_perturbation(
    model: CompiledModel, *, backend: str | None = None, **options: Any
) -> PerturbationSolution:
    """Solve a ``dsge`` problem to first order (see :class:`PerturbationConfig`)."""
    cfg = resolve_config(PerturbationConfig, options)
    prob = model.problem
    if not isinstance(prob, DSGEProblem):
        raise ValidationError("perturbation needs a 'dsge' problem")
    t0 = perf_counter()
    y_ss = steady_state(prob, model.p, cfg.ss_tol)
    if backend == "jax":
        from dynecon.solvers.perturbation.jacobian_jax import linearize_jax

        A, B, sigma = linearize_jax(prob, model.p, y_ss)
    else:
        A, B, sigma = linearize(prob, model.p, y_ss, cfg.fd_eps)
    F, P, eig = klein_solve(A, B, len(prob.states))
    return PerturbationSolution(
        method="perturbation",
        converged=True,
        iterations=0,
        residual=0.0,
        elapsed=perf_counter() - t0,
        backend="jax" if backend == "jax" else "numpy",
        model_name=model.name,
        params=model.parameters.values,
        grids={},
        policy={"F": F, "P": P, "Sigma": sigma},
        value=None,
        diagnostics={
            "n_stable_roots": int(np.sum(eig < 1.0 - 1e-9)),
            "n_predetermined": len(prob.states),
            "max_stable_root": float(np.max(eig[eig < 1.0 - 1e-9], initial=0.0)),
        },
        metadata={
            "fingerprint": model.fingerprint(),
            "provenance": collect_provenance(),
            "config": cfg,
        },
        model=model,
        variables=prob.variables,
        states=prob.states,
        shocks=prob.shocks,
        log_variables=prob.log_variables,
        steady_state=y_ss,
    )


SOLVERS["perturbation"] = solve_perturbation  # type: ignore[assignment]
