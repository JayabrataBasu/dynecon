"""Lucas-tree asset pricing by fixed point on the price functional.

Economics note:
In an endowment economy the price of a claim to the dividend stream equals the expected
discounted value of next period's price plus dividend, weighted by marginal utility growth.
The price function is the fixed point of that operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt

from dynecon.core.errors import ValidationError
from dynecon.core.solution import Solution
from dynecon.model.compiled import CompiledModel
from dynecon.model.problems import PricingProblem
from dynecon.numerics.fixedpoint import fixed_point
from dynecon.solvers.registry import SOLVERS, resolve_config
from dynecon.utils.provenance import collect_provenance

__all__ = ["PricingConfig", "price_one_period_claim", "solve_pricing"]

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PricingConfig:
    """Options of :func:`solve_pricing`."""

    tol: float = 1e-13
    max_iter: int = 10000

    def __post_init__(self) -> None:
        if not self.tol > 0.0 or self.max_iter < 1:
            raise ValidationError("tol must be positive and max_iter >= 1")


def _parts(model: CompiledModel) -> tuple[PricingProblem, Array, Array, Array, float]:
    prob = model.problem
    if not isinstance(prob, PricingProblem):
        raise ValidationError("asset pricing needs a 'pricing' problem")
    shock = model.shock_process(prob.shock)
    mu = np.asarray(prob.marginal_utility(shock.values, model.p), dtype=np.float64)
    return prob, shock.values, shock.P, mu, model.parameters.values[prob.discount]


def price_one_period_claim(solution: Solution, payoff: npt.ArrayLike) -> Array:
    """Price today (by state) of a claim paying ``payoff[state']`` next period only."""
    assert solution.model is not None
    _, _, P, mu, beta = _parts(solution.model)
    pay = np.asarray(payoff, dtype=np.float64)
    return np.asarray(beta / mu * (P @ (mu * pay)), dtype=np.float64)


def solve_pricing(model: CompiledModel, *, backend: str | None = None, **options: Any) -> Solution:
    """Price function of the Lucas tree (policy: ``price``, ``price_dividend``, ``risk_free``)."""
    del backend
    cfg = resolve_config(PricingConfig, options)
    prob, y, P, mu, beta = _parts(model)
    t0 = perf_counter()

    def operator(p: Array) -> Array:
        return np.asarray(beta / mu * (P @ (mu * (p + y))), dtype=np.float64)

    res = fixed_point(
        operator, np.zeros_like(y), tol=cfg.tol, max_iter=cfg.max_iter, name="pricing"
    )
    q = beta / mu * (P @ mu)  # price of a riskless one-period bond
    return Solution(
        method="pricing",
        converged=res.converged,
        iterations=res.iterations,
        residual=res.residual,
        elapsed=perf_counter() - t0,
        backend="numpy",
        model_name=model.name,
        params=model.parameters.values,
        grids={prob.shock: y},
        policy={"price": res.x, "price_dividend": res.x / y, "risk_free": 1.0 / q - 1.0},
        value=None,
        diagnostics={"residual_history": list(res.history)},
        metadata={
            "fingerprint": model.fingerprint(),
            "provenance": collect_provenance(),
            "config": cfg,
        },
        model=model,
    )


SOLVERS["pricing"] = solve_pricing  # type: ignore[assignment]
