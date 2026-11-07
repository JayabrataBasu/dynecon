"""Generic stationary-equilibrium engine for heterogeneous-agent economies.

Economics note:
In a Bewley-Huggett-Aiyagari economy households face an exogenous price (an interest rate);
the equilibrium price is the one at which the stationary cross-section of household
choices clears the market (assets supplied = capital demanded, or bonds in zero supply).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt

from dynecon.core.errors import ConvergenceError, ValidationError
from dynecon.core.solution import Solution
from dynecon.model.compiled import CompiledModel
from dynecon.simulation.distribution import stationary_distribution_young
from dynecon.solvers.api import solve

__all__ = ["EquilibriumSolution", "solve_stationary_equilibrium"]

Array = npt.NDArray[np.float64]
ExcessDemand = Callable[[float, Solution, Array], tuple[float, dict[str, float]]]


@dataclass(kw_only=True)
class EquilibriumSolution(Solution):
    """A :class:`Solution` of the household problem at the market-clearing price."""

    price_name: str = ""
    price: float = float("nan")
    excess_demand: float = float("nan")
    aggregates: dict[str, float] = field(default_factory=dict)
    household_solution: Solution | None = None
    distribution: Array | None = None
    history: dict[str, list[float]] = field(default_factory=dict)


def solve_stationary_equilibrium(
    model_factory: Callable[[float], CompiledModel],
    price_name: str,
    bracket: tuple[float, float],
    excess_demand: ExcessDemand,
    *,
    tol: float = 1e-8,
    warm_start: bool = True,
    method: str = "auto",
    solve_options: dict[str, Any] | None = None,
) -> EquilibriumSolution:
    """Find the price clearing a market, given the household problem at each price.

    ``model_factory(p)`` builds the household model at price ``p``; ``excess_demand(p, sol, mu)``
    returns ``(excess, aggregates)`` from the household solution ``sol`` and stationary mass
    ``mu``. Household solves are warm-started from the previous price and cached by price.
    Raises :class:`ConvergenceError` (with the endpoint excess demands) if the bracket has no
    sign change.
    """
    from scipy.optimize import brentq

    t0 = perf_counter()
    lo, hi = bracket
    if not lo < hi:
        raise ValidationError("bracket must satisfy lo < hi")
    opts = dict(solve_options or {})
    cache: dict[float, tuple[float, Solution, Array, dict[str, float]]] = {}
    history: dict[str, list[float]] = {"price": [], "excess_demand": [], "inner_iterations": []}
    last: dict[str, Any] = {"init": None}

    def evaluate(price: float) -> float:
        key = float(price)
        if key not in cache:
            model = model_factory(key)
            kw = dict(opts)
            if warm_start and last["init"] is not None:
                kw["initial"] = last["init"]
            sol = solve(model, method=method, **kw)
            if not sol.converged:
                raise ConvergenceError(f"household problem did not converge at {price_name}={key}")
            mu = stationary_distribution_young(sol)
            excess, agg = excess_demand(key, sol, mu)
            if sol.value is not None:
                last["init"] = sol.value
            else:
                last["init"] = sol.policy[sol.model.problem.control]  # type: ignore[union-attr]
            cache[key] = (float(excess), sol, mu, agg)
            history["price"].append(key)
            history["excess_demand"].append(float(excess))
            history["inner_iterations"].append(float(sol.iterations))
        return cache[key][0]

    f_lo, f_hi = evaluate(lo), evaluate(hi)
    if f_lo * f_hi > 0.0:
        raise ConvergenceError(
            f"no sign change of excess demand on {price_name} in [{lo}, {hi}]: "
            f"excess demand is {f_lo:.6g} at the lower end and {f_hi:.6g} at the upper end"
        )
    root = float(brentq(evaluate, lo, hi, xtol=tol, rtol=4 * np.finfo(float).eps, maxiter=200))
    evaluate(root)
    excess, hh, mu, agg = cache[root]
    history["inner_iterations_total"] = [float(sum(history["inner_iterations"]))]
    return EquilibriumSolution(
        method=f"equilibrium({hh.method})",
        converged=True,
        iterations=len(history["price"]),
        residual=abs(excess),
        elapsed=perf_counter() - t0,
        backend=hh.backend,
        model_name=hh.model_name,
        params=hh.params,
        grids=hh.grids,
        policy=hh.policy,
        value=hh.value,
        diagnostics=dict(hh.diagnostics),
        metadata=dict(hh.metadata),
        model=hh.model,
        price_name=price_name,
        price=root,
        excess_demand=excess,
        aggregates=agg,
        household_solution=hh,
        distribution=mu,
        history=history,
    )
