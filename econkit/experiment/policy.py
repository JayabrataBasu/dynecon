"""Counterfactual policy experiments with consumption-equivalent welfare.

Economics note:
A policy experiment solves the household problem under a baseline and a reform and
compares aggregates, distributions and welfare across the two stationary economies.
Transition costs between them are out of scope: the comparison is stationary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from econkit.core.errors import ValidationError
from econkit.core.solution import Solution
from econkit.experiment.welfare import cev
from econkit.model import Model
from econkit.model.compiled import CompiledModel
from econkit.model.problems import SavingProblem
from econkit.simulation.distribution import (
    aggregate,
    gini,
    marginal_distribution,
    stationary_distribution_young,
)
from econkit.solvers.api import solve

__all__ = ["PolicyComparison", "PolicyExperiment"]

Reform = Mapping[str, float] | Callable[[Model], Model]


@dataclass
class PolicyComparison:
    """Baseline and reform solutions, distributions, aggregates and welfare."""

    baseline: Solution
    reform: Solution
    distributions: dict[str, Any]
    aggregates: dict[str, dict[str, float]]
    welfare: dict[str, Any] = field(default_factory=dict)


def _aggregates(sol: Solution, mu: Any) -> dict[str, float]:
    model = sol.model
    assert model is not None and isinstance(model.problem, SavingProblem)
    a = sol.grids[model.problem.asset]
    cons = sol.policy[model.problem.control]
    wealth = marginal_distribution(mu, 0)
    shifted = a - a[0]
    out = {
        "mean_assets": aggregate(mu, a[:, None]),
        "mean_consumption": aggregate(mu, cons),
        "borrowing_constrained_share": float(
            np.sum(mu[sol.policy[f"{model.problem.asset}_next"] <= a[0] + 1e-10])
        ),
    }
    if shifted.sum() > 0 and np.sum(wealth * shifted) > 0:
        out["wealth_gini"] = float(gini(shifted, wealth))
    return out


@dataclass
class PolicyExperiment:
    """Compare a baseline household economy with a reform.

    ``reform`` is a dict of parameter overrides, or a function ``Model -> Model`` (use
    :meth:`Model.copy` and :meth:`Model.replace_state`); the latter needs ``baseline`` as a
    :class:`Model`.
    """

    baseline: Model | CompiledModel
    reform: Reform
    welfare: str = "cev"
    solve_options: dict[str, Any] = field(default_factory=dict)

    def _models(self) -> tuple[CompiledModel, CompiledModel]:
        base = (
            self.baseline if isinstance(self.baseline, CompiledModel) else self.baseline.compile()
        )
        if callable(self.reform):
            if not isinstance(self.baseline, Model):
                raise ValidationError(
                    "a reform function needs the baseline as a Model, not compiled"
                )
            return base, self.reform(self.baseline.copy()).compile()
        return base, base.with_params(**dict(self.reform))

    def run(self) -> PolicyComparison:
        if self.welfare != "cev":
            raise ValidationError("only welfare='cev' is implemented")
        base_m, reform_m = self._models()
        sols, mus = [], []
        for m in (base_m, reform_m):
            sol = solve(m, **self.solve_options)
            sols.append(sol)
            mus.append(stationary_distribution_young(sol))
        agg = {"baseline": _aggregates(sols[0], mus[0]), "reform": _aggregates(sols[1], mus[1])}
        welfare: dict[str, Any] = {"cev": cev(sols[0], sols[1], mus[0])}
        same_grids = all(
            k in sols[1].grids and np.array_equal(sols[0].grids[k], sols[1].grids[k])
            for k in sols[0].grids
        )
        if same_grids:
            welfare["cev_state"] = cev(sols[0], sols[1])
        return PolicyComparison(
            sols[0], sols[1], {"baseline": mus[0], "reform": mus[1]}, agg, welfare
        )
