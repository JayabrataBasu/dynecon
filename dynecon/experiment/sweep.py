"""Parameter sweeps and elasticities.

Economics note:
Comparative statics ask how outcomes move with a parameter (a tax rate, a discount factor);
a sweep solves the model on a grid of parameters and records chosen outcomes at each point.
"""

from __future__ import annotations

import itertools
import multiprocessing
import warnings
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dynecon.core.errors import DyneconError, ValidationError
from dynecon.core.solution import Solution
from dynecon.experiment._build import build_model
from dynecon.model.compiled import CompiledModel
from dynecon.solvers.api import solve

__all__ = ["ParameterSweep", "SweepResult", "elasticities"]

Metric = Callable[[Solution], Mapping[str, float]]


@dataclass(frozen=True)
class SweepResult:
    """Results in grid order: parameter points, statuses, messages and metrics."""

    points: list[dict[str, float]]
    status: list[str]
    messages: list[str]
    metrics: list[dict[str, float] | None]

    def to_dataframe(self) -> Any:
        """One row per grid point (parameters, metrics, status); needs pandas."""
        import pandas as pd

        rows = []
        for pt, st, msg, met in zip(
            self.points, self.status, self.messages, self.metrics, strict=True
        ):
            rows.append({**pt, **(met or {}), "status": st, "message": msg})
        return pd.DataFrame(rows)

    def best(self, metric: str, maximize: bool = True) -> dict[str, float]:
        """The parameter point with the largest (or smallest) successful ``metric``."""
        ok = [(m[metric], p) for p, m in zip(self.points, self.metrics, strict=True) if m]
        if not ok:
            raise ValidationError("no successful points")
        return dict((max if maximize else min)(ok, key=lambda t: t[0])[1])

    def column(self, metric: str) -> np.ndarray:
        """Metric values in grid order (NaN for failed points)."""
        return np.array([np.nan if m is None else m[metric] for m in self.metrics])


def _run_point(
    task: tuple[Any, dict[str, float], str, str | None, dict[str, Any], Metric],
) -> tuple[str, str, Any]:
    spec, point, method, backend, opts, function = task
    try:
        model = spec if isinstance(spec, CompiledModel) else build_model(spec).compile()
        sol = solve(model.with_params(**point), method=method, backend=backend, **opts)
        return "ok", "", {k: float(v) for k, v in function(sol).items()}
    except (DyneconError, ValueError, ArithmeticError) as exc:
        return "failed", f"{type(exc).__name__}: {exc}", None


@dataclass
class ParameterSweep:
    """Solve a model over a grid of parameters and evaluate ``function(solution)`` at each.

    ``grid`` is a dict of lists (Cartesian product, first key slowest) or a list of dicts.
    ``model`` is a :class:`CompiledModel` (``n_jobs=1``) or a picklable build spec
    (``"name"`` / ``("name", overrides)`` / ``"module:function"``) required for ``n_jobs > 1``;
    ``function`` must then be an importable module-level function. Failed points are recorded
    (``status="failed"``) unless ``fail_fast``. With ``backend="jax"`` and ``method="egm"`` the
    whole grid is solved in one batch.
    """

    model: Any
    grid: Mapping[str, Sequence[float]] | Sequence[Mapping[str, float]]
    function: Metric
    solve_options: dict[str, Any] = field(default_factory=dict)
    n_jobs: int = 1
    backend: str | None = None
    method: str = "auto"
    fail_fast: bool = False

    def points(self) -> list[dict[str, float]]:
        if isinstance(self.grid, Mapping):
            names = list(self.grid)
            return [
                dict(zip(names, vals, strict=True))
                for vals in itertools.product(*(list(self.grid[n]) for n in names))
            ]
        return [dict(p) for p in self.grid]

    def run(self) -> SweepResult:
        pts = self.points()
        if not pts:
            raise ValidationError("the parameter grid is empty")
        if self.n_jobs > 1 and isinstance(self.model, CompiledModel):
            raise ValidationError(
                "n_jobs > 1 needs a build spec (compiled models cannot be pickled)"
            )
        if self.backend == "jax" and self.method == "egm":
            return self._run_batched(pts)
        tasks = [
            (self.model, pt, self.method, self.backend, self.solve_options, self.function)
            for pt in pts
        ]
        if self.n_jobs > 1:
            with ProcessPoolExecutor(
                max_workers=self.n_jobs, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                outcomes = list(pool.map(_run_point, tasks))
        else:
            outcomes = [_run_point(t) for t in tasks]
        if self.fail_fast and any(o[0] == "failed" for o in outcomes):
            first = next(o for o in outcomes if o[0] == "failed")
            raise DyneconError(f"sweep failed: {first[1]}")
        return SweepResult(
            pts, [o[0] for o in outcomes], [o[1] for o in outcomes], [o[2] for o in outcomes]
        )

    def _run_batched(self, pts: list[dict[str, float]]) -> SweepResult:
        from dynecon.solvers.batch import solve_many

        model = (
            self.model
            if isinstance(self.model, CompiledModel)
            else build_model(self.model).compile()
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sols = solve_many(
                model,
                pts,
                **{k: v for k, v in self.solve_options.items() if k in ("tol", "max_iter")},
            )
        status: list[str] = []
        messages: list[str] = []
        metrics: list[dict[str, float] | None] = []
        for sol in sols:
            try:
                metrics.append({k: float(v) for k, v in self.function(sol).items()})
                status.append("ok")
                messages.append("")
            except (DyneconError, ValueError, ArithmeticError) as exc:
                if self.fail_fast:
                    raise
                metrics.append(None)
                status.append("failed")
                messages.append(f"{type(exc).__name__}: {exc}")
        return SweepResult(pts, status, messages, metrics)


def elasticities(
    model: Any,
    params: Sequence[str],
    metric: Callable[[Solution], float],
    rel_change: float = 0.01,
    solve_options: dict[str, Any] | None = None,
    method: str = "auto",
) -> dict[str, float]:
    """Central-difference elasticities ``d ln metric / d ln param`` (needs positive values)."""
    compiled = model if isinstance(model, CompiledModel) else build_model(model).compile()
    base = compiled.parameters.values
    pts: list[dict[str, float]] = []
    for name in params:
        if name not in base:
            raise ValidationError(f"unknown parameter {name!r}")
        pts += [{name: base[name] * (1 - rel_change)}, {name: base[name] * (1 + rel_change)}]
    sweep = ParameterSweep(
        compiled,
        pts,
        lambda sol: {"m": metric(sol)},
        solve_options or {},
        method=method,
        fail_fast=True,
    )
    res = sweep.run()
    out: dict[str, float] = {}
    for i, name in enumerate(params):
        lo, hi = res.metrics[2 * i], res.metrics[2 * i + 1]
        assert lo is not None and hi is not None
        out[name] = float(
            (np.log(hi["m"]) - np.log(lo["m"])) / (np.log(1 + rel_change) - np.log(1 - rel_change))
        )
    return out
