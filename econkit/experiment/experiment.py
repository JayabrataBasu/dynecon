"""Reproducible experiments: configuration, provenance, results and ``reproduce``.

Economics note:
A result is only scientific if it can be regenerated. An experiment records exactly how a
model was built, solved and simulated, together with the software that produced it, and
can re-run itself and compare numbers.

Reproducibility contract: same machine, library versions, backend and thread count.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from econkit.core.errors import ValidationError
from econkit.core.solution import Solution
from econkit.estimation.moments import model_moments
from econkit.experiment._build import build_model, normalise_build
from econkit.simulation.result import SimulationResult
from econkit.simulation.simulate import simulate
from econkit.solvers.api import solve
from econkit.utils.provenance import collect_provenance
from econkit.utils.serialization import decode, encode, numeric_leaves

__all__ = ["Experiment", "ExperimentResult", "ReproductionReport"]


@dataclass
class ExperimentResult:
    """Outcome of :meth:`Experiment.run`."""

    path: Path
    solution: Solution
    simulation: SimulationResult | None
    moments: dict[str, float]
    result: dict[str, Any]


@dataclass(frozen=True)
class ReproductionReport:
    """Comparison of a re-run against the saved ``result.json``."""

    passed: bool
    differences: list[str]
    n_compared: int


def _summarise(
    sol: Solution, sim: SimulationResult | None, moments: dict[str, float]
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "solution": {
            "method": sol.method,
            "converged": sol.converged,
            "iterations": sol.iterations,
            "residual": sol.residual,
            "policy": {
                k: {
                    "mean": float(np.mean(v)),
                    "norm": float(np.linalg.norm(np.asarray(v, dtype=float))),
                }
                for k, v in sol.policy.items()
            },
        },
        "moments": moments,
    }
    if sim is not None:
        out["simulation"] = {
            k: float(np.mean(v))
            for k, v in sim.paths.items()
            if np.issubdtype(v.dtype, np.floating)
        }
    return out


@dataclass
class Experiment:
    """A named, reproducible solve (and optional simulate / moments) run.

    ``build`` is a built-in model name plus overrides, ``("aiyagari", {"beta": 0.95})``, or an
    importable ``"module:function"``. Models are never serialised as code.
    """

    name: str
    build: Any
    solve_options: dict[str, Any] = field(default_factory=dict)
    simulate_options: dict[str, Any] | None = None
    moments: list[str] | None = None
    outdir: str | Path = "runs"

    def config(self, rng: int) -> dict[str, Any]:
        name, overrides = normalise_build(self.build)
        return {
            "name": self.name,
            "build": [name, overrides],
            "solve_options": self.solve_options,
            "simulate_options": self.simulate_options,
            "moments": self.moments,
            "rng": rng,
        }

    @staticmethod
    def _execute(
        config: dict[str, Any],
    ) -> tuple[Any, Solution, SimulationResult | None, dict[str, float]]:
        model = build_model(config["build"]).compile()
        sol = solve(model, **config["solve_options"])
        sim = None
        moments: dict[str, float] = {}
        if config["simulate_options"] is not None:
            sim = simulate(sol, rng=config["rng"], **config["simulate_options"])
            if config["moments"]:
                vec = model_moments(sim, config["moments"])
                moments = dict(zip(config["moments"], map(float, vec), strict=True))
        elif config["moments"]:
            raise ValidationError("moments need simulate_options")
        return model, sol, sim, moments

    def run(self, rng: int = 0) -> ExperimentResult:
        """Execute and write config, provenance, solution, result and log files to a new run folder."""
        config = self.config(rng)
        model, sol, sim, moments = self._execute(config)
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        run_dir = Path(self.outdir) / self.name / f"{stamp}-{model.fingerprint()[:8]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        result = _summarise(sol, sim, moments)
        prov = collect_provenance() | {"model_fingerprint": model.fingerprint()}
        (run_dir / "config.json").write_text(json.dumps(encode(config), indent=2), encoding="utf-8")
        (run_dir / "provenance.json").write_text(
            json.dumps(encode(prov), indent=2), encoding="utf-8"
        )
        (run_dir / "result.json").write_text(json.dumps(encode(result), indent=2), encoding="utf-8")
        sol.save(run_dir / "solution")
        (run_dir / "log.txt").write_text(sol.summary() + "\n", encoding="utf-8")
        return ExperimentResult(run_dir, sol, sim, moments, result)

    @staticmethod
    def reproduce(path: str | Path, rtol: float = 1e-10) -> ReproductionReport:
        """Re-run the experiment stored in ``path`` and compare with its saved result."""
        run_dir = Path(path)
        config = decode(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
        saved = decode(json.loads((run_dir / "result.json").read_text(encoding="utf-8")))
        prov = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        model, sol, sim, moments = Experiment._execute(config)
        new = decode(encode(_summarise(sol, sim, moments)))
        a, b = numeric_leaves(saved), numeric_leaves(new)
        diffs: list[str] = []
        if model.fingerprint() != prov.get("model_fingerprint"):
            diffs.append("model_fingerprint")
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                diffs.append(f"{key} (missing)")
            elif not (abs(a[key] - b[key]) <= rtol * max(abs(a[key]), abs(b[key])) + 1e-14):
                diffs.append(f"{key}: saved {a[key]!r} vs rerun {b[key]!r}")
        return ReproductionReport(not diffs, diffs, len(set(a) | set(b)))
