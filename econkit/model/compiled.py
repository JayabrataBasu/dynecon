"""Immutable compiled model consumed by solvers.

Economics note:
Compilation turns declarative primitives into arrays (grids, shock chains) and
evaluates every equation once, so a solver starts from a validated, reproducible model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from econkit.core.errors import ModelSpecificationError, ValidationError
from econkit.model.fingerprint import fingerprint_model
from econkit.model.problems import (
    DSGEProblem,
    FiniteDPProblem,
    PricingProblem,
    SavingProblem,
)
from econkit.model.shocks import discretize_shock
from econkit.model.specs import Control, ParameterSet, ParamView, ShockProcess, State
from econkit.model.validation import validate_model
from econkit.numerics.grids import make_grid

__all__ = ["CompiledModel", "ModelSpec", "compile_spec"]

Problem = SavingProblem | FiniteDPProblem | DSGEProblem | PricingProblem


@dataclass(frozen=True)
class ModelSpec:
    """Frozen snapshot of a model builder (everything except parameter values)."""

    name: str
    states: tuple[State, ...]
    controls: tuple[Control, ...]
    problem: Problem | None

    @property
    def state_map(self) -> dict[str, State]:
        return {s.name: s for s in self.states}

    @property
    def control_map(self) -> dict[str, Control]:
        return {c.name: c for c in self.controls}


@dataclass(frozen=True, eq=False)
class CompiledModel:
    """Validated, immutable model: parameters, grids, discretised shocks and a problem."""

    spec: ModelSpec
    parameters: ParameterSet
    grids: Mapping[str, np.ndarray]
    shocks: Mapping[str, ShockProcess]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def problem(self) -> Problem:
        assert self.spec.problem is not None  # guaranteed by compile_spec
        return self.spec.problem

    @property
    def p(self) -> ParamView:
        """Read-only parameter view (``model.p.beta``)."""
        return self.parameters.view

    @property
    def states(self) -> Mapping[str, State]:
        return MappingProxyType(self.spec.state_map)

    @property
    def controls(self) -> Mapping[str, Control]:
        return MappingProxyType(self.spec.control_map)

    def with_params(self, **overrides: float) -> CompiledModel:
        """A *new* compiled model with parameter overrides (the original is unchanged)."""
        return compile_spec(self.spec, self.parameters.replace(**overrides))

    def shock_process(self, name: str | None) -> ShockProcess:
        """Shock process of state ``name``; ``None`` gives the degenerate shock ``z = 1``."""
        if name is None:
            return ShockProcess(np.array([1.0]), np.array([[1.0]]))
        try:
            return self.shocks[name]
        except KeyError:
            raise ValidationError(f"state {name!r} has no shock process") from None

    def fingerprint(self) -> str:
        """SHA-256 over parameters, grids, shocks and equation source text."""
        return fingerprint_model(
            self.name, self.parameters.values, self.grids, self.shocks, self.spec.problem
        )

    def describe(self) -> str:
        """Human-readable summary of the model."""
        lines = [f"CompiledModel '{self.name}' ({type(self.problem).__name__})"]
        lines.append(
            "  parameters: " + ", ".join(f"{k}={v:g}" for k, v in self.parameters.values.items())
        )
        for k, g in self.grids.items():
            lines.append(f"  grid {k}: n={g.shape[0]}, [{g[0]:g}, {g[-1]:g}]")
        for k, s in self.shocks.items():
            lines.append(f"  shock {k}: {s.values.shape[0]} states")
        lines.append(f"  fingerprint: {self.fingerprint()[:12]}")
        return "\n".join(lines)

    __repr__ = describe


def compile_spec(spec: ModelSpec, parameters: ParameterSet) -> CompiledModel:
    """Discretise shocks, build grids, validate, and return an immutable model."""
    errors: list[str] = []
    values = parameters.values
    grids: dict[str, np.ndarray] = {}
    shocks: dict[str, ShockProcess] = {}
    names = [s.name for s in spec.states]
    errors += [
        f"duplicate state name {n!r}" for n in sorted({n for n in names if names.count(n) > 1})
    ]
    for st in spec.states:
        if st.shock is not None:
            try:
                shocks[st.name] = discretize_shock(st.shock, values)
            except ValidationError as exc:
                errors.append(f"state {st.name!r} shock: {exc}")
        elif st.bounds is None or st.n_points is None:
            errors.append(f"state {st.name!r}: needs bounds and n_points (or a shock)")
        else:
            try:
                kw = {"curvature": st.grid_curvature} if st.grid_type == "exp" else {}
                grids[st.name] = make_grid(
                    st.grid_type, st.bounds[0], st.bounds[1], st.n_points, **kw
                )
            except ValidationError as exc:
                errors.append(f"state {st.name!r} grid: {exc}")
    errors += validate_model(spec, parameters, grids, shocks)
    if errors:
        raise ModelSpecificationError(
            f"model {spec.name!r} is invalid ({len(errors)} problem(s)):\n  - "
            + "\n  - ".join(errors)
        )
    return CompiledModel(
        spec=spec,
        parameters=parameters,
        grids=MappingProxyType(grids),
        shocks=MappingProxyType(shocks),
    )
