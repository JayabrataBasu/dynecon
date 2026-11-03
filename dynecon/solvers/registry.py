"""Solver registry, the ``Solver`` protocol and option resolution.

Economics note:
All solution methods (VFI, EGM, perturbation, equilibrium) are interchangeable
behind one call signature, so a model can be re-solved by a different method without
changing user code.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol, TypeVar

from dynecon.core.errors import ValidationError
from dynecon.core.solution import Solution
from dynecon.model.compiled import CompiledModel

__all__ = ["SOLVERS", "Solver", "resolve_config"]

C = TypeVar("C")


class Solver(Protocol):
    """Implemented by every solution method."""

    def __call__(
        self, model: CompiledModel, *, backend: str | None = None, **options: Any
    ) -> Solution: ...


SOLVERS: dict[str, Solver] = {}


def resolve_config(cfg_cls: type[C], options: dict[str, Any]) -> C:
    """Build a frozen config dataclass from keyword options; unknown names are an error."""
    valid = [f.name for f in dataclasses.fields(cfg_cls)]  # type: ignore[arg-type]
    unknown = sorted(set(options) - set(valid))
    if unknown:
        raise ValidationError(f"unknown option(s) {unknown} for {cfg_cls.__name__}; valid: {valid}")
    return cfg_cls(**options)
