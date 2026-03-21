"""Compiled model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from econkit.model.specs import Control, EquationSpec, Param, State


@dataclass(frozen=True)
class CompiledModel:
    """Immutable compiled model artifact consumed by solvers."""

    name: str
    parameters: Mapping[str, Param]
    states: Mapping[str, State]
    controls: Mapping[str, Control]
    equations: Mapping[str, EquationSpec]

    @classmethod
    def from_parts(
        cls,
        name: str,
        parameters: dict[str, Param],
        states: dict[str, State],
        controls: dict[str, Control],
        equations: dict[str, EquationSpec],
    ) -> "CompiledModel":
        return cls(
            name=name,
            parameters=MappingProxyType(dict(parameters)),
            states=MappingProxyType(dict(states)),
            controls=MappingProxyType(dict(controls)),
            equations=MappingProxyType(dict(equations)),
        )

    def call_equation(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.equations[name].func(*args, **kwargs)
