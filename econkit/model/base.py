"""Python-native model builder for EconKit.

Economics note:
The builder lets researchers state primitives (parameters, states, controls and the
decision problem) in a readable structure before any numerical solving.
"""

from __future__ import annotations

from typing import Any

from econkit.core.errors import ModelSpecificationError, ValidationError
from econkit.model.compiled import CompiledModel, ModelSpec, compile_spec
from econkit.model.problems import (
    DSGEProblem,
    FiniteDPProblem,
    PricingProblem,
    SavingProblem,
)
from econkit.model.specs import Control, Param, ParameterSet, State

__all__ = ["Model"]

_TEMPLATES = ("saving", "finite_dp", "dsge", "pricing")


class Model:
    """Mutable model builder; :meth:`compile` returns an immutable :class:`CompiledModel`."""

    def __init__(self, name: str):
        self.name = name
        self._parameters: dict[str, Param] = {}
        self._states: dict[str, State] = {}
        self._controls: dict[str, Control] = {}
        self._problem: SavingProblem | FiniteDPProblem | DSGEProblem | PricingProblem | None = None

    def params(self, **kwargs: Param) -> dict[str, Param]:
        """Declare parameters by keyword; returns all parameters declared so far."""
        for key, param in kwargs.items():
            if not isinstance(param, Param):
                raise TypeError("params() values must be Param objects")
            self._parameters[key] = param
        return dict(self._parameters)

    def add_state(self, state: State) -> State:
        if state.name in self._states:
            raise ValidationError(f"Duplicate state name: {state.name}")
        self._states[state.name] = state
        return state

    def get_state(self, name: str) -> State:
        """The declared state called ``name``."""
        try:
            return self._states[name]
        except KeyError:
            raise ValidationError(f"no state named {name!r}; have {list(self._states)}") from None

    def replace_state(self, state: State) -> State:
        """Replace an existing state of the same name (used by reform functions)."""
        self.get_state(state.name)
        self._states[state.name] = state
        return state

    def add_control(self, control: Control) -> Control:
        if control.name in self._controls:
            raise ValidationError(f"Duplicate control name: {control.name}")
        self._controls[control.name] = control
        return control

    def set_problem(self, kind: str, **fields: Any) -> None:
        """Declare the problem: ``"saving"``, ``"finite_dp"``, ``"dsge"`` or ``"pricing"``."""
        try:
            if kind == "saving":
                self._problem = SavingProblem(**fields)
            elif kind == "finite_dp":
                self._problem = FiniteDPProblem(**fields)
            elif kind == "pricing":
                self._problem = PricingProblem(**fields)
            elif kind == "dsge":
                for key in ("variables", "states", "shocks", "log_variables"):
                    if key in fields:
                        fields[key] = tuple(fields[key])
                self._problem = DSGEProblem(**fields)
            else:
                raise ModelSpecificationError(
                    f"unknown problem template {kind!r}; use {_TEMPLATES}"
                )
        except TypeError as exc:
            raise ModelSpecificationError(f"problem {kind!r}: {exc}") from exc

    def copy(self, name: str | None = None) -> Model:
        """Independent copy of the builder (used by reform functions)."""
        new = Model(name or self.name)
        new._parameters = dict(self._parameters)
        new._states = dict(self._states)
        new._controls = dict(self._controls)
        new._problem = self._problem
        return new

    def compile(self) -> CompiledModel:
        """Validate and return an immutable compiled model."""
        spec = ModelSpec(
            self.name, tuple(self._states.values()), tuple(self._controls.values()), self._problem
        )
        return compile_spec(spec, ParameterSet(self._parameters))

    def __repr__(self) -> str:
        return (
            f"Model('{self.name}', params={list(self._parameters)}, states={list(self._states)}, "
            f"controls={list(self._controls)}, problem={type(self._problem).__name__})"
        )
