"""Python-native model DSL for EconKit."""

from __future__ import annotations

from collections.abc import Callable

from econkit.model.compiled import CompiledModel
from econkit.model.specs import Control, EquationSpec, Param, State


class Model:
    """Economic model builder.

    Economics note:
    The DSL lets researchers define primitives (states, controls, shocks,
    parameters, equations) in a readable structure before numerical solving.
    """

    def __init__(self, name: str):
        self.name = name
        self._parameters: dict[str, Param] = {}
        self._states: dict[str, State] = {}
        self._controls: dict[str, Control] = {}
        self._equations: dict[str, EquationSpec] = {}

    def params(self, **kwargs: Param) -> dict[str, Param]:
        for key, param in kwargs.items():
            if not isinstance(param, Param):
                raise TypeError("params() values must be Param objects")
            self._parameters[key] = param
        return dict(self._parameters)

    def add_state(self, state: State) -> State:
        if state.name in self._states:
            raise ValueError(f"Duplicate state name: {state.name}")
        self._states[state.name] = state
        return state

    def add_control(self, control: Control) -> Control:
        if control.name in self._controls:
            raise ValueError(f"Duplicate control name: {control.name}")
        self._controls[control.name] = control
        return control

    def equation(self, name: str) -> Callable[[Callable[..., object]], Callable[..., object]]:
        def _decorate(func: Callable[..., object]) -> Callable[..., object]:
            if name in self._equations:
                raise ValueError(f"Duplicate equation name: {name}")
            self._equations[name] = EquationSpec(name=name, func=func)
            return func

        return _decorate

    def compile(self) -> CompiledModel:
        if not self._equations:
            raise ValueError("Model must define at least one equation before compile()")
        if not self._states:
            raise ValueError("Model must define at least one state before compile()")
        if not self._controls:
            raise ValueError("Model must define at least one control before compile()")
        return CompiledModel.from_parts(
            name=self.name,
            parameters=self._parameters,
            states=self._states,
            controls=self._controls,
            equations=self._equations,
        )
