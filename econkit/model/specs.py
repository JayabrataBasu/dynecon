"""Model specification objects for EconKit DSL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Param:
    """Named model parameter.

    Economics note:
    Parameters map structural primitives such as preferences, technology, and
    policy constants into executable model code.
    """

    value: float
    bounds: tuple[float, float] | None = None
    calibrated: bool = False
    description: str = ""


@dataclass(frozen=True)
class AR1Shock:
    """AR(1) shock process descriptor."""

    rho: float
    sigma_e: float
    n_states: int


@dataclass(frozen=True)
class State:
    """State variable descriptor for model DSL."""

    name: str
    bounds: tuple[float, float] | None = None
    grid_type: str = "uniform"
    n_points: int | None = None
    shock: AR1Shock | None = None


@dataclass(frozen=True)
class Control:
    """Control/choice variable descriptor for model DSL."""

    name: str
    bounds: tuple[float | None, float | None] | None = None


@dataclass(frozen=True)
class EquationSpec:
    """Registered model equation function metadata."""

    name: str
    func: Any
