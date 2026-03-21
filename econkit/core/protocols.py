"""Protocol-first core abstractions for EconKit."""

from typing import Any, Protocol


class Model(Protocol):
    """Economic model interface consumed by solvers.

    Economics note:
    A model encapsulates preferences, constraints, shocks, and equilibrium
    relationships that define agent decision problems.
    """

    def compile(self) -> Any: ...


class Solver(Protocol):
    """Solver interface for dynamic economic models."""

    def solve(self, model: Model) -> Any: ...


class Solution(Protocol):
    """Serialized output of a model solution routine."""

    metadata: dict[str, Any]


class Simulation(Protocol):
    """Simulation artifact generated from a solved model."""

    def to_dataframe(self) -> Any: ...


class Estimator(Protocol):
    """Estimator interface for structural inference."""

    def fit(self, model: Model, data: Any) -> Any: ...
