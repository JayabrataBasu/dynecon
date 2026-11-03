"""Solvers: the ``solve`` entry point, registered methods and diagnostics."""

from econkit.solvers.api import solve
from econkit.solvers.registry import SOLVERS, Solver

__all__ = ["SOLVERS", "Solver", "solve"]
