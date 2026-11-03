"""Solvers: the ``solve`` entry point, registered methods and diagnostics."""

from dynecon.solvers.api import solve
from dynecon.solvers.registry import SOLVERS, Solver

__all__ = ["SOLVERS", "Solver", "solve"]
