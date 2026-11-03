"""Error hierarchy for Dynecon.

Economics note:
Distinct error types separate user mistakes (bad primitives), numerical failures
(non-convergence) and environment problems (missing backend) so that a failed
solve is never mistaken for an economically meaningful result.
"""

from __future__ import annotations

__all__ = [
    "BackendError",
    "BackendUnavailableError",
    "ConvergenceError",
    "ConvergenceWarning",
    "DeterminacyError",
    "DyneconError",
    "IdentificationError",
    "ModelSpecificationError",
    "ValidationError",
]


class DyneconError(Exception):
    """Base class of every Dynecon exception."""


class ValidationError(DyneconError, ValueError):
    """An argument failed validation (shape, finiteness, range)."""


class ModelSpecificationError(ValidationError):
    """A model is mis-specified (missing fields, invalid shocks, bad callables)."""


class BackendError(DyneconError):
    """A backend problem, e.g. an operation with no implementation in strict mode."""


class BackendUnavailableError(BackendError):
    """The requested backend is known but not installed."""


class ConvergenceError(DyneconError):
    """An iteration did not converge (raised only in strict mode)."""


class DeterminacyError(DyneconError):
    """A linear rational-expectations model has no unique stable solution (Blanchard-Kahn)."""


class IdentificationError(DyneconError):
    """The moment Jacobian is rank-deficient: some parameters are not identified."""


class ConvergenceWarning(UserWarning):
    """An iteration did not converge; the returned result is flagged unconverged."""
