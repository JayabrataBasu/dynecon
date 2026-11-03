"""Error hierarchy for EconKit.

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
    "EconKitError",
    "ModelSpecificationError",
    "ValidationError",
]


class EconKitError(Exception):
    """Base class of every EconKit exception."""


class ValidationError(EconKitError, ValueError):
    """An argument failed validation (shape, finiteness, range)."""


class ModelSpecificationError(ValidationError):
    """A model is mis-specified (missing fields, invalid shocks, bad callables)."""


class BackendError(EconKitError):
    """A backend problem, e.g. an operation with no implementation in strict mode."""


class BackendUnavailableError(BackendError):
    """The requested backend is known but not installed."""


class ConvergenceError(EconKitError):
    """An iteration did not converge (raised only in strict mode)."""


class DeterminacyError(EconKitError):
    """A linear rational-expectations model has no unique stable solution (Blanchard-Kahn)."""


class ConvergenceWarning(UserWarning):
    """An iteration did not converge; the returned result is flagged unconverged."""
