"""Core abstractions for EconKit: errors (and, from P5, the Solution result type)."""

from econkit.core.errors import (
    BackendError,
    BackendUnavailableError,
    ConvergenceError,
    ConvergenceWarning,
    DeterminacyError,
    EconKitError,
    IdentificationError,
    ModelSpecificationError,
    ValidationError,
)

__all__ = [
    "BackendError",
    "BackendUnavailableError",
    "ConvergenceError",
    "ConvergenceWarning",
    "DeterminacyError",
    "EconKitError",
    "IdentificationError",
    "ModelSpecificationError",
    "ValidationError",
]
