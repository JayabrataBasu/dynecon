"""Core abstractions for Dynecon: errors (and, from P5, the Solution result type)."""

from dynecon.core.errors import (
    BackendError,
    BackendUnavailableError,
    ConvergenceError,
    ConvergenceWarning,
    DeterminacyError,
    DyneconError,
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
    "DyneconError",
    "IdentificationError",
    "ModelSpecificationError",
    "ValidationError",
]
