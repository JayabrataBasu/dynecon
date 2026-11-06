"""Core abstractions for Dynecon: errors (and, from P5, the Solution result type)."""

from dynecon.core.errors import (
    BackendError,
    BackendUnavailableError,
    ConvergenceError,
    ConvergenceWarning,
    DeterminacyError,
    DyneconError,
    GridWarning,
    IdentificationError,
    ModelSpecificationError,
    ValidationError,
)

__all__ = [
    "BackendError",
    "BackendUnavailableError",
    "ConvergenceError",
    "ConvergenceWarning",
    "GridWarning",
    "DeterminacyError",
    "DyneconError",
    "IdentificationError",
    "ModelSpecificationError",
    "ValidationError",
]
