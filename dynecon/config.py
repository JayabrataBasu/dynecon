"""Runtime configuration for backend dispatch.

Economics note:
Switching backend must never change the economics; this object only chooses the
numerical engine.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from dynecon import backends
from dynecon.core.errors import BackendUnavailableError, ValidationError

__all__ = ["Config", "config"]


@dataclass
class Config:
    """Global runtime config: active backend (``"auto"`` prefers Numba) and strict modes."""

    active_backend: str = "auto"
    strict_backend: bool = False
    strict_convergence: bool = False

    def set_backend(self, backend: str) -> None:
        """Select the global backend (validated).

        Unknown name -> ``ValidationError``. Known but not installed ->
        ``BackendUnavailableError`` in strict mode, otherwise one warning and NumPy.
        """
        if backend == "auto":
            self.active_backend = "auto"
            return
        if backend not in backends.KNOWN_BACKENDS:
            raise ValidationError(
                f"unknown backend {backend!r}; known backends: {list(backends.KNOWN_BACKENDS)}"
            )
        if not backends.is_available(backend):
            if self.strict_backend:
                raise BackendUnavailableError(f"backend {backend!r} is not installed")
            warnings.warn(f"backend {backend!r} is not installed; using 'numpy'", stacklevel=2)
            backend = "numpy"
        self.active_backend = backend


config = Config()
