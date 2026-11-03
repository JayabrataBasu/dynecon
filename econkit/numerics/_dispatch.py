"""Backend dispatch infrastructure for numerical kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from econkit.config import config
from econkit.core.errors import BackendError
from econkit.numerics._registry import get


def resolve_backend(operation: str, backend: str | None = None) -> tuple[str, Callable[..., Any]]:
    """Resolve ``(backend_used, implementation)`` for an operation.

    Uses the requested backend if it has an implementation (``"auto"`` prefers Numba).
    Otherwise raises ``BackendError`` in strict mode, or falls back to the NumPy reference.
    """
    selected = backend or config.active_backend
    if selected == "auto":
        selected = "numba" if get(operation, "numba") is not None else "numpy"
    impl = get(operation, selected)
    if impl is not None:
        return selected, impl
    if config.strict_backend:
        raise BackendError(f"no {selected!r} implementation of operation {operation!r}")
    fallback = get(operation, "numpy")
    if fallback is None:
        raise BackendError(f"operation {operation!r} has no NumPy reference implementation")
    return "numpy", fallback
