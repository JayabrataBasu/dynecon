"""Backend dispatch infrastructure for numerical kernels."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from typing import Any

from dynecon import backends
from dynecon.config import config
from dynecon.core.errors import BackendError, BackendUnavailableError
from dynecon.numerics._registry import get

__all__ = ["ensure_jax", "resolve_backend"]

_jax_state = {"loaded": False, "warned": False}


def ensure_jax() -> bool:
    """Import and register the JAX kernels on first use; ``False`` if JAX is not installed."""
    if _jax_state["loaded"]:
        return True
    if not backends.is_available("jax"):
        return False
    importlib.import_module("dynecon.numerics.backends.jax_kernels")
    _jax_state["loaded"] = True
    return True


def resolve_backend(operation: str, backend: str | None = None) -> tuple[str, Callable[..., Any]]:
    """Resolve ``(backend_used, implementation)`` for an operation.

    Uses the requested backend if it has an implementation (``"auto"`` prefers Numba and
    never JAX). Otherwise raises ``BackendError`` in strict mode, or falls back to the NumPy
    reference. Requesting ``"jax"`` without JAX installed raises
    :class:`BackendUnavailableError` in strict mode and otherwise warns once and uses NumPy.
    """
    selected = backend or config.active_backend
    if selected == "auto":
        selected = "numba" if get(operation, "numba") is not None else "numpy"
    if selected == "jax" and get(operation, "jax") is None and not ensure_jax():
        if config.strict_backend:
            raise BackendUnavailableError("backend 'jax' is not installed")
        if not _jax_state["warned"]:
            _jax_state["warned"] = True
            warnings.warn("backend 'jax' is not installed; using 'numpy'", stacklevel=3)
        selected = "numpy"
    impl = get(operation, selected)
    if impl is not None:
        return selected, impl
    if config.strict_backend:
        raise BackendError(f"no {selected!r} implementation of operation {operation!r}")
    fallback = get(operation, "numpy")
    if fallback is None:
        raise BackendError(f"operation {operation!r} has no NumPy reference implementation")
    return "numpy", fallback
